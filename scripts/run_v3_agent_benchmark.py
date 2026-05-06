"""Run the V3 Claude agent benchmark in mock or explicitly confirmed API mode."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Dict, Iterable, Sequence

from _path_setup import RESULTS_DIR

from v3_agent_dataset import build_v3_agent_samples
from v3_agent_metrics import (
    safe_normalize_agent_output,
    score_agent_output,
    summarize_agent_records,
)
from v3_claude_client import (
    CLAUDE_HIGH_MODEL_SET,
    DEFAULT_CLAUDE_MODEL,
    INPUT_CONDITIONS,
    PROMPT_POLICIES,
    TOOL_LEVELS,
    ClaudeRunConfig,
    run_claude_agent,
)
from v3_visual_tools import V3VisualToolbox


PROFILE_CONFIGS = {
    "smoke": {
        "mode": "debug",
        "samples_per_strength": {0.0: 2, 0.5: 2, 1.0: 2},
        "image_size": 64,
    },
    "pilot": {
        "mode": "full",
        "samples_per_strength": {0.0: 10, 0.5: 20, 1.0: 20},
        "image_size": 64,
    },
}


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=tuple(PROFILE_CONFIGS), default="smoke")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=21)
    parser.add_argument("--model", default=DEFAULT_CLAUDE_MODEL)
    parser.add_argument("--model-set", choices=("single", "claude_high"), default="single")
    parser.add_argument(
        "--input-condition",
        choices=(*INPUT_CONDITIONS, "all"),
        default="all",
        help="Evidence condition to run. Default runs image_only, hic_only, and both_modalities.",
    )
    parser.add_argument(
        "--prompt-policy",
        choices=(*PROMPT_POLICIES, "all"),
        default="baseline",
        help="Prompt policy to run. Use evidence_separation to force per-modality evidence before reconciliation.",
    )
    parser.add_argument(
        "--tool-level",
        choices=TOOL_LEVELS,
        default="visual",
        help="Tool level to expose. scientific includes visual tools plus fixed public feature tools.",
    )
    parser.add_argument("--thinking", choices=("none", "high"), default="high")
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Ignored for thinking-enabled calls, which Anthropic requires to use temperature 1.0.",
    )
    parser.add_argument("--thinking-budget-tokens", type=int, default=2048)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--max-tool-rounds", type=int, default=4)
    parser.add_argument(
        "--max-workers",
        type=int,
        default=1,
        help="Number of samples to query concurrently per model/prompt/input group. Use a small value to avoid rate limits.",
    )
    parser.add_argument("--mock", action="store_true", help="Force mock mode. This is the default unless --real-api is set.")
    parser.add_argument("--real-api", action="store_true", help="Enable real Claude API calls.")
    parser.add_argument(
        "--confirm-api-call",
        action="store_true",
        help="Required together with --real-api to make provider API calls.",
    )
    return parser.parse_args()


def _run_id(profile: str) -> str:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return f"{timestamp}_{profile}"


def _models_from_args(args: argparse.Namespace) -> Sequence[str]:
    if args.model_set == "claude_high":
        return CLAUDE_HIGH_MODEL_SET
    return (str(args.model),)


def _input_conditions_from_args(args: argparse.Namespace) -> Sequence[str]:
    if args.input_condition == "all":
        return INPUT_CONDITIONS
    return (str(args.input_condition),)


def _prompt_policies_from_args(args: argparse.Namespace) -> Sequence[str]:
    if args.prompt_policy == "all":
        return PROMPT_POLICIES
    return (str(args.prompt_policy),)


def _write_jsonl(path: Path, rows: Iterable[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def _write_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames = [
        "model",
        "prompt_policy",
        "tool_level",
        "input_condition",
        "contradiction_strength",
        "n",
        "classification_accuracy",
        "contradiction_accuracy",
        "contradiction_binary_f1",
        "recommended_action_accuracy",
        "malformed_fraction",
        "abstention_rate",
        "ece_10bin",
        "mean_latency_seconds",
        "mean_tool_call_count",
        "total_input_tokens",
        "total_output_tokens",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def _flat_summary_rows(scored_rows: Sequence[Dict[str, object]]) -> list[Dict[str, object]]:
    """Build flat summary rows by model and contradiction strength."""
    grouped: Dict[tuple[str, str, str, str, str], list[Dict[str, object]]] = defaultdict(list)
    for row in scored_rows:
        input_condition = str(row["input_condition"])
        prompt_policy = str(row["prompt_policy"])
        tool_level = str(row["tool_level"])
        grouped[(str(row["model"]), prompt_policy, tool_level, input_condition, "all")].append(row)
        grouped[(str(row["model"]), prompt_policy, tool_level, input_condition, f"{float(row['contradiction_strength']):.2f}")].append(row)

    summary_rows = []
    for (model, prompt_policy, tool_level, input_condition, strength), rows in sorted(grouped.items()):
        summary = summarize_agent_records(rows)
        summary_rows.append(
            {
                "model": model,
                "prompt_policy": prompt_policy,
                "tool_level": tool_level,
                "input_condition": input_condition,
                "contradiction_strength": strength,
                "n": summary["n"],
                "classification_accuracy": summary.get("classification_accuracy"),
                "contradiction_accuracy": summary.get("contradiction_accuracy"),
                "contradiction_binary_f1": summary.get("contradiction_binary_f1"),
                "recommended_action_accuracy": summary.get("recommended_action_accuracy"),
                "malformed_fraction": summary.get("malformed_fraction"),
                "abstention_rate": summary.get("abstention_rate"),
                "ece_10bin": summary.get("ece_10bin"),
                "mean_latency_seconds": summary.get("mean_latency_seconds"),
                "mean_tool_call_count": summary.get("mean_tool_call_count"),
                "total_input_tokens": summary.get("total_input_tokens"),
                "total_output_tokens": summary.get("total_output_tokens"),
            }
        )
    return summary_rows


def _write_report(path: Path, args: argparse.Namespace, summary_rows: Sequence[Dict[str, object]]) -> None:
    lines = [
        "V3 agent benchmark report",
        f"profile={args.profile}",
        f"real_api={bool(args.real_api)}",
        f"model_set={args.model_set}",
        f"input_condition={args.input_condition}",
        f"prompt_policy={args.prompt_policy}",
        f"tool_level={args.tool_level}",
        f"thinking={args.thinking}",
        f"max_workers={args.max_workers}",
        "",
        "Summary rows",
    ]
    for row in summary_rows:
        lines.append(
            "model={model} prompt={prompt_policy} tool={tool_level} input={input_condition} cs={cs} n={n} "
            "acc={acc:.3f} contradiction_acc={cacc:.3f} action_acc={aacc:.3f}".format(
                model=row["model"],
                prompt_policy=row["prompt_policy"],
                tool_level=row["tool_level"],
                input_condition=row["input_condition"],
                cs=row["contradiction_strength"],
                n=int(row["n"]),
                acc=float(row["classification_accuracy"]),
                cacc=float(row["contradiction_accuracy"]),
                aacc=float(row["recommended_action_accuracy"]),
            )
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _score_one_sample(
    sample,
    toolbox: V3VisualToolbox,
    config: ClaudeRunConfig,
    model: str,
    input_condition: str,
    prompt_policy: str,
    thinking: str,
    tool_level: str,
) -> tuple[Dict[str, object], Dict[str, object]]:
    """Run and score one sample for a fixed benchmark cell."""
    result = run_claude_agent(sample=sample, toolbox=toolbox, config=config)
    normalized = safe_normalize_agent_output(result.raw_output)
    scored = score_agent_output(
        sample=sample,
        normalized_output=normalized,
        model=model,
        raw_output=result.raw_output,
        resolved_model=result.resolved_model,
        input_condition=input_condition,
        prompt_policy=prompt_policy,
        thinking=thinking,
        tool_level=tool_level,
        latency_seconds=result.latency_seconds,
        input_tokens=result.input_tokens,
        output_tokens=result.output_tokens,
        tool_call_count=result.tool_call_count,
    )
    return scored, {**scored, "public_sample": sample.public_dict()}


def _score_sample_batch(
    samples,
    toolbox: V3VisualToolbox,
    config: ClaudeRunConfig,
    model: str,
    input_condition: str,
    prompt_policy: str,
    thinking: str,
    tool_level: str,
    max_workers: int,
) -> list[tuple[Dict[str, object], Dict[str, object]]]:
    """Run samples sequentially or with bounded concurrency while preserving output order."""
    worker_count = max(1, min(int(max_workers), len(samples)))
    if worker_count == 1:
        return [
            _score_one_sample(
                sample=sample,
                toolbox=toolbox,
                config=config,
                model=model,
                input_condition=input_condition,
                prompt_policy=prompt_policy,
                thinking=thinking,
                tool_level=tool_level,
            )
            for sample in samples
        ]

    results: list[tuple[Dict[str, object], Dict[str, object]] | None] = [None] * len(samples)
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = {
            executor.submit(
                _score_one_sample,
                sample,
                toolbox,
                config,
                model,
                input_condition,
                prompt_policy,
                thinking,
                tool_level,
            ): idx
            for idx, sample in enumerate(samples)
        }
        for future in as_completed(futures):
            results[futures[future]] = future.result()

    return [result for result in results if result is not None]


def main() -> None:
    """Run the V3 benchmark."""
    args = parse_args()
    if args.real_api and not args.confirm_api_call:
        raise SystemExit("Refusing real API calls without --confirm-api-call.")

    profile = PROFILE_CONFIGS[args.profile]
    output_dir = args.output_dir or (RESULTS_DIR / "v3" / "runs" / _run_id(args.profile))
    output_dir.mkdir(parents=True, exist_ok=True)

    samples = build_v3_agent_samples(
        output_dir=output_dir,
        samples_per_strength=profile["samples_per_strength"],
        mode=profile["mode"],
        image_size=int(profile["image_size"]),
        seed=int(args.seed),
    )
    toolbox = V3VisualToolbox(samples=samples, work_dir=output_dir / "tool_outputs")
    models = _models_from_args(args)
    input_conditions = _input_conditions_from_args(args)
    prompt_policies = _prompt_policies_from_args(args)
    scored_rows = []
    prediction_rows = []

    for model in models:
        for prompt_policy in prompt_policies:
            for input_condition in input_conditions:
                config = ClaudeRunConfig(
                    model=model,
                    max_tokens=int(args.max_tokens),
                    temperature=float(args.temperature),
                    thinking=str(args.thinking),
                    thinking_budget_tokens=int(args.thinking_budget_tokens),
                    max_tool_rounds=int(args.max_tool_rounds),
                    real_api=bool(args.real_api),
                    confirm_api_call=bool(args.confirm_api_call),
                    input_condition=input_condition,
                    prompt_policy=prompt_policy,
                    tool_level=str(args.tool_level),
                )
                for scored, prediction in _score_sample_batch(
                    samples=samples,
                    toolbox=toolbox,
                    config=config,
                    model=model,
                    input_condition=input_condition,
                    prompt_policy=prompt_policy,
                    thinking=str(args.thinking),
                    tool_level=str(args.tool_level),
                    max_workers=int(args.max_workers),
                ):
                    scored_rows.append(scored)
                    prediction_rows.append(prediction)

    summary_rows = _flat_summary_rows(scored_rows)
    _write_jsonl(output_dir / "predictions.jsonl", prediction_rows)
    _write_jsonl(output_dir / "summary.jsonl", summary_rows)
    _write_csv(output_dir / "summary.csv", summary_rows)
    _write_report(output_dir / "report.txt", args=args, summary_rows=summary_rows)

    print(f"wrote V3 benchmark run: {output_dir}")
    for row in summary_rows:
        if row["contradiction_strength"] == "all":
            print(
                f"{row['model']} [{row['prompt_policy']}/{row['tool_level']}/{row['input_condition']}]: "
                f"n={row['n']} acc={float(row['classification_accuracy']):.3f} "
                f"contradiction_acc={float(row['contradiction_accuracy']):.3f} "
                f"action_acc={float(row['recommended_action_accuracy']):.3f}"
            )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
