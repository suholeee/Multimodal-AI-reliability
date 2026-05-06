"""Summarize V3 agent benchmark run directories.

This script handles run directories whose ``predictions.jsonl`` contains more
than one input condition, which is the default for ``--input-condition all``.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, DefaultDict, Iterable


METRICS_BY_CONDITION = {
    "both_modalities": ("label_correct", "contradiction_correct", "action_correct"),
    "image_only": ("label_correct",),
    "hic_only": ("label_correct",),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "run_roots",
        nargs="+",
        type=Path,
        help="One or more V3 run directories, or directories containing V3 run directories.",
    )
    parser.add_argument(
        "--input-cost-per-mtok",
        type=float,
        default=0.0,
        help="Optional input-token price in dollars per million tokens.",
    )
    parser.add_argument(
        "--output-cost-per-mtok",
        type=float,
        default=0.0,
        help="Optional output-token price in dollars per million tokens.",
    )
    return parser.parse_args()


def prediction_files(root: Path) -> list[Path]:
    if (root / "predictions.jsonl").exists():
        return [root / "predictions.jsonl"]
    return sorted(root.rglob("predictions.jsonl"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def infer_thinking(path: Path) -> str:
    parts = path.parent.parts
    for part in reversed(parts):
        if part.startswith("high_"):
            return "high"
        if part.startswith("none_"):
            return "none"
    return "unknown"


def row_thinking(row: dict[str, Any], path: Path) -> str:
    thinking = str(row.get("thinking", "")).strip()
    if thinking:
        return thinking
    return infer_thinking(path)


def row_prompt_policy(row: dict[str, Any]) -> str:
    return str(row.get("prompt_policy", "baseline")).strip() or "baseline"


def row_tool_level(row: dict[str, Any]) -> str:
    return str(row.get("tool_level", "visual")).strip() or "visual"


def row_model(row: dict[str, Any]) -> str:
    return str(row.get("model", row.get("resolved_model", "unknown"))).strip() or "unknown"


def infer_seed(path: Path) -> int | None:
    match = re.search(r"seed_(\d+)", path.parent.name)
    if not match:
        return None
    return int(match.group(1))


def metric_name(field: str) -> str:
    return {
        "label_correct": "acc",
        "contradiction_correct": "contr",
        "action_correct": "action",
    }[field]


def mean_sd(values: list[float]) -> tuple[float, float]:
    if not values:
        return math.nan, math.nan
    return statistics.mean(values), statistics.pstdev(values)


def fmt_float(value: float) -> str:
    if math.isnan(value):
        return "nan"
    return f"{value:.3f}"


def print_per_run_summary(
    grouped_runs: DefaultDict[tuple[str, str, str, str, str], DefaultDict[str, list[float]]],
    samples_per_run: Counter[tuple[str, str, str, str, str]],
) -> None:
    print("\nPer-run means across run directories")
    for key in sorted(grouped_runs):
        model, thinking, prompt_policy, tool_level, condition = key
        print(f"\n{model} / {thinking} / {prompt_policy} / {tool_level} / {condition}")
        n_runs = len(next(iter(grouped_runs[key].values()))) if grouped_runs[key] else 0
        n_samples = samples_per_run[key]
        for field in METRICS_BY_CONDITION.get(condition, ("label_correct",)):
            name = metric_name(field)
            values = grouped_runs[key].get(name, [])
            mean, sd = mean_sd(values)
            print(f"  {name}: mean={fmt_float(mean)}, sd={fmt_float(sd)}, n_runs={n_runs}, n_samples={n_samples}")


def print_pooled_summary(pooled: DefaultDict[tuple[str, str, str, str, str], DefaultDict[str, list[bool]]]) -> None:
    print("\nPooled sample metrics")
    for key in sorted(pooled):
        model, thinking, prompt_policy, tool_level, condition = key
        print(f"\n{model} / {thinking} / {prompt_policy} / {tool_level} / {condition}")
        for field in METRICS_BY_CONDITION.get(condition, ("label_correct",)):
            name = metric_name(field)
            values = pooled[key].get(name, [])
            if not values:
                continue
            successes = sum(values)
            print(f"  {name}: {successes}/{len(values)} = {successes / len(values):.3f}")


def paired_delta(
    rows_by_key: dict[tuple[str, int, str, str, str, str, str], dict[str, Any]],
    left: tuple[str, str, str, str, str],
    right: tuple[str, str, str, str, str],
    field: str,
) -> None:
    left_model, left_thinking, left_prompt, left_tool, left_condition = left
    right_model, right_thinking, right_prompt, right_tool, right_condition = right
    left_ids = {
        (seed, sample_id)
        for model, seed, sample_id, condition, thinking, prompt_policy, tool_level in rows_by_key
        if (
            model == left_model
            and condition == left_condition
            and thinking == left_thinking
            and prompt_policy == left_prompt
            and tool_level == left_tool
        )
    }
    right_ids = {
        (seed, sample_id)
        for model, seed, sample_id, condition, thinking, prompt_policy, tool_level in rows_by_key
        if (
            model == right_model
            and condition == right_condition
            and thinking == right_thinking
            and prompt_policy == right_prompt
            and tool_level == right_tool
        )
    }
    common = sorted(left_ids & right_ids)
    if not common:
        return

    left_only = right_only = both = neither = 0
    deltas = []
    for seed, sample_id in common:
        left_ok = bool(rows_by_key[(left_model, seed, sample_id, left_condition, left_thinking, left_prompt, left_tool)].get(field))
        right_ok = bool(rows_by_key[(right_model, seed, sample_id, right_condition, right_thinking, right_prompt, right_tool)].get(field))
        deltas.append(int(left_ok) - int(right_ok))
        if left_ok and right_ok:
            both += 1
        elif left_ok:
            left_only += 1
        elif right_ok:
            right_only += 1
        else:
            neither += 1

    mean = statistics.mean(deltas)
    if len(deltas) > 1:
        sd = statistics.stdev(deltas)
        se = sd / math.sqrt(len(deltas))
        low = mean - 1.96 * se
        high = mean + 1.96 * se
    else:
        low = high = math.nan
    print(
        f"  {left_model}/{left_thinking}/{left_prompt}/{left_tool}/{left_condition} - "
        f"{right_model}/{right_thinking}/{right_prompt}/{right_tool}/{right_condition} "
        f"{metric_name(field)}: delta={mean:+.3f}, approx95=[{low:+.3f},{high:+.3f}], "
        f"n={len(common)}, left_only={left_only}, right_only={right_only}, both={both}, neither={neither}"
    )


def print_paired_summaries(rows_by_key: dict[tuple[str, int, str, str, str, str, str], dict[str, Any]]) -> None:
    print("\nPaired deltas on matching seed/sample_id")
    sonnet = "claude-sonnet-4-6"
    paired_delta(
        rows_by_key,
        (sonnet, "high", "baseline", "visual", "both_modalities"),
        (sonnet, "none", "baseline", "visual", "both_modalities"),
        "label_correct",
    )
    paired_delta(
        rows_by_key,
        (sonnet, "high", "baseline", "visual", "both_modalities"),
        (sonnet, "none", "baseline", "visual", "both_modalities"),
        "contradiction_correct",
    )
    paired_delta(
        rows_by_key,
        (sonnet, "high", "baseline", "visual", "both_modalities"),
        (sonnet, "none", "baseline", "visual", "both_modalities"),
        "action_correct",
    )
    paired_delta(
        rows_by_key,
        (sonnet, "none", "evidence_separation", "visual", "both_modalities"),
        (sonnet, "none", "baseline", "visual", "both_modalities"),
        "label_correct",
    )
    paired_delta(
        rows_by_key,
        (sonnet, "none", "evidence_separation", "visual", "both_modalities"),
        (sonnet, "none", "baseline", "visual", "both_modalities"),
        "contradiction_correct",
    )
    paired_delta(
        rows_by_key,
        (sonnet, "none", "evidence_separation", "visual", "both_modalities"),
        (sonnet, "none", "baseline", "visual", "both_modalities"),
        "action_correct",
    )
    paired_delta(
        rows_by_key,
        (sonnet, "none", "evidence_separation", "scientific", "both_modalities"),
        (sonnet, "none", "evidence_separation", "visual", "both_modalities"),
        "label_correct",
    )
    paired_delta(
        rows_by_key,
        (sonnet, "none", "evidence_separation", "scientific", "both_modalities"),
        (sonnet, "none", "evidence_separation", "visual", "both_modalities"),
        "contradiction_correct",
    )
    paired_delta(
        rows_by_key,
        (sonnet, "none", "evidence_separation", "scientific", "both_modalities"),
        (sonnet, "none", "evidence_separation", "visual", "both_modalities"),
        "action_correct",
    )
    paired_delta(
        rows_by_key,
        (sonnet, "none", "baseline", "visual", "both_modalities"),
        (sonnet, "none", "baseline", "visual", "image_only"),
        "label_correct",
    )
    paired_delta(
        rows_by_key,
        (sonnet, "none", "baseline", "visual", "both_modalities"),
        (sonnet, "none", "baseline", "visual", "hic_only"),
        "label_correct",
    )
    paired_delta(
        rows_by_key,
        (sonnet, "none", "baseline", "visual", "hic_only"),
        (sonnet, "none", "baseline", "visual", "image_only"),
        "label_correct",
    )


def print_confusion_matrices(
    contradiction_cm: DefaultDict[tuple[str, str, str, str], Counter[tuple[str, str]]],
    action_cm: DefaultDict[tuple[str, str, str, str], Counter[tuple[str, str]]],
) -> None:
    print("\nBoth-modality contradiction confusion")
    for model, thinking, prompt_policy, tool_level in sorted(contradiction_cm):
        print(f"\n{model} / {thinking} / {prompt_policy} / {tool_level}")
        for key, value in sorted(contradiction_cm[(model, thinking, prompt_policy, tool_level)].items()):
            print(f"  true={key[0]} pred={key[1]}: {value}")

    print("\nBoth-modality action confusion")
    for model, thinking, prompt_policy, tool_level in sorted(action_cm):
        print(f"\n{model} / {thinking} / {prompt_policy} / {tool_level}")
        for key, value in sorted(action_cm[(model, thinking, prompt_policy, tool_level)].items()):
            print(f"  true={key[0]} pred={key[1]}: {value}")


def print_modality_separation_diagnostics(
    agreement_counts: DefaultDict[tuple[str, str, str, str, str], Counter[str]],
    agreement_by_true: DefaultDict[tuple[str, str, str, str, str], Counter[tuple[str, str]]],
    modality_label_correct: DefaultDict[tuple[str, str, str, str, str], Counter[str]],
    modality_label_total: DefaultDict[tuple[str, str, str, str, str], Counter[str]],
    modality_label_disagreement: DefaultDict[tuple[str, str, str, str, str], Counter[tuple[str, bool]]],
) -> None:
    print("\nModality-separation diagnostics")
    for key in sorted(agreement_counts):
        model, thinking, prompt_policy, tool_level, condition = key
        if condition != "both_modalities":
            continue
        print(f"\n{model} / {thinking} / {prompt_policy} / {tool_level} / {condition}")
        print("  modality_agreement counts:")
        for value, count in sorted(agreement_counts[key].items()):
            print(f"    {value}: {count}")
        print("  modality_agreement by true contradiction:")
        for (true_status, agreement), count in sorted(agreement_by_true[key].items()):
            print(f"    true={true_status} agreement={agreement}: {count}")
        for field in ("image_predicted_label", "hic_predicted_label"):
            total = modality_label_total[key][field]
            if total:
                correct = modality_label_correct[key][field]
                print(f"  {field} accuracy: {correct}/{total} = {correct / total:.3f}")
        print("  image/hic predicted-label disagreement by true contradiction:")
        for (true_status, disagreed), count in sorted(modality_label_disagreement[key].items()):
            label = "disagree" if disagreed else "same_or_unscored"
            print(f"    true={true_status} {label}: {count}")


def print_usage(
    usage: DefaultDict[tuple[str, str, str, str, str], Counter[str]],
    input_cost_per_mtok: float,
    output_cost_per_mtok: float,
) -> None:
    print("\nToken usage")
    total_input = 0
    total_output = 0
    total_cost = 0.0
    for key in sorted(usage):
        model, thinking, prompt_policy, tool_level, condition = key
        input_tokens = usage[key]["input_tokens"]
        output_tokens = usage[key]["output_tokens"]
        total_input += input_tokens
        total_output += output_tokens
        cost = (input_tokens / 1_000_000) * input_cost_per_mtok + (output_tokens / 1_000_000) * output_cost_per_mtok
        total_cost += cost
        suffix = f", cost=${cost:.2f}" if input_cost_per_mtok or output_cost_per_mtok else ""
        print(f"  {model} / {thinking} / {prompt_policy} / {tool_level} / {condition}: input={input_tokens}, output={output_tokens}{suffix}")
    if input_cost_per_mtok or output_cost_per_mtok:
        print(f"  total: input={total_input}, output={total_output}, cost=${total_cost:.2f}")


def summarize(paths: Iterable[Path], input_cost_per_mtok: float, output_cost_per_mtok: float) -> None:
    grouped_runs: DefaultDict[tuple[str, str, str, str, str], DefaultDict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    pooled: DefaultDict[tuple[str, str, str, str, str], DefaultDict[str, list[bool]]] = defaultdict(lambda: defaultdict(list))
    samples_per_run: Counter[tuple[str, str, str, str, str]] = Counter()
    contradiction_cm: DefaultDict[tuple[str, str, str, str], Counter[tuple[str, str]]] = defaultdict(Counter)
    action_cm: DefaultDict[tuple[str, str, str, str], Counter[tuple[str, str]]] = defaultdict(Counter)
    agreement_counts: DefaultDict[tuple[str, str, str, str, str], Counter[str]] = defaultdict(Counter)
    agreement_by_true: DefaultDict[tuple[str, str, str, str, str], Counter[tuple[str, str]]] = defaultdict(Counter)
    modality_label_correct: DefaultDict[tuple[str, str, str, str, str], Counter[str]] = defaultdict(Counter)
    modality_label_total: DefaultDict[tuple[str, str, str, str, str], Counter[str]] = defaultdict(Counter)
    modality_label_disagreement: DefaultDict[tuple[str, str, str, str, str], Counter[tuple[str, bool]]] = defaultdict(Counter)
    rows_by_key: dict[tuple[str, int, str, str, str, str, str], dict[str, Any]] = {}
    usage: DefaultDict[tuple[str, str, str, str, str], Counter[str]] = defaultdict(Counter)

    for path in paths:
        seed = infer_seed(path)
        rows_by_run_group: DefaultDict[tuple[str, str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
        for row in load_jsonl(path):
            model = row_model(row)
            thinking = row_thinking(row, path)
            prompt_policy = row_prompt_policy(row)
            tool_level = row_tool_level(row)
            condition = str(row.get("input_condition", "both_modalities"))
            key = (model, thinking, prompt_policy, tool_level, condition)
            rows_by_run_group[key].append(row)
            for field in METRICS_BY_CONDITION.get(condition, ("label_correct",)):
                pooled[key][metric_name(field)].append(bool(row.get(field)))
            usage[key]["input_tokens"] += int(row.get("input_tokens") or 0)
            usage[key]["output_tokens"] += int(row.get("output_tokens") or 0)
            if seed is not None:
                rows_by_key[(model, seed, str(row["sample_id"]), condition, thinking, prompt_policy, tool_level)] = row
            if condition == "both_modalities":
                contradiction_cm[(model, thinking, prompt_policy, tool_level)][
                    (str(row.get("true_contradiction_status")), str(row.get("predicted_contradiction_status")))
                ] += 1
                action_cm[(model, thinking, prompt_policy, tool_level)][
                    (str(row.get("true_recommended_action")), str(row.get("predicted_recommended_action")))
                ] += 1
                agreement = str(row.get("modality_agreement", "not_reported"))
                agreement_counts[key][agreement] += 1
                agreement_by_true[key][(str(row.get("true_contradiction_status")), agreement)] += 1
                image_label = str(row.get("image_predicted_label", "not_reported"))
                hic_label = str(row.get("hic_predicted_label", "not_reported"))
                true_label = str(row.get("true_label"))
                for field, predicted in (
                    ("image_predicted_label", image_label),
                    ("hic_predicted_label", hic_label),
                ):
                    if predicted in {"normal", "cancer"}:
                        modality_label_total[key][field] += 1
                        modality_label_correct[key][field] += int(predicted == true_label)
                comparable = image_label in {"normal", "cancer"} and hic_label in {"normal", "cancer"}
                modality_label_disagreement[key][
                    (str(row.get("true_contradiction_status")), bool(comparable and image_label != hic_label))
                ] += 1

        for key, rows in rows_by_run_group.items():
            condition = key[4]
            samples_per_run[key] += len(rows)
            for field in METRICS_BY_CONDITION.get(condition, ("label_correct",)):
                grouped_runs[key][metric_name(field)].append(sum(bool(row.get(field)) for row in rows) / len(rows))

    print_per_run_summary(grouped_runs, samples_per_run)
    print_pooled_summary(pooled)
    print_paired_summaries(rows_by_key)
    print_confusion_matrices(contradiction_cm, action_cm)
    print_modality_separation_diagnostics(
        agreement_counts,
        agreement_by_true,
        modality_label_correct,
        modality_label_total,
        modality_label_disagreement,
    )
    print_usage(usage, input_cost_per_mtok, output_cost_per_mtok)


def main() -> None:
    args = parse_args()
    paths: list[Path] = []
    for root in args.run_roots:
        paths.extend(prediction_files(root))
    if not paths:
        roots = ", ".join(str(root) for root in args.run_roots)
        raise SystemExit(f"No predictions.jsonl files found under {roots}")
    roots = ", ".join(str(root) for root in args.run_roots)
    print(f"Found {len(paths)} prediction files under {roots}")
    summarize(paths, args.input_cost_per_mtok, args.output_cost_per_mtok)


if __name__ == "__main__":
    main()
