"""Run V4 public task folders through Claude Code or a mock path."""

from __future__ import annotations

import argparse
import time
from datetime import UTC, datetime
from pathlib import Path

from _path_setup import RESULTS_DIR
from v4_terminal_agent import INPUT_CONDITIONS, MODEL_ALIASES, has_complete_v4_result, list_task_dirs, run_one_task

USAGE_LIMIT_SLEEP_SECONDS = 90 * 60


def _run_id() -> str:
    return datetime.now(UTC).strftime("v4_terminal_%Y%m%dT%H%M%SZ")


def _split_csv(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.split(",") if item.strip())


def _conditions(value: str) -> tuple[str, ...]:
    if value == "all":
        return INPUT_CONDITIONS
    requested = _split_csv(value)
    unknown = sorted(set(requested) - set(INPUT_CONDITIONS))
    if unknown:
        raise SystemExit(f"Unknown input condition(s): {unknown}")
    return requested


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-root", type=Path, required=True, help="Public task root from build_v4_agent_tasks.py.")
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--models", default=",".join(MODEL_ALIASES), help="Comma-separated Claude Code model aliases.")
    parser.add_argument(
        "--conditions",
        default="both_modalities",
        help="Comma-separated conditions or all. Default runs only the primary both_modalities condition.",
    )
    parser.add_argument("--effort", default="high", help="Claude Code effort level. Default: high.")
    parser.add_argument(
        "--max-budget-usd",
        type=float,
        default=None,
        help="Optional Claude Code per-call budget cap. Omit for subscription-auth runs.",
    )
    parser.add_argument("--timeout-seconds", type=int, default=300)
    parser.add_argument("--limit", type=int, default=None, help="Optional per-condition task limit for smoke tests.")
    parser.add_argument(
        "--usage-limit-sleep-seconds",
        type=int,
        default=USAGE_LIMIT_SLEEP_SECONDS,
        help="Seconds to sleep before retrying after a Claude Code usage limit. Default: 5400.",
    )
    parser.add_argument(
        "--no-usage-limit-retry",
        action="store_true",
        help="Exit after a Claude Code usage limit instead of sleeping and resuming.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip task outputs that already have successful metadata and result.json.",
    )
    parser.add_argument("--mock", action="store_true", help="Write schema-valid mock outputs without calling Claude Code.")
    return parser.parse_args()


def _run_pass(
    args: argparse.Namespace,
    output_root: Path,
    models: tuple[str, ...],
    conditions: tuple[str, ...],
    resume_completed: bool,
) -> tuple[int, int, bool]:
    count = 0
    skipped = 0
    for model in models:
        for condition in conditions:
            task_dirs = list_task_dirs(args.task_root, condition=condition, limit=args.limit)
            for task_dir in task_dirs:
                result_dir = output_root / str(model) / condition / task_dir.name
                if resume_completed and has_complete_v4_result(result_dir):
                    skipped += 1
                    print(f"{model}/{condition}/{task_dir.name}: skipped=complete")
                    continue
                metadata = run_one_task(
                    task_dir=task_dir,
                    output_dir=result_dir,
                    model=str(model),
                    condition=condition,
                    effort=str(args.effort),
                    max_budget_usd=args.max_budget_usd,
                    timeout_seconds=int(args.timeout_seconds),
                    mock=bool(args.mock),
                )
                count += 1
                status = "timeout" if metadata.get("timed_out") else metadata.get("returncode")
                usage_limited = bool(metadata.get("usage_limited"))
                suffix = " usage_limited=True" if usage_limited else ""
                print(f"{model}/{condition}/{task_dir.name}: status={status}{suffix}")
                if usage_limited:
                    return count, skipped, True
    return count, skipped, False


def main() -> None:
    args = parse_args()
    output_root = args.output_root or (RESULTS_DIR / "v4" / "runs" / _run_id())
    output_root.mkdir(parents=True, exist_ok=True)
    models = _split_csv(str(args.models))
    conditions = _conditions(str(args.conditions))

    count = 0
    skipped = 0
    passes = 0
    usage_limit_pauses = 0
    while True:
        passes += 1
        pass_count, pass_skipped, usage_limited = _run_pass(
            args=args,
            output_root=output_root,
            models=models,
            conditions=conditions,
            resume_completed=bool(args.resume) or passes > 1,
        )
        count += pass_count
        skipped += pass_skipped
        if not usage_limited:
            break
        if bool(args.no_usage_limit_retry):
            print("Claude Code usage limit detected; exiting because --no-usage-limit-retry was set.")
            break
        sleep_seconds = max(0, int(args.usage_limit_sleep_seconds))
        usage_limit_pauses += 1
        print(
            "Claude Code usage limit detected; "
            f"sleeping {sleep_seconds} seconds before retrying with completed outputs skipped."
        )
        time.sleep(sleep_seconds)

    print(f"wrote V4 terminal-agent outputs: {output_root}")
    print(
        f"invocations={count} skipped={skipped} mock={bool(args.mock)} effort={args.effort} "
        f"passes={passes} usage_limit_pauses={usage_limit_pauses}"
    )


if __name__ == "__main__":
    main()
