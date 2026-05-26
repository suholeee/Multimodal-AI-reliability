"""Build public V4 Claude Code task folders and a hidden evaluator manifest."""

from __future__ import annotations

import argparse
import tempfile
from datetime import UTC, datetime
from pathlib import Path

from _path_setup import RESULTS_DIR
from v4_terminal_agent import build_v4_task_set


PROFILE_CONFIGS = {
    "smoke": {"samples_per_status": 2, "mode": "debug", "image_size": 64},
    "compact_final": {"samples_per_status": 30, "mode": "full", "image_size": 64},
}


def _run_id(profile: str) -> str:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return f"v4_{profile}_{timestamp}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=tuple(PROFILE_CONFIGS), default="smoke")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--public-root", type=Path, default=None)
    parser.add_argument("--evaluator-root", type=Path, default=None)
    parser.add_argument(
        "--conditions",
        default="all",
        help="Comma-separated conditions to export, or all. Default exports both_modalities, image_only, and hic_only.",
    )
    parser.add_argument("--samples-per-status", type=int, default=None)
    parser.add_argument("--seed-start", type=int, default=101)
    parser.add_argument("--max-seeds", type=int, default=250)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    profile = PROFILE_CONFIGS[args.profile]
    run_id = args.run_id or _run_id(args.profile)
    public_root = args.public_root or (Path(tempfile.gettempdir()) / "v4_public_tasks" / run_id)
    evaluator_root = args.evaluator_root or (RESULTS_DIR / "v4" / "evaluator" / run_id)
    samples_per_status = int(args.samples_per_status or profile["samples_per_status"])

    records = build_v4_task_set(
        public_root=public_root,
        evaluator_root=evaluator_root,
        run_id=run_id,
        samples_per_status=samples_per_status,
        mode=str(profile["mode"]),
        image_size=int(profile["image_size"]),
        seed_start=int(args.seed_start),
        max_seeds=int(args.max_seeds),
        conditions=str(args.conditions),
    )
    print(f"wrote V4 public tasks: {public_root}")
    print(f"wrote V4 hidden evaluator manifest: {evaluator_root / 'hidden_manifest.jsonl'}")
    print(f"samples={len(records)} samples_per_status={samples_per_status} conditions={args.conditions}")


if __name__ == "__main__":
    main()
