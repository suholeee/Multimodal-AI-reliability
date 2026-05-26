"""Evaluate V4 Claude Code terminal-agent outputs against a hidden manifest."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
from pathlib import Path

from _path_setup import RESULTS_DIR
from v4_terminal_agent import evaluate_v4_outputs


def _run_id() -> str:
    return datetime.now(UTC).strftime("v4_eval_%Y%m%dT%H%M%SZ")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_roots", nargs="+", type=Path, help="V4 run roots or per-task result directories.")
    parser.add_argument("--hidden-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir or (RESULTS_DIR / "v4" / "evaluations" / _run_id())
    result = evaluate_v4_outputs(
        run_roots=args.run_roots,
        hidden_manifest=args.hidden_manifest,
        output_dir=output_dir,
    )
    print(f"wrote V4 evaluation: {output_dir}")
    print(f"n_scored={result['n_scored']}")
    for row in result["summary_rows"]:
        print(
            f"{row['model']} / {row['input_condition']}: "
            f"n={row['n']} acc={float(row['classification_accuracy']):.3f} "
            f"contradiction_acc={float(row['contradiction_accuracy']):.3f} "
            f"action_acc={float(row['recommended_action_accuracy']):.3f}"
        )


if __name__ == "__main__":
    main()
