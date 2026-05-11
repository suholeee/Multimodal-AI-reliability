"""Tests for the V4 Claude Code terminal-agent evaluation harness."""

from __future__ import annotations

import json
import subprocess
import sys
from collections import Counter
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from v4_terminal_agent import (  # noqa: E402
    audit_public_task_root,
    build_v4_task_set,
    claude_code_command,
    evaluate_v4_outputs,
    extract_agent_payload,
    is_claude_usage_limited,
    load_hidden_manifest,
)


def test_v4_task_builder_exports_stratified_public_tasks(tmp_path: Path) -> None:
    public_root = tmp_path / "public"
    evaluator_root = tmp_path / "evaluator"
    records = build_v4_task_set(
        public_root=public_root,
        evaluator_root=evaluator_root,
        run_id="test_v4",
        samples_per_status=2,
        mode="debug",
        seed_start=101,
        max_seeds=40,
        conditions="all",
    )
    assert len(records) == 6
    assert (evaluator_root / "hidden_manifest.jsonl").exists()
    assert (public_root / "public_manifest.jsonl").exists()
    assert audit_public_task_root(public_root)["finding_count"] == 0

    by_status_label = Counter((row.true_contradiction_status, row.true_label_name) for row in records)
    assert by_status_label == {
        ("none", "normal"): 1,
        ("none", "cancer"): 1,
        ("weak", "normal"): 1,
        ("weak", "cancer"): 1,
        ("strong", "normal"): 1,
        ("strong", "cancer"): 1,
    }
    both_task = public_root / "both_modalities" / records[0].sample_id
    assert (both_task / "polymer_image.png").exists()
    assert (both_task / "hic_contact_map.png").exists()
    assert not (both_task / "evidence_panel.png").exists()
    image_task = public_root / "image_only" / records[0].sample_id
    assert (image_task / "polymer_image.png").exists()
    assert not (image_task / "hic_contact_map.png").exists()
    hic_task = public_root / "hic_only" / records[0].sample_id
    assert not (hic_task / "polymer_image.png").exists()
    assert (hic_task / "hic_contact_map.png").exists()


def test_v4_claude_command_uses_subscription_auth_mode_and_high_effort(tmp_path: Path) -> None:
    public_root = tmp_path / "public"
    evaluator_root = tmp_path / "evaluator"
    records = build_v4_task_set(
        public_root=public_root,
        evaluator_root=evaluator_root,
        run_id="test_command",
        samples_per_status=2,
        mode="debug",
        seed_start=102,
        max_seeds=40,
        conditions="both_modalities",
    )
    task_dir = public_root / "both_modalities" / records[0].sample_id
    command = claude_code_command(task_dir, model="sonnet", effort="high", max_budget_usd=0.25)
    assert "--bare" not in command
    assert command[command.index("--model") + 1] == "sonnet"
    assert command[command.index("--effort") + 1] == "high"
    assert command[command.index("--output-format") + 1] == "json"
    assert command[command.index("--max-budget-usd") + 1] == "0.2500"
    assert "Read,Bash" in command


def test_v4_claude_command_omits_budget_cap_by_default(tmp_path: Path) -> None:
    public_root = tmp_path / "public"
    evaluator_root = tmp_path / "evaluator"
    records = build_v4_task_set(
        public_root=public_root,
        evaluator_root=evaluator_root,
        run_id="test_command_no_budget",
        samples_per_status=2,
        mode="debug",
        seed_start=102,
        max_seeds=40,
        conditions="both_modalities",
    )
    task_dir = public_root / "both_modalities" / records[0].sample_id
    command = claude_code_command(task_dir, model="sonnet", effort="high")
    assert "--max-budget-usd" not in command


def test_v4_extract_agent_payload_handles_claude_wrapper() -> None:
    payload = {
        "predicted_label": "normal",
        "label_confidence": 0.6,
        "contradiction_status": "none",
        "trusted_evidence": "both",
        "recommended_action": "use_both",
        "image_evidence": "image evidence",
        "hic_evidence": "hic evidence",
        "image_predicted_label": "normal",
        "hic_predicted_label": "normal",
        "modality_agreement": "agree",
        "rationale": "visible evidence agrees",
    }
    wrapped = {"type": "result", "result": json.dumps(payload)}
    assert extract_agent_payload(json.dumps(wrapped)) == payload


def test_v4_extract_agent_payload_handles_structured_output_wrapper() -> None:
    payload = {
        "predicted_label": "cancer",
        "label_confidence": 0.65,
        "contradiction_status": "none",
        "trusted_evidence": "both",
        "recommended_action": "use_both",
        "image_evidence": "image evidence",
        "hic_evidence": "hic evidence",
        "image_predicted_label": "cancer",
        "hic_predicted_label": "cancer",
        "modality_agreement": "agree",
        "rationale": "visible evidence agrees",
    }
    wrapped = {
        "type": "result",
        "result": "Plain wrapper text is not the benchmark payload.",
        "structured_output": payload,
    }
    assert extract_agent_payload(json.dumps(wrapped)) == payload


def test_v4_detects_claude_usage_limit_output() -> None:
    usage_limited = {
        "type": "result",
        "is_error": True,
        "result": "Claude AI usage limit reached. Your limit will reset later.",
    }
    connection_error = {
        "type": "result",
        "is_error": True,
        "result": "API Error: Unable to connect to API (ConnectionRefused)",
    }
    assert is_claude_usage_limited(json.dumps(usage_limited))
    assert not is_claude_usage_limited(json.dumps(connection_error))


def test_v4_mock_runner_and_evaluator_cli(tmp_path: Path) -> None:
    public_root = tmp_path / "public"
    evaluator_root = tmp_path / "evaluator"
    output_root = tmp_path / "runs"
    eval_root = tmp_path / "eval"

    build = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "build_v4_agent_tasks.py"),
            "--profile",
            "smoke",
            "--run-id",
            "test_cli",
            "--public-root",
            str(public_root),
            "--evaluator-root",
            str(evaluator_root),
            "--conditions",
            "all",
            "--max-seeds",
            "40",
        ],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert "samples=6" in build.stdout

    run = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "run_v4_terminal_agent_benchmark.py"),
            "--task-root",
            str(public_root),
            "--output-root",
            str(output_root),
            "--models",
            "haiku,sonnet,opus",
            "--conditions",
            "both_modalities",
            "--effort",
            "high",
            "--limit",
            "2",
            "--mock",
        ],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert "invocations=6" in run.stdout

    resume_run = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "run_v4_terminal_agent_benchmark.py"),
            "--task-root",
            str(public_root),
            "--output-root",
            str(output_root),
            "--models",
            "haiku,sonnet,opus",
            "--conditions",
            "both_modalities",
            "--effort",
            "high",
            "--limit",
            "2",
            "--mock",
            "--resume",
        ],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert "invocations=0 skipped=6" in resume_run.stdout

    evaluate = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "evaluate_v4_agent_outputs.py"),
            str(output_root),
            "--hidden-manifest",
            str(evaluator_root / "hidden_manifest.jsonl"),
            "--output-dir",
            str(eval_root),
        ],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert "n_scored=6" in evaluate.stdout
    assert (eval_root / "summary.csv").exists()
    assert (eval_root / "scored_predictions.jsonl").exists()
    text = (eval_root / "summary.csv").read_text(encoding="utf-8")
    assert "both_modalities" in text
    assert "haiku" in text
    assert "sonnet" in text
    assert "opus" in text


def test_v4_evaluator_scores_in_memory_mock_outputs(tmp_path: Path) -> None:
    public_root = tmp_path / "public"
    evaluator_root = tmp_path / "evaluator"
    run_root = tmp_path / "run"
    eval_root = tmp_path / "eval"
    records = build_v4_task_set(
        public_root=public_root,
        evaluator_root=evaluator_root,
        run_id="test_eval",
        samples_per_status=2,
        mode="debug",
        seed_start=103,
        max_seeds=40,
        conditions="both_modalities",
    )
    hidden = load_hidden_manifest(evaluator_root / "hidden_manifest.jsonl")
    assert set(hidden) == {row.sample_id for row in records}

    for record in records[:2]:
        result_dir = run_root / "sonnet" / "both_modalities" / record.sample_id
        result_dir.mkdir(parents=True)
        output = {
            "image_evidence": "image evidence",
            "hic_evidence": "hic evidence",
            "image_predicted_label": record.true_label_name,
            "hic_predicted_label": record.true_label_name,
            "modality_agreement": "agree",
            "predicted_label": record.true_label_name,
            "label_confidence": 0.7,
            "contradiction_status": record.true_contradiction_status,
            "trusted_evidence": "both",
            "recommended_action": record.recommended_action,
            "rationale": "mock correct output",
        }
        (result_dir / "result.json").write_text(json.dumps(output), encoding="utf-8")
        (result_dir / "metadata.json").write_text(
            json.dumps(
                {
                    "task_id": record.sample_id,
                    "model": "sonnet",
                    "input_condition": "both_modalities",
                    "effort": "high",
                    "elapsed_seconds": 0.0,
                    "returncode": 0,
                    "timed_out": False,
                }
            ),
            encoding="utf-8",
        )

    result = evaluate_v4_outputs(
        run_roots=[run_root],
        hidden_manifest=evaluator_root / "hidden_manifest.jsonl",
        output_dir=eval_root,
    )
    assert result["n_scored"] == 2
    assert result["summary_rows"][0]["classification_accuracy"] == 1.0
    assert result["summary_rows"][0]["recommended_action_accuracy"] == 1.0
