"""Tests for the V3 agent benchmark harness."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from v3_agent_dataset import build_v3_agent_samples  # noqa: E402
from v3_agent_metrics import safe_normalize_agent_output, score_agent_output, summarize_agent_records  # noqa: E402
from v3_claude_client import (  # noqa: E402
    ClaudeRunConfig,
    model_candidates,
    run_claude_agent,
    system_prompt_for_policy,
    visual_tool_definitions,
)
from v3_python_sandbox import assert_python_tool_payload_is_safe  # noqa: E402
from v3_scientific_tools import assert_scientific_tool_payload_is_safe  # noqa: E402
from v3_visual_tools import V3VisualToolbox, assert_visual_tool_payload_is_safe  # noqa: E402


def test_v3_dataset_exports_balanced_visual_assets(tmp_path: Path) -> None:
    samples = build_v3_agent_samples(
        output_dir=tmp_path,
        contradiction_strengths=(0.0, 0.5),
        samples_per_strength={0.0: 2, 0.5: 2},
        mode="debug",
        seed=3,
    )
    assert len(samples) == 4
    assert {sample.contradiction_strength for sample in samples} == {0.0, 0.5}
    for sample in samples:
        assert Path(sample.image_path).exists()
        assert Path(sample.hic_path).exists()
        assert Path(sample.panel_path).exists()
        assert "cs" not in sample.sample_id
        assert "0p" not in sample.sample_id
        assert sample.true_label_name in {"normal", "cancer"}
        assert sample.recommended_action in {"use_both", "use_image", "use_hic", "abstain"}


def test_visual_tools_do_not_expose_hidden_metadata(tmp_path: Path) -> None:
    samples = build_v3_agent_samples(
        output_dir=tmp_path / "data",
        contradiction_strengths=(0.0,),
        samples_per_strength=2,
        mode="debug",
        seed=4,
    )
    toolbox = V3VisualToolbox(samples=samples, work_dir=tmp_path / "tools")
    payload = toolbox.load_sample_panel(samples[0].sample_id)
    assert_visual_tool_payload_is_safe(payload)
    crop = toolbox.crop_modality_image(samples[0].sample_id, "hic", 0, 0, 10, 12)
    assert_visual_tool_payload_is_safe(crop)
    assert Path(str(crop["image_path"])).exists()


def test_scientific_tools_do_not_expose_hidden_metadata(tmp_path: Path) -> None:
    samples = build_v3_agent_samples(
        output_dir=tmp_path / "data",
        contradiction_strengths=(0.0,),
        samples_per_strength=2,
        mode="debug",
        seed=10,
    )
    toolbox = V3VisualToolbox(samples=samples, work_dir=tmp_path / "tools")
    for tool_name in (
        "measure_image_features",
        "measure_hic_features",
        "compare_modalities",
        "generate_feature_report",
    ):
        payload = toolbox.call_tool(tool_name, {"sample_id": samples[0].sample_id})
        assert_scientific_tool_payload_is_safe(payload)
        assert payload["sample_id"] == samples[0].sample_id


def test_python_tool_runs_only_on_public_sample_files(tmp_path: Path) -> None:
    samples = build_v3_agent_samples(
        output_dir=tmp_path / "data",
        contradiction_strengths=(0.0,),
        samples_per_strength=2,
        mode="debug",
        seed=11,
    )
    toolbox = V3VisualToolbox(samples=samples, work_dir=tmp_path / "tools")
    payload = toolbox.call_tool(
        "run_python_analysis",
        {
            "sample_id": samples[0].sample_id,
            "_input_condition": "both_modalities",
            "code": (
                "from PIL import Image\n"
                "import json\n"
                "out = {}\n"
                "for name in ['polymer_image.png', 'hic_contact_map.png']:\n"
                "    with Image.open(name) as im:\n"
                "        out[name] = im.size\n"
                "print(json.dumps(out, sort_keys=True))\n"
            ),
        },
    )
    assert_python_tool_payload_is_safe(payload)
    assert payload["returncode"] == 0
    assert "polymer_image.png" in str(payload["stdout"])
    assert "hic_contact_map.png" in str(payload["stdout"])


def test_python_tool_rejects_hidden_manifest_access(tmp_path: Path) -> None:
    samples = build_v3_agent_samples(
        output_dir=tmp_path / "data",
        contradiction_strengths=(0.0,),
        samples_per_strength=2,
        mode="debug",
        seed=12,
    )
    toolbox = V3VisualToolbox(samples=samples, work_dir=tmp_path / "tools")
    payload = toolbox.call_tool(
        "run_python_analysis",
        {
            "sample_id": samples[0].sample_id,
            "_input_condition": "both_modalities",
            "code": "print(open('../sample_manifest.jsonl').read())",
        },
    )
    assert payload["returncode"] == 2
    assert "forbidden fragment" in str(payload["stderr"])


def test_output_parser_and_metrics_on_toy_response(tmp_path: Path) -> None:
    samples = build_v3_agent_samples(
        output_dir=tmp_path,
        contradiction_strengths=(0.0,),
        samples_per_strength=2,
        mode="debug",
        seed=5,
    )
    sample = samples[0]
    raw = json.dumps(
        {
            "predicted_label": sample.true_label_name,
            "label_confidence": 0.8,
            "contradiction_status": sample.true_contradiction_status,
            "trusted_evidence": "both",
            "recommended_action": sample.recommended_action,
            "rationale": "visible evidence is consistent",
        }
    )
    normalized = safe_normalize_agent_output(raw)
    scored = score_agent_output(sample, normalized, model="mock", raw_output=raw)
    assert scored["label_correct"] is True
    assert scored["contradiction_correct"] is True
    assert scored["action_correct"] is True
    summary = summarize_agent_records([scored])
    assert summary["classification_accuracy"] == 1.0
    assert summary["recommended_action_accuracy"] == 1.0


def test_evidence_separation_fields_are_preserved(tmp_path: Path) -> None:
    samples = build_v3_agent_samples(
        output_dir=tmp_path,
        contradiction_strengths=(0.0,),
        samples_per_strength=2,
        mode="debug",
        seed=9,
    )
    sample = samples[0]
    raw = json.dumps(
        {
            "image_evidence": "image looks compact",
            "hic_evidence": "Hi-C map has a strong diagonal",
            "image_predicted_label": sample.true_label_name,
            "hic_predicted_label": sample.true_label_name,
            "modality_agreement": "agree",
            "predicted_label": sample.true_label_name,
            "label_confidence": 0.7,
            "contradiction_status": sample.true_contradiction_status,
            "trusted_evidence": "both",
            "recommended_action": sample.recommended_action,
            "rationale": "separate evidence points to the same label",
        }
    )
    scored = score_agent_output(
        sample,
        safe_normalize_agent_output(raw),
        model="mock",
        raw_output=raw,
        prompt_policy="evidence_separation",
        thinking="none",
    )
    assert scored["prompt_policy"] == "evidence_separation"
    assert scored["thinking"] == "none"
    assert scored["image_predicted_label"] == sample.true_label_name
    assert scored["hic_predicted_label"] == sample.true_label_name
    assert scored["modality_agreement"] == "agree"


def test_unimodal_records_do_not_report_reconciliation_metrics(tmp_path: Path) -> None:
    samples = build_v3_agent_samples(
        output_dir=tmp_path,
        contradiction_strengths=(0.0,),
        samples_per_strength=2,
        mode="debug",
        seed=8,
    )
    sample = samples[0]
    raw = json.dumps(
        {
            "predicted_label": sample.true_label_name,
            "label_confidence": 0.7,
            "contradiction_status": "unclear",
            "trusted_evidence": "image",
            "recommended_action": "use_image",
            "rationale": "single modality",
        }
    )
    scored = score_agent_output(
        sample,
        safe_normalize_agent_output(raw),
        model="mock",
        raw_output=raw,
        input_condition="image_only",
    )
    summary = summarize_agent_records([scored])
    assert summary["classification_accuracy"] == 1.0
    assert str(summary["contradiction_accuracy"]) == "nan"
    assert str(summary["recommended_action_accuracy"]) == "nan"


def test_mock_claude_path_never_requires_api_key(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    samples = build_v3_agent_samples(
        output_dir=tmp_path,
        contradiction_strengths=(0.0,),
        samples_per_strength=2,
        mode="debug",
        seed=6,
    )
    toolbox = V3VisualToolbox(samples=samples, work_dir=tmp_path / "tools")
    result = run_claude_agent(samples[0], toolbox, ClaudeRunConfig(model="mock-sonnet", real_api=False, tool_level="python"))
    normalized = safe_normalize_agent_output(result.raw_output)
    assert normalized["predicted_label"] in {"normal", "cancer"}
    assert result.resolved_model == "mock-sonnet"


def test_claude_model_candidates_prefer_short_names() -> None:
    assert model_candidates("claude-sonnet-4-6")[0] == "claude-sonnet-4-6"
    assert model_candidates("claude-sonnet-4-6") == ("claude-sonnet-4-6",)
    assert model_candidates("claude-opus-4-7") == ("claude-opus-4-7",)
    assert "claude-haiku-4-5-20251001" in model_candidates("claude-haiku-4-5")
    assert model_candidates("custom-model") == ("custom-model",)


def test_unimodal_tool_definitions_expose_only_one_modality() -> None:
    image_tools = visual_tool_definitions("image_only")
    hic_tools = visual_tool_definitions("hic_only")
    assert {tool["name"] for tool in image_tools} == {"load_modality_image", "crop_modality_image"}
    assert {tool["name"] for tool in hic_tools} == {"load_modality_image", "crop_modality_image"}
    assert image_tools[0]["input_schema"]["properties"]["modality"]["enum"] == ["image"]
    assert hic_tools[0]["input_schema"]["properties"]["modality"]["enum"] == ["hic"]
    image_scientific_tools = visual_tool_definitions("image_only", tool_level="scientific")
    hic_scientific_tools = visual_tool_definitions("hic_only", tool_level="scientific")
    both_scientific_tools = visual_tool_definitions("both_modalities", tool_level="scientific")
    both_python_tools = visual_tool_definitions("both_modalities", tool_level="python")
    assert "measure_image_features" in {tool["name"] for tool in image_scientific_tools}
    assert "measure_hic_features" not in {tool["name"] for tool in image_scientific_tools}
    assert "measure_hic_features" in {tool["name"] for tool in hic_scientific_tools}
    assert "compare_modalities" in {tool["name"] for tool in both_scientific_tools}
    assert "run_python_analysis" in {tool["name"] for tool in both_python_tools}


def test_evidence_separation_prompt_requests_modality_fields() -> None:
    prompt = system_prompt_for_policy("evidence_separation")
    assert "image_predicted_label" in prompt
    assert "hic_predicted_label" in prompt
    assert "modality_agreement" in prompt


def test_runner_mock_smoke_writes_outputs(tmp_path: Path) -> None:
    output_dir = tmp_path / "run"
    command = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "run_v3_agent_benchmark.py"),
        "--profile",
        "smoke",
        "--mock",
        "--prompt-policy",
        "all",
        "--tool-level",
        "python",
        "--max-workers",
        "2",
        "--output-dir",
        str(output_dir),
    ]
    completed = subprocess.run(command, cwd=REPO_ROOT, check=True, capture_output=True, text=True)
    assert "wrote V3 benchmark run" in completed.stdout
    assert (output_dir / "sample_manifest.jsonl").exists()
    assert (output_dir / "predictions.jsonl").exists()
    assert (output_dir / "summary.csv").exists()
    assert (output_dir / "report.txt").exists()
    text = (output_dir / "summary.csv").read_text(encoding="utf-8")
    assert "python" in text
    assert "evidence_separation" in text
    assert "image_only" in text
    assert "hic_only" in text
    assert "both_modalities" in text

    summarize = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "summarize_v3_agent_runs.py"),
            str(output_dir),
        ],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert "Pooled sample metrics" in summarize.stdout
    assert "both_modalities" in summarize.stdout
