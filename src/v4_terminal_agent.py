"""Utilities for the V4 Claude Code terminal-agent evaluation.

V4 deliberately exports public task folders that contain only the visible
evidence for one sample. Hidden labels and evaluator metadata stay in a
separate manifest used only after model outputs are written.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
import shutil
import subprocess
import time
import uuid
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from v2_regime import ContradictionRegimeConfig, build_v2_paired_dataset
from v3_agent_dataset import (
    LABEL_NAMES,
    _contradiction_status,
    _recommended_action,
    _save_panel,
    _to_uint8_image,
)
from v3_agent_metrics import safe_normalize_agent_output, summarize_agent_records


INPUT_CONDITIONS = ("both_modalities", "image_only", "hic_only")
MODEL_ALIASES = ("haiku", "sonnet", "opus")
CONTRADICTION_STATUSES = ("none", "weak", "strong")
ACTION_VALUES = ("use_both", "use_image", "use_hic", "abstain")
CONTRADICTION_STRENGTHS = (0.0, 0.5, 1.0)

PUBLIC_FORBIDDEN_FRAGMENTS = (
    "true_label",
    "true_contradiction",
    "true_recommended",
    "latent",
    "reliability",
    "source_index",
    "split_indices",
    "sample_manifest",
    "results/v3",
    "results/v4/evaluator",
    "contradiction_strength",
    "anthropic_api_key",
    "openai_api_key",
)


@dataclass(frozen=True)
class V4TaskRecord:
    """Hidden evaluator row for one selected V4 sample."""

    sample_id: str
    contradiction_strength: float
    source_index: int
    source_seed: int
    split: str
    true_label: int
    true_label_name: str
    true_contradiction_status: str
    recommended_action: str
    image_reliability: float
    hic_reliability: float
    structural_contradiction_score: float
    public_sample_dir: str
    public_tasks: dict[str, str]


def output_schema() -> dict[str, object]:
    """Return the structured JSON schema expected from terminal agents."""
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "image_evidence": {"type": "string"},
            "hic_evidence": {"type": "string"},
            "image_predicted_label": {
                "type": "string",
                "enum": ["normal", "cancer", "unclear", "not_visible"],
            },
            "hic_predicted_label": {
                "type": "string",
                "enum": ["normal", "cancer", "unclear", "not_visible"],
            },
            "modality_agreement": {
                "type": "string",
                "enum": [
                    "agree",
                    "weak_disagreement",
                    "strong_disagreement",
                    "unclear",
                    "not_applicable",
                ],
            },
            "predicted_label": {"type": "string", "enum": ["normal", "cancer"]},
            "label_confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
            "contradiction_status": {
                "type": "string",
                "enum": ["none", "weak", "strong", "unclear"],
            },
            "trusted_evidence": {
                "type": "string",
                "enum": ["image", "hic", "both", "neither", "unclear"],
            },
            "recommended_action": {"type": "string", "enum": list(ACTION_VALUES)},
            "rationale": {"type": "string"},
        },
        "required": [
            "image_evidence",
            "hic_evidence",
            "image_predicted_label",
            "hic_predicted_label",
            "modality_agreement",
            "predicted_label",
            "label_confidence",
            "contradiction_status",
            "trusted_evidence",
            "recommended_action",
            "rationale",
        ],
    }


def instructions_for_condition(condition: str) -> str:
    """Build task-only instructions for one input condition."""
    if condition not in INPUT_CONDITIONS:
        raise ValueError(f"condition must be one of {INPUT_CONDITIONS}")

    if condition == "both_modalities":
        evidence_text = (
            "Visible files: polymer_image.png and hic_contact_map.png.\n"
            "Classify the sample as normal or cancer, then reconcile the two modalities. "
            "Judge the polymer image and Hi-C contact map separately before deciding whether "
            "they agree, weakly disagree, or strongly disagree."
        )
        contradiction_text = (
            "For contradiction_status, use none only when the modalities are visually compatible; "
            "use weak or strong when their visible evidence points to conflicting structure or labels."
        )
    elif condition == "image_only":
        evidence_text = (
            "Visible file: polymer_image.png only.\n"
            "Classify the sample as normal or cancer using this image modality alone."
        )
        contradiction_text = (
            "The paired Hi-C modality is not visible. Set hic_evidence and hic_predicted_label "
            "to not_visible, modality_agreement to not_applicable, contradiction_status to unclear, "
            "and recommended_action to use_image unless the image evidence is insufficient."
        )
    else:
        evidence_text = (
            "Visible file: hic_contact_map.png only.\n"
            "Classify the sample as normal or cancer using this Hi-C modality alone."
        )
        contradiction_text = (
            "The paired polymer image modality is not visible. Set image_evidence and "
            "image_predicted_label to not_visible, modality_agreement to not_applicable, "
            "contradiction_status to unclear, and recommended_action to use_hic unless the "
            "Hi-C evidence is insufficient."
        )

    return (
        "You are participating in an evaluation of biological multimodal evidence reconciliation.\n"
        "This is evaluation work: do not optimize the benchmark, infer hidden metadata, or use "
        "anything outside this task folder.\n\n"
        f"{evidence_text}\n\n"
        "You may use Python or shell commands to inspect the visible files. Write temporary files "
        "only inside work/. Do not read parent directories, repository files, result folders, "
        "manifests, credentials, or network resources.\n\n"
        f"{contradiction_text}\n\n"
        "Return exactly one JSON object matching output_schema.json. Do not include markdown or "
        "extra prose outside the JSON object."
    )


def _safe_public_id(run_id: str, seed: int, strength: float, source_index: int, salt: str) -> str:
    raw = f"{run_id}:{seed}:{strength:.6f}:{source_index}:{salt}".encode("utf-8")
    return "task_" + hashlib.sha256(raw).hexdigest()[:16]


def _condition_values(values: Sequence[str] | str) -> tuple[str, ...]:
    if values == "all":
        return INPUT_CONDITIONS
    if isinstance(values, str):
        requested = tuple(item.strip() for item in values.split(",") if item.strip())
    else:
        requested = tuple(str(item).strip() for item in values if str(item).strip())
    unknown = sorted(set(requested) - set(INPUT_CONDITIONS))
    if unknown:
        raise ValueError(f"Unknown input condition(s): {unknown}")
    return requested


def _copy_public_files(sample_dir: Path, task_dir: Path, condition: str) -> None:
    task_dir.mkdir(parents=True, exist_ok=True)
    (task_dir / "work").mkdir(exist_ok=True)
    if condition in {"both_modalities", "image_only"}:
        shutil.copyfile(sample_dir / "polymer_image.png", task_dir / "polymer_image.png")
    if condition in {"both_modalities", "hic_only"}:
        shutil.copyfile(sample_dir / "hic_contact_map.png", task_dir / "hic_contact_map.png")
    (task_dir / "instructions.md").write_text(instructions_for_condition(condition), encoding="utf-8")
    (task_dir / "output_schema.json").write_text(json.dumps(output_schema(), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def build_v4_task_set(
    public_root: Path | str,
    evaluator_root: Path | str,
    run_id: str,
    samples_per_status: int = 30,
    mode: str = "full",
    image_size: int = 64,
    seed_start: int = 101,
    max_seeds: int = 250,
    split: str = "test",
    conditions: Sequence[str] | str = INPUT_CONDITIONS,
    structural_strong_threshold: float = 0.40,
) -> list[V4TaskRecord]:
    """Export a hidden-stratified V4 task set and hidden evaluator manifest."""
    if samples_per_status % 2 != 0:
        raise ValueError("samples_per_status must be even so labels can be balanced")

    public_path = Path(public_root)
    evaluator_path = Path(evaluator_root)
    sample_asset_root = evaluator_path / "selected_public_assets"
    public_path.mkdir(parents=True, exist_ok=True)
    evaluator_path.mkdir(parents=True, exist_ok=True)
    sample_asset_root.mkdir(parents=True, exist_ok=True)

    requested_conditions = _condition_values(conditions)
    per_label = samples_per_status // 2
    remaining = Counter(
        {
            (status, label): per_label
            for status in CONTRADICTION_STATUSES
            for label in ("normal", "cancer")
        }
    )
    selected: list[V4TaskRecord] = []
    salt = uuid.uuid4().hex

    for seed in range(int(seed_start), int(seed_start) + int(max_seeds)):
        strength_order = list(CONTRADICTION_STRENGTHS)
        np.random.default_rng(seed).shuffle(strength_order)
        for strength in strength_order:
            if not remaining:
                break
            dataset = build_v2_paired_dataset(
                regime=ContradictionRegimeConfig(
                    contradiction_strength=float(strength),
                    modality_asymmetry=0.0,
                    structural_strong_threshold=float(structural_strong_threshold),
                ),
                mode=mode,
                image_size=int(image_size),
                genomic_representation="matrix",
                seed=seed + int(round(1000.0 * float(strength))),
            )
            arrays = dataset["arrays"]
            split_indices = np.asarray(dataset["split_indices"][split], dtype=np.int64)
            labels = np.asarray(arrays["label"], dtype=np.int64)
            rng = np.random.default_rng(seed * 1009 + int(round(1000.0 * float(strength))))
            indices = np.array(split_indices, copy=True)
            rng.shuffle(indices)

            for source_index in indices:
                source_index_int = int(source_index)
                true_label = int(labels[source_index_int])
                true_label_name = LABEL_NAMES[true_label]
                score = float(arrays["latent_structural_contradiction_score"][source_index_int])
                status = _contradiction_status(score, strong_threshold=float(structural_strong_threshold))
                key = (status, true_label_name)
                if remaining.get(key, 0) <= 0:
                    continue

                sample_id = _safe_public_id(run_id, seed, float(strength), source_index_int, salt)
                sample_dir = sample_asset_root / sample_id
                sample_dir.mkdir(parents=True, exist_ok=True)
                image_path = sample_dir / "polymer_image.png"
                hic_path = sample_dir / "hic_contact_map.png"
                panel_path = sample_dir / "evidence_panel.png"
                _to_uint8_image(arrays["image"][source_index_int]).save(image_path)
                _to_uint8_image(arrays["genomic_matrix"][source_index_int]).save(hic_path)
                _save_panel(image_path=image_path, hic_path=hic_path, panel_path=panel_path)

                public_tasks: dict[str, str] = {}
                for condition in requested_conditions:
                    task_dir = public_path / condition / sample_id
                    _copy_public_files(sample_dir=sample_dir, task_dir=task_dir, condition=condition)
                    public_tasks[condition] = str(task_dir)

                image_reliability = float(arrays["latent_image_observability_drive"][source_index_int])
                hic_reliability = float(arrays["latent_genomic_observability_drive"][source_index_int])
                selected.append(
                    V4TaskRecord(
                        sample_id=sample_id,
                        contradiction_strength=float(strength),
                        source_index=source_index_int,
                        source_seed=int(seed),
                        split=split,
                        true_label=true_label,
                        true_label_name=true_label_name,
                        true_contradiction_status=status,
                        recommended_action=_recommended_action(status, image_reliability, hic_reliability),
                        image_reliability=image_reliability,
                        hic_reliability=hic_reliability,
                        structural_contradiction_score=score,
                        public_sample_dir=str(sample_dir),
                        public_tasks=public_tasks,
                    )
                )
                remaining[key] -= 1
                if remaining[key] <= 0:
                    del remaining[key]
                if not remaining:
                    break
        if not remaining:
            break

    if remaining:
        raise RuntimeError(f"Could not fill requested V4 strata; remaining={dict(remaining)}")

    manifest_path = evaluator_path / "hidden_manifest.jsonl"
    with manifest_path.open("w", encoding="utf-8") as handle:
        for record in selected:
            handle.write(json.dumps(asdict(record), sort_keys=True) + "\n")

    public_manifest_path = public_path / "public_manifest.jsonl"
    with public_manifest_path.open("w", encoding="utf-8") as handle:
        for record in selected:
            for condition, task_dir in record.public_tasks.items():
                handle.write(
                    json.dumps(
                        {
                            "sample_id": record.sample_id,
                            "input_condition": condition,
                            "task_dir": task_dir,
                        },
                        sort_keys=True,
                    )
                    + "\n"
                )

    metadata = {
        "run_id": run_id,
        "samples_per_status": int(samples_per_status),
        "total_samples": len(selected),
        "mode": mode,
        "image_size": int(image_size),
        "seed_start": int(seed_start),
        "max_seeds": int(max_seeds),
        "split": split,
        "conditions": list(requested_conditions),
        "public_root": str(public_path),
        "evaluator_root": str(evaluator_path),
    }
    (evaluator_path / "build_metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    audit = audit_public_task_root(public_path)
    (evaluator_path / "contamination_audit.json").write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if audit["finding_count"]:
        raise RuntimeError(f"Public task contamination audit failed: {audit['findings'][:5]}")
    return selected


def load_hidden_manifest(path: Path | str) -> dict[str, V4TaskRecord]:
    """Load hidden V4 evaluator rows keyed by public sample ID."""
    rows: dict[str, V4TaskRecord] = {}
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            data = json.loads(line)
            record = V4TaskRecord(**data)
            rows[record.sample_id] = record
    return rows


def audit_public_task_root(public_root: Path | str) -> dict[str, object]:
    """Scan public task files for hidden-metadata fragments."""
    root = Path(public_root)
    findings = []
    for path in sorted(root.rglob("*")):
        relative = str(path.relative_to(root)).lower()
        for fragment in PUBLIC_FORBIDDEN_FRAGMENTS:
            if fragment in relative:
                findings.append({"path": str(path), "fragment": fragment, "location": "path"})
        if not path.is_file():
            continue
        if path.suffix.lower() in {".png", ".jpg", ".jpeg"}:
            continue
        text = path.read_text(encoding="utf-8", errors="ignore").lower()
        for fragment in PUBLIC_FORBIDDEN_FRAGMENTS:
            if fragment in text:
                findings.append({"path": str(path), "fragment": fragment, "location": "content"})
    return {"finding_count": len(findings), "findings": findings}


def claude_code_command(
    task_dir: Path | str,
    model: str,
    effort: str = "high",
    max_budget_usd: float | None = None,
) -> list[str]:
    """Build the Claude Code command for one V4 task."""
    task_path = Path(task_dir)
    prompt = (task_path / "instructions.md").read_text(encoding="utf-8")
    schema = (task_path / "output_schema.json").read_text(encoding="utf-8")
    command = [
        "claude",
        "-p",
        prompt,
        "--model",
        str(model),
        "--effort",
        str(effort),
        "--output-format",
        "json",
        "--json-schema",
        schema,
        "--tools",
        "Read,Bash",
        "--allowedTools",
        "Read,Bash(python *),Bash(python3 *),Bash(ls),Bash(ls *),Bash(pwd),Bash(file *),Bash(shasum *)",
        "--disallowedTools",
        "Edit,Write,WebFetch,WebSearch",
        "--permission-mode",
        "dontAsk",
        "--no-session-persistence",
    ]
    if max_budget_usd is not None:
        command.extend(["--max-budget-usd", f"{float(max_budget_usd):.4f}"])
    return command


def is_claude_usage_limited(stdout: str, stderr: str = "") -> bool:
    """Return True when Claude Code reports a recoverable usage-limit stop."""
    text = f"{stdout}\n{stderr}".lower()
    markers = (
        "usage limit",
        "usage_limit",
        "rate limit",
        "rate_limit",
        "too many requests",
        "quota exceeded",
    )
    return any(marker in text for marker in markers)


def list_task_dirs(task_root: Path | str, condition: str, limit: int | None = None) -> list[Path]:
    """List public task directories for one condition."""
    if condition not in INPUT_CONDITIONS:
        raise ValueError(f"condition must be one of {INPUT_CONDITIONS}")
    paths = sorted(path for path in (Path(task_root) / condition).iterdir() if path.is_dir())
    if limit is not None:
        return paths[: int(limit)]
    return paths


def mock_agent_output(condition: str) -> dict[str, object]:
    """Return a schema-valid mock output for runner/evaluator dry runs."""
    if condition == "image_only":
        return {
            "image_evidence": "mock image evidence",
            "hic_evidence": "not_visible",
            "image_predicted_label": "normal",
            "hic_predicted_label": "not_visible",
            "modality_agreement": "not_applicable",
            "predicted_label": "normal",
            "label_confidence": 0.5,
            "contradiction_status": "unclear",
            "trusted_evidence": "image",
            "recommended_action": "use_image",
            "rationale": "mock single-modality image decision",
        }
    if condition == "hic_only":
        return {
            "image_evidence": "not_visible",
            "hic_evidence": "mock Hi-C evidence",
            "image_predicted_label": "not_visible",
            "hic_predicted_label": "normal",
            "modality_agreement": "not_applicable",
            "predicted_label": "normal",
            "label_confidence": 0.5,
            "contradiction_status": "unclear",
            "trusted_evidence": "hic",
            "recommended_action": "use_hic",
            "rationale": "mock single-modality Hi-C decision",
        }
    return {
        "image_evidence": "mock image evidence",
        "hic_evidence": "mock Hi-C evidence",
        "image_predicted_label": "normal",
        "hic_predicted_label": "normal",
        "modality_agreement": "agree",
        "predicted_label": "normal",
        "label_confidence": 0.5,
        "contradiction_status": "none",
        "trusted_evidence": "both",
        "recommended_action": "use_both",
        "rationale": "mock both-modality reconciliation decision",
    }


def run_one_task(
    task_dir: Path | str,
    output_dir: Path | str,
    model: str,
    condition: str,
    effort: str = "high",
    max_budget_usd: float | None = None,
    timeout_seconds: int = 300,
    mock: bool = False,
) -> dict[str, object]:
    """Run or mock one Claude Code task and write replayable artifacts."""
    task_path = Path(task_dir)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    metadata: dict[str, object] = {
        "task_id": task_path.name,
        "task_dir": str(task_path),
        "model": model,
        "input_condition": condition,
        "effort": effort,
        "mock": bool(mock),
        "timeout_seconds": int(timeout_seconds),
    }
    start = time.time()
    if mock:
        payload = mock_agent_output(condition)
        (out_path / "result.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        (out_path / "stdout.txt").write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
        (out_path / "stderr.txt").write_text("", encoding="utf-8")
        metadata.update({"returncode": 0, "timed_out": False, "elapsed_seconds": 0.0})
    else:
        command = claude_code_command(task_path, model=model, effort=effort, max_budget_usd=max_budget_usd)
        inline_arg_indices = {idx + 1 for idx, item in enumerate(command) if item in {"-p", "--json-schema"}}
        metadata["command_preview"] = [
            "<inline>" if idx in inline_arg_indices else item
            for idx, item in enumerate(command)
        ]
        try:
            completed = subprocess.run(
                command,
                cwd=task_path,
                capture_output=True,
                text=True,
                timeout=int(timeout_seconds),
                check=False,
            )
            stdout = completed.stdout
            stderr = completed.stderr
            returncode = int(completed.returncode)
            timed_out = False
        except subprocess.TimeoutExpired as exc:
            stdout = exc.stdout or ""
            stderr = (exc.stderr or "") + f"\nTimed out after {timeout_seconds} seconds."
            returncode = 124
            timed_out = True
        (out_path / "stdout.txt").write_text(str(stdout), encoding="utf-8")
        (out_path / "stderr.txt").write_text(str(stderr), encoding="utf-8")
        metadata.update(
            {
                "returncode": returncode,
                "timed_out": timed_out,
                "usage_limited": is_claude_usage_limited(str(stdout), str(stderr)),
            }
        )
        try:
            payload = extract_agent_payload(str(stdout))
            (out_path / "result.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        except Exception as exc:  # noqa: BLE001 - malformed outputs are benchmark data.
            metadata["extract_error"] = str(exc)

    metadata["elapsed_seconds"] = float(time.time() - start)
    (out_path / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return metadata


def has_complete_v4_result(result_dir: Path | str) -> bool:
    """Return True when a result directory has a successful parsed task output."""
    path = Path(result_dir)
    metadata_path = path / "metadata.json"
    result_path = path / "result.json"
    if not metadata_path.exists() or not result_path.exists():
        return False
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return int(metadata.get("returncode", 1)) == 0 and not bool(metadata.get("timed_out"))


def _extract_json_object(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    if not cleaned:
        raise ValueError("empty output")
    try:
        data = json.loads(cleaned)
        if isinstance(data, dict):
            return data
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
    if not match:
        raise ValueError("no JSON object found")
    data = json.loads(match.group(0))
    if not isinstance(data, dict):
        raise ValueError("JSON output is not an object")
    return data


def extract_agent_payload(raw_output: str | Mapping[str, Any]) -> dict[str, Any]:
    """Extract the model's schema object from raw Claude Code output."""
    data = dict(raw_output) if isinstance(raw_output, Mapping) else _extract_json_object(str(raw_output))
    if "predicted_label" in data:
        return data
    for key in ("structured_output", "result", "text", "response", "output"):
        value = data.get(key)
        if isinstance(value, Mapping):
            try:
                return extract_agent_payload(value)
            except ValueError:
                pass
        if isinstance(value, str) and value.strip():
            try:
                return extract_agent_payload(value)
            except ValueError:
                pass
    if isinstance(data.get("content"), list):
        texts = []
        for block in data["content"]:
            if isinstance(block, Mapping) and isinstance(block.get("text"), str):
                texts.append(str(block["text"]))
        if texts:
            return extract_agent_payload("\n".join(texts))
    raise ValueError("could not find schema payload in output")


def _load_output_payload(result_dir: Path) -> tuple[dict[str, Any] | str, str]:
    result_path = result_dir / "result.json"
    if result_path.exists():
        return json.loads(result_path.read_text(encoding="utf-8")), str(result_path)
    stdout_path = result_dir / "stdout.txt"
    if stdout_path.exists():
        raw = stdout_path.read_text(encoding="utf-8")
        try:
            return extract_agent_payload(raw), str(stdout_path)
        except Exception:
            return raw, str(stdout_path)
    return "", ""


def _cost_from_raw_stdout(result_dir: Path) -> float:
    stdout_path = result_dir / "stdout.txt"
    if not stdout_path.exists():
        return 0.0
    try:
        data = _extract_json_object(stdout_path.read_text(encoding="utf-8"))
    except Exception:
        return 0.0
    for key in ("total_cost_usd", "cost_usd", "cost"):
        try:
            return float(data.get(key) or 0.0)
        except (TypeError, ValueError):
            continue
    return 0.0


def prediction_result_dirs(run_root: Path | str) -> list[Path]:
    """Find V4 per-task result directories under a run root."""
    root = Path(run_root)
    if (root / "metadata.json").exists():
        return [root]
    return sorted(path.parent for path in root.rglob("metadata.json"))


def score_v4_result_dir(result_dir: Path, hidden_rows: Mapping[str, V4TaskRecord]) -> dict[str, object]:
    """Score one V4 result directory against hidden evaluator metadata."""
    metadata = json.loads((result_dir / "metadata.json").read_text(encoding="utf-8"))
    task_id = str(metadata["task_id"])
    if task_id not in hidden_rows:
        raise KeyError(f"Task {task_id} is not present in hidden manifest")
    hidden = hidden_rows[task_id]
    raw_payload, output_source = _load_output_payload(result_dir)
    normalized = safe_normalize_agent_output(raw_payload)
    predicted_label = str(normalized["predicted_label"])
    contradiction_status = str(normalized["contradiction_status"])
    recommended_action = str(normalized["recommended_action"])
    condition = str(metadata["input_condition"])

    return {
        "sample_id": hidden.sample_id,
        "model": str(metadata["model"]),
        "resolved_model": str(metadata.get("resolved_model") or metadata["model"]),
        "input_condition": condition,
        "prompt_policy": "terminal_agent_task_only",
        "thinking": str(metadata.get("effort", "unknown")),
        "tool_level": "terminal_python",
        "contradiction_strength": hidden.contradiction_strength,
        "true_label": hidden.true_label_name,
        "predicted_label": predicted_label,
        "label_confidence": normalized["label_confidence"],
        "label_correct": predicted_label == hidden.true_label_name,
        "true_contradiction_status": hidden.true_contradiction_status,
        "predicted_contradiction_status": contradiction_status,
        "contradiction_correct": contradiction_status == hidden.true_contradiction_status,
        "true_recommended_action": hidden.recommended_action,
        "predicted_recommended_action": recommended_action,
        "action_correct": recommended_action == hidden.recommended_action,
        "trusted_evidence": normalized["trusted_evidence"],
        "rationale": normalized["rationale"],
        "image_evidence": normalized.get("image_evidence", ""),
        "hic_evidence": normalized.get("hic_evidence", ""),
        "image_predicted_label": normalized.get("image_predicted_label", "not_reported"),
        "hic_predicted_label": normalized.get("hic_predicted_label", "not_reported"),
        "modality_agreement": normalized.get("modality_agreement", "not_reported"),
        "malformed": bool(normalized["malformed"]),
        "latency_seconds": float(metadata.get("elapsed_seconds") or 0.0),
        "tool_call_count": int(metadata.get("tool_call_count") or 0),
        "input_tokens": int(metadata.get("input_tokens") or 0),
        "output_tokens": int(metadata.get("output_tokens") or 0),
        "timed_out": bool(metadata.get("timed_out", False)),
        "returncode": int(metadata.get("returncode") or 0),
        "cost_usd": _cost_from_raw_stdout(result_dir),
        "output_source": output_source,
        "raw_output": raw_payload if isinstance(raw_payload, str) else json.dumps(raw_payload, sort_keys=True),
    }


def evaluate_v4_outputs(
    run_roots: Iterable[Path | str],
    hidden_manifest: Path | str,
    output_dir: Path | str,
) -> dict[str, object]:
    """Score V4 outputs, write summary artifacts, and return aggregate data."""
    hidden_rows = load_hidden_manifest(hidden_manifest)
    scored_rows = []
    for run_root in run_roots:
        for result_dir in prediction_result_dirs(run_root):
            scored_rows.append(score_v4_result_dir(result_dir, hidden_rows))

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    _write_jsonl(out_path / "scored_predictions.jsonl", scored_rows)
    summary_rows = _summary_rows(scored_rows)
    _write_summary_csv(out_path / "summary.csv", summary_rows)
    report = _report_text(scored_rows, summary_rows)
    (out_path / "report.txt").write_text(report, encoding="utf-8")
    paired_rows = _paired_delta_rows(scored_rows)
    _write_paired_csv(out_path / "paired_deltas.csv", paired_rows)
    confusion = _confusion_tables(scored_rows)
    (out_path / "confusion.json").write_text(json.dumps(confusion, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {
        "n_scored": len(scored_rows),
        "summary_rows": summary_rows,
        "paired_rows": paired_rows,
        "confusion": confusion,
    }


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, object]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), sort_keys=True) + "\n")


def _summary_rows(rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str], list[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["model"]), str(row["input_condition"]))].append(row)
    summary_rows = []
    for (model, condition), group in sorted(grouped.items()):
        summary = summarize_agent_records(group)
        timeouts = sum(bool(row.get("timed_out")) for row in group)
        nonzero_returncodes = sum(int(row.get("returncode") or 0) != 0 for row in group)
        total_cost = float(sum(float(row.get("cost_usd") or 0.0) for row in group))
        summary_rows.append(
            {
                "model": model,
                "input_condition": condition,
                "n": int(summary["n"]),
                "classification_accuracy": summary.get("classification_accuracy"),
                "contradiction_accuracy": summary.get("contradiction_accuracy"),
                "contradiction_binary_f1": summary.get("contradiction_binary_f1"),
                "recommended_action_accuracy": summary.get("recommended_action_accuracy"),
                "malformed_fraction": summary.get("malformed_fraction"),
                "ece_10bin": summary.get("ece_10bin"),
                "timeout_rate": timeouts / len(group) if group else float("nan"),
                "nonzero_returncode_rate": nonzero_returncodes / len(group) if group else float("nan"),
                "mean_latency_seconds": summary.get("mean_latency_seconds"),
                "total_cost_usd": total_cost,
            }
        )
    return summary_rows


def _write_summary_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    fieldnames = [
        "model",
        "input_condition",
        "n",
        "classification_accuracy",
        "contradiction_accuracy",
        "contradiction_binary_f1",
        "recommended_action_accuracy",
        "malformed_fraction",
        "ece_10bin",
        "timeout_rate",
        "nonzero_returncode_rate",
        "mean_latency_seconds",
        "total_cost_usd",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def _confusion_tables(rows: Sequence[Mapping[str, object]]) -> dict[str, dict[str, int]]:
    tables: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        if str(row["input_condition"]) != "both_modalities":
            continue
        model = str(row["model"])
        tables[f"{model}:contradiction"][f"{row['true_contradiction_status']}->{row['predicted_contradiction_status']}"] += 1
        tables[f"{model}:action"][f"{row['true_recommended_action']}->{row['predicted_recommended_action']}"] += 1
    return {key: dict(counter) for key, counter in sorted(tables.items())}


def _paired_delta_rows(rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    by_key = {
        (str(row["model"]), str(row["input_condition"]), str(row["sample_id"])): row
        for row in rows
    }
    models = sorted({str(row["model"]) for row in rows})
    pairs = [
        ("both_modalities", "image_only"),
        ("both_modalities", "hic_only"),
        ("hic_only", "image_only"),
    ]
    deltas = []
    for model in models:
        for left, right in pairs:
            sample_ids = sorted(
                {
                    sample_id
                    for m, condition, sample_id in by_key
                    if m == model and condition == left
                }
                & {
                    sample_id
                    for m, condition, sample_id in by_key
                    if m == model and condition == right
                }
            )
            if not sample_ids:
                continue
            values = [
                int(bool(by_key[(model, left, sample_id)]["label_correct"]))
                - int(bool(by_key[(model, right, sample_id)]["label_correct"]))
                for sample_id in sample_ids
            ]
            mean = float(np.mean(values))
            if len(values) > 1:
                sd = float(np.std(values, ddof=1))
                se = sd / math.sqrt(len(values))
                low = mean - 1.96 * se
                high = mean + 1.96 * se
            else:
                low = high = float("nan")
            deltas.append(
                {
                    "model": model,
                    "left_condition": left,
                    "right_condition": right,
                    "metric": "label_correct",
                    "delta": mean,
                    "approx95_low": low,
                    "approx95_high": high,
                    "n": len(values),
                }
            )
    return deltas


def _write_paired_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    fieldnames = ["model", "left_condition", "right_condition", "metric", "delta", "approx95_low", "approx95_high", "n"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def _fmt(value: object) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if math.isnan(number):
        return "nan"
    return f"{number:.3f}"


def _report_text(rows: Sequence[Mapping[str, object]], summary_rows: Sequence[Mapping[str, object]]) -> str:
    lines = [
        "V4 Claude Code terminal-agent evaluation",
        f"n_scored={len(rows)}",
        "",
        "Summary",
    ]
    for row in summary_rows:
        lines.append(
            "model={model} input={condition} n={n} acc={acc} contradiction_acc={contr} action_acc={action} "
            "malformed={malformed} timeout={timeout} cost_usd={cost}".format(
                model=row["model"],
                condition=row["input_condition"],
                n=row["n"],
                acc=_fmt(row["classification_accuracy"]),
                contr=_fmt(row["contradiction_accuracy"]),
                action=_fmt(row["recommended_action_accuracy"]),
                malformed=_fmt(row["malformed_fraction"]),
                timeout=_fmt(row["timeout_rate"]),
                cost=_fmt(row["total_cost_usd"]),
            )
        )
    lines.append("")
    lines.append("Interpretation note: do not claim multimodal improvement unless clean image_only and hic_only rows exist for the same samples.")
    return "\n".join(lines) + "\n"
