"""Scoring utilities for V3 agent benchmark outputs."""

from __future__ import annotations

import json
import math
import re
from typing import Dict, Iterable, Mapping

import numpy as np

from v3_agent_dataset import VALID_RECOMMENDED_ACTIONS, V3AgentSample


LABEL_VALUES = ("normal", "cancer")
CONTRADICTION_VALUES = ("none", "weak", "strong", "unclear")
TRUSTED_EVIDENCE_VALUES = ("image", "hic", "both", "neither", "unclear")
ACTION_VALUES = ("use_both", "use_image", "use_hic", "abstain")
MODALITY_LABEL_VALUES = ("normal", "cancer", "unclear", "not_visible")
MODALITY_AGREEMENT_VALUES = ("agree", "weak_disagreement", "strong_disagreement", "unclear", "not_applicable")


def _extract_json_object(text: str) -> Dict[str, object]:
    """Extract the first JSON object from a model response."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?", "", cleaned.strip(), flags=re.IGNORECASE).strip()
        cleaned = re.sub(r"```$", "", cleaned.strip()).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
        if not match:
            raise
        return json.loads(match.group(0))


def normalize_agent_output(raw_output: str | Mapping[str, object]) -> Dict[str, object]:
    """Parse and normalize one structured agent answer."""
    data = dict(raw_output) if isinstance(raw_output, Mapping) else _extract_json_object(str(raw_output))

    predicted_label = str(data.get("predicted_label", "")).strip().lower()
    contradiction_status = str(data.get("contradiction_status", "")).strip().lower()
    trusted_evidence = str(data.get("trusted_evidence", "")).strip().lower()
    recommended_action = str(data.get("recommended_action", "")).strip().lower()
    rationale = str(data.get("rationale", "")).strip()
    image_predicted_label = str(data.get("image_predicted_label", "")).strip().lower()
    hic_predicted_label = str(data.get("hic_predicted_label", "")).strip().lower()
    modality_agreement = str(data.get("modality_agreement", "")).strip().lower()
    try:
        confidence = float(data.get("label_confidence", float("nan")))
    except (TypeError, ValueError):
        confidence = float("nan")

    return {
        "predicted_label": predicted_label if predicted_label in LABEL_VALUES else "invalid",
        "label_confidence": float(np.clip(confidence, 0.0, 1.0)) if np.isfinite(confidence) else float("nan"),
        "contradiction_status": contradiction_status
        if contradiction_status in CONTRADICTION_VALUES
        else "invalid",
        "trusted_evidence": trusted_evidence if trusted_evidence in TRUSTED_EVIDENCE_VALUES else "invalid",
        "recommended_action": recommended_action if recommended_action in ACTION_VALUES else "invalid",
        "rationale": rationale,
        "image_evidence": str(data.get("image_evidence", "")).strip(),
        "hic_evidence": str(data.get("hic_evidence", "")).strip(),
        "image_predicted_label": image_predicted_label
        if image_predicted_label in MODALITY_LABEL_VALUES
        else "not_reported",
        "hic_predicted_label": hic_predicted_label
        if hic_predicted_label in MODALITY_LABEL_VALUES
        else "not_reported",
        "modality_agreement": modality_agreement
        if modality_agreement in MODALITY_AGREEMENT_VALUES
        else "not_reported",
        "malformed": False,
    }


def malformed_agent_output(error: Exception | str) -> Dict[str, object]:
    """Represent an unparseable answer in the same schema."""
    return {
        "predicted_label": "invalid",
        "label_confidence": float("nan"),
        "contradiction_status": "invalid",
        "trusted_evidence": "invalid",
        "recommended_action": "invalid",
        "rationale": f"malformed: {error}",
        "image_evidence": "",
        "hic_evidence": "",
        "image_predicted_label": "invalid",
        "hic_predicted_label": "invalid",
        "modality_agreement": "invalid",
        "malformed": True,
    }


def _binary_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute binary F1 with zero-safe behavior."""
    true_positive = float(np.sum(y_true & y_pred))
    false_positive = float(np.sum(~y_true & y_pred))
    false_negative = float(np.sum(y_true & ~y_pred))
    denom = 2.0 * true_positive + false_positive + false_negative
    if denom == 0.0:
        return float("nan")
    return float((2.0 * true_positive) / denom)


def expected_calibration_error(records: list[Mapping[str, object]], n_bins: int = 10) -> float:
    """Compute ECE over parsed label confidence values."""
    confidences = np.asarray([float(record.get("label_confidence", np.nan)) for record in records], dtype=float)
    correct = np.asarray([bool(record.get("label_correct", False)) for record in records], dtype=bool)
    finite = np.isfinite(confidences)
    if not np.any(finite):
        return float("nan")
    confidences = confidences[finite]
    correct = correct[finite]
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for idx in range(n_bins):
        lower = edges[idx]
        upper = edges[idx + 1]
        if idx == n_bins - 1:
            mask = (confidences >= lower) & (confidences <= upper)
        else:
            mask = (confidences >= lower) & (confidences < upper)
        if not np.any(mask):
            continue
        ece += float(mask.mean()) * abs(float(correct[mask].mean()) - float(confidences[mask].mean()))
    return float(ece)


def score_agent_output(
    sample: V3AgentSample,
    normalized_output: Mapping[str, object],
    model: str,
    raw_output: str,
    resolved_model: str = "",
    input_condition: str = "both_modalities",
    prompt_policy: str = "baseline",
    thinking: str = "unknown",
    tool_level: str = "visual",
    latency_seconds: float = 0.0,
    input_tokens: int = 0,
    output_tokens: int = 0,
    tool_call_count: int = 0,
) -> Dict[str, object]:
    """Score one model answer against hidden evaluator labels."""
    predicted_label = str(normalized_output["predicted_label"])
    contradiction_status = str(normalized_output["contradiction_status"])
    recommended_action = str(normalized_output["recommended_action"])
    label_correct = predicted_label == sample.true_label_name
    contradiction_correct = contradiction_status == sample.true_contradiction_status
    action_correct = recommended_action == sample.recommended_action

    return {
        "sample_id": sample.sample_id,
        "model": model,
        "resolved_model": resolved_model or model,
        "input_condition": input_condition,
        "prompt_policy": prompt_policy,
        "thinking": thinking,
        "tool_level": tool_level,
        "contradiction_strength": sample.contradiction_strength,
        "true_label": sample.true_label_name,
        "predicted_label": predicted_label,
        "label_confidence": normalized_output["label_confidence"],
        "label_correct": label_correct,
        "true_contradiction_status": sample.true_contradiction_status,
        "predicted_contradiction_status": contradiction_status,
        "contradiction_correct": contradiction_correct,
        "true_recommended_action": sample.recommended_action,
        "predicted_recommended_action": recommended_action,
        "action_correct": action_correct,
        "trusted_evidence": normalized_output["trusted_evidence"],
        "rationale": normalized_output["rationale"],
        "image_evidence": normalized_output.get("image_evidence", ""),
        "hic_evidence": normalized_output.get("hic_evidence", ""),
        "image_predicted_label": normalized_output.get("image_predicted_label", "not_reported"),
        "hic_predicted_label": normalized_output.get("hic_predicted_label", "not_reported"),
        "modality_agreement": normalized_output.get("modality_agreement", "not_reported"),
        "malformed": bool(normalized_output["malformed"]),
        "latency_seconds": float(latency_seconds),
        "input_tokens": int(input_tokens),
        "output_tokens": int(output_tokens),
        "tool_call_count": int(tool_call_count),
        "raw_output": raw_output,
    }


def summarize_agent_records(records: Iterable[Mapping[str, object]]) -> Dict[str, object]:
    """Aggregate scored records into benchmark metrics."""
    rows = list(records)
    if not rows:
        return {"n": 0}

    label_correct = np.asarray([bool(row["label_correct"]) for row in rows], dtype=bool)
    contradiction_correct = np.asarray([bool(row["contradiction_correct"]) for row in rows], dtype=bool)
    action_correct = np.asarray([bool(row["action_correct"]) for row in rows], dtype=bool)
    malformed = np.asarray([bool(row["malformed"]) for row in rows], dtype=bool)
    reconciliation_mask = np.asarray(
        [str(row.get("input_condition", "both_modalities")) == "both_modalities" for row in rows],
        dtype=bool,
    )
    true_conflict = np.asarray([row["true_contradiction_status"] != "none" for row in rows], dtype=bool)
    pred_conflict = np.asarray(
        [row["predicted_contradiction_status"] in {"weak", "strong"} for row in rows],
        dtype=bool,
    )

    summary: Dict[str, object] = {
        "n": len(rows),
        "classification_accuracy": float(label_correct.mean()),
        "contradiction_accuracy": float(contradiction_correct[reconciliation_mask].mean())
        if np.any(reconciliation_mask)
        else float("nan"),
        "contradiction_binary_f1": _binary_f1(true_conflict[reconciliation_mask], pred_conflict[reconciliation_mask])
        if np.any(reconciliation_mask)
        else float("nan"),
        "recommended_action_accuracy": float(action_correct[reconciliation_mask].mean())
        if np.any(reconciliation_mask)
        else float("nan"),
        "malformed_fraction": float(malformed.mean()),
        "abstention_rate": float(np.mean([row["predicted_recommended_action"] == "abstain" for row in rows])),
        "ece_10bin": expected_calibration_error(rows, n_bins=10),
        "mean_latency_seconds": float(np.mean([float(row["latency_seconds"]) for row in rows])),
        "mean_tool_call_count": float(np.mean([int(row["tool_call_count"]) for row in rows])),
        "total_input_tokens": int(np.sum([int(row["input_tokens"]) for row in rows])),
        "total_output_tokens": int(np.sum([int(row["output_tokens"]) for row in rows])),
    }

    by_strength = {}
    for strength in sorted({float(row["contradiction_strength"]) for row in rows}):
        subset = [row for row in rows if math.isclose(float(row["contradiction_strength"]), strength)]
        by_strength[f"{strength:.2f}"] = summarize_agent_records(subset) if len(subset) < len(rows) else {
            "n": len(subset),
            "classification_accuracy": float(np.mean([bool(row["label_correct"]) for row in subset])),
            "contradiction_accuracy": float(np.mean([bool(row["contradiction_correct"]) for row in subset])),
            "recommended_action_accuracy": float(np.mean([bool(row["action_correct"]) for row in subset])),
        }
    summary["by_contradiction_strength"] = by_strength
    return summary


def safe_normalize_agent_output(raw_output: str | Mapping[str, object]) -> Dict[str, object]:
    """Normalize model output and convert parse failures to malformed records."""
    try:
        return normalize_agent_output(raw_output)
    except Exception as exc:  # noqa: BLE001 - parser errors should become benchmark data.
        return malformed_agent_output(exc)
