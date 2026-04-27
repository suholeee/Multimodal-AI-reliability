"""V2 evaluation helpers for contradiction-conditioned reliability analysis."""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from ml_train_eval import evaluate_probability_predictions


def binary_probabilities_from_positive(positive_probability: np.ndarray) -> np.ndarray:
    """Convert binary positive-class probabilities into a two-column table."""
    positive = np.clip(np.asarray(positive_probability, dtype=float), 1e-8, 1.0 - 1e-8)
    return np.column_stack((1.0 - positive, positive))


def brier_score_binary(positive_probability: np.ndarray, labels: np.ndarray) -> float:
    """Compute the binary Brier score for positive-class probabilities."""
    positive = np.asarray(positive_probability, dtype=float)
    label_array = np.asarray(labels, dtype=float)
    return float(np.mean((positive - label_array) ** 2))


def expected_calibration_error_binary(
    positive_probability: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 10,
) -> float:
    """Estimate binary expected calibration error from confidence bins."""
    probabilities = binary_probabilities_from_positive(positive_probability)
    predictions = probabilities.argmax(axis=1)
    confidence = probabilities.max(axis=1)
    correct = predictions == np.asarray(labels, dtype=int)

    edges = np.linspace(0.5, 1.0, n_bins + 1)
    total_weight = float(len(confidence))
    calibration_error = 0.0

    for bin_idx in range(n_bins):
        lower = edges[bin_idx]
        upper = edges[bin_idx + 1]
        if bin_idx == n_bins - 1:
            mask = (confidence >= lower) & (confidence <= upper)
        else:
            mask = (confidence >= lower) & (confidence < upper)
        if not np.any(mask):
            continue
        bin_accuracy = float(correct[mask].mean())
        bin_confidence = float(confidence[mask].mean())
        calibration_error += float(mask.mean()) * abs(bin_accuracy - bin_confidence)

    return float(calibration_error if total_weight > 0 else float("nan"))


def _masked_mean(values: np.ndarray, mask: np.ndarray) -> float:
    """Compute a mean over one Boolean mask."""
    array = np.asarray(values, dtype=float)
    active = np.asarray(mask, dtype=bool)
    if not np.any(active):
        return float("nan")
    return float(array[active].mean())


def _masked_accuracy(correct_mask: np.ndarray, mask: np.ndarray) -> float:
    """Compute accuracy over one subset."""
    return _masked_mean(np.asarray(correct_mask, dtype=float), mask)


def build_split_sample_table(
    dataset: Dict[str, object],
    split_name: str,
    image_eval: Dict[str, object],
    genomic_eval: Dict[str, object],
    fusion_eval: Optional[Dict[str, object]] = None,
    seed: Optional[int] = None,
) -> Dict[str, np.ndarray]:
    """Collect per-sample split outputs and latent diagnostics for V2 evaluation."""
    split_indices = dataset["splits"][split_name]["index"].cpu().numpy().astype(np.int64)
    arrays = dataset["arrays"]

    image_probability = np.asarray(image_eval["positive_probability"], dtype=float)
    genomic_probability = np.asarray(genomic_eval["positive_probability"], dtype=float)
    image_confidence = np.asarray(image_eval["confidence"], dtype=float)
    genomic_confidence = np.asarray(genomic_eval["confidence"], dtype=float)
    image_entropy = np.asarray(image_eval["entropy"], dtype=float)
    genomic_entropy = np.asarray(genomic_eval["entropy"], dtype=float)
    image_prediction = np.asarray(image_eval["predictions"], dtype=int)
    genomic_prediction = np.asarray(genomic_eval["predictions"], dtype=int)
    labels = np.asarray(image_eval["labels"], dtype=int)
    image_correct = np.asarray(image_eval["correct_mask"], dtype=bool)
    genomic_correct = np.asarray(genomic_eval["correct_mask"], dtype=bool)
    disagreement = image_prediction != genomic_prediction

    table = {
        "sample_index": split_indices,
        "true_label": labels,
        "image_prediction": image_prediction,
        "genomic_prediction": genomic_prediction,
        "image_probability_cancer": image_probability,
        "genomic_probability_cancer": genomic_probability,
        "image_confidence": image_confidence,
        "genomic_confidence": genomic_confidence,
        "image_entropy": image_entropy,
        "genomic_entropy": genomic_entropy,
        "image_correct": image_correct.astype(np.int64),
        "genomic_correct": genomic_correct.astype(np.int64),
        "disagreement": disagreement.astype(np.int64),
        "probability_gap": np.abs(image_probability - genomic_probability),
        "signed_probability_gap": image_probability - genomic_probability,
        "confidence_gap": image_confidence - genomic_confidence,
        "entropy_gap": image_entropy - genomic_entropy,
    }
    if seed is not None:
        table["seed"] = np.full(len(labels), int(seed), dtype=np.int64)

    condition_array = np.asarray(arrays.get("condition", np.asarray([], dtype=str)))
    if condition_array.size > 0:
        table["condition_name"] = condition_array[split_indices]

    for key, value in arrays.items():
        if key.startswith("latent_"):
            table[key] = np.asarray(value)[split_indices]

    if fusion_eval is not None:
        fusion_probability = np.asarray(fusion_eval["positive_probability"], dtype=float)
        table["fusion_prediction"] = np.asarray(fusion_eval["predictions"], dtype=int)
        table["fusion_probability_cancer"] = fusion_probability
        table["fusion_confidence"] = np.asarray(fusion_eval["confidence"], dtype=float)
        table["fusion_entropy"] = np.asarray(fusion_eval["entropy"], dtype=float)
        table["fusion_correct"] = np.asarray(fusion_eval["correct_mask"], dtype=bool).astype(np.int64)
        table["fusion_brier"] = np.full(
            len(fusion_probability),
            brier_score_binary(fusion_probability, labels),
            dtype=float,
        )

    return table


def summarize_strategy(
    sample_table: Dict[str, np.ndarray],
    positive_probability: np.ndarray,
    high_confidence_threshold: float = 0.70,
    strong_contradiction_threshold: float = 0.40,
    abstain_mask: Optional[np.ndarray] = None,
) -> Dict[str, object]:
    """Summarize one strategy with conditional reliability and regret metrics."""
    labels = np.asarray(sample_table["true_label"], dtype=int)
    probabilities = binary_probabilities_from_positive(positive_probability)
    full_metrics = evaluate_probability_predictions(
        probabilities=probabilities,
        labels=labels,
        high_confidence_threshold=high_confidence_threshold,
    )

    evaluation_mask = np.ones(len(labels), dtype=bool)
    if abstain_mask is not None:
        evaluation_mask = ~np.asarray(abstain_mask, dtype=bool)

    if np.any(evaluation_mask):
        selected_probabilities = probabilities[evaluation_mask]
        selected_labels = labels[evaluation_mask]
        selected_metrics = evaluate_probability_predictions(
            probabilities=selected_probabilities,
            labels=selected_labels,
            high_confidence_threshold=high_confidence_threshold,
        )
    else:
        selected_metrics = {
            "accuracy": float("nan"),
            "mean_confidence": float("nan"),
            "mean_confidence_correct": float("nan"),
            "mean_confidence_incorrect": float("nan"),
            "high_confidence_fraction": float("nan"),
            "high_confidence_error_rate": float("nan"),
            "predictions": np.asarray([], dtype=int),
            "confidence": np.asarray([], dtype=float),
            "correct_mask": np.asarray([], dtype=bool),
        }

    full_correct = np.asarray(full_metrics["correct_mask"], dtype=bool)
    disagreement_mask = np.asarray(sample_table["disagreement"], dtype=bool)
    agreement_mask = ~disagreement_mask
    strong_mask = np.asarray(
        sample_table.get("latent_structural_contradiction_score", np.zeros(len(labels), dtype=float)),
        dtype=float,
    ) >= float(strong_contradiction_threshold)
    oracle_correct = np.asarray(sample_table["image_correct"], dtype=bool) | np.asarray(
        sample_table["genomic_correct"], dtype=bool
    )

    active_disagreement = disagreement_mask & evaluation_mask
    active_agreement = agreement_mask & evaluation_mask
    active_strong = strong_mask & evaluation_mask
    active_correct = full_correct & evaluation_mask
    active_incorrect = (~full_correct) & evaluation_mask
    regret_mask = (~full_correct) & oracle_correct & evaluation_mask

    positive_array = np.asarray(positive_probability, dtype=float)

    return {
        "accuracy": float(selected_metrics["accuracy"]),
        "coverage": float(evaluation_mask.mean()),
        "abstained_fraction": float((~evaluation_mask).mean()),
        "full_coverage_accuracy": float(full_metrics["accuracy"]),
        "mean_confidence": float(selected_metrics["mean_confidence"]),
        "mean_confidence_correct": _masked_mean(full_metrics["confidence"], active_correct),
        "mean_confidence_incorrect": _masked_mean(full_metrics["confidence"], active_incorrect),
        "high_confidence_fraction": float(selected_metrics["high_confidence_fraction"]),
        "high_confidence_error_rate": float(selected_metrics["high_confidence_error_rate"]),
        "brier": brier_score_binary(positive_array[evaluation_mask], labels[evaluation_mask])
        if np.any(evaluation_mask)
        else float("nan"),
        "ece_10bin": expected_calibration_error_binary(
            positive_array[evaluation_mask],
            labels[evaluation_mask],
            n_bins=10,
        )
        if np.any(evaluation_mask)
        else float("nan"),
        "agreement_accuracy": _masked_accuracy(full_correct, active_agreement),
        "disagreement_accuracy": _masked_accuracy(full_correct, active_disagreement),
        "strong_contradiction_accuracy": _masked_accuracy(full_correct, active_strong),
        "disagreement_error_rate": _masked_mean(~full_correct, active_disagreement),
        "regret": _masked_mean(regret_mask, evaluation_mask),
        "regret_count": int(regret_mask.sum()),
        "oracle_accuracy": _masked_mean(oracle_correct, evaluation_mask),
        "agreement_fraction": float(active_agreement.mean()),
        "disagreement_fraction": float(active_disagreement.mean()),
        "strong_contradiction_fraction": float(active_strong.mean()),
    }
