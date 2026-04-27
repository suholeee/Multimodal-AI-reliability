"""V2 decision-policy baselines for contradiction-aware multimodal routing."""

from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np

from fusion_safety import (
    calibrate_oversight_thresholds,
    compute_risk_score,
    flag_unreliable_samples,
    selective_accuracy_curve,
)
from v2_metrics import binary_probabilities_from_positive


DEFAULT_POLICY_THRESHOLD_GRID = np.linspace(0.0, 1.0, 21)
DEFAULT_SELECTIVE_COVERAGE_LEVELS = np.asarray((0.50, 0.60, 0.70, 0.80, 0.90, 1.00), dtype=float)


def _prediction_accuracy(positive_probability: np.ndarray, labels: np.ndarray) -> float:
    """Compute hard-label accuracy from positive-class probabilities."""
    probabilities = binary_probabilities_from_positive(positive_probability)
    predictions = probabilities.argmax(axis=1)
    return float((predictions == np.asarray(labels, dtype=int)).mean())


def _prediction_confidence(positive_probability: np.ndarray) -> np.ndarray:
    """Convert binary probabilities into scalar confidence values."""
    probabilities = binary_probabilities_from_positive(positive_probability)
    return probabilities.max(axis=1)


def _prediction_labels(positive_probability: np.ndarray) -> np.ndarray:
    """Convert binary probabilities into class predictions."""
    probabilities = binary_probabilities_from_positive(positive_probability)
    return probabilities.argmax(axis=1)


def build_safety_signals_from_table(sample_table: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """Reconstruct the interpretable fusion-safety signals from a split sample table."""
    image_probability = np.asarray(sample_table["image_probability_cancer"], dtype=float)
    genomic_probability = np.asarray(sample_table["genomic_probability_cancer"], dtype=float)
    fusion_probability = np.asarray(sample_table["fusion_probability_cancer"], dtype=float)
    image_confidence = np.asarray(sample_table["image_confidence"], dtype=float)
    genomic_confidence = np.asarray(sample_table["genomic_confidence"], dtype=float)
    fusion_confidence = np.asarray(sample_table["fusion_confidence"], dtype=float)
    image_entropy = np.asarray(sample_table["image_entropy"], dtype=float)
    genomic_entropy = np.asarray(sample_table["genomic_entropy"], dtype=float)
    fusion_entropy = np.asarray(sample_table["fusion_entropy"], dtype=float)

    return {
        "labels": np.asarray(sample_table["true_label"], dtype=int),
        "image_probability": image_probability,
        "genomic_probability": genomic_probability,
        "fusion_probability": fusion_probability,
        "image_confidence": image_confidence,
        "genomic_confidence": genomic_confidence,
        "fusion_confidence": fusion_confidence,
        "image_entropy": image_entropy,
        "genomic_entropy": genomic_entropy,
        "fusion_entropy": fusion_entropy,
        "image_prediction": np.asarray(sample_table["image_prediction"], dtype=int),
        "genomic_prediction": np.asarray(sample_table["genomic_prediction"], dtype=int),
        "fusion_prediction": np.asarray(sample_table["fusion_prediction"], dtype=int),
        "fusion_correct": np.asarray(sample_table["fusion_correct"], dtype=bool),
        "prediction_disagreement": np.asarray(sample_table["disagreement"], dtype=bool),
        "disagreement_score": np.abs(image_probability - genomic_probability),
        "max_unimodal_confidence": np.maximum(image_confidence, genomic_confidence),
        "mean_unimodal_confidence": 0.5 * (image_confidence + genomic_confidence),
        "fusion_confidence_gap": fusion_confidence - np.maximum(image_confidence, genomic_confidence),
    }


def tune_genomic_first_policy(
    validation_table: Dict[str, np.ndarray],
    threshold_grid: Optional[Sequence[float]] = None,
) -> Dict[str, float]:
    """Tune the disagreement threshold for the genomic-first fallback policy."""
    thresholds = DEFAULT_POLICY_THRESHOLD_GRID if threshold_grid is None else np.asarray(threshold_grid, dtype=float)
    labels = np.asarray(validation_table["true_label"], dtype=int)
    disagreement = np.asarray(validation_table["disagreement"], dtype=bool)
    probability_gap = np.asarray(validation_table["probability_gap"], dtype=float)
    fusion_positive = np.asarray(validation_table["fusion_probability_cancer"], dtype=float)
    genomic_positive = np.asarray(validation_table["genomic_probability_cancer"], dtype=float)

    best_threshold = float(thresholds[0])
    best_accuracy = -np.inf
    best_route_fraction = float("inf")

    for threshold in thresholds:
        use_genomic = disagreement & (probability_gap >= float(threshold))
        candidate_positive = np.where(use_genomic, genomic_positive, fusion_positive)
        candidate_accuracy = _prediction_accuracy(candidate_positive, labels)
        candidate_route_fraction = float(use_genomic.mean())
        if (
            candidate_accuracy > best_accuracy + 1e-8
            or (
                abs(candidate_accuracy - best_accuracy) <= 1e-8
                and candidate_route_fraction < best_route_fraction - 1e-8
            )
        ):
            best_threshold = float(threshold)
            best_accuracy = float(candidate_accuracy)
            best_route_fraction = candidate_route_fraction

    return {
        "threshold": best_threshold,
        "validation_accuracy": best_accuracy,
        "route_fraction": best_route_fraction,
    }


def apply_genomic_first_policy(
    sample_table: Dict[str, np.ndarray],
    threshold: float,
) -> Dict[str, np.ndarray | float]:
    """Use fusion by default and fall back to genomics under strong disagreement."""
    disagreement = np.asarray(sample_table["disagreement"], dtype=bool)
    probability_gap = np.asarray(sample_table["probability_gap"], dtype=float)
    fusion_positive = np.asarray(sample_table["fusion_probability_cancer"], dtype=float)
    genomic_positive = np.asarray(sample_table["genomic_probability_cancer"], dtype=float)

    use_genomic = disagreement & (probability_gap >= float(threshold))
    positive_probability = np.where(use_genomic, genomic_positive, fusion_positive)
    return {
        "positive_probability": positive_probability,
        "prediction": _prediction_labels(positive_probability),
        "confidence": _prediction_confidence(positive_probability),
        "use_genomic_mask": use_genomic,
        "use_fusion_mask": ~use_genomic,
        "threshold": float(threshold),
    }


def tune_confidence_routing_policy(
    validation_table: Dict[str, np.ndarray],
    threshold_grid: Optional[Sequence[float]] = None,
) -> Dict[str, float]:
    """Tune the confidence-gap threshold for routing to the more confident branch."""
    thresholds = DEFAULT_POLICY_THRESHOLD_GRID if threshold_grid is None else np.asarray(threshold_grid, dtype=float)
    labels = np.asarray(validation_table["true_label"], dtype=int)
    disagreement = np.asarray(validation_table["disagreement"], dtype=bool)
    confidence_gap = np.abs(np.asarray(validation_table["confidence_gap"], dtype=float))
    image_confidence = np.asarray(validation_table["image_confidence"], dtype=float)
    genomic_confidence = np.asarray(validation_table["genomic_confidence"], dtype=float)
    image_positive = np.asarray(validation_table["image_probability_cancer"], dtype=float)
    genomic_positive = np.asarray(validation_table["genomic_probability_cancer"], dtype=float)
    fusion_positive = np.asarray(validation_table["fusion_probability_cancer"], dtype=float)

    best_threshold = float(thresholds[0])
    best_accuracy = -np.inf
    best_route_fraction = float("inf")

    for threshold in thresholds:
        route_mask = disagreement & (confidence_gap >= float(threshold))
        use_image = route_mask & (image_confidence >= genomic_confidence)
        use_genomic = route_mask & ~use_image
        candidate_positive = fusion_positive.copy()
        candidate_positive[use_image] = image_positive[use_image]
        candidate_positive[use_genomic] = genomic_positive[use_genomic]

        candidate_accuracy = _prediction_accuracy(candidate_positive, labels)
        candidate_route_fraction = float(route_mask.mean())
        if (
            candidate_accuracy > best_accuracy + 1e-8
            or (
                abs(candidate_accuracy - best_accuracy) <= 1e-8
                and candidate_route_fraction < best_route_fraction - 1e-8
            )
        ):
            best_threshold = float(threshold)
            best_accuracy = float(candidate_accuracy)
            best_route_fraction = candidate_route_fraction

    return {
        "threshold": best_threshold,
        "validation_accuracy": best_accuracy,
        "route_fraction": best_route_fraction,
    }


def apply_confidence_routing_policy(
    sample_table: Dict[str, np.ndarray],
    threshold: float,
) -> Dict[str, np.ndarray | float]:
    """Route disagreement cases to the more confident unimodal branch."""
    disagreement = np.asarray(sample_table["disagreement"], dtype=bool)
    confidence_gap = np.abs(np.asarray(sample_table["confidence_gap"], dtype=float))
    image_confidence = np.asarray(sample_table["image_confidence"], dtype=float)
    genomic_confidence = np.asarray(sample_table["genomic_confidence"], dtype=float)
    image_positive = np.asarray(sample_table["image_probability_cancer"], dtype=float)
    genomic_positive = np.asarray(sample_table["genomic_probability_cancer"], dtype=float)
    fusion_positive = np.asarray(sample_table["fusion_probability_cancer"], dtype=float)

    route_mask = disagreement & (confidence_gap >= float(threshold))
    use_image = route_mask & (image_confidence >= genomic_confidence)
    use_genomic = route_mask & ~use_image

    positive_probability = fusion_positive.copy()
    positive_probability[use_image] = image_positive[use_image]
    positive_probability[use_genomic] = genomic_positive[use_genomic]

    return {
        "positive_probability": positive_probability,
        "prediction": _prediction_labels(positive_probability),
        "confidence": _prediction_confidence(positive_probability),
        "use_image_mask": use_image,
        "use_genomic_mask": use_genomic,
        "use_fusion_mask": ~route_mask,
        "threshold": float(threshold),
    }


def calibrate_abstention_policy(validation_table: Dict[str, np.ndarray]) -> Dict[str, object]:
    """Fit the fixed-rule abstention thresholds from validation data only."""
    validation_signals = build_safety_signals_from_table(validation_table)
    thresholds = calibrate_oversight_thresholds(validation_signals)
    flagged_output = flag_unreliable_samples(validation_signals, thresholds)
    return {
        "thresholds": thresholds,
        "validation_flagged_fraction": float(np.asarray(flagged_output["flags"], dtype=bool).mean()),
        "validation_risk_score_mean": float(np.asarray(flagged_output["risk_score"], dtype=float).mean()),
    }


def evaluate_abstention_policy(
    sample_table: Dict[str, np.ndarray],
    thresholds: Dict[str, float],
    coverage_levels: Sequence[float] = DEFAULT_SELECTIVE_COVERAGE_LEVELS,
) -> Dict[str, object]:
    """Evaluate fixed-threshold abstention and the full selective-accuracy curve."""
    signals = build_safety_signals_from_table(sample_table)
    flagged_output = flag_unreliable_samples(signals, thresholds)
    flags = np.asarray(flagged_output["flags"], dtype=bool)
    risk_score = np.asarray(flagged_output["risk_score"], dtype=float)
    fusion_correct = np.asarray(sample_table["fusion_correct"], dtype=bool)
    retained = ~flags
    curve = selective_accuracy_curve(
        risk_score=risk_score,
        correct_mask=fusion_correct,
        coverage_levels=coverage_levels,
    )

    return {
        "flags": flags,
        "risk_score": risk_score,
        "coverage": float(retained.mean()),
        "abstained_fraction": float(flags.mean()),
        "selective_accuracy": float(fusion_correct[retained].mean()) if np.any(retained) else float("nan"),
        "error_rate_flagged": float((~fusion_correct[flags]).mean()) if np.any(flags) else float("nan"),
        "curve": curve,
        "thresholds": dict(thresholds),
    }


def oracle_branch_accuracy(sample_table: Dict[str, np.ndarray]) -> float:
    """Upper bound from choosing a correct unimodal branch whenever available."""
    image_correct = np.asarray(sample_table["image_correct"], dtype=bool)
    genomic_correct = np.asarray(sample_table["genomic_correct"], dtype=bool)
    return float((image_correct | genomic_correct).mean())


def oracle_branch_regret(sample_table: Dict[str, np.ndarray]) -> float:
    """The oracle branch incurs no regret by construction."""
    _ = sample_table
    return 0.0


def contradiction_risk_score(sample_table: Dict[str, np.ndarray]) -> np.ndarray:
    """Expose the abstention risk score for downstream reporting."""
    return compute_risk_score(build_safety_signals_from_table(sample_table))
