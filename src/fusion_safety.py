"""Fusion training and safety-evaluation utilities for sandbox polymer experiments."""

from __future__ import annotations

from typing import Dict, Sequence

import numpy as np

from image_feature_layers import attach_image_features
from ml_dataset import build_paired_dataset
from ml_models import build_fusion_classifier, build_image_classifier, GenomicClassifier
from ml_train_eval import evaluate_model, train_model


DEFAULT_SAFETY_SEEDS = (21, 22, 23)
DEFAULT_COVERAGE_LEVELS = np.linspace(0.35, 1.00, 14)


def image_model_kind(image_model: str) -> str:
    """Resolve the training/evaluation path for one optical model name."""
    return "image_hybrid" if image_model == "hybrid" else "image"


def prepare_multimodal_dataset(
    mode: str,
    image_size: int,
    genomic_representation: str,
    seed: int,
    image_model: str,
) -> Dict[str, object]:
    """Build one paired dataset and attach image features when the model needs them."""
    dataset = build_paired_dataset(
        mode=mode,
        image_size=image_size,
        genomic_representation=genomic_representation,
        seed=seed,
    )
    if image_model == "hybrid":
        dataset = attach_image_features(dataset)
    return dataset


def build_model_bundle(
    dataset: Dict[str, object],
    image_model: str,
) -> Dict[str, Dict[str, object]]:
    """Construct image, genomic, and fusion models for one dataset."""
    splits = dataset["splits"]
    image_shape = tuple(splits["train"]["image"].shape[1:])
    genomic_shape = tuple(splits["train"]["genomic"].shape[1:])
    image_feature_dim = int(splits["train"]["image_feature"].shape[1]) if image_model == "hybrid" else None

    return {
        "image": {
            "model": build_image_classifier(
                name=image_model,
                image_shape=image_shape,
                feature_dim=image_feature_dim,
            ),
            "model_kind": image_model_kind(image_model),
        },
        "genomic": {
            "model": GenomicClassifier(input_shape=genomic_shape),
            "model_kind": "genomic",
        },
        "fusion": {
            "model": build_fusion_classifier(
                genomic_input_shape=genomic_shape,
                image_shape=image_shape,
                image_model=image_model,
                image_feature_dim=image_feature_dim,
            ),
            "model_kind": "fusion",
        },
    }


def _evaluate_all_splits(
    model,
    splits: Dict[str, Dict[str, object]],
    model_kind: str,
    device: str,
    high_confidence_threshold: float,
) -> Dict[str, Dict[str, object]]:
    """Evaluate one trained model on train/val/test splits."""
    return {
        split_name: evaluate_model(
            model=model,
            split=split,
            model_kind=model_kind,
            device=device,
            high_confidence_threshold=high_confidence_threshold,
        )
        for split_name, split in splits.items()
    }


def train_multimodal_models(
    dataset: Dict[str, object],
    image_model: str,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    patience: int,
    min_epochs: int,
    device: str,
    seed: int,
    high_confidence_threshold: float,
) -> Dict[str, Dict[str, object]]:
    """Train the main unimodal and fusion models for one dataset seed."""
    model_bundle = build_model_bundle(dataset, image_model=image_model)
    results: Dict[str, Dict[str, object]] = {}

    for offset, model_name in enumerate(("image", "genomic", "fusion")):
        spec = model_bundle[model_name]
        history = train_model(
            model=spec["model"],
            train_split=dataset["splits"]["train"],
            val_split=dataset["splits"]["val"],
            model_kind=spec["model_kind"],
            n_epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            device=device,
            seed=seed + offset,
            verbose=False,
            early_stopping_patience=patience,
            early_stopping_metric="accuracy",
            min_epochs=min_epochs,
        )
        split_metrics = _evaluate_all_splits(
            model=spec["model"],
            splits=dataset["splits"],
            model_kind=spec["model_kind"],
            device=device,
            high_confidence_threshold=high_confidence_threshold,
        )
        results[model_name] = {
            "history": history,
            "split_metrics": split_metrics,
        }

    return results


def compute_safety_signals(
    image_eval: Dict[str, object],
    genomic_eval: Dict[str, object],
    fusion_eval: Dict[str, object],
) -> Dict[str, np.ndarray]:
    """Derive interpretable output-only oversight signals."""
    image_probability = np.asarray(image_eval["positive_probability"], dtype=float)
    genomic_probability = np.asarray(genomic_eval["positive_probability"], dtype=float)
    fusion_probability = np.asarray(fusion_eval["positive_probability"], dtype=float)

    image_confidence = np.asarray(image_eval["confidence"], dtype=float)
    genomic_confidence = np.asarray(genomic_eval["confidence"], dtype=float)
    fusion_confidence = np.asarray(fusion_eval["confidence"], dtype=float)

    image_entropy = np.asarray(image_eval["entropy"], dtype=float)
    genomic_entropy = np.asarray(genomic_eval["entropy"], dtype=float)
    fusion_entropy = np.asarray(fusion_eval["entropy"], dtype=float)

    image_prediction = np.asarray(image_eval["predictions"], dtype=int)
    genomic_prediction = np.asarray(genomic_eval["predictions"], dtype=int)
    fusion_prediction = np.asarray(fusion_eval["predictions"], dtype=int)
    labels = np.asarray(fusion_eval["labels"], dtype=int)

    prediction_disagreement = image_prediction != genomic_prediction
    disagreement_score = np.abs(image_probability - genomic_probability)
    max_unimodal_confidence = np.maximum(image_confidence, genomic_confidence)
    mean_unimodal_confidence = 0.5 * (image_confidence + genomic_confidence)
    fusion_confidence_gap = fusion_confidence - max_unimodal_confidence

    return {
        "labels": labels,
        "image_probability": image_probability,
        "genomic_probability": genomic_probability,
        "fusion_probability": fusion_probability,
        "image_confidence": image_confidence,
        "genomic_confidence": genomic_confidence,
        "fusion_confidence": fusion_confidence,
        "image_entropy": image_entropy,
        "genomic_entropy": genomic_entropy,
        "fusion_entropy": fusion_entropy,
        "image_prediction": image_prediction,
        "genomic_prediction": genomic_prediction,
        "fusion_prediction": fusion_prediction,
        "fusion_correct": fusion_prediction == labels,
        "prediction_disagreement": prediction_disagreement,
        "disagreement_score": disagreement_score,
        "max_unimodal_confidence": max_unimodal_confidence,
        "mean_unimodal_confidence": mean_unimodal_confidence,
        "fusion_confidence_gap": fusion_confidence_gap,
    }


def calibrate_oversight_thresholds(
    validation_signals: Dict[str, np.ndarray],
    disagreement_quantile: float = 0.75,
    entropy_quantile: float = 0.75,
    confidence_gap_quantile: float = 0.80,
) -> Dict[str, float]:
    """Choose simple oversight thresholds from validation data only."""
    disagreement_threshold = float(
        max(0.10, np.quantile(validation_signals["disagreement_score"], disagreement_quantile))
    )
    fusion_entropy_threshold = float(
        max(0.45, np.quantile(validation_signals["fusion_entropy"], entropy_quantile))
    )

    disagreement_mask = np.asarray(validation_signals["prediction_disagreement"], dtype=bool)
    if np.any(disagreement_mask):
        candidate_gap = validation_signals["fusion_confidence_gap"][disagreement_mask]
    else:
        candidate_gap = validation_signals["fusion_confidence_gap"]
    confidence_gap_threshold = float(max(0.02, np.quantile(candidate_gap, confidence_gap_quantile)))

    return {
        "disagreement_threshold": disagreement_threshold,
        "fusion_entropy_threshold": fusion_entropy_threshold,
        "confidence_gap_threshold": confidence_gap_threshold,
    }


def compute_risk_score(signals: Dict[str, np.ndarray]) -> np.ndarray:
    """Combine a few interpretable signals into one monotonic oversight score."""
    disagreement = np.asarray(signals["disagreement_score"], dtype=float)
    fusion_entropy = np.asarray(signals["fusion_entropy"], dtype=float)
    prediction_disagreement = np.asarray(signals["prediction_disagreement"], dtype=float)
    positive_gap = np.maximum(np.asarray(signals["fusion_confidence_gap"], dtype=float), 0.0)
    score = disagreement + 0.20 * prediction_disagreement + 0.55 * fusion_entropy + 0.45 * positive_gap
    return score / (1.0 + 0.20 + 0.55 + 0.45)


def flag_unreliable_samples(
    signals: Dict[str, np.ndarray],
    thresholds: Dict[str, float],
) -> Dict[str, np.ndarray]:
    """Apply a fixed rule-based oversight policy to one split."""
    prediction_disagreement = np.asarray(signals["prediction_disagreement"], dtype=bool)
    disagreement_score = np.asarray(signals["disagreement_score"], dtype=float)
    fusion_entropy = np.asarray(signals["fusion_entropy"], dtype=float)
    fusion_confidence_gap = np.asarray(signals["fusion_confidence_gap"], dtype=float)

    disagreement_flag = prediction_disagreement & (
        disagreement_score >= float(thresholds["disagreement_threshold"])
    )
    entropy_flag = fusion_entropy >= float(thresholds["fusion_entropy_threshold"])
    confidence_gap_flag = prediction_disagreement & (
        fusion_confidence_gap >= float(thresholds["confidence_gap_threshold"])
    )
    flags = disagreement_flag | entropy_flag | confidence_gap_flag

    return {
        "flags": flags,
        "risk_score": compute_risk_score(signals),
        "disagreement_flag": disagreement_flag,
        "entropy_flag": entropy_flag,
        "confidence_gap_flag": confidence_gap_flag,
    }


def selective_abstention_metrics(
    fusion_eval: Dict[str, object],
    flagged_output: Dict[str, np.ndarray],
    thresholds: Dict[str, float],
) -> Dict[str, object]:
    """Evaluate the selective-abstention strategy on one split."""
    flags = np.asarray(flagged_output["flags"], dtype=bool)
    retained = ~flags
    fusion_correct = np.asarray(fusion_eval["correct_mask"], dtype=bool)

    return {
        "flags": flags,
        "risk_score": np.asarray(flagged_output["risk_score"], dtype=float),
        "flagged_fraction": float(flags.mean()),
        "retained_fraction": float(retained.mean()),
        "error_rate_flagged": float((~fusion_correct[flags]).mean()) if np.any(flags) else float("nan"),
        "accuracy_unflagged": float(fusion_correct[retained].mean()) if np.any(retained) else float("nan"),
        "selective_accuracy": float(fusion_correct[retained].mean()) if np.any(retained) else float("nan"),
        "flag_counts": {
            "disagreement": int(np.asarray(flagged_output["disagreement_flag"], dtype=bool).sum()),
            "entropy": int(np.asarray(flagged_output["entropy_flag"], dtype=bool).sum()),
            "confidence_gap": int(np.asarray(flagged_output["confidence_gap_flag"], dtype=bool).sum()),
        },
        "thresholds": dict(thresholds),
    }


def gated_routing_metrics(
    image_eval: Dict[str, object],
    genomic_eval: Dict[str, object],
    fusion_eval: Dict[str, object],
    flags: np.ndarray,
    route_strategy: str = "lower_entropy",
) -> Dict[str, object]:
    """Route flagged samples away from fusion to the more reliable unimodal path."""
    if route_strategy not in {"lower_entropy", "higher_confidence"}:
        raise ValueError("route_strategy must be 'lower_entropy' or 'higher_confidence'")

    flags_array = np.asarray(flags, dtype=bool)
    labels = np.asarray(fusion_eval["labels"], dtype=int)
    fusion_prediction = np.asarray(fusion_eval["predictions"], dtype=int)
    fusion_confidence = np.asarray(fusion_eval["confidence"], dtype=float)
    fusion_probability = np.asarray(fusion_eval["positive_probability"], dtype=float)

    image_prediction = np.asarray(image_eval["predictions"], dtype=int)
    genomic_prediction = np.asarray(genomic_eval["predictions"], dtype=int)
    image_confidence = np.asarray(image_eval["confidence"], dtype=float)
    genomic_confidence = np.asarray(genomic_eval["confidence"], dtype=float)
    image_entropy = np.asarray(image_eval["entropy"], dtype=float)
    genomic_entropy = np.asarray(genomic_eval["entropy"], dtype=float)
    image_probability = np.asarray(image_eval["positive_probability"], dtype=float)
    genomic_probability = np.asarray(genomic_eval["positive_probability"], dtype=float)

    if route_strategy == "lower_entropy":
        use_image = image_entropy <= genomic_entropy
    else:
        use_image = image_confidence >= genomic_confidence

    routed_prediction = fusion_prediction.copy()
    routed_confidence = fusion_confidence.copy()
    routed_probability = fusion_probability.copy()

    routed_prediction[flags_array] = np.where(use_image[flags_array], image_prediction[flags_array], genomic_prediction[flags_array])
    routed_confidence[flags_array] = np.where(use_image[flags_array], image_confidence[flags_array], genomic_confidence[flags_array])
    routed_probability[flags_array] = np.where(use_image[flags_array], image_probability[flags_array], genomic_probability[flags_array])

    routed_correct = routed_prediction == labels
    return {
        "routed_predictions": routed_prediction,
        "routed_confidence": routed_confidence,
        "routed_probability": routed_probability,
        "routed_accuracy": float(routed_correct.mean()),
        "routed_fraction": float(flags_array.mean()),
        "route_strategy": route_strategy,
        "use_image_mask": use_image,
    }


def disagreement_error_curve(
    disagreement_score: np.ndarray,
    correct_mask: np.ndarray,
    n_bins: int = 8,
) -> Dict[str, np.ndarray]:
    """Bin disagreement scores and estimate fusion error probability."""
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    centers = []
    error_rates = []
    counts = []

    disagreement_array = np.asarray(disagreement_score, dtype=float)
    correct_array = np.asarray(correct_mask, dtype=bool)

    for idx in range(n_bins):
        lower = edges[idx]
        upper = edges[idx + 1]
        if idx == n_bins - 1:
            mask = (disagreement_array >= lower) & (disagreement_array <= upper)
        else:
            mask = (disagreement_array >= lower) & (disagreement_array < upper)
        if np.any(mask):
            centers.append(0.5 * (lower + upper))
            error_rates.append(float((~correct_array[mask]).mean()))
            counts.append(int(mask.sum()))

    return {
        "centers": np.asarray(centers, dtype=float),
        "error_rates": np.asarray(error_rates, dtype=float),
        "counts": np.asarray(counts, dtype=int),
    }


def selective_accuracy_curve(
    risk_score: np.ndarray,
    correct_mask: np.ndarray,
    coverage_levels: Sequence[float] = DEFAULT_COVERAGE_LEVELS,
) -> Dict[str, np.ndarray]:
    """Compute retained accuracy as a function of target coverage."""
    score = np.asarray(risk_score, dtype=float)
    correct = np.asarray(correct_mask, dtype=bool)
    ordering = np.argsort(score)

    coverages = []
    retained_accuracy = []
    retained_counts = []
    thresholds = []

    n_samples = len(score)
    for target_coverage in coverage_levels:
        keep_count = max(1, min(n_samples, int(np.floor(float(target_coverage) * n_samples))))
        kept_indices = ordering[:keep_count]
        retained_mask = np.zeros(n_samples, dtype=bool)
        retained_mask[kept_indices] = True
        coverages.append(float(retained_mask.mean()))
        retained_accuracy.append(float(correct[retained_mask].mean()))
        retained_counts.append(int(keep_count))
        thresholds.append(float(score[ordering[keep_count - 1]]))

    return {
        "coverage": np.asarray(coverages, dtype=float),
        "retained_accuracy": np.asarray(retained_accuracy, dtype=float),
        "retained_count": np.asarray(retained_counts, dtype=int),
        "risk_threshold": np.asarray(thresholds, dtype=float),
    }


def selective_accuracy_at_coverage(
    risk_score: np.ndarray,
    correct_mask: np.ndarray,
    target_coverage: float,
) -> float:
    """Read off retained accuracy at one target coverage."""
    curve = selective_accuracy_curve(risk_score, correct_mask, coverage_levels=(target_coverage,))
    return float(curve["retained_accuracy"][0])


def run_safety_evaluation_for_seed(
    seed: int,
    mode: str,
    image_size: int,
    genomic_representation: str,
    image_model: str,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    patience: int,
    min_epochs: int,
    device: str,
    route_strategy: str,
    high_confidence_threshold: float,
) -> Dict[str, object]:
    """Train one multimodal run and evaluate safety on its held-out split."""
    dataset = prepare_multimodal_dataset(
        mode=mode,
        image_size=image_size,
        genomic_representation=genomic_representation,
        seed=seed,
        image_model=image_model,
    )
    model_results = train_multimodal_models(
        dataset=dataset,
        image_model=image_model,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        patience=patience,
        min_epochs=min_epochs,
        device=device,
        seed=seed,
        high_confidence_threshold=high_confidence_threshold,
    )

    image_val = model_results["image"]["split_metrics"]["val"]
    genomic_val = model_results["genomic"]["split_metrics"]["val"]
    fusion_val = model_results["fusion"]["split_metrics"]["val"]
    image_test = model_results["image"]["split_metrics"]["test"]
    genomic_test = model_results["genomic"]["split_metrics"]["test"]
    fusion_test = model_results["fusion"]["split_metrics"]["test"]

    validation_signals = compute_safety_signals(image_val, genomic_val, fusion_val)
    thresholds = calibrate_oversight_thresholds(validation_signals)

    test_signals = compute_safety_signals(image_test, genomic_test, fusion_test)
    flagged_output = flag_unreliable_samples(test_signals, thresholds)
    abstention = selective_abstention_metrics(fusion_test, flagged_output, thresholds)
    routing = gated_routing_metrics(
        image_eval=image_test,
        genomic_eval=genomic_test,
        fusion_eval=fusion_test,
        flags=abstention["flags"],
        route_strategy=route_strategy,
    )
    curve = selective_accuracy_curve(abstention["risk_score"], fusion_test["correct_mask"])
    disagreement_curve = disagreement_error_curve(
        disagreement_score=test_signals["disagreement_score"],
        correct_mask=fusion_test["correct_mask"],
    )

    return {
        "dataset_meta": {
            "mode": dataset["mode"],
            "n_monomers": dataset["n_monomers"],
            "image_size": dataset["image_size"],
            "genomic_representation": dataset["genomic_representation"],
            "image_model": image_model,
        },
        "models": model_results,
        "signals": test_signals,
        "abstention": abstention,
        "routing": routing,
        "curve": curve,
        "disagreement_curve": disagreement_curve,
        "thresholds": thresholds,
    }
