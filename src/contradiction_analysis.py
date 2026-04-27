"""Cross-modal contradiction analysis for the sandbox multimodal classifiers."""

from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np
import torch

from image_feature_layers import attach_image_features
from ml_dataset import build_paired_dataset
from ml_models import GenomicClassifier, build_fusion_classifier, build_image_classifier
from ml_train_eval import evaluate_model, train_model


DEFAULT_CONTRADICTION_SEEDS = (21, 22, 23, 24, 25)
DEFAULT_CONTRADICTION_THRESHOLDS = (0.60, 0.70, 0.75, 0.80)
DEFAULT_IMAGE_MODEL = "hybrid"


def image_model_kind(image_model: str) -> str:
    """Resolve the forward-path variant for one image model."""
    return "image_hybrid" if image_model == "hybrid" else "image"


def default_epochs(mode: str) -> int:
    """Match the current diagnostic epoch budget."""
    if mode == "debug":
        return 35
    if mode == "full":
        return 45
    return 50


def default_batch_size(mode: str) -> int:
    """Scale batch size with dataset size."""
    return 16 if mode == "debug" else 32


def default_min_epochs(mode: str) -> int:
    """Require a modest warm-up before early stopping."""
    if mode == "debug":
        return 10
    if mode == "full":
        return 15
    return 18


def format_threshold_label(threshold: Optional[float]) -> str:
    """Format one contradiction threshold for printing and plotting."""
    if threshold is None:
        return "any disagreement"
    return f"conf>={threshold:.2f}"


def contradiction_level_specs(
    thresholds: Sequence[float] = DEFAULT_CONTRADICTION_THRESHOLDS,
) -> list[tuple[str, Optional[float]]]:
    """List the contradiction levels requested by the analysis."""
    return [("any_disagreement", None)] + [
        (f"conf_ge_{threshold:.2f}".replace(".", "_"), float(threshold))
        for threshold in thresholds
    ]


def _safe_mean(values: np.ndarray) -> float:
    """Average an array when it is non-empty."""
    array = np.asarray(values, dtype=float)
    finite = array[np.isfinite(array)]
    return float(finite.mean()) if finite.size > 0 else float("nan")


def _evaluate_all_splits(
    model,
    splits: Dict[str, Dict[str, torch.Tensor]],
    model_kind: str,
    device: str,
    high_confidence_threshold: float,
) -> Dict[str, Dict[str, object]]:
    """Evaluate one trained model on train/val/test."""
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


def prepare_dataset(
    mode: str,
    image_size: int,
    genomic_representation: str,
    seed: int,
    image_model: str,
) -> Dict[str, object]:
    """Build one paired dataset with the image-feature path attached when needed."""
    dataset = build_paired_dataset(
        mode=mode,
        image_size=image_size,
        genomic_representation=genomic_representation,
        seed=seed,
    )
    if image_model == "hybrid":
        dataset = attach_image_features(dataset)
    return dataset


def train_models_for_seed(
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
    include_fusion: bool = True,
) -> Dict[str, Dict[str, object]]:
    """Train the current image, genomic, and optional fusion models for one seed."""
    splits = dataset["splits"]
    image_shape = tuple(splits["train"]["image"].shape[1:])
    genomic_shape = tuple(splits["train"]["genomic"].shape[1:])
    image_feature_dim = int(splits["train"]["image_feature"].shape[1]) if image_model == "hybrid" else None

    model_specs = {
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
    }
    if include_fusion:
        model_specs["fusion"] = {
            "model": build_fusion_classifier(
                genomic_input_shape=genomic_shape,
                image_shape=image_shape,
                image_model=image_model,
                image_feature_dim=image_feature_dim,
            ),
            "model_kind": "fusion",
        }

    results: Dict[str, Dict[str, object]] = {}
    for offset, model_name in enumerate(model_specs):
        spec = model_specs[model_name]
        history = train_model(
            model=spec["model"],
            train_split=splits["train"],
            val_split=splits["val"],
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
            splits=splits,
            model_kind=spec["model_kind"],
            device=device,
            high_confidence_threshold=high_confidence_threshold,
        )
        results[model_name] = {
            "history": history,
            "split_metrics": split_metrics,
        }
    return results


def build_test_sample_table(
    seed: int,
    dataset: Dict[str, object],
    model_results: Dict[str, Dict[str, object]],
) -> Dict[str, np.ndarray]:
    """Collect the required per-sample test outputs for contradiction analysis."""
    test_indices = dataset["splits"]["test"]["index"].cpu().numpy().astype(np.int64)
    image_test = model_results["image"]["split_metrics"]["test"]
    genomic_test = model_results["genomic"]["split_metrics"]["test"]
    fusion_test = model_results["fusion"]["split_metrics"]["test"] if "fusion" in model_results else None

    image_probability = np.asarray(image_test["positive_probability"], dtype=float)
    genomic_probability = np.asarray(genomic_test["positive_probability"], dtype=float)
    image_confidence = np.asarray(image_test["confidence"], dtype=float)
    genomic_confidence = np.asarray(genomic_test["confidence"], dtype=float)
    image_prediction = np.asarray(image_test["predictions"], dtype=int)
    genomic_prediction = np.asarray(genomic_test["predictions"], dtype=int)
    labels = np.asarray(image_test["labels"], dtype=int)
    image_correct = np.asarray(image_test["correct_mask"], dtype=bool)
    genomic_correct = np.asarray(genomic_test["correct_mask"], dtype=bool)
    disagreement = image_prediction != genomic_prediction
    arrays = dataset["arrays"]

    table = {
        "seed": np.full(len(labels), int(seed), dtype=np.int64),
        "sample_index": test_indices,
        "true_label": labels,
        "image_prediction": image_prediction,
        "genomic_prediction": genomic_prediction,
        "image_probability_cancer": image_probability,
        "genomic_probability_cancer": genomic_probability,
        "image_confidence": image_confidence,
        "genomic_confidence": genomic_confidence,
        "image_correct": image_correct.astype(np.int64),
        "genomic_correct": genomic_correct.astype(np.int64),
        "disagreement": disagreement.astype(np.int64),
        "probability_gap": np.abs(image_probability - genomic_probability),
        "signed_probability_gap": image_probability - genomic_probability,
        "confidence_gap": image_confidence - genomic_confidence,
        "image_higher_confidence": (image_confidence > genomic_confidence).astype(np.int64),
        "genomic_higher_confidence": (genomic_confidence > image_confidence).astype(np.int64),
        "equal_confidence": np.isclose(image_confidence, genomic_confidence, atol=1e-8).astype(np.int64),
    }

    condition_array = np.asarray(arrays.get("condition", np.asarray([], dtype=str)))
    if condition_array.size > 0:
        table["condition_name"] = condition_array[test_indices]

    for key, value in arrays.items():
        if not key.startswith("latent_"):
            continue
        table[key] = np.asarray(value)[test_indices]

    if fusion_test is not None:
        table["fusion_prediction"] = np.asarray(fusion_test["predictions"], dtype=int)
        table["fusion_probability_cancer"] = np.asarray(fusion_test["positive_probability"], dtype=float)
        table["fusion_confidence"] = np.asarray(fusion_test["confidence"], dtype=float)
        table["fusion_correct"] = np.asarray(fusion_test["correct_mask"], dtype=bool).astype(np.int64)

    return table


def contradiction_mask(
    sample_table: Dict[str, np.ndarray],
    threshold: Optional[float] = None,
) -> np.ndarray:
    """Flag contradiction samples at one confidence level."""
    disagreement = np.asarray(sample_table["disagreement"], dtype=bool)
    if threshold is None:
        return disagreement
    return disagreement & (
        np.asarray(sample_table["image_confidence"], dtype=float) >= float(threshold)
    ) & (
        np.asarray(sample_table["genomic_confidence"], dtype=float) >= float(threshold)
    )


def summarize_contradiction_level(
    sample_table: Dict[str, np.ndarray],
    threshold: Optional[float],
) -> Dict[str, object]:
    """Summarize one contradiction level for a single seed."""
    mask = contradiction_mask(sample_table, threshold=threshold)
    not_mask = ~mask

    labels = np.asarray(sample_table["true_label"], dtype=int)
    image_correct = np.asarray(sample_table["image_correct"], dtype=bool)
    genomic_correct = np.asarray(sample_table["genomic_correct"], dtype=bool)
    fusion_correct = (
        np.asarray(sample_table["fusion_correct"], dtype=bool)
        if "fusion_correct" in sample_table
        else None
    )

    image_correct_genomic_wrong = image_correct & ~genomic_correct
    genomic_correct_image_wrong = genomic_correct & ~image_correct
    both_wrong = ~image_correct & ~genomic_correct
    both_correct = image_correct & genomic_correct

    def masked_mean(values: np.ndarray, active_mask: np.ndarray) -> float:
        return _safe_mean(np.asarray(values, dtype=float)[active_mask])

    summary = {
        "threshold": float(threshold) if threshold is not None else None,
        "label": format_threshold_label(threshold),
        "count": int(mask.sum()),
        "fraction": float(mask.mean()),
        "image_correct_genomic_wrong_fraction": masked_mean(image_correct_genomic_wrong, mask),
        "genomic_correct_image_wrong_fraction": masked_mean(genomic_correct_image_wrong, mask),
        "both_wrong_fraction": masked_mean(both_wrong, mask),
        "both_correct_fraction": masked_mean(both_correct, mask),
        "mean_probability_gap": masked_mean(sample_table["probability_gap"], mask),
        "mean_signed_probability_gap": masked_mean(sample_table["signed_probability_gap"], mask),
        "mean_image_confidence": masked_mean(sample_table["image_confidence"], mask),
        "mean_genomic_confidence": masked_mean(sample_table["genomic_confidence"], mask),
        "mean_confidence_gap": masked_mean(sample_table["confidence_gap"], mask),
        "image_higher_confidence_fraction": masked_mean(sample_table["image_higher_confidence"], mask),
        "genomic_higher_confidence_fraction": masked_mean(sample_table["genomic_higher_confidence"], mask),
        "image_accuracy_contradiction": masked_mean(image_correct, mask),
        "genomic_accuracy_contradiction": masked_mean(genomic_correct, mask),
        "image_accuracy_non_contradiction": masked_mean(image_correct, not_mask),
        "genomic_accuracy_non_contradiction": masked_mean(genomic_correct, not_mask),
        "image_error_contradiction": masked_mean(~image_correct, mask),
        "genomic_error_contradiction": masked_mean(~genomic_correct, mask),
        "image_error_non_contradiction": masked_mean(~image_correct, not_mask),
        "genomic_error_non_contradiction": masked_mean(~genomic_correct, not_mask),
        "normal_contradiction_fraction": float(mask[labels == 0].mean()) if np.any(labels == 0) else float("nan"),
        "cancer_contradiction_fraction": float(mask[labels == 1].mean()) if np.any(labels == 1) else float("nan"),
    }
    if fusion_correct is not None:
        summary["fusion_accuracy_contradiction"] = masked_mean(fusion_correct, mask)
        summary["fusion_accuracy_non_contradiction"] = masked_mean(fusion_correct, not_mask)
        summary["fusion_error_contradiction"] = masked_mean(~fusion_correct, mask)
        summary["fusion_error_non_contradiction"] = masked_mean(~fusion_correct, not_mask)

    if "latent_shared_severity" in sample_table:
        shared_severity = np.asarray(sample_table["latent_shared_severity"], dtype=float)
        discordance = np.asarray(sample_table["latent_discordance"], dtype=float)
        image_pathology = np.asarray(sample_table["latent_image_pathology"], dtype=float)
        genomic_pathology = np.asarray(sample_table["latent_genomic_pathology"], dtype=float)
        pathology_gap = np.asarray(sample_table["latent_pathology_gap"], dtype=float)

        summary.update(
            {
                "mean_shared_severity": masked_mean(shared_severity, mask),
                "mean_abs_discordance": masked_mean(np.abs(discordance), mask),
                "mean_abs_discordance_non_contradiction": masked_mean(np.abs(discordance), not_mask),
                "mean_discordance": masked_mean(discordance, mask),
                "mean_image_pathology": masked_mean(image_pathology, mask),
                "mean_genomic_pathology": masked_mean(genomic_pathology, mask),
                "mean_pathology_gap": masked_mean(pathology_gap, mask),
                "mean_pathology_gap_non_contradiction": masked_mean(pathology_gap, not_mask),
                "image_correct_genomic_wrong_mean_discordance": masked_mean(
                    discordance,
                    mask & image_correct_genomic_wrong,
                ),
                "genomic_correct_image_wrong_mean_discordance": masked_mean(
                    discordance,
                    mask & genomic_correct_image_wrong,
                ),
                "both_wrong_mean_discordance": masked_mean(discordance, mask & both_wrong),
            }
        )
    return summary


def summarize_contradictions_for_seed(
    sample_table: Dict[str, np.ndarray],
    thresholds: Sequence[float] = DEFAULT_CONTRADICTION_THRESHOLDS,
) -> Dict[str, Dict[str, object]]:
    """Summarize all contradiction levels for one seed."""
    summaries = {}
    for level_name, threshold in contradiction_level_specs(thresholds):
        summaries[level_name] = summarize_contradiction_level(sample_table, threshold=threshold)
    return summaries


def run_contradiction_analysis_for_seed(
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
    thresholds: Sequence[float] = DEFAULT_CONTRADICTION_THRESHOLDS,
    high_confidence_threshold: float = 0.70,
    include_fusion: bool = True,
) -> Dict[str, object]:
    """Run the contradiction analysis for one seed."""
    dataset = prepare_dataset(
        mode=mode,
        image_size=image_size,
        genomic_representation=genomic_representation,
        seed=seed,
        image_model=image_model,
    )
    model_results = train_models_for_seed(
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
        include_fusion=include_fusion,
    )
    sample_table = build_test_sample_table(seed=seed, dataset=dataset, model_results=model_results)
    contradiction_summaries = summarize_contradictions_for_seed(sample_table, thresholds=thresholds)

    return {
        "dataset_meta": {
            "mode": dataset["mode"],
            "image_size": dataset["image_size"],
            "genomic_representation": dataset["genomic_representation"],
            "image_model": image_model,
            "n_monomers": dataset["n_monomers"],
        },
        "models": model_results,
        "sample_table": sample_table,
        "contradictions": contradiction_summaries,
    }


def pooled_sample_table(
    per_seed_results: Dict[int, Dict[str, object]],
    thresholds: Sequence[float] = DEFAULT_CONTRADICTION_THRESHOLDS,
) -> Dict[str, np.ndarray]:
    """Concatenate per-seed test sample outputs and threshold masks."""
    sample_tables = [per_seed_results[seed]["sample_table"] for seed in per_seed_results]
    base_keys = list(sample_tables[0].keys())
    pooled = {
        key: np.concatenate([np.asarray(table[key]) for table in sample_tables], axis=0)
        for key in base_keys
    }
    for level_name, threshold in contradiction_level_specs(thresholds):
        pooled[f"mask_{level_name}"] = contradiction_mask(pooled, threshold=threshold).astype(np.int64)
    return pooled


def collect_contradiction_metric(
    per_seed_results: Dict[int, Dict[str, object]],
    level_name: str,
    metric_name: str,
) -> list[float]:
    """Collect one contradiction summary metric across seeds."""
    return [
        float(per_seed_results[seed]["contradictions"][level_name][metric_name])
        for seed in per_seed_results
    ]


def preferred_analysis_level_name(
    per_seed_results: Dict[int, Dict[str, object]],
    thresholds: Sequence[float] = DEFAULT_CONTRADICTION_THRESHOLDS,
    preferred_threshold: float = 0.70,
) -> str:
    """Choose the strongest non-empty contradiction level for subset analysis."""
    preferred_name = f"conf_ge_{preferred_threshold:.2f}".replace(".", "_")
    available_names = {name for name, _ in contradiction_level_specs(thresholds)}
    if preferred_name in available_names:
        preferred_fraction = _safe_mean(
            np.asarray(collect_contradiction_metric(per_seed_results, preferred_name, "fraction"), dtype=float)
        )
        if preferred_fraction > 0.0:
            return preferred_name

    for threshold in sorted((float(value) for value in thresholds), reverse=True):
        level_name = f"conf_ge_{threshold:.2f}".replace(".", "_")
        level_fraction = _safe_mean(
            np.asarray(collect_contradiction_metric(per_seed_results, level_name, "fraction"), dtype=float)
        )
        if level_fraction > 0.0:
            return level_name
    return "any_disagreement"


def print_seed_summary(seed: int, seed_result: Dict[str, object]) -> None:
    """Print one short per-seed contradiction summary."""
    any_fraction = seed_result["contradictions"]["any_disagreement"]["fraction"]
    moderate_fraction = seed_result["contradictions"]["conf_ge_0_60"]["fraction"]
    strong_fraction = seed_result["contradictions"]["conf_ge_0_70"]["fraction"]
    image_acc = seed_result["models"]["image"]["split_metrics"]["test"]["accuracy"]
    genomic_acc = seed_result["models"]["genomic"]["split_metrics"]["test"]["accuracy"]
    fusion_acc = (
        seed_result["models"]["fusion"]["split_metrics"]["test"]["accuracy"]
        if "fusion" in seed_result["models"]
        else float("nan")
    )
    print(
        f"seed {seed}: image={image_acc:.3f}  genomic={genomic_acc:.3f}  fusion={fusion_acc:.3f}  "
        f"any={any_fraction:.3f}  conf>=0.60={moderate_fraction:.3f}  conf>=0.70={strong_fraction:.3f}"
    )


def interpret_contradiction_results(
    per_seed_results: Dict[int, Dict[str, object]],
    thresholds: Sequence[float] = DEFAULT_CONTRADICTION_THRESHOLDS,
) -> list[str]:
    """Produce a short rule-based contradiction interpretation."""
    strong_level = "conf_ge_0_70"
    moderate_level = "conf_ge_0_60"
    any_level = "any_disagreement"
    analysis_level = preferred_analysis_level_name(per_seed_results, thresholds=thresholds)
    analysis_threshold = per_seed_results[next(iter(per_seed_results))]["contradictions"][analysis_level]["threshold"]
    analysis_label = format_threshold_label(analysis_threshold)

    any_fraction = _safe_mean(np.asarray(collect_contradiction_metric(per_seed_results, any_level, "fraction")))
    moderate_fraction = _safe_mean(
        np.asarray(collect_contradiction_metric(per_seed_results, moderate_level, "fraction"))
    )
    strong_fraction = _safe_mean(
        np.asarray(collect_contradiction_metric(per_seed_results, strong_level, "fraction"))
    )
    image_wins = _safe_mean(
        np.asarray(
            collect_contradiction_metric(
                per_seed_results,
                analysis_level,
                "image_correct_genomic_wrong_fraction",
            )
        )
    )
    genomic_wins = _safe_mean(
        np.asarray(
            collect_contradiction_metric(
                per_seed_results,
                analysis_level,
                "genomic_correct_image_wrong_fraction",
            )
        )
    )
    both_wrong = _safe_mean(
        np.asarray(collect_contradiction_metric(per_seed_results, analysis_level, "both_wrong_fraction"))
    )
    fusion_acc_contra = _safe_mean(
        np.asarray(collect_contradiction_metric(per_seed_results, analysis_level, "fusion_accuracy_contradiction"))
    )
    fusion_acc_non = _safe_mean(
        np.asarray(collect_contradiction_metric(per_seed_results, analysis_level, "fusion_accuracy_non_contradiction"))
    )

    lines = []
    if strong_fraction < 0.05:
        lines.append("Strong contradiction is rare; most disagreements are weaker than confident opposite signals.")
    elif strong_fraction >= 0.15:
        lines.append("Strong contradiction is common enough to matter for multimodal fusion.")
    else:
        lines.append("Strong contradiction exists but is not the dominant test-time pattern.")

    if any_fraction >= moderate_fraction + 0.10 and moderate_fraction >= strong_fraction + 0.05:
        lines.append("Most disagreement comes from unequal-strength evidence rather than two strongly opposed modalities.")

    if genomic_wins >= image_wins + 0.10:
        lines.append(f"On {analysis_label} cases, genomics is more often correct.")
    elif image_wins >= genomic_wins + 0.10:
        lines.append(f"On {analysis_label} cases, the image path is more often correct.")
    else:
        lines.append(f"{analysis_label.capitalize()} cases do not resolve consistently in favor of one modality.")

    if both_wrong >= 0.20:
        lines.append("Contradiction-heavy samples also behave like hard samples rather than clean handoffs between modalities.")

    if np.isfinite(fusion_acc_contra) and np.isfinite(fusion_acc_non):
        if strong_fraction < 0.05 and fusion_acc_contra <= fusion_acc_non - 0.08:
            lines.append("Fusion degrades on the rare strong-contradiction subset, but that subset is too small to explain broad fusion instability on its own.")
        elif strong_fraction >= 0.10 and fusion_acc_contra <= fusion_acc_non - 0.08:
            lines.append("Fusion behaves worse on contradiction-heavy samples in a way that is consistent with genuine cross-modal opposition.")
        else:
            lines.append("Current fusion behavior is only weakly coupled to the measured contradiction rate.")

    analysis_summary = per_seed_results[next(iter(per_seed_results))]["contradictions"][analysis_level]
    if "mean_abs_discordance" in analysis_summary:
        contradiction_abs_d = _safe_mean(
            np.asarray(collect_contradiction_metric(per_seed_results, analysis_level, "mean_abs_discordance"))
        )
        non_contradiction_abs_d = _safe_mean(
            np.asarray(
                collect_contradiction_metric(
                    per_seed_results,
                    analysis_level,
                    "mean_abs_discordance_non_contradiction",
                )
            )
        )
        image_win_d = _safe_mean(
            np.asarray(
                collect_contradiction_metric(
                    per_seed_results,
                    analysis_level,
                    "image_correct_genomic_wrong_mean_discordance",
                )
            )
        )
        genomic_win_d = _safe_mean(
            np.asarray(
                collect_contradiction_metric(
                    per_seed_results,
                    analysis_level,
                    "genomic_correct_image_wrong_mean_discordance",
                )
            )
        )

        if contradiction_abs_d >= non_contradiction_abs_d + 0.15:
            lines.append(
                f"{analysis_label.capitalize()} samples have larger |d| ({contradiction_abs_d:.2f} vs "
                f"{non_contradiction_abs_d:.2f}), so strong opposition concentrates where the shared latent splits "
                "local texture from distal contacts."
            )
        if image_win_d > 0.10 and genomic_win_d < -0.10:
            lines.append(
                "Positive d produces image-abnormal/genomic-normal evidence and negative d produces the reverse, "
                "which keeps contradiction tied to one coherent latent biology rather than modality corruption."
            )

    if not lines:
        lines.append("The dataset contains some contradiction, but the current measurements do not support a strong oppositional-modality story.")
    return lines
