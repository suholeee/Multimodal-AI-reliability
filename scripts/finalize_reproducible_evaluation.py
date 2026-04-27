"""Run one clean, reproducible final evaluation for the polymer sandbox."""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Dict, Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch

from _path_setup import RESULTS_DIR

from contradiction_analysis import (
    build_test_sample_table,
    contradiction_level_specs,
    format_threshold_label,
    summarize_contradictions_for_seed,
)
from fusion_debug_utils import (
    LEARNED_FUSION_VARIANTS,
    evaluate_late_fusion_variants,
    train_learned_fusion_variant,
    train_unimodal_controls,
)
from fusion_safety import prepare_multimodal_dataset


SEEDS = (21, 22, 23, 24, 25)
MODE = "safety"
IMAGE_SIZE = 32
GENOMIC_REPRESENTATION = "matrix"
IMAGE_MODEL = "hybrid"
EPOCHS = 45
MIN_EPOCHS = 15
PATIENCE = 10
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
HIGH_CONFIDENCE_THRESHOLD = 0.70
CONTRADICTION_THRESHOLDS = (0.60, 0.70, 0.75)
DEVICE = "auto"

VARIANT_ORDER = (
    "image",
    "genomic",
    "late_average",
    "late_weighted",
    "late_route",
    "fusion_scratch",
    "fusion_pretrained_full",
    "fusion_frozen",
    "fusion_partial",
)
VARIANT_LABELS = {
    "image": "image",
    "genomic": "genomic",
    "late_average": "late_avg",
    "late_weighted": "late_weighted",
    "late_route": "route_max_conf",
    "fusion_scratch": "fusion_scratch",
    "fusion_pretrained_full": "fusion_pretrained_full",
    "fusion_frozen": "fusion_frozen",
    "fusion_partial": "fusion_partial",
}
BENCHMARK_FUSION_VARIANTS = tuple(name for name in VARIANT_ORDER if name not in {"image", "genomic"})

OUTPUT_ACCURACY_FIG = RESULTS_DIR / "final" / "final_accuracy_summary.png"
OUTPUT_CONTRADICTION_FIG = RESULTS_DIR / "final" / "final_contradiction_summary.png"
OUTPUT_CONTRADICTION_NPZ = RESULTS_DIR / "final" / "final_contradiction_samples.npz"
OUTPUT_SUMMARY_TXT = RESULTS_DIR / "reports" / "final_results_summary.txt"


def _safe_mean(values: Sequence[float]) -> float:
    array = np.asarray(values, dtype=float)
    finite = array[np.isfinite(array)]
    return float(finite.mean()) if finite.size > 0 else float("nan")


def _safe_std(values: Sequence[float]) -> float:
    array = np.asarray(values, dtype=float)
    finite = array[np.isfinite(array)]
    return float(finite.std(ddof=0)) if finite.size > 0 else float("nan")


def _format_mean_std(values: Sequence[float]) -> str:
    return f"{_safe_mean(values):.3f} ± {_safe_std(values):.3f}"


def _clone_state_dict(module: torch.nn.Module) -> Dict[str, torch.Tensor]:
    return {
        name: tensor.detach().cpu().clone()
        for name, tensor in module.state_dict().items()
    }


def _state_dict_max_abs_diff(
    state_a: Dict[str, torch.Tensor],
    state_b: Dict[str, torch.Tensor],
) -> float:
    if set(state_a) != set(state_b):
        return float("inf")
    max_abs_diff = 0.0
    for name in state_a:
        tensor_a = state_a[name]
        tensor_b = state_b[name]
        if tensor_a.shape != tensor_b.shape:
            return float("inf")
        if tensor_a.numel() == 0:
            continue
        diff = torch.abs(tensor_a.to(dtype=torch.float64) - tensor_b.to(dtype=torch.float64))
        max_abs_diff = max(max_abs_diff, float(diff.max().item()))
    return float(max_abs_diff)


def _dataset_fingerprint(dataset: Dict[str, object]) -> str:
    digest = hashlib.sha256()
    arrays = dataset["arrays"]
    split_indices = dataset["split_indices"]
    ordered_keys = (
        "image",
        "genomic",
        "genomic_matrix",
        "label",
        "condition",
        "latent_shared_severity",
        "latent_discordance",
        "latent_image_pathology",
        "latent_genomic_pathology",
        "latent_label_probability",
        "latent_compaction",
        "latent_long_range_mixing",
        "latent_heterogeneity",
        "latent_local_texture",
        "latent_distal_bridge_strength",
        "latent_bridge_balance",
        "latent_pathology_gap",
    )
    for key in ordered_keys:
        array = np.ascontiguousarray(np.asarray(arrays[key]))
        digest.update(key.encode("utf-8"))
        digest.update(str(array.dtype).encode("utf-8"))
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(array.view(np.uint8).tobytes())
    for split_name in ("train", "val", "test"):
        indices = np.ascontiguousarray(np.asarray(split_indices[split_name], dtype=np.int64))
        digest.update(split_name.encode("utf-8"))
        digest.update(indices.tobytes())
    return digest.hexdigest()


def _collect_test_accuracy(seed_result: Dict[str, object], variant_name: str) -> float:
    if variant_name in seed_result["controls"]:
        return float(seed_result["controls"][variant_name]["split_metrics"]["test"]["accuracy"])
    return float(seed_result["variants"][variant_name]["split_metrics"]["test"]["accuracy"])


def _collect_test_metric(seed_result: Dict[str, object], variant_name: str, metric_name: str) -> float:
    if variant_name in seed_result["controls"]:
        return float(seed_result["controls"][variant_name]["split_metrics"]["test"][metric_name])
    return float(seed_result["variants"][variant_name]["split_metrics"]["test"][metric_name])


def _check_prediction_health(
    model_eval: Dict[str, object],
    label: str,
    warnings: list[str],
) -> None:
    probabilities = np.asarray(model_eval["probabilities"], dtype=float)
    predictions = np.asarray(model_eval["predictions"], dtype=int)
    if np.isnan(probabilities).any():
        warnings.append(f"{label}: NaN probabilities detected")
    if not np.isfinite(probabilities).all():
        warnings.append(f"{label}: non-finite probabilities detected")
    if np.unique(predictions).size < 2:
        warnings.append(f"{label}: constant test predictions detected")


def _pooled_fraction(sample_tables: Sequence[Dict[str, np.ndarray]], key: str) -> float:
    pooled = np.concatenate([np.asarray(table[key], dtype=bool) for table in sample_tables], axis=0)
    return float(pooled.mean()) if pooled.size > 0 else float("nan")


def _build_accuracy_figure(
    per_seed_results: Dict[int, Dict[str, object]],
    save_path: Path,
) -> dict[str, float]:
    labels = [VARIANT_LABELS[name] for name in VARIANT_ORDER]
    means = [
        _safe_mean([_collect_test_accuracy(per_seed_results[seed], name) for seed in per_seed_results])
        for name in VARIANT_ORDER
    ]
    stds = [
        _safe_std([_collect_test_accuracy(per_seed_results[seed], name) for seed in per_seed_results])
        for name in VARIANT_ORDER
    ]
    per_seed_points = {
        name: np.asarray([_collect_test_accuracy(per_seed_results[seed], name) for seed in per_seed_results], dtype=float)
        for name in VARIANT_ORDER
    }

    fig, ax = plt.subplots(figsize=(13, 6))
    x_positions = np.arange(len(VARIANT_ORDER))
    colors = [
        "#4C72B0",
        "#55A868",
        "#8172B2",
        "#C44E52",
        "#DD8452",
        "#937860",
        "#64B5CD",
        "#CCB974",
        "#8C8C8C",
    ]
    ax.bar(x_positions, means, yerr=stds, color=colors, alpha=0.88, capsize=4)
    for idx, name in enumerate(VARIANT_ORDER):
        y_values = per_seed_points[name]
        ax.scatter(
            np.full_like(y_values, idx, dtype=float),
            y_values,
            color="black",
            s=22,
            alpha=0.65,
            zorder=3,
        )
    ax.set_xticks(x_positions)
    ax.set_xticklabels(labels, rotation=28, ha="right")
    ax.set_ylabel("Test accuracy")
    ax.set_ylim(0.0, 1.0)
    ax.set_title("Final Accuracy Summary")
    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return {
        name: mean
        for name, mean in zip(VARIANT_ORDER, means)
    }


def _build_contradiction_figure(
    contradiction_results: Dict[int, Dict[str, object]],
    sample_tables: Sequence[Dict[str, np.ndarray]],
    primary_fusion_name: str,
    primary_fusion_label: str,
    save_path: Path,
) -> None:
    level_specs = contradiction_level_specs(CONTRADICTION_THRESHOLDS)
    level_names = [name for name, _ in level_specs]
    level_labels = ["any"] + [f">={threshold:.2f}" for _, threshold in level_specs[1:]]

    frequency_means = [
        _safe_mean([float(contradiction_results[seed]["contradictions"][name]["fraction"]) for seed in contradiction_results])
        for name in level_names
    ]
    frequency_stds = [
        _safe_std([float(contradiction_results[seed]["contradictions"][name]["fraction"]) for seed in contradiction_results])
        for name in level_names
    ]
    image_only = [
        _safe_mean(
            [float(contradiction_results[seed]["contradictions"][name]["image_correct_genomic_wrong_fraction"]) for seed in contradiction_results]
        )
        for name in level_names
    ]
    genomic_only = [
        _safe_mean(
            [float(contradiction_results[seed]["contradictions"][name]["genomic_correct_image_wrong_fraction"]) for seed in contradiction_results]
        )
        for name in level_names
    ]
    both_wrong = [
        _safe_mean(
            [float(contradiction_results[seed]["contradictions"][name]["both_wrong_fraction"]) for seed in contradiction_results]
        )
        for name in level_names
    ]
    image_acc = [
        _safe_mean(
            [float(contradiction_results[seed]["contradictions"][name]["image_accuracy_contradiction"]) for seed in contradiction_results]
        )
        for name in level_names
    ]
    genomic_acc = [
        _safe_mean(
            [float(contradiction_results[seed]["contradictions"][name]["genomic_accuracy_contradiction"]) for seed in contradiction_results]
        )
        for name in level_names
    ]
    fusion_acc = [
        _safe_mean(
            [float(contradiction_results[seed]["contradictions"][name]["fusion_accuracy_contradiction"]) for seed in contradiction_results]
        )
        for name in level_names
    ]

    pooled_disagreement = np.concatenate(
        [np.asarray(table["disagreement"], dtype=bool) for table in sample_tables],
        axis=0,
    )
    pooled_fusion_correct = np.concatenate(
        [np.asarray(table["fusion_correct"], dtype=bool) for table in sample_tables],
        axis=0,
    )
    pooled_fusion_conf = np.concatenate(
        [np.asarray(table["fusion_confidence"], dtype=float) for table in sample_tables],
        axis=0,
    )
    agreement_mask = ~pooled_disagreement
    fusion_acc_agree = float(pooled_fusion_correct[agreement_mask].mean()) if np.any(agreement_mask) else float("nan")
    fusion_acc_disagree = float(pooled_fusion_correct[pooled_disagreement].mean()) if np.any(pooled_disagreement) else float("nan")
    mean_conf_agree = float(pooled_fusion_conf[agreement_mask].mean()) if np.any(agreement_mask) else float("nan")
    mean_conf_disagree = float(pooled_fusion_conf[pooled_disagreement].mean()) if np.any(pooled_disagreement) else float("nan")

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    x_positions = np.arange(len(level_names))

    axes[0, 0].bar(
        x_positions,
        frequency_means,
        yerr=frequency_stds,
        color=["#4C72B0", "#55A868", "#C44E52", "#8172B2"],
        alpha=0.88,
        capsize=4,
    )
    axes[0, 0].set_xticks(x_positions)
    axes[0, 0].set_xticklabels(level_labels)
    axes[0, 0].set_ylim(0.0, 1.0)
    axes[0, 0].set_ylabel("Fraction of test samples")
    axes[0, 0].set_title("Contradiction Frequency")

    axes[0, 1].bar(x_positions, image_only, color="#4C72B0", label="image correct / genomic wrong")
    axes[0, 1].bar(x_positions, genomic_only, bottom=image_only, color="#55A868", label="genomic correct / image wrong")
    axes[0, 1].bar(
        x_positions,
        both_wrong,
        bottom=np.asarray(image_only) + np.asarray(genomic_only),
        color="#C44E52",
        label="both wrong",
    )
    axes[0, 1].set_xticks(x_positions)
    axes[0, 1].set_xticklabels(level_labels)
    axes[0, 1].set_ylim(0.0, 1.0)
    axes[0, 1].set_ylabel("Fraction within contradiction subset")
    axes[0, 1].set_title("Outcome Breakdown")
    axes[0, 1].legend(frameon=False, fontsize=9)

    axes[1, 0].plot(x_positions, image_acc, marker="o", color="#4C72B0", label="image")
    axes[1, 0].plot(x_positions, genomic_acc, marker="o", color="#55A868", label="genomic")
    axes[1, 0].plot(x_positions, fusion_acc, marker="o", color="#C44E52", label=primary_fusion_label)
    axes[1, 0].set_xticks(x_positions)
    axes[1, 0].set_xticklabels(level_labels)
    axes[1, 0].set_ylim(0.0, 1.0)
    axes[1, 0].set_ylabel("Accuracy on contradiction subset")
    axes[1, 0].set_title("Subset Accuracy")
    axes[1, 0].legend(frameon=False)

    axes[1, 1].bar(
        [0, 1, 2, 3],
        [fusion_acc_agree, fusion_acc_disagree, mean_conf_agree, mean_conf_disagree],
        color=["#4C72B0", "#C44E52", "#64B5CD", "#DD8452"],
        alpha=0.88,
    )
    axes[1, 1].set_xticks([0, 1, 2, 3])
    axes[1, 1].set_xticklabels(
        [
            "acc\nagreement",
            "acc\ndisagreement",
            "conf\nagreement",
            "conf\ndisagreement",
        ]
    )
    axes[1, 1].set_ylim(0.0, 1.0)
    axes[1, 1].set_ylabel("Value")
    axes[1, 1].set_title(f"{primary_fusion_label} Agreement vs Disagreement")

    fig.suptitle(f"Final Contradiction Summary ({primary_fusion_label})", fontsize=14)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _save_contradiction_archive(
    contradiction_results: Dict[int, Dict[str, object]],
    primary_fusion_name: str,
    fingerprints: Dict[int, str],
    save_path: Path,
) -> Dict[str, np.ndarray]:
    level_specs = contradiction_level_specs(CONTRADICTION_THRESHOLDS)
    sample_tables = [contradiction_results[seed]["sample_table"] for seed in contradiction_results]
    base_keys = list(sample_tables[0].keys())
    pooled = {
        key: np.concatenate([np.asarray(table[key]) for table in sample_tables], axis=0)
        for key in base_keys
    }
    for level_name, threshold in level_specs:
        if threshold is None:
            pooled[f"mask_{level_name}"] = np.asarray(pooled["disagreement"], dtype=np.int64)
        else:
            mask = (
                np.asarray(pooled["disagreement"], dtype=bool)
                & (np.asarray(pooled["image_confidence"], dtype=float) >= float(threshold))
                & (np.asarray(pooled["genomic_confidence"], dtype=float) >= float(threshold))
            )
            pooled[f"mask_{level_name}"] = mask.astype(np.int64)
    pooled["dataset_fingerprint"] = np.asarray(
        [fingerprints[int(seed)] for seed in pooled["seed"]],
        dtype="<U64",
    )
    pooled["primary_fusion_name"] = np.asarray([primary_fusion_name] * len(pooled["seed"]), dtype="<U64")
    np.savez(save_path, **pooled)
    return pooled


def _choose_summary_conclusion(
    genomic_mean: float,
    best_fusion_mean: float,
    any_disagreement_fraction: float,
    strong_contradiction_fraction: float,
) -> str:
    if strong_contradiction_fraction >= 0.10 and best_fusion_mean < genomic_mean - 0.03:
        return "fusion fails under strong contradiction"
    if any_disagreement_fraction >= strong_contradiction_fraction + 0.10 and best_fusion_mean >= genomic_mean + 0.005:
        return "fusion provides modest gain in weak-contradiction regime"
    return "modalities differ in strength rather than oppose each other"


def main() -> None:
    per_seed_results: Dict[int, Dict[str, object]] = {}
    warnings: list[str] = []
    fingerprints: Dict[int, str] = {}
    OUTPUT_ACCURACY_FIG.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_CONTRADICTION_FIG.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_CONTRADICTION_NPZ.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_SUMMARY_TXT.parent.mkdir(parents=True, exist_ok=True)

    print("Final reproducible evaluation")
    print(f"mode={MODE}  image_model={IMAGE_MODEL}  epochs={EPOCHS}  min_epochs={MIN_EPOCHS}")
    print(f"seeds={list(SEEDS)}")

    for seed in SEEDS:
        print(f"\nRunning seed {seed}")
        dataset = prepare_multimodal_dataset(
            mode=MODE,
            image_size=IMAGE_SIZE,
            genomic_representation=GENOMIC_REPRESENTATION,
            seed=seed,
            image_model=IMAGE_MODEL,
        )
        fingerprint = _dataset_fingerprint(dataset)
        fingerprints[seed] = fingerprint

        controls = train_unimodal_controls(
            dataset=dataset,
            image_model=IMAGE_MODEL,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            learning_rate=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY,
            patience=PATIENCE,
            min_epochs=MIN_EPOCHS,
            device=DEVICE,
            seed=seed,
            high_confidence_threshold=HIGH_CONFIDENCE_THRESHOLD,
        )
        variants = evaluate_late_fusion_variants(
            controls=controls,
            high_confidence_threshold=HIGH_CONFIDENCE_THRESHOLD,
        )
        for offset, variant_name in enumerate(LEARNED_FUSION_VARIANTS):
            variants[variant_name] = train_learned_fusion_variant(
                dataset=dataset,
                controls=controls,
                image_model=IMAGE_MODEL,
                variant_name=variant_name,
                epochs=EPOCHS,
                batch_size=BATCH_SIZE,
                learning_rate=LEARNING_RATE,
                weight_decay=WEIGHT_DECAY,
                patience=PATIENCE,
                min_epochs=MIN_EPOCHS,
                device=DEVICE,
                seed=seed + 20 + offset,
                high_confidence_threshold=HIGH_CONFIDENCE_THRESHOLD,
            )

        seed_result = {
            "dataset": dataset,
            "dataset_fingerprint": fingerprint,
            "controls": controls,
            "variants": variants,
        }
        per_seed_results[seed] = seed_result

        for variant_name in ("image", "genomic"):
            _check_prediction_health(
                controls[variant_name]["split_metrics"]["test"],
                label=f"seed {seed} / {variant_name}",
                warnings=warnings,
            )
        for variant_name in BENCHMARK_FUSION_VARIANTS:
            _check_prediction_health(
                variants[variant_name]["split_metrics"]["test"],
                label=f"seed {seed} / {variant_name}",
                warnings=warnings,
            )

        scratch = variants["fusion_scratch"]
        pretrained = variants["fusion_pretrained_full"]
        if scratch is pretrained:
            warnings.append(f"seed {seed}: fusion_scratch and fusion_pretrained_full share the same result object")
        if scratch["split_metrics"] is pretrained["split_metrics"]:
            warnings.append(f"seed {seed}: fusion_scratch and fusion_pretrained_full share split_metrics")
        if scratch["model"] is pretrained["model"]:
            warnings.append(f"seed {seed}: fusion_scratch and fusion_pretrained_full share the same model object")
        final_weight_gap = _state_dict_max_abs_diff(
            _clone_state_dict(scratch["model"]),
            _clone_state_dict(pretrained["model"]),
        )
        if final_weight_gap == 0.0:
            warnings.append(f"seed {seed}: fusion_scratch and fusion_pretrained_full ended with identical weights")

        accuracy_line = "  ".join(
            f"{VARIANT_LABELS[name]}={_collect_test_accuracy(seed_result, name):.3f}"
            for name in VARIANT_ORDER
        )
        print(accuracy_line)

    print("\nPer-seed test accuracy")
    for seed in SEEDS:
        seed_result = per_seed_results[seed]
        line = "  ".join(
            f"{VARIANT_LABELS[name]}={_collect_test_accuracy(seed_result, name):.3f}"
            for name in VARIANT_ORDER
        )
        print(f"seed {seed}: {line}")

    print("\nMean ± std across seeds")
    benchmark_summary = {}
    for variant_name in VARIANT_ORDER:
        values = [_collect_test_accuracy(per_seed_results[seed], variant_name) for seed in SEEDS]
        benchmark_summary[variant_name] = {
            "mean": _safe_mean(values),
            "std": _safe_std(values),
            "values": values,
        }
        print(f"{VARIANT_LABELS[variant_name]:>21}: {_format_mean_std(values)}")

    plot_means = _build_accuracy_figure(per_seed_results=per_seed_results, save_path=OUTPUT_ACCURACY_FIG)
    for variant_name in VARIANT_ORDER:
        if not math.isclose(
            plot_means[variant_name],
            benchmark_summary[variant_name]["mean"],
            rel_tol=1e-9,
            abs_tol=1e-9,
        ):
            warnings.append(f"accuracy plot mismatch for {variant_name}")

    primary_fusion_name = max(
        BENCHMARK_FUSION_VARIANTS,
        key=lambda name: benchmark_summary[name]["mean"],
    )
    primary_fusion_label = VARIANT_LABELS[primary_fusion_name]
    print(f"\nPrimary fusion for contradiction analysis: {primary_fusion_label}")

    contradiction_results: Dict[int, Dict[str, object]] = {}
    contradiction_sample_tables = []
    for seed in SEEDS:
        seed_result = per_seed_results[seed]
        dataset = seed_result["dataset"]
        if _dataset_fingerprint(dataset) != seed_result["dataset_fingerprint"]:
            warnings.append(f"seed {seed}: dataset fingerprint changed before contradiction analysis")
        model_results = {
            "image": {
                "split_metrics": seed_result["controls"]["image"]["split_metrics"],
            },
            "genomic": {
                "split_metrics": seed_result["controls"]["genomic"]["split_metrics"],
            },
            "fusion": {
                "split_metrics": seed_result["variants"][primary_fusion_name]["split_metrics"],
            },
        }
        sample_table = build_test_sample_table(
            seed=seed,
            dataset=dataset,
            model_results=model_results,
        )
        sample_table["dataset_fingerprint"] = np.asarray(
            [seed_result["dataset_fingerprint"]] * len(sample_table["true_label"]),
            dtype="<U64",
        )
        contradiction_summary = summarize_contradictions_for_seed(
            sample_table=sample_table,
            thresholds=CONTRADICTION_THRESHOLDS,
        )
        contradiction_results[seed] = {
            "sample_table": sample_table,
            "contradictions": contradiction_summary,
        }
        contradiction_sample_tables.append(sample_table)

    print("\nContradiction analysis")
    for level_name, threshold in contradiction_level_specs(CONTRADICTION_THRESHOLDS):
        label = "any disagreement" if threshold is None else format_threshold_label(threshold)
        fraction_values = [
            float(contradiction_results[seed]["contradictions"][level_name]["fraction"])
            for seed in SEEDS
        ]
        print(f"{label:>18}: {_format_mean_std(fraction_values)}")

    print("\nOutcome breakdown within contradiction subsets")
    for level_name, threshold in contradiction_level_specs(CONTRADICTION_THRESHOLDS):
        label = "any disagreement" if threshold is None else format_threshold_label(threshold)
        image_only_values = [
            float(contradiction_results[seed]["contradictions"][level_name]["image_correct_genomic_wrong_fraction"])
            for seed in SEEDS
        ]
        genomic_only_values = [
            float(contradiction_results[seed]["contradictions"][level_name]["genomic_correct_image_wrong_fraction"])
            for seed in SEEDS
        ]
        both_wrong_values = [
            float(contradiction_results[seed]["contradictions"][level_name]["both_wrong_fraction"])
            for seed in SEEDS
        ]
        print(
            f"{label:>18}: "
            f"img_correct/gen_wrong={_format_mean_std(image_only_values)}  "
            f"gen_correct/img_wrong={_format_mean_std(genomic_only_values)}  "
            f"both_wrong={_format_mean_std(both_wrong_values)}"
        )

    print("\nAccuracy on contradiction vs non-contradiction subsets")
    for level_name, threshold in contradiction_level_specs(CONTRADICTION_THRESHOLDS):
        label = "any disagreement" if threshold is None else format_threshold_label(threshold)
        image_contra = [
            float(contradiction_results[seed]["contradictions"][level_name]["image_accuracy_contradiction"])
            for seed in SEEDS
        ]
        genomic_contra = [
            float(contradiction_results[seed]["contradictions"][level_name]["genomic_accuracy_contradiction"])
            for seed in SEEDS
        ]
        fusion_contra = [
            float(contradiction_results[seed]["contradictions"][level_name]["fusion_accuracy_contradiction"])
            for seed in SEEDS
        ]
        image_non = [
            float(contradiction_results[seed]["contradictions"][level_name]["image_accuracy_non_contradiction"])
            for seed in SEEDS
        ]
        genomic_non = [
            float(contradiction_results[seed]["contradictions"][level_name]["genomic_accuracy_non_contradiction"])
            for seed in SEEDS
        ]
        fusion_non = [
            float(contradiction_results[seed]["contradictions"][level_name]["fusion_accuracy_non_contradiction"])
            for seed in SEEDS
        ]
        print(
            f"{label:>18}: "
            f"image={_format_mean_std(image_contra)} vs {_format_mean_std(image_non)}  "
            f"genomic={_format_mean_std(genomic_contra)} vs {_format_mean_std(genomic_non)}  "
            f"{primary_fusion_label}={_format_mean_std(fusion_contra)} vs {_format_mean_std(fusion_non)}"
        )

    pooled_contradiction = _save_contradiction_archive(
        contradiction_results=contradiction_results,
        primary_fusion_name=primary_fusion_name,
        fingerprints=fingerprints,
        save_path=OUTPUT_CONTRADICTION_NPZ,
    )
    _build_contradiction_figure(
        contradiction_results=contradiction_results,
        sample_tables=contradiction_sample_tables,
        primary_fusion_name=primary_fusion_name,
        primary_fusion_label=primary_fusion_label,
        save_path=OUTPUT_CONTRADICTION_FIG,
    )

    pooled_disagreement = np.asarray(pooled_contradiction["disagreement"], dtype=bool)
    pooled_fusion_correct = np.asarray(pooled_contradiction["fusion_correct"], dtype=bool)
    pooled_fusion_confidence = np.asarray(pooled_contradiction["fusion_confidence"], dtype=float)
    pooled_agreement = ~pooled_disagreement

    any_gap = (
        float(pooled_fusion_correct[pooled_agreement].mean()) - float(pooled_fusion_correct[pooled_disagreement].mean())
        if np.any(pooled_agreement) and np.any(pooled_disagreement)
        else float("nan")
    )
    mean_conf_agreement = float(pooled_fusion_confidence[pooled_agreement].mean()) if np.any(pooled_agreement) else float("nan")
    mean_conf_disagreement = float(pooled_fusion_confidence[pooled_disagreement].mean()) if np.any(pooled_disagreement) else float("nan")

    print("\nLinking contradiction to performance")
    print(f"{primary_fusion_label} accuracy on disagreement samples: {float(pooled_fusion_correct[pooled_disagreement].mean()):.3f}")
    print(f"{primary_fusion_label} accuracy on non-disagreement samples: {float(pooled_fusion_correct[pooled_agreement].mean()):.3f}")
    print(f"performance gap (non-disagreement - disagreement): {any_gap:.3f}")
    print(f"{primary_fusion_label} mean confidence on agreement samples: {mean_conf_agreement:.3f}")
    print(f"{primary_fusion_label} mean confidence on disagreement samples: {mean_conf_disagreement:.3f}")

    strong_fraction = float(np.asarray(pooled_contradiction["mask_conf_ge_0_70"], dtype=bool).mean())
    any_fraction = float(np.asarray(pooled_contradiction["mask_any_disagreement"], dtype=bool).mean())
    conclusion = _choose_summary_conclusion(
        genomic_mean=benchmark_summary["genomic"]["mean"],
        best_fusion_mean=benchmark_summary[primary_fusion_name]["mean"],
        any_disagreement_fraction=any_fraction,
        strong_contradiction_fraction=strong_fraction,
    )

    summary_lines = [
        f"Configuration: mode={MODE}, image_model={IMAGE_MODEL}, seeds={list(SEEDS)}, epochs={EPOCHS}, min_epochs={MIN_EPOCHS}",
        "",
        "Unimodal performance",
        f"- image: {_format_mean_std(benchmark_summary['image']['values'])}",
        f"- genomic: {_format_mean_std(benchmark_summary['genomic']['values'])}",
        "",
        "Fusion result",
        f"- best fusion variant: {primary_fusion_label}",
        f"- {primary_fusion_label}: {_format_mean_std(benchmark_summary[primary_fusion_name]['values'])}",
        f"- genomic baseline: {_format_mean_std(benchmark_summary['genomic']['values'])}",
        "",
        "Contradiction statistics",
        f"- any disagreement: {any_fraction:.3f}",
        f"- conf>=0.60 disagreement: {float(np.asarray(pooled_contradiction['mask_conf_ge_0_60'], dtype=bool).mean()):.3f}",
        f"- conf>=0.70 disagreement: {strong_fraction:.3f}",
        f"- conf>=0.75 disagreement: {float(np.asarray(pooled_contradiction['mask_conf_ge_0_75'], dtype=bool).mean()):.3f}",
        "",
        "Key conclusion",
        conclusion,
    ]
    OUTPUT_SUMMARY_TXT.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    print("\nConsistency check")
    if warnings:
        for message in warnings:
            print(f"WARNING: {message}")
    else:
        print("No warnings")

    dataset_size = per_seed_results[SEEDS[0]]["dataset"]["arrays"]["label"].shape[0]
    test_samples_per_seed = int(per_seed_results[SEEDS[0]]["dataset"]["splits"]["test"]["label"].shape[0])
    print("\nFinal checklist")
    print(f"- dataset size: {per_seed_results[SEEDS[0]]['dataset']['n_monomers']} monomers, 300 polymers per condition ({dataset_size} total samples per seed)")
    print(f"- seeds used: {list(SEEDS)}")
    print(f"- number of test samples: {test_samples_per_seed} per seed, {test_samples_per_seed * len(SEEDS)} pooled")
    print(f"- saved file: {OUTPUT_ACCURACY_FIG}")
    print(f"- saved file: {OUTPUT_CONTRADICTION_FIG}")
    print(f"- saved file: {OUTPUT_CONTRADICTION_NPZ}")
    print(f"- saved file: {OUTPUT_SUMMARY_TXT}")


if __name__ == "__main__":
    main()
