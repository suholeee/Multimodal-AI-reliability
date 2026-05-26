"""Run the V2 contradiction-regime sweep for multimodal reliability analysis."""

from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np

from _path_setup import RESULTS_DIR

from fusion_debug_utils import (
    LEARNED_FUSION_VARIANTS,
    attach_variant_safety_diagnostics,
    evaluate_late_fusion_variants,
    train_learned_fusion_variant,
    train_unimodal_controls,
)
from image_feature_layers import attach_image_features
from ml_models import BEST_IMAGE_MODEL_NAMES
from v2_metrics import build_split_sample_table, summarize_strategy
from v2_policies import (
    DEFAULT_SELECTIVE_COVERAGE_LEVELS,
    apply_confidence_routing_policy,
    apply_genomic_first_policy,
    calibrate_abstention_policy,
    evaluate_abstention_policy,
    oracle_branch_accuracy,
    tune_confidence_routing_policy,
    tune_genomic_first_policy,
)
from v2_regime import ContradictionRegimeConfig, build_v2_paired_dataset


PROFILES = {
    "debug": {
        "mode": "debug",
        "contradiction_strengths": (0.0, 0.5, 1.0),
        "modality_asymmetries": (0.0,),
        "seeds": (21,),
        "epochs": 18,
        "min_epochs": 5,
        "patience": 5,
        "batch_size": 16,
    },
    "default": {
        "mode": "full",
        "contradiction_strengths": (0.0, 0.25, 0.5, 0.75, 1.0),
        "modality_asymmetries": (0.0,),
        "seeds": (21, 22, 23),
        "epochs": 45,
        "min_epochs": 15,
        "patience": 10,
        "batch_size": 32,
    },
    "final": {
        "mode": "safety",
        "contradiction_strengths": (0.0, 0.25, 0.5, 0.75, 1.0),
        "modality_asymmetries": (0.0,),
        "seeds": (21, 22, 23, 24, 25),
        "epochs": 45,
        "min_epochs": 15,
        "patience": 10,
        "batch_size": 32,
    },
}

DEFAULT_LEARNING_RATE = 1e-3
DEFAULT_WEIGHT_DECAY = 1e-4
DEFAULT_HIGH_CONFIDENCE_THRESHOLD = 0.70
DEFAULT_STRONG_CONTRADICTION_THRESHOLD = 0.40
DEFAULT_IMAGE_SIZE = 32
DEFAULT_GENOMIC_REPRESENTATION = "matrix"
DEFAULT_IMAGE_MODEL = "hybrid"
VALIDATION_CONTRADICTION_STRENGTHS = (0.0, 0.5, 1.0)
VALIDATION_CONFIDENCE_THRESHOLDS = (0.60, 0.70)
VALIDATION_MIN_IMAGE_ACCURACY = 0.60
VALIDATION_MIN_GENOMIC_ACCURACY = 0.60
VALIDATION_MIN_STRONG_CONTRADICTION_RATE = 0.05
VALIDATION_MAX_CORRECTNESS_SPLIT = 0.70


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the V2 contradiction sweep."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=tuple(PROFILES), default="default")
    parser.add_argument("--mode", choices=("debug", "full", "safety"), default=None)
    parser.add_argument("--image-size", type=int, default=DEFAULT_IMAGE_SIZE, choices=(32, 64))
    parser.add_argument(
        "--genomic-representation",
        choices=("matrix", "vector", "summary"),
        default=DEFAULT_GENOMIC_REPRESENTATION,
    )
    parser.add_argument("--image-model", choices=BEST_IMAGE_MODEL_NAMES, default=DEFAULT_IMAGE_MODEL)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--min-epochs", type=int, default=None)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument("--weight-decay", type=float, default=DEFAULT_WEIGHT_DECAY)
    parser.add_argument("--high-confidence-threshold", type=float, default=DEFAULT_HIGH_CONFIDENCE_THRESHOLD)
    parser.add_argument(
        "--structural-strong-threshold",
        type=float,
        default=DEFAULT_STRONG_CONTRADICTION_THRESHOLD,
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--contradiction-strengths", nargs="*", type=float, default=None)
    parser.add_argument("--modality-asymmetries", nargs="*", type=float, default=None)
    parser.add_argument("--seeds", nargs="*", type=int, default=None)
    parser.add_argument("--no-show", action="store_true")
    return parser.parse_args()


def _profile_value(args: argparse.Namespace, key: str):
    """Resolve one configuration value from explicit args or the selected profile."""
    profile_value = PROFILES[args.profile][key]
    arg_name = key.replace("-", "_")
    explicit_value = getattr(args, arg_name)
    return profile_value if explicit_value is None else explicit_value


def _safe_mean(values: Sequence[float]) -> float:
    """Compute a NaN-safe mean."""
    array = np.asarray(values, dtype=float)
    finite = array[np.isfinite(array)]
    return float(finite.mean()) if finite.size > 0 else float("nan")


def _safe_std(values: Sequence[float]) -> float:
    """Compute a NaN-safe standard deviation."""
    array = np.asarray(values, dtype=float)
    finite = array[np.isfinite(array)]
    return float(finite.std(ddof=0)) if finite.size > 0 else float("nan")


def _format_mean_std(values: Sequence[float]) -> str:
    """Format one metric as mean +- std."""
    return f"{_safe_mean(values):.3f} +/- {_safe_std(values):.3f}"


def _safe_setting_slug(contradiction_strength: float, modality_asymmetry: float, seed: int) -> str:
    """Create a stable filesystem-safe identifier for one sweep row."""
    return (
        f"cs_{contradiction_strength:0.2f}".replace(".", "p")
        + f"_asym_{modality_asymmetry:+0.2f}".replace(".", "p").replace("+", "pos").replace("-", "neg")
        + f"_seed_{seed}"
    )


def _select_best_variant(variants: Dict[str, Dict[str, object]]) -> str:
    """Choose the validation-best fusion variant using loss as a tie-breaker."""
    best_name = None
    best_accuracy = -np.inf
    best_loss = np.inf

    for variant_name, result in variants.items():
        val_metrics = result["split_metrics"]["val"]
        accuracy = float(val_metrics["accuracy"])
        loss = float(val_metrics["loss"])
        if (
            accuracy > best_accuracy + 1e-8
            or (abs(accuracy - best_accuracy) <= 1e-8 and loss < best_loss - 1e-8)
        ):
            best_name = variant_name
            best_accuracy = accuracy
            best_loss = loss

    if best_name is None:
        raise RuntimeError("Failed to select a best fusion variant")
    return best_name


def _curve_value_at_coverage(curve: Dict[str, np.ndarray], target_coverage: float) -> float:
    """Read one retained-accuracy value from a fixed-coverage curve."""
    coverage = np.asarray(curve["coverage"], dtype=float)
    retained_accuracy = np.asarray(curve["retained_accuracy"], dtype=float)
    matches = np.isclose(coverage, float(target_coverage), atol=1e-8)
    if np.any(matches):
        return float(retained_accuracy[matches][0])
    if coverage.size == 0:
        return float("nan")
    nearest_idx = int(np.argmin(np.abs(coverage - float(target_coverage))))
    return float(retained_accuracy[nearest_idx])


def _threshold_slug(confidence_threshold: float) -> str:
    """Create a short stable slug for one confidence threshold."""
    return f"conf_{int(round(100.0 * float(confidence_threshold))):03d}"


def _strong_contradiction_summary(
    sample_table: Dict[str, np.ndarray],
    confidence_threshold: float,
) -> Dict[str, float]:
    """Summarize disagreement where both modalities are confident."""
    disagreement = np.asarray(sample_table["disagreement"], dtype=bool)
    image_confidence = np.asarray(sample_table["image_confidence"], dtype=float)
    genomic_confidence = np.asarray(sample_table["genomic_confidence"], dtype=float)
    image_correct = np.asarray(sample_table["image_correct"], dtype=bool)
    genomic_correct = np.asarray(sample_table["genomic_correct"], dtype=bool)

    mask = (
        disagreement
        & (image_confidence >= float(confidence_threshold))
        & (genomic_confidence >= float(confidence_threshold))
    )
    count = int(mask.sum())
    frequency = float(mask.mean())
    if count == 0:
        return {
            "threshold": float(confidence_threshold),
            "count": 0.0,
            "fraction": frequency,
            "image_correct_genomic_wrong_fraction": 0.0,
            "genomic_correct_image_wrong_fraction": 0.0,
            "split_balance": float("nan"),
        }

    image_only = float((image_correct & ~genomic_correct & mask).sum() / count)
    genomic_only = float((genomic_correct & ~image_correct & mask).sum() / count)
    return {
        "threshold": float(confidence_threshold),
        "count": float(count),
        "fraction": frequency,
        "image_correct_genomic_wrong_fraction": image_only,
        "genomic_correct_image_wrong_fraction": genomic_only,
        "split_balance": max(image_only, genomic_only),
    }


def _build_strategy_rows(
    seed: int,
    contradiction_strength: float,
    modality_asymmetry: float,
    best_late_name: str,
    best_learned_name: str,
    image_summary: Dict[str, object],
    genomic_summary: Dict[str, object],
    late_summary: Dict[str, object],
    learned_summary: Dict[str, object],
    genomic_first_summary: Dict[str, object],
    confidence_route_summary: Dict[str, object],
    abstain_summary: Dict[str, object],
    abstain_eval: Dict[str, object],
    oracle_accuracy: float,
    contradiction_summaries: Dict[float, Dict[str, float]],
) -> Dict[str, object]:
    """Flatten one per-seed evaluation into a CSV-ready row."""
    best_policy_name = (
        "genomic_first"
        if float(genomic_first_summary["accuracy"]) >= float(confidence_route_summary["accuracy"])
        else "confidence_route"
    )
    best_policy_summary = genomic_first_summary if best_policy_name == "genomic_first" else confidence_route_summary

    row = {
        "seed": seed,
        "contradiction_strength": contradiction_strength,
        "modality_asymmetry": modality_asymmetry,
        "best_late_name": best_late_name,
        "best_learned_name": best_learned_name,
        "image_accuracy": float(image_summary["accuracy"]),
        "genomic_accuracy": float(genomic_summary["accuracy"]),
        "late_fusion_accuracy": float(late_summary["accuracy"]),
        "learned_fusion_accuracy": float(learned_summary["accuracy"]),
        "genomic_first_accuracy": float(genomic_first_summary["accuracy"]),
        "confidence_route_accuracy": float(confidence_route_summary["accuracy"]),
        "best_policy_name": best_policy_name,
        "best_policy_accuracy": float(best_policy_summary["accuracy"]),
        "best_policy_gain_over_learned": float(best_policy_summary["accuracy"]) - float(learned_summary["accuracy"]),
        "oracle_accuracy": float(oracle_accuracy),
        "prediction_disagreement_frequency": float(learned_summary["disagreement_fraction"]),
        "moderate_prediction_contradiction_frequency": float(contradiction_summaries[0.60]["fraction"]),
        "strong_prediction_contradiction_frequency": float(contradiction_summaries[0.70]["fraction"]),
        "strong_structural_contradiction_frequency": float(learned_summary["strong_contradiction_fraction"]),
        "learned_fusion_regret": float(learned_summary["regret"]),
        "best_policy_regret": float(best_policy_summary["regret"]),
        "learned_fusion_disagreement_error": float(learned_summary["disagreement_error_rate"]),
        "best_policy_disagreement_error": float(best_policy_summary["disagreement_error_rate"]),
        "learned_fusion_high_conf_error": float(learned_summary["high_confidence_error_rate"]),
        "best_policy_high_conf_error": float(best_policy_summary["high_confidence_error_rate"]),
        "learned_fusion_brier": float(learned_summary["brier"]),
        "learned_fusion_ece_10bin": float(learned_summary["ece_10bin"]),
        "best_policy_brier": float(best_policy_summary["brier"]),
        "best_policy_ece_10bin": float(best_policy_summary["ece_10bin"]),
        "abstain_coverage": float(abstain_summary["coverage"]),
        "abstain_selective_accuracy": float(abstain_summary["accuracy"]),
        "abstain_flagged_error": float(abstain_eval["error_rate_flagged"]),
        "selective_accuracy_at_050": _curve_value_at_coverage(abstain_eval["curve"], 0.50),
        "selective_accuracy_at_060": _curve_value_at_coverage(abstain_eval["curve"], 0.60),
        "selective_accuracy_at_070": _curve_value_at_coverage(abstain_eval["curve"], 0.70),
        "selective_accuracy_at_080": _curve_value_at_coverage(abstain_eval["curve"], 0.80),
        "selective_accuracy_at_090": _curve_value_at_coverage(abstain_eval["curve"], 0.90),
        "selective_accuracy_at_100": _curve_value_at_coverage(abstain_eval["curve"], 1.00),
    }
    for confidence_threshold, summary in contradiction_summaries.items():
        slug = _threshold_slug(confidence_threshold)
        row[f"{slug}_contradiction_count"] = float(summary["count"])
        row[f"{slug}_contradiction_frequency"] = float(summary["fraction"])
        row[f"{slug}_image_correct_genomic_wrong_fraction"] = float(
            summary["image_correct_genomic_wrong_fraction"]
        )
        row[f"{slug}_genomic_correct_image_wrong_fraction"] = float(
            summary["genomic_correct_image_wrong_fraction"]
        )
        row[f"{slug}_split_balance"] = float(summary["split_balance"])
    return row


def _aggregate_rows(rows: Sequence[Dict[str, object]]) -> list[Dict[str, object]]:
    """Aggregate per-seed rows into one summary per contradiction regime."""
    grouped: Dict[tuple[float, float], list[Dict[str, object]]] = defaultdict(list)
    for row in rows:
        key = (float(row["contradiction_strength"]), float(row["modality_asymmetry"]))
        grouped[key].append(row)

    summary_rows = []
    for key in sorted(grouped):
        setting_rows = grouped[key]
        contradiction_strength, modality_asymmetry = key
        best_policy_counts = Counter(str(row["best_policy_name"]) for row in setting_rows)
        best_policy_name = best_policy_counts.most_common(1)[0][0]

        summary = {
            "contradiction_strength": contradiction_strength,
            "modality_asymmetry": modality_asymmetry,
            "n_seeds": len(setting_rows),
            "best_policy_name_mode": best_policy_name,
        }
        metric_names = (
            "image_accuracy",
            "genomic_accuracy",
            "late_fusion_accuracy",
            "learned_fusion_accuracy",
            "genomic_first_accuracy",
            "confidence_route_accuracy",
            "best_policy_accuracy",
            "best_policy_gain_over_learned",
            "oracle_accuracy",
            "prediction_disagreement_frequency",
            "moderate_prediction_contradiction_frequency",
            "strong_prediction_contradiction_frequency",
            "strong_structural_contradiction_frequency",
            "learned_fusion_regret",
            "best_policy_regret",
            "learned_fusion_disagreement_error",
            "best_policy_disagreement_error",
            "learned_fusion_high_conf_error",
            "best_policy_high_conf_error",
            "learned_fusion_brier",
            "learned_fusion_ece_10bin",
            "best_policy_brier",
            "best_policy_ece_10bin",
            "abstain_coverage",
            "abstain_selective_accuracy",
            "abstain_flagged_error",
            "selective_accuracy_at_050",
            "selective_accuracy_at_060",
            "selective_accuracy_at_070",
            "selective_accuracy_at_080",
            "selective_accuracy_at_090",
            "selective_accuracy_at_100",
            "conf_060_contradiction_count",
            "conf_060_contradiction_frequency",
            "conf_060_image_correct_genomic_wrong_fraction",
            "conf_060_genomic_correct_image_wrong_fraction",
            "conf_060_split_balance",
            "conf_070_contradiction_count",
            "conf_070_contradiction_frequency",
            "conf_070_image_correct_genomic_wrong_fraction",
            "conf_070_genomic_correct_image_wrong_fraction",
            "conf_070_split_balance",
        )
        for metric_name in metric_names:
            values = [float(row[metric_name]) for row in setting_rows]
            summary[f"{metric_name}_mean"] = _safe_mean(values)
            summary[f"{metric_name}_std"] = _safe_std(values)
        summary_rows.append(summary)

    return summary_rows


def _write_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    """Write a machine-readable CSV table."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"No rows available for CSV output: {path}")
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _save_selective_curve_bundle(
    path: Path,
    rows: Sequence[Dict[str, object]],
    curve_bundle: Sequence[Dict[str, object]],
) -> None:
    """Save selective-accuracy curves in one compact NPZ archive."""
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        contradiction_strength=np.asarray([float(row["contradiction_strength"]) for row in rows], dtype=float),
        modality_asymmetry=np.asarray([float(row["modality_asymmetry"]) for row in rows], dtype=float),
        seed=np.asarray([int(row["seed"]) for row in rows], dtype=int),
        coverage=np.stack([np.asarray(bundle["coverage"], dtype=float) for bundle in curve_bundle], axis=0),
        retained_accuracy=np.stack(
            [np.asarray(bundle["retained_accuracy"], dtype=float) for bundle in curve_bundle],
            axis=0,
        ),
        retained_count=np.stack([np.asarray(bundle["retained_count"], dtype=int) for bundle in curve_bundle], axis=0),
    )


def build_summary_figure(summary_rows: Sequence[Dict[str, object]], save_path: Path) -> None:
    """Save the main V2 contradiction-sweep figure."""
    one_dimensional = sorted(summary_rows, key=lambda row: float(row["contradiction_strength"]))
    strengths = np.asarray([float(row["contradiction_strength"]) for row in one_dimensional], dtype=float)

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    accuracy_series = (
        ("image", "image_accuracy_mean", "#4C72B0"),
        ("genomic", "genomic_accuracy_mean", "#55A868"),
        ("late fusion", "late_fusion_accuracy_mean", "#8172B2"),
        ("learned fusion", "learned_fusion_accuracy_mean", "#C44E52"),
        ("best policy", "best_policy_accuracy_mean", "#DD8452"),
    )
    for label, key, color in accuracy_series:
        axes[0, 0].plot(
            strengths,
            [float(row[key]) for row in one_dimensional],
            marker="o",
            linewidth=2.0,
            color=color,
            label=label,
        )
    axes[0, 0].set_ylim(0.0, 1.0)
    axes[0, 0].set_xlabel("Contradiction strength")
    axes[0, 0].set_ylabel("Accuracy")
    axes[0, 0].set_title("Accuracy vs Contradiction Strength")
    axes[0, 0].legend(frameon=False)

    axes[0, 1].plot(
        strengths,
        [float(row["prediction_disagreement_frequency_mean"]) for row in one_dimensional],
        marker="o",
        linewidth=2.0,
        color="#4C72B0",
        label="prediction disagreement",
    )
    axes[0, 1].plot(
        strengths,
        [float(row["conf_060_contradiction_frequency_mean"]) for row in one_dimensional],
        marker="s",
        linewidth=2.0,
        color="#55A868",
        label="conf>=0.60 contradiction",
    )
    axes[0, 1].plot(
        strengths,
        [float(row["conf_070_contradiction_frequency_mean"]) for row in one_dimensional],
        marker="^",
        linewidth=2.0,
        color="#C44E52",
        label="conf>=0.70 contradiction",
    )
    axes[0, 1].set_ylim(0.0, 1.0)
    axes[0, 1].set_xlabel("Contradiction strength")
    axes[0, 1].set_ylabel("Frequency")
    axes[0, 1].set_title("Contradiction Frequency")
    axes[0, 1].legend(frameon=False)

    axes[0, 2].plot(
        strengths,
        [float(row["learned_fusion_regret_mean"]) for row in one_dimensional],
        marker="o",
        linewidth=2.0,
        color="#C44E52",
        label="learned fusion",
    )
    axes[0, 2].plot(
        strengths,
        [float(row["best_policy_regret_mean"]) for row in one_dimensional],
        marker="s",
        linewidth=2.0,
        color="#DD8452",
        label="best policy",
    )
    axes[0, 2].axhline(0.0, linestyle="--", color="0.5", linewidth=1.0)
    axes[0, 2].set_ylim(0.0, 1.0)
    axes[0, 2].set_xlabel("Contradiction strength")
    axes[0, 2].set_ylabel("Regret")
    axes[0, 2].set_title("Regret vs Oracle Branch")
    axes[0, 2].legend(frameon=False)

    axes[1, 0].plot(
        strengths,
        [float(row["learned_fusion_disagreement_error_mean"]) for row in one_dimensional],
        marker="o",
        linewidth=2.0,
        color="#C44E52",
        label="learned fusion",
    )
    axes[1, 0].plot(
        strengths,
        [float(row["best_policy_disagreement_error_mean"]) for row in one_dimensional],
        marker="s",
        linewidth=2.0,
        color="#DD8452",
        label="best policy",
    )
    axes[1, 0].set_ylim(0.0, 1.0)
    axes[1, 0].set_xlabel("Contradiction strength")
    axes[1, 0].set_ylabel("Error rate on disagreement")
    axes[1, 0].set_title("Conditional Error on Disagreement")
    axes[1, 0].legend(frameon=False)

    axes[1, 1].plot(
        strengths,
        [float(row["selective_accuracy_at_080_mean"]) for row in one_dimensional],
        marker="o",
        linewidth=2.0,
        color="#8172B2",
        label="selective acc@0.80",
    )
    axes[1, 1].plot(
        strengths,
        [float(row["selective_accuracy_at_090_mean"]) for row in one_dimensional],
        marker="s",
        linewidth=2.0,
        color="#4C72B0",
        label="selective acc@0.90",
    )
    axes[1, 1].plot(
        strengths,
        [float(row["abstain_coverage_mean"]) for row in one_dimensional],
        marker="^",
        linewidth=2.0,
        color="#55A868",
        label="fixed abstention coverage",
    )
    axes[1, 1].set_ylim(0.0, 1.0)
    axes[1, 1].set_xlabel("Contradiction strength")
    axes[1, 1].set_ylabel("Selective metric")
    axes[1, 1].set_title("Abstention Performance")
    axes[1, 1].legend(frameon=False)

    policy_gain = np.asarray(
        [float(row["best_policy_accuracy_mean"]) - float(row["learned_fusion_accuracy_mean"]) for row in one_dimensional],
        dtype=float,
    )
    axes[1, 2].bar(strengths, policy_gain, width=0.10, color="#DD8452", alpha=0.85)
    axes[1, 2].axhline(0.0, linestyle="--", color="0.5", linewidth=1.0)
    axes[1, 2].set_xlabel("Contradiction strength")
    axes[1, 2].set_ylabel("Accuracy gain")
    axes[1, 2].set_title("Best Policy Minus Learned Fusion")

    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def build_phase_diagram(summary_rows: Sequence[Dict[str, object]], save_path: Path) -> None:
    """Save an optional 2D phase diagram when multiple asymmetry values are present."""
    strengths = sorted({float(row["contradiction_strength"]) for row in summary_rows})
    asymmetries = sorted({float(row["modality_asymmetry"]) for row in summary_rows})
    if len(asymmetries) <= 1:
        return

    strategy_names = ("learned_fusion", "genomic_first", "confidence_route")
    strategy_colors = {
        "learned_fusion": "#C44E52",
        "genomic_first": "#55A868",
        "confidence_route": "#DD8452",
    }
    lookup = {
        (float(row["contradiction_strength"]), float(row["modality_asymmetry"])): row
        for row in summary_rows
    }
    strategy_index = {name: idx for idx, name in enumerate(strategy_names)}
    color_grid = np.zeros((len(asymmetries), len(strengths)), dtype=int)

    for row_idx, asymmetry in enumerate(asymmetries):
        for col_idx, contradiction_strength in enumerate(strengths):
            row = lookup[(contradiction_strength, asymmetry)]
            strategy_scores = {
                "learned_fusion": float(row["learned_fusion_accuracy_mean"]),
                "genomic_first": float(row["genomic_first_accuracy_mean"]),
                "confidence_route": float(row["confidence_route_accuracy_mean"]),
            }
            best_strategy = max(strategy_scores, key=strategy_scores.get)
            color_grid[row_idx, col_idx] = strategy_index[best_strategy]

    from matplotlib.colors import ListedColormap

    cmap = ListedColormap([strategy_colors[name] for name in strategy_names])
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.imshow(color_grid, cmap=cmap, origin="lower", aspect="auto")
    ax.set_xticks(np.arange(len(strengths)))
    ax.set_xticklabels([f"{value:.2f}" for value in strengths])
    ax.set_yticks(np.arange(len(asymmetries)))
    ax.set_yticklabels([f"{value:+.2f}" for value in asymmetries])
    ax.set_xlabel("Contradiction strength")
    ax.set_ylabel("Modality asymmetry")
    ax.set_title("Best Full-Coverage Strategy")

    legend_handles = [
        plt.Line2D([0], [0], marker="s", color="w", markerfacecolor=strategy_colors[name], markersize=10, label=name)
        for name in strategy_names
    ]
    ax.legend(handles=legend_handles, frameon=False, loc="upper left")
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def write_report(
    summary_rows: Sequence[Dict[str, object]],
    save_path: Path,
    settings: Dict[str, object],
) -> None:
    """Write a concise text summary of the V2 findings."""
    strongest_row = max(summary_rows, key=lambda row: float(row["contradiction_strength"]))
    max_policy_gain_row = max(
        summary_rows,
        key=lambda row: float(row["best_policy_accuracy_mean"]) - float(row["learned_fusion_accuracy_mean"]),
    )
    policy_win_rows = [
        row
        for row in summary_rows
        if float(row["best_policy_accuracy_mean"]) > float(row["learned_fusion_accuracy_mean"]) + 1e-8
    ]

    lines = [
        "V2 contradiction sweep summary",
        f"profile={settings['profile']} mode={settings['mode']} image_model={settings['image_model']}",
        (
            "contradiction_strengths="
            + ",".join(f"{value:.2f}" for value in settings["contradiction_strengths"])
            + " modality_asymmetries="
            + ",".join(f"{value:+.2f}" for value in settings["modality_asymmetries"])
        ),
        "seeds=" + ",".join(str(seed) for seed in settings["seeds"]),
        "",
        "Aggregate findings",
        (
            f"- strongest tested regime: cs={strongest_row['contradiction_strength']:.2f}, "
            f"asym={strongest_row['modality_asymmetry']:+.2f}"
        ),
        (
            "- strongest regime contradiction: disagreement="
            f"{strongest_row['prediction_disagreement_frequency_mean']:.3f} +/- "
            f"{strongest_row['prediction_disagreement_frequency_std']:.3f}, "
            "conf>=0.60="
            f"{strongest_row['conf_060_contradiction_frequency_mean']:.3f} +/- "
            f"{strongest_row['conf_060_contradiction_frequency_std']:.3f}, "
            "conf>=0.70="
            f"{strongest_row['conf_070_contradiction_frequency_mean']:.3f} +/- "
            f"{strongest_row['conf_070_contradiction_frequency_std']:.3f}"
        ),
        (
            "- strongest regime accuracies: image="
            f"{strongest_row['image_accuracy_mean']:.3f}, genomic={strongest_row['genomic_accuracy_mean']:.3f}, "
            f"late={strongest_row['late_fusion_accuracy_mean']:.3f}, "
            f"learned={strongest_row['learned_fusion_accuracy_mean']:.3f}, "
            f"best_policy={strongest_row['best_policy_accuracy_mean']:.3f}"
        ),
        (
            "- strongest regime regret: learned="
            f"{strongest_row['learned_fusion_regret_mean']:.3f}, "
            f"best_policy={strongest_row['best_policy_regret_mean']:.3f}, "
            f"oracle={strongest_row['oracle_accuracy_mean']:.3f}"
        ),
        (
            "- largest policy gain over learned fusion: "
            f"{max_policy_gain_row['best_policy_accuracy_mean'] - max_policy_gain_row['learned_fusion_accuracy_mean']:.3f} "
            f"at cs={max_policy_gain_row['contradiction_strength']:.2f}, "
            f"asym={max_policy_gain_row['modality_asymmetry']:+.2f}"
        ),
        (
            "- strongest regime contradiction split @conf>=0.70: image-only="
            f"{strongest_row['conf_070_image_correct_genomic_wrong_fraction_mean']:.3f}, "
            f"genomic-only={strongest_row['conf_070_genomic_correct_image_wrong_fraction_mean']:.3f}"
        ),
        (
            f"- policy beats learned fusion in {len(policy_win_rows)} / {len(summary_rows)} settings"
        ),
        (
            "- abstention at strongest regime: fixed coverage="
            f"{strongest_row['abstain_coverage_mean']:.3f}, "
            f"selective_accuracy={strongest_row['abstain_selective_accuracy_mean']:.3f}, "
            f"selective_accuracy@0.90={strongest_row['selective_accuracy_at_090_mean']:.3f}"
        ),
        "",
        "Interpretation",
        "- The sweep should be read as a controlled study of when disagreement becomes common, when regret rises, and whether simple policies recover avoidable fusion mistakes.",
        "- If strong structural contradiction remains low even at high contradiction strength, that is a real negative finding rather than a failed run.",
        "- Best-policy values are descriptive maxima over the implemented full-coverage policies; deployable policy selection should still be validated separately.",
    ]

    save_path.parent.mkdir(parents=True, exist_ok=True)
    save_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def evaluate_single_setting(
    seed: int,
    contradiction_strength: float,
    modality_asymmetry: float,
    args: argparse.Namespace,
) -> Dict[str, object]:
    """Run one seed at one contradiction setting and keep rich diagnostics."""
    regime = ContradictionRegimeConfig(
        contradiction_strength=contradiction_strength,
        modality_asymmetry=modality_asymmetry,
        structural_strong_threshold=args.structural_strong_threshold,
    )
    dataset = build_v2_paired_dataset(
        regime=regime,
        mode=args.mode,
        image_size=args.image_size,
        genomic_representation=args.genomic_representation,
        seed=seed,
    )
    if args.image_model == "hybrid":
        dataset = attach_image_features(dataset)

    controls = train_unimodal_controls(
        dataset=dataset,
        image_model=args.image_model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        patience=args.patience,
        min_epochs=args.min_epochs,
        device=args.device,
        seed=seed,
        high_confidence_threshold=args.high_confidence_threshold,
    )
    late_variants = evaluate_late_fusion_variants(
        controls=controls,
        high_confidence_threshold=args.high_confidence_threshold,
    )
    learned_variants = {}
    for offset, variant_name in enumerate(LEARNED_FUSION_VARIANTS):
        variant_result = train_learned_fusion_variant(
            dataset=dataset,
            controls=controls,
            image_model=args.image_model,
            variant_name=variant_name,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            patience=args.patience,
            min_epochs=args.min_epochs,
            device=args.device,
            seed=seed + 10 + offset,
            high_confidence_threshold=args.high_confidence_threshold,
        )
        learned_variants[variant_name] = attach_variant_safety_diagnostics(
            variant_result,
            image_metrics=controls["image"]["split_metrics"],
            genomic_metrics=controls["genomic"]["split_metrics"],
        )

    best_late_name = _select_best_variant(late_variants)
    best_learned_name = _select_best_variant(learned_variants)
    best_late = late_variants[best_late_name]
    best_learned = learned_variants[best_learned_name]

    val_table = build_split_sample_table(
        dataset=dataset,
        split_name="val",
        image_eval=controls["image"]["split_metrics"]["val"],
        genomic_eval=controls["genomic"]["split_metrics"]["val"],
        fusion_eval=best_learned["split_metrics"]["val"],
        seed=seed,
    )
    test_table_learned = build_split_sample_table(
        dataset=dataset,
        split_name="test",
        image_eval=controls["image"]["split_metrics"]["test"],
        genomic_eval=controls["genomic"]["split_metrics"]["test"],
        fusion_eval=best_learned["split_metrics"]["test"],
        seed=seed,
    )
    test_table_late = build_split_sample_table(
        dataset=dataset,
        split_name="test",
        image_eval=controls["image"]["split_metrics"]["test"],
        genomic_eval=controls["genomic"]["split_metrics"]["test"],
        fusion_eval=best_late["split_metrics"]["test"],
        seed=seed,
    )

    image_summary = summarize_strategy(
        sample_table=test_table_learned,
        positive_probability=test_table_learned["image_probability_cancer"],
        high_confidence_threshold=args.high_confidence_threshold,
        strong_contradiction_threshold=args.structural_strong_threshold,
    )
    genomic_summary = summarize_strategy(
        sample_table=test_table_learned,
        positive_probability=test_table_learned["genomic_probability_cancer"],
        high_confidence_threshold=args.high_confidence_threshold,
        strong_contradiction_threshold=args.structural_strong_threshold,
    )
    late_summary = summarize_strategy(
        sample_table=test_table_late,
        positive_probability=test_table_late["fusion_probability_cancer"],
        high_confidence_threshold=args.high_confidence_threshold,
        strong_contradiction_threshold=args.structural_strong_threshold,
    )
    learned_summary = summarize_strategy(
        sample_table=test_table_learned,
        positive_probability=test_table_learned["fusion_probability_cancer"],
        high_confidence_threshold=args.high_confidence_threshold,
        strong_contradiction_threshold=args.structural_strong_threshold,
    )

    genomic_first_fit = tune_genomic_first_policy(val_table)
    confidence_route_fit = tune_confidence_routing_policy(val_table)
    genomic_first_output = apply_genomic_first_policy(test_table_learned, threshold=genomic_first_fit["threshold"])
    confidence_route_output = apply_confidence_routing_policy(
        test_table_learned,
        threshold=confidence_route_fit["threshold"],
    )
    genomic_first_summary = summarize_strategy(
        sample_table=test_table_learned,
        positive_probability=genomic_first_output["positive_probability"],
        high_confidence_threshold=args.high_confidence_threshold,
        strong_contradiction_threshold=args.structural_strong_threshold,
    )
    confidence_route_summary = summarize_strategy(
        sample_table=test_table_learned,
        positive_probability=confidence_route_output["positive_probability"],
        high_confidence_threshold=args.high_confidence_threshold,
        strong_contradiction_threshold=args.structural_strong_threshold,
    )

    abstention_fit = calibrate_abstention_policy(val_table)
    abstention_eval = evaluate_abstention_policy(
        sample_table=test_table_learned,
        thresholds=abstention_fit["thresholds"],
        coverage_levels=DEFAULT_SELECTIVE_COVERAGE_LEVELS,
    )
    abstain_summary = summarize_strategy(
        sample_table=test_table_learned,
        positive_probability=test_table_learned["fusion_probability_cancer"],
        high_confidence_threshold=args.high_confidence_threshold,
        strong_contradiction_threshold=args.structural_strong_threshold,
        abstain_mask=abstention_eval["flags"],
    )
    contradiction_summaries = {
        threshold: _strong_contradiction_summary(test_table_learned, confidence_threshold=threshold)
        for threshold in VALIDATION_CONFIDENCE_THRESHOLDS
    }

    row = _build_strategy_rows(
        seed=seed,
        contradiction_strength=contradiction_strength,
        modality_asymmetry=modality_asymmetry,
        best_late_name=best_late_name,
        best_learned_name=best_learned_name,
        image_summary=image_summary,
        genomic_summary=genomic_summary,
        late_summary=late_summary,
        learned_summary=learned_summary,
        genomic_first_summary=genomic_first_summary,
        confidence_route_summary=confidence_route_summary,
        abstain_summary=abstain_summary,
        abstain_eval=abstention_eval,
        oracle_accuracy=oracle_branch_accuracy(test_table_learned),
        contradiction_summaries=contradiction_summaries,
    )

    curve_bundle = {
        "coverage": np.asarray(abstention_eval["curve"]["coverage"], dtype=float),
        "retained_accuracy": np.asarray(abstention_eval["curve"]["retained_accuracy"], dtype=float),
        "retained_count": np.asarray(abstention_eval["curve"]["retained_count"], dtype=int),
    }
    return {
        "row": row,
        "curve_bundle": curve_bundle,
        "test_table": test_table_learned,
        "contradiction_summaries": contradiction_summaries,
        "best_late_name": best_late_name,
        "best_learned_name": best_learned_name,
    }


def run_single_setting(
    seed: int,
    contradiction_strength: float,
    modality_asymmetry: float,
    args: argparse.Namespace,
) -> tuple[Dict[str, object], Dict[str, np.ndarray]]:
    """Run one seed at one contradiction setting."""
    result = evaluate_single_setting(
        seed=seed,
        contradiction_strength=contradiction_strength,
        modality_asymmetry=modality_asymmetry,
        args=args,
    )
    row = result["row"]

    print(
        f"seed={seed} cs={contradiction_strength:.2f} asym={modality_asymmetry:+.2f} "
        f"image={row['image_accuracy']:.3f} genomic={row['genomic_accuracy']:.3f} "
        f"late={row['late_fusion_accuracy']:.3f} learned={row['learned_fusion_accuracy']:.3f} "
        f"best_policy={row['best_policy_name']}:{row['best_policy_accuracy']:.3f} "
        f"disagreement={row['prediction_disagreement_frequency']:.3f} "
        "conflict@0.60="
        f"{row['moderate_prediction_contradiction_frequency']:.3f} "
        "strong_conflict@0.70="
        f"{row['strong_prediction_contradiction_frequency']:.3f} "
        f"structural_strong={row['strong_structural_contradiction_frequency']:.3f}"
    )
    return row, result["curve_bundle"]


def _validation_args(args: argparse.Namespace) -> argparse.Namespace:
    """Build a lightweight validation configuration before the full sweep."""
    validation = argparse.Namespace(**vars(args))
    validation.mode = "full"
    validation.contradiction_strengths = VALIDATION_CONTRADICTION_STRENGTHS
    validation.modality_asymmetries = (float(args.modality_asymmetries[0]),)
    validation.seeds = (int(args.seeds[0]),)
    validation.epochs = min(int(args.epochs), 18)
    validation.min_epochs = min(int(args.min_epochs), 5)
    validation.patience = min(int(args.patience), 5)
    validation.batch_size = min(int(args.batch_size), 16)
    return validation


def _validation_rows_to_report(rows: Sequence[Dict[str, object]]) -> str:
    """Format the validation sweep into a short text report."""
    lines = [
        "V2 validation gate",
        (
            "criteria: "
            f"image@cs1>={VALIDATION_MIN_IMAGE_ACCURACY:.2f}, "
            f"genomic@cs1>={VALIDATION_MIN_GENOMIC_ACCURACY:.2f}, "
            f"conf>=0.70 contradiction>={VALIDATION_MIN_STRONG_CONTRADICTION_RATE:.2f}, "
            f"correctness split<={VALIDATION_MAX_CORRECTNESS_SPLIT:.2f}/{1.0 - VALIDATION_MAX_CORRECTNESS_SPLIT:.2f}"
        ),
        "",
    ]
    for row in rows:
        lines.extend(
            [
                (
                    f"cs={float(row['contradiction_strength']):.2f} "
                    f"image={float(row['image_accuracy']):.3f} "
                    f"genomic={float(row['genomic_accuracy']):.3f} "
                    f"late={float(row['late_fusion_accuracy']):.3f} "
                    f"learned={float(row['learned_fusion_accuracy']):.3f} "
                    f"best_policy={float(row['best_policy_accuracy']):.3f} "
                    f"policy_gain={float(row['best_policy_gain_over_learned']):.3f}"
                ),
                (
                    f"  disagreement={float(row['prediction_disagreement_frequency']):.3f} "
                    f"conf>=0.60={float(row['conf_060_contradiction_frequency']):.3f} "
                    f"split={float(row['conf_060_image_correct_genomic_wrong_fraction']):.3f}/"
                    f"{float(row['conf_060_genomic_correct_image_wrong_fraction']):.3f}"
                ),
                (
                    f"  conf>=0.70={float(row['conf_070_contradiction_frequency']):.3f} "
                    f"split={float(row['conf_070_image_correct_genomic_wrong_fraction']):.3f}/"
                    f"{float(row['conf_070_genomic_correct_image_wrong_fraction']):.3f}"
                ),
            ]
        )
    return "\n".join(lines) + "\n"


def run_validation_gate(args: argparse.Namespace) -> tuple[bool, list[Dict[str, object]]]:
    """Run the required validation gate before the full sweep."""
    validation_args = _validation_args(args)
    validation_rows = []

    print(
        "\nRunning validation gate with "
        f"mode={validation_args.mode}, strengths={list(validation_args.contradiction_strengths)}, "
        f"seeds={list(validation_args.seeds)}"
    )
    for contradiction_strength in validation_args.contradiction_strengths:
        result = evaluate_single_setting(
            seed=int(validation_args.seeds[0]),
            contradiction_strength=float(contradiction_strength),
            modality_asymmetry=float(validation_args.modality_asymmetries[0]),
            args=validation_args,
        )
        row = result["row"]
        validation_rows.append(row)
        print(
            f"validation cs={contradiction_strength:.2f} "
            f"image={float(row['image_accuracy']):.3f} "
            f"genomic={float(row['genomic_accuracy']):.3f} "
            f"late={float(row['late_fusion_accuracy']):.3f} "
            f"learned={float(row['learned_fusion_accuracy']):.3f} "
            f"best_policy={float(row['best_policy_accuracy']):.3f} "
            f"disagreement={float(row['prediction_disagreement_frequency']):.3f} "
            f"conf>=0.60={float(row['conf_060_contradiction_frequency']):.3f} "
            f"conf>=0.70={float(row['conf_070_contradiction_frequency']):.3f}"
        )

    strongest = max(validation_rows, key=lambda row: float(row["contradiction_strength"]))
    checks = [
        (
            "image accuracy at cs=1.0",
            float(strongest["image_accuracy"]) >= VALIDATION_MIN_IMAGE_ACCURACY,
            float(strongest["image_accuracy"]),
            VALIDATION_MIN_IMAGE_ACCURACY,
        ),
        (
            "genomic accuracy at cs=1.0",
            float(strongest["genomic_accuracy"]) >= VALIDATION_MIN_GENOMIC_ACCURACY,
            float(strongest["genomic_accuracy"]),
            VALIDATION_MIN_GENOMIC_ACCURACY,
        ),
        (
            "conf>=0.70 contradiction rate at cs=1.0",
            float(strongest["conf_070_contradiction_frequency"]) >= VALIDATION_MIN_STRONG_CONTRADICTION_RATE,
            float(strongest["conf_070_contradiction_frequency"]),
            VALIDATION_MIN_STRONG_CONTRADICTION_RATE,
        ),
        (
            "conf>=0.70 contradiction balance at cs=1.0",
            float(strongest["conf_070_split_balance"]) <= VALIDATION_MAX_CORRECTNESS_SPLIT,
            float(strongest["conf_070_split_balance"]),
            VALIDATION_MAX_CORRECTNESS_SPLIT,
        ),
    ]
    validation_pass = all(check[1] for check in checks)

    report_lines = [_validation_rows_to_report(validation_rows), "Gate checks"]
    for label, passed, value, target in checks:
        comparator = ">=" if "balance" not in label else "<="
        report_lines.append(
            f"- {label}: {'PASS' if passed else 'FAIL'} ({value:.3f} {comparator} {target:.3f})"
        )
    report_lines.append("")
    report_lines.append(f"validation_pass={validation_pass}")

    base_dir = RESULTS_DIR / "v2"
    validation_table_path = base_dir / "tables" / "v2_validation_metrics.csv"
    validation_report_path = base_dir / "reports" / "v2_validation_report.txt"
    _write_csv(validation_table_path, validation_rows)
    validation_report_path.parent.mkdir(parents=True, exist_ok=True)
    validation_report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    print(f"validation report: {validation_report_path}")
    print(f"validation table: {validation_table_path}")

    if not validation_pass:
        print("Validation gate failed. Full sweep aborted.")
    return validation_pass, validation_rows


def main() -> None:
    """Run the V2 contradiction sweep and save artifacts."""
    args = parse_args()
    args.mode = _profile_value(args, "mode")
    args.epochs = _profile_value(args, "epochs")
    args.min_epochs = _profile_value(args, "min_epochs")
    args.patience = _profile_value(args, "patience")
    args.batch_size = _profile_value(args, "batch_size")
    args.contradiction_strengths = tuple(
        _profile_value(args, "contradiction_strengths")
        if args.contradiction_strengths is None
        else args.contradiction_strengths
    )
    args.modality_asymmetries = tuple(
        _profile_value(args, "modality_asymmetries")
        if args.modality_asymmetries is None
        else args.modality_asymmetries
    )
    args.seeds = tuple(_profile_value(args, "seeds") if args.seeds is None else args.seeds)

    print(
        "Running V2 contradiction sweep with "
        f"profile={args.profile}, mode={args.mode}, image_model={args.image_model}, "
        f"strengths={list(args.contradiction_strengths)}, "
        f"asymmetries={list(args.modality_asymmetries)}, seeds={list(args.seeds)}"
    )

    if args.profile != "debug":
        validation_pass, _ = run_validation_gate(args)
        if not validation_pass:
            raise RuntimeError("V2 validation gate failed. Full sweep was not executed.")

    result_rows = []
    curve_rows = []
    for contradiction_strength in args.contradiction_strengths:
        for modality_asymmetry in args.modality_asymmetries:
            for seed in args.seeds:
                row, curve_bundle = run_single_setting(
                    seed=seed,
                    contradiction_strength=float(contradiction_strength),
                    modality_asymmetry=float(modality_asymmetry),
                    args=args,
                )
                result_rows.append(row)
                curve_rows.append(curve_bundle)

    summary_rows = _aggregate_rows(result_rows)

    base_dir = RESULTS_DIR / "v2"
    seed_table_path = base_dir / "tables" / "v2_sweep_metrics.csv"
    summary_table_path = base_dir / "tables" / "v2_sweep_summary.csv"
    curve_path = base_dir / "tables" / "v2_selective_curves.npz"
    report_path = base_dir / "reports" / "v2_main_findings.txt"
    figure_path = base_dir / "figures" / "v2_contradiction_sweep.png"
    phase_path = base_dir / "figures" / "v2_phase_diagram.png"

    _write_csv(seed_table_path, result_rows)
    _write_csv(summary_table_path, summary_rows)
    _save_selective_curve_bundle(curve_path, result_rows, curve_rows)
    build_summary_figure(
        [row for row in summary_rows if float(row["modality_asymmetry"]) == float(args.modality_asymmetries[0])],
        figure_path,
    )
    build_phase_diagram(summary_rows, phase_path)
    write_report(
        summary_rows=summary_rows,
        save_path=report_path,
        settings={
            "profile": args.profile,
            "mode": args.mode,
            "image_model": args.image_model,
            "contradiction_strengths": args.contradiction_strengths,
            "modality_asymmetries": args.modality_asymmetries,
            "seeds": args.seeds,
        },
    )

    print("\nSaved artifacts")
    print(f"- seed table: {seed_table_path}")
    print(f"- summary table: {summary_table_path}")
    print(f"- selective curves: {curve_path}")
    print(f"- report: {report_path}")
    print(f"- figure: {figure_path}")
    if len(args.modality_asymmetries) > 1:
        print(f"- phase diagram: {phase_path}")

    if not args.no_show:
        image = plt.imread(figure_path)
        plt.figure(figsize=(12, 6))
        plt.imshow(image)
        plt.axis("off")
        plt.show()


if __name__ == "__main__":
    main()
