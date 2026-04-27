"""Hand-crafted feature baselines for diagnosing the sandbox ML pipeline."""

from __future__ import annotations

from typing import Dict, Sequence

import numpy as np
import torch
from torch import nn

from polymer_image_renderer import image_metrics


IMAGE_FEATURE_NAMES = (
    "mean_intensity",
    "intensity_entropy",
    "local_texture_variation",
    "patch_density_variation",
    "radial_intensity_std",
    "occupancy_fraction",
)

GENOMIC_FEATURE_NAMES = (
    "mean_contact",
    "distal_contact_mass",
    "near_far_ratio",
    "contact_decay_slope",
    "row_sum_cv",
    "band_profile_entropy",
    "distal_patch_variation",
    "distal_contact_cv",
)


class FeatureClassifier(nn.Module):
    """A tiny classifier for hand-crafted features."""

    def __init__(self, input_dim: int, hidden_dim: int = 0) -> None:
        super().__init__()
        if hidden_dim <= 0:
            self.network = nn.Linear(input_dim, 2)
        else:
            self.network = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 2),
            )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Predict binary logits from standardized feature vectors."""
        return self.network(features)


def _safe_ratio(numerator: float, denominator: float) -> float:
    """Compute a finite ratio with a small epsilon."""
    return float(numerator / (denominator + 1e-8))


def _coarse_patch_variation(matrix: np.ndarray, n_bins: int = 4) -> float:
    """Measure heterogeneity after coarse spatial pooling."""
    matrix_array = np.asarray(matrix, dtype=float)
    usable_size = (matrix_array.shape[0] // n_bins) * n_bins
    if usable_size == 0:
        return 0.0

    trimmed = matrix_array[:usable_size, :usable_size]
    patch_size = usable_size // n_bins
    pooled = trimmed.reshape(n_bins, patch_size, n_bins, patch_size).mean(axis=(1, 3))
    return float(pooled.std(ddof=0) / (pooled.mean() + 1e-8))


def _distal_patch_variation(matrix: np.ndarray, min_separation: int) -> float:
    """Measure blockiness away from the diagonal."""
    matrix_array = np.asarray(matrix, dtype=float)
    indices = np.arange(matrix_array.shape[0])
    separation = np.abs(indices[:, None] - indices[None, :])
    masked = np.where(separation >= min_separation, matrix_array, 0.0)
    return _coarse_patch_variation(masked, n_bins=4)


def _band_values(matrix: np.ndarray, min_separation: int, max_separation: int | None = None) -> np.ndarray:
    """Collect upper-triangle values within a range of genomic separations."""
    n_monomers = matrix.shape[0]
    collected = []
    upper = n_monomers if max_separation is None else min(max_separation + 1, n_monomers)
    for separation in range(min_separation, upper):
        diagonal = np.diag(matrix, k=separation)
        if diagonal.size > 0:
            collected.append(diagonal)
    if not collected:
        return np.empty(0, dtype=float)
    return np.concatenate(collected, axis=0).astype(float)


def _contact_decay_slope(matrix: np.ndarray) -> float:
    """Fit a simple log-log slope for contact probability vs genomic distance."""
    separations = []
    band_means = []

    for separation in range(1, matrix.shape[0]):
        diagonal = np.diag(matrix, k=separation)
        if diagonal.size == 0:
            continue
        separations.append(float(separation))
        band_means.append(float(diagonal.mean()) + 1e-4)

    if len(separations) < 2:
        return 0.0
    slope, _ = np.polyfit(np.log(separations), np.log(band_means), deg=1)
    return float(slope)


def _band_profile_entropy(matrix: np.ndarray) -> float:
    """Measure how concentrated contacts are across genomic separations."""
    band_profile = np.asarray([np.diag(matrix, k=separation).mean() for separation in range(1, matrix.shape[0])])
    total = float(band_profile.sum())
    if total <= 1e-8:
        return 0.0
    probabilities = band_profile / total
    nonzero = probabilities > 0.0
    entropy = -np.sum(probabilities[nonzero] * np.log(probabilities[nonzero]))
    return float(entropy / np.log(len(probabilities)))


def extract_image_features(images: Sequence[np.ndarray]) -> tuple[np.ndarray, list[str]]:
    """Extract a small set of interpretable optical-space features."""
    feature_rows = []
    for image in images:
        metrics = image_metrics(np.asarray(image, dtype=float))
        feature_rows.append([metrics[name] for name in IMAGE_FEATURE_NAMES])
    return np.asarray(feature_rows, dtype=np.float32), list(IMAGE_FEATURE_NAMES)


def extract_genomic_features(contact_matrices: Sequence[np.ndarray]) -> tuple[np.ndarray, list[str]]:
    """Extract a small set of interpretable genomic/contact-map features."""
    feature_rows = []

    for matrix in contact_matrices:
        matrix_array = np.asarray(matrix, dtype=float)
        n_monomers = matrix_array.shape[0]
        near_values = _band_values(matrix_array, min_separation=1, max_separation=max(4, n_monomers // 12))
        far_min_separation = max(8, n_monomers // 5)
        far_values = _band_values(matrix_array, min_separation=far_min_separation)
        upper_triangle = matrix_array[np.triu_indices_from(matrix_array, k=1)]
        row_sums = matrix_array.sum(axis=1)

        mean_contact = float(upper_triangle.mean()) if upper_triangle.size > 0 else 0.0
        near_contact = float(near_values.mean()) if near_values.size > 0 else 0.0
        far_contact = float(far_values.mean()) if far_values.size > 0 else 0.0

        feature_rows.append(
            [
                mean_contact,
                far_contact,
                _safe_ratio(near_contact, far_contact),
                _contact_decay_slope(matrix_array),
                float(row_sums.std(ddof=0) / (row_sums.mean() + 1e-8)),
                _band_profile_entropy(matrix_array),
                _distal_patch_variation(matrix_array, min_separation=far_min_separation),
                float(far_values.std(ddof=0) / (far_contact + 1e-8)) if far_values.size > 0 else 0.0,
            ]
        )

    return np.asarray(feature_rows, dtype=np.float32), list(GENOMIC_FEATURE_NAMES)


def _standardize_feature_matrix(
    feature_matrix: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
) -> np.ndarray:
    """Apply train-split z-scoring with safe variance handling."""
    return (feature_matrix - mean) / std


def _build_feature_split(
    feature_matrix: np.ndarray,
    labels: np.ndarray,
    indices: np.ndarray,
) -> Dict[str, torch.Tensor]:
    """Convert one feature split into tensors."""
    return {
        "feature": torch.as_tensor(feature_matrix, dtype=torch.float32),
        "label": torch.as_tensor(labels[indices], dtype=torch.long),
        "index": torch.as_tensor(indices.astype(np.int64), dtype=torch.long),
    }


def build_feature_baseline_dataset(dataset: Dict[str, object]) -> Dict[str, object]:
    """Create train/val/test feature splits for each baseline modality."""
    arrays = dataset["arrays"]
    split_indices = dataset["split_indices"]
    labels = np.asarray(arrays["label"], dtype=np.int64)

    image_feature_array, image_feature_names = extract_image_features(arrays["image"])
    genomic_feature_array, genomic_feature_names = extract_genomic_features(arrays["genomic_matrix"])
    fusion_feature_array = np.concatenate((image_feature_array, genomic_feature_array), axis=1)

    feature_arrays = {
        "image_feature": image_feature_array,
        "genomic_feature": genomic_feature_array,
        "fusion_feature": fusion_feature_array,
    }
    feature_names = {
        "image_feature": image_feature_names,
        "genomic_feature": genomic_feature_names,
        "fusion_feature": image_feature_names + genomic_feature_names,
    }

    split_tensors: Dict[str, Dict[str, Dict[str, torch.Tensor]]] = {}
    for baseline_name, feature_matrix in feature_arrays.items():
        train_indices = split_indices["train"]
        train_mean = feature_matrix[train_indices].mean(axis=0, keepdims=True)
        train_std = feature_matrix[train_indices].std(axis=0, keepdims=True)
        train_std = np.where(train_std < 1e-6, 1.0, train_std)

        standardized = _standardize_feature_matrix(feature_matrix, train_mean, train_std).astype(np.float32)
        split_tensors[baseline_name] = {
            split_name: _build_feature_split(standardized[indices], labels, indices)
            for split_name, indices in split_indices.items()
        }

    return {
        "feature_arrays": feature_arrays,
        "feature_names": feature_names,
        "splits": split_tensors,
    }
