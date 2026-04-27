"""Simple sanity-check metrics for stylized polymer conformations."""

from __future__ import annotations

from typing import Dict, Iterable, Sequence

import numpy as np


DEFAULT_CONTACT_THRESHOLD_SCALE = 2.35
DEFAULT_MIN_CONTACT_SEPARATION = 8
DEFAULT_LOCAL_DENSITY_RADIUS_SCALE = 3.25
DEFAULT_SOFT_CONTACT_DISTANCE_SCALE = 2.20
DEFAULT_SOFT_CONTACT_EXPONENT = 1.60


def mean_bond_length(coords: np.ndarray) -> float:
    """Compute the mean adjacent bond length."""
    if len(coords) < 2:
        return 1.0
    bond_lengths = np.linalg.norm(np.diff(coords, axis=0), axis=1)
    return float(np.clip(bond_lengths.mean(), 1e-8, None))


def pairwise_distances(coords: np.ndarray) -> np.ndarray:
    """Compute all pairwise Euclidean distances."""
    deltas = coords[:, None, :] - coords[None, :, :]
    return np.linalg.norm(deltas, axis=-1)


def radius_of_gyration(coords: np.ndarray) -> float:
    """Measure global polymer spread around the center of mass."""
    centered = coords - coords.mean(axis=0, keepdims=True)
    squared_radius = np.sum(centered**2, axis=1)
    return float(np.sqrt(np.mean(squared_radius)))


def contact_matrix(
    coords: np.ndarray,
    threshold: float | None = None,
    threshold_scale: float = DEFAULT_CONTACT_THRESHOLD_SCALE,
) -> np.ndarray:
    """Convert distances to a binary contact matrix."""
    distances = pairwise_distances(coords)
    effective_threshold = threshold
    if effective_threshold is None:
        effective_threshold = threshold_scale * mean_bond_length(coords)
    contacts = (distances <= effective_threshold).astype(float)
    np.fill_diagonal(contacts, 0.0)
    return contacts


def soft_contact_matrix(
    coords: np.ndarray,
    distance_scale: float = DEFAULT_SOFT_CONTACT_DISTANCE_SCALE,
    exponent: float = DEFAULT_SOFT_CONTACT_EXPONENT,
) -> np.ndarray:
    """Convert distances to a smooth contact-intensity matrix."""
    distances = pairwise_distances(coords)
    effective_scale = max(distance_scale * mean_bond_length(coords), 1e-8)
    contacts = np.exp(-0.5 * (distances / effective_scale) ** exponent)
    np.fill_diagonal(contacts, 0.0)
    return contacts


def observable_contact_matrix(
    coords: np.ndarray,
    distance_scale: float = DEFAULT_SOFT_CONTACT_DISTANCE_SCALE,
    exponent: float = DEFAULT_SOFT_CONTACT_EXPONENT,
    separation_scale: float | None = None,
) -> np.ndarray:
    """Build a Hi-C-like observation matrix with reduced diagonal dominance.

    The matrix stays faithful to geometric proximity while softly downweighting
    very short genomic separations so long-range folding differences are easier
    to observe.
    """
    soft_contacts = soft_contact_matrix(
        coords,
        distance_scale=distance_scale,
        exponent=exponent,
    )
    indices = np.arange(len(coords))
    separation = np.abs(indices[:, None] - indices[None, :]).astype(float)
    effective_scale = float(separation_scale) if separation_scale is not None else max(4.0, len(coords) / 10.0)
    separation_weight = np.sqrt(separation / (separation + effective_scale + 1e-8))
    normalized_separation = separation / max(len(coords) - 1, 1)
    distal_emphasis = 0.75 + 0.75 * normalized_separation**0.85
    observed = soft_contacts * separation_weight * distal_emphasis
    np.fill_diagonal(observed, 0.0)
    return observed


def off_diagonal_contact_fraction(
    coords: np.ndarray,
    threshold: float | None = None,
    min_separation: int = DEFAULT_MIN_CONTACT_SEPARATION,
    threshold_scale: float = DEFAULT_CONTACT_THRESHOLD_SCALE,
) -> float:
    """Measure contacts between genomically distant monomers."""
    contacts = contact_matrix(coords, threshold=threshold, threshold_scale=threshold_scale)
    indices = np.arange(len(coords))
    separation = np.abs(indices[:, None] - indices[None, :])
    valid_mask = np.triu(separation >= min_separation, k=1)

    if not np.any(valid_mask):
        return 0.0
    return float(contacts[valid_mask].mean())


def box_counting_fractal_proxy(
    coords: np.ndarray,
    box_sizes: Sequence[int] = (2, 4, 8, 16, 32, 64),
) -> float:
    """Approximate geometric complexity with a lightweight box-counting slope."""
    # Interpolate along the backbone so the estimate reflects the curve itself,
    # not only the discrete monomer locations.
    interpolation_weights = np.linspace(0.0, 1.0, 8, endpoint=False)[:, None]
    dense_segments = [
        start + interpolation_weights * (end - start)
        for start, end in zip(coords[:-1], coords[1:])
    ]
    dense_coords = np.vstack(dense_segments + [coords[-1:]])

    shifted = dense_coords - dense_coords.min(axis=0, keepdims=True)
    max_span = float(np.clip(np.ptp(shifted, axis=0).max(), 1e-8, None))
    normalized = np.clip(shifted / max_span, 0.0, 1.0 - 1e-9)

    occupied_counts = []
    for n_boxes in box_sizes:
        box_indices = np.floor(normalized * n_boxes).astype(int)
        linear_index = box_indices[:, 0] * n_boxes + box_indices[:, 1]
        occupied_counts.append(max(np.unique(linear_index).size, 1))

    slope, _ = np.polyfit(np.log(box_sizes), np.log(occupied_counts), deg=1)
    return float(slope)


def local_density_variation(
    coords: np.ndarray,
    radius: float | None = None,
    min_separation: int = 2,
    radius_scale: float = DEFAULT_LOCAL_DENSITY_RADIUS_SCALE,
) -> float:
    """Quantify clumpiness by measuring variation in local neighborhood counts."""
    distances = pairwise_distances(coords)
    indices = np.arange(len(coords))
    separation = np.abs(indices[:, None] - indices[None, :])
    effective_radius = radius
    if effective_radius is None:
        effective_radius = radius_scale * mean_bond_length(coords)

    density_mask = (distances <= effective_radius) & (separation >= min_separation)
    local_counts = density_mask.sum(axis=1).astype(float)
    mean_count = float(local_counts.mean())

    if mean_count < 1e-8:
        return 0.0
    return float(local_counts.std(ddof=0) / mean_count)


def polymer_metrics(
    coords: np.ndarray,
    threshold: float | None = None,
    min_separation: int = DEFAULT_MIN_CONTACT_SEPARATION,
) -> Dict[str, float]:
    """Compute scalar summary metrics for a single polymer."""
    return {
        "radius_of_gyration": radius_of_gyration(coords),
        "off_diagonal_contact_fraction": off_diagonal_contact_fraction(
            coords,
            threshold=threshold,
            min_separation=min_separation,
        ),
        "box_counting_fractal_proxy": box_counting_fractal_proxy(coords),
        "local_density_variation": local_density_variation(coords),
    }


def stack_metric(metric_dicts: Iterable[Dict[str, float]], metric_name: str) -> np.ndarray:
    """Collect one metric across an ensemble."""
    return np.asarray([metric[metric_name] for metric in metric_dicts], dtype=float)
