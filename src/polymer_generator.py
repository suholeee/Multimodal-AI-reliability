"""Standalone latent polymer generator for stylized chromatin ensembles."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class LatentParameters:
    """Interpretable latent state and derived polymer parameters."""

    shared_severity: float
    discordance: float
    image_pathology: float
    genomic_pathology: float
    label_probability: float

    compaction: float
    long_range_mixing: float
    heterogeneity: float
    local_texture: float
    distal_bridge_strength: float
    bridge_balance: float

    def as_dict(self) -> Dict[str, float]:
        """Return the parameters as a plain dictionary."""
        return {
            "shared_severity": float(self.shared_severity),
            "discordance": float(self.discordance),
            "image_pathology": float(self.image_pathology),
            "genomic_pathology": float(self.genomic_pathology),
            "label_probability": float(self.label_probability),
            "compaction": float(self.compaction),
            "long_range_mixing": float(self.long_range_mixing),
            "heterogeneity": float(self.heterogeneity),
            "local_texture": float(self.local_texture),
            "distal_bridge_strength": float(self.distal_bridge_strength),
            "bridge_balance": float(self.bridge_balance),
        }


CONDITION_TO_LABEL = {"normal": 0, "cancer": 1}
LATENT_BOUNDS = (0.05, 0.95)
MAX_LATENT_ATTEMPTS = 2048
SEVERITY_DISTRIBUTIONS = {
    "normal": (-1.10, 0.55),
    "cancer": (1.10, 0.55),
}


def _coerce_rng(rng: Optional[np.random.Generator] = None) -> np.random.Generator:
    """Create a generator when one is not provided."""
    return rng if rng is not None else np.random.default_rng()


def _sigmoid(value: float) -> float:
    """Convert an unconstrained latent into a bounded level."""
    return float(1.0 / (1.0 + np.exp(-value)))


def _center_coordinates(coords: np.ndarray) -> np.ndarray:
    """Shift coordinates so the center of mass sits at the origin."""
    return coords - coords.mean(axis=0, keepdims=True)


def _mean_bond_length(coords: np.ndarray) -> float:
    """Compute the mean adjacent bond length."""
    if len(coords) < 2:
        return 1.0

    bond_lengths = np.linalg.norm(np.diff(coords, axis=0), axis=1)
    return float(np.clip(bond_lengths.mean(), 1e-8, None))


def _normalize_mean_bond_length(coords: np.ndarray, target: float = 1.0) -> np.ndarray:
    """Rescale coordinates so the mean adjacent bond length is stable."""
    centered = _center_coordinates(coords)
    if len(centered) < 2:
        return centered

    return centered * (target / _mean_bond_length(centered))


def _radius_of_gyration(coords: np.ndarray) -> float:
    """Compute radius of gyration for internal normalization."""
    centered = _center_coordinates(coords)
    squared_radius = np.sum(centered**2, axis=1)
    return float(np.sqrt(np.mean(squared_radius)))


def _normalize_radius_of_gyration(coords: np.ndarray, target: float) -> np.ndarray:
    """Normalize global spread while preserving internal folding patterns."""
    centered = _center_coordinates(coords)
    current = _radius_of_gyration(centered)
    if current < 1e-8:
        return centered
    return centered * (target / current)


def _resample_along_contour(coords: np.ndarray) -> np.ndarray:
    """Re-parameterize the chain so bond lengths stay uniform after deformations."""
    if len(coords) < 3:
        return _center_coordinates(coords)

    segment_lengths = np.linalg.norm(np.diff(coords, axis=0), axis=1)
    cumulative_length = np.concatenate(([0.0], np.cumsum(segment_lengths)))
    total_length = float(cumulative_length[-1])
    if total_length < 1e-8:
        return _center_coordinates(coords)

    target_positions = np.linspace(0.0, total_length, len(coords))
    resampled_x = np.interp(target_positions, cumulative_length, coords[:, 0])
    resampled_y = np.interp(target_positions, cumulative_length, coords[:, 1])
    return _center_coordinates(np.column_stack((resampled_x, resampled_y)))


def _initialize_random_walk(
    n_monomers: int,
    rng: np.random.Generator,
    turning_noise: float = 0.55,
) -> np.ndarray:
    """Create a smooth random-walk-like backbone."""
    angles = np.empty(n_monomers - 1, dtype=float)
    angles[0] = rng.uniform(0.0, 2.0 * np.pi)

    for step_idx in range(1, n_monomers - 1):
        angles[step_idx] = angles[step_idx - 1] + rng.normal(scale=turning_noise)

    step_lengths = np.clip(rng.normal(loc=1.0, scale=0.08, size=n_monomers - 1), 0.75, 1.25)
    steps = np.column_stack((np.cos(angles), np.sin(angles))) * step_lengths[:, None]
    coords = np.vstack((np.zeros((1, 2), dtype=float), np.cumsum(steps, axis=0)))
    coords = _normalize_mean_bond_length(coords)
    return _resample_along_contour(coords)


def _smooth_polymer(coords: np.ndarray, weight: float = 0.30, n_steps: int = 1) -> np.ndarray:
    """Apply a light smoothing pass so the chain remains visually polymer-like."""
    smoothed = coords.copy()

    for _ in range(n_steps):
        neighbor_average = smoothed.copy()
        neighbor_average[1:-1] = (
            0.25 * smoothed[:-2] + 0.50 * smoothed[1:-1] + 0.25 * smoothed[2:]
        )
        smoothed = (1.0 - weight) * smoothed + weight * neighbor_average

    return smoothed


def _apply_local_compaction(
    coords: np.ndarray,
    compaction: float,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
    """Apply mild region-specific compaction rather than global collapse."""
    n_monomers = len(coords)
    indices = np.arange(n_monomers)
    updated = coords.copy()
    compacted_regions: List[Tuple[int, int]] = []

    n_regions = 1 + int(np.round(3.2 * compaction))
    min_width = max(4, n_monomers // 18)
    max_width = max(min_width + 1, n_monomers // 10)

    for _ in range(n_regions):
        center_idx = int(rng.integers(0, n_monomers))
        half_width = int(rng.integers(min_width, max_width + 1))
        sigma = max(1.8, half_width / 1.6)
        weights = np.exp(-0.5 * ((indices - center_idx) / sigma) ** 2)[:, None]

        local_center = np.average(updated, axis=0, weights=weights[:, 0])
        target = 0.70 * local_center + 0.30 * updated.mean(axis=0)
        strength = 0.05 + 0.10 * compaction * rng.uniform(0.8, 1.2)
        updated += weights * strength * (target - updated)
        compacted_regions.append((center_idx, half_width))

    return updated, compacted_regions


def _apply_local_texture_domains(
    coords: np.ndarray,
    local_texture: float,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, List[Dict[str, int | str]]]:
    """Create local patchiness that mostly affects image-visible texture."""
    n_monomers = len(coords)
    indices = np.arange(n_monomers)
    updated = coords.copy()
    regions: List[Dict[str, int | str]] = []

    region_scale = max(1.0, n_monomers / 64.0)
    n_regions = int(np.round(region_scale * (2.2 + 8.4 * local_texture)))
    min_width = max(3, n_monomers // 36)
    max_width = max(min_width + 1, n_monomers // 16)

    for _ in range(n_regions):
        center_idx = int(rng.integers(0, n_monomers))
        half_width = int(rng.integers(min_width, max_width + 1))
        sigma = max(1.5, half_width / 1.7)
        weights = np.exp(-0.5 * ((indices - center_idx) / sigma) ** 2)[:, None]

        local_center = np.average(updated, axis=0, weights=weights[:, 0])
        condense_region = rng.random() < (0.58 + 0.22 * local_texture)
        strength = 0.07 + 0.20 * local_texture * rng.uniform(0.8, 1.25)

        if condense_region:
            updated += weights * strength * (local_center - updated)
            region_mode = "condense"
        else:
            updated += weights * 0.75 * strength * (updated - local_center)
            region_mode = "loosen"

        flank_offset = int(rng.integers(max(2, half_width), max(3, 2 * half_width + 1)))
        flank_idx = int(np.clip(center_idx + rng.choice((-flank_offset, flank_offset)), 0, n_monomers - 1))
        flank_weights = np.exp(-0.5 * ((indices - flank_idx) / (sigma + 0.6)) ** 2)[:, None]
        flank_center = np.average(updated, axis=0, weights=flank_weights[:, 0])
        updated += flank_weights * (0.05 + 0.10 * local_texture) * (updated - flank_center)

        left_idx = max(center_idx - 1, 0)
        right_idx = min(center_idx + 1, n_monomers - 1)
        tangent = updated[right_idx] - updated[left_idx]
        tangent /= np.linalg.norm(tangent) + 1e-8
        normal = np.array([-tangent[1], tangent[0]])
        normal_noise = rng.normal(scale=0.04 + 0.11 * local_texture, size=(n_monomers, 1))
        tangent_noise = rng.normal(scale=0.02 + 0.06 * local_texture, size=(n_monomers, 1))
        updated += weights * (0.65 * normal_noise * normal + 0.35 * tangent_noise * tangent)

        regions.append(
            {
                "center_idx": center_idx,
                "half_width": half_width,
                "flank_idx": flank_idx,
                "mode": region_mode,
            }
        )

    return updated, regions


def _apply_long_range_attractions(
    coords: np.ndarray,
    long_range_mixing: float,
    rng: np.random.Generator,
    distal_bridge_strength: float = 0.5,
) -> Tuple[np.ndarray, List[Tuple[int, int, int]]]:
    """Create contacts between distant neighborhoods without global collapse."""
    n_monomers = len(coords)
    min_gap = max(12, n_monomers // 4)
    sigma = 1.1 + 1.5 * long_range_mixing + 1.1 * distal_bridge_strength
    length_scale = max(1.0, n_monomers / 64.0)
    n_pairs = int(
        np.round(length_scale * (1.8 + 5.4 * long_range_mixing + 3.6 * distal_bridge_strength))
    )
    if rng.random() < long_range_mixing + 0.30 * distal_bridge_strength:
        n_pairs += 1

    indices = np.arange(n_monomers)
    updated = coords.copy()
    selected_pairs: List[Tuple[int, int, int]] = []

    for _ in range(n_pairs):
        left_idx = int(rng.integers(0, n_monomers - min_gap))
        max_gap = n_monomers - left_idx - 1
        relative_gap = rng.power(1.2 + 1.8 * long_range_mixing + 2.2 * distal_bridge_strength)
        gap = min_gap + int(relative_gap * max(max_gap - min_gap, 1))
        right_idx = min(left_idx + gap, n_monomers - 1)

        left_weight = np.exp(-0.5 * ((indices - left_idx) / sigma) ** 2)
        right_weight = np.exp(-0.5 * ((indices - right_idx) / sigma) ** 2)
        separation_vector = updated[right_idx] - updated[left_idx]
        midpoint = 0.5 * (updated[left_idx] + updated[right_idx])

        attraction_strength = (
            0.12
            + 0.20 * long_range_mixing
            + 0.22 * distal_bridge_strength * rng.uniform(0.9, 1.2)
        )
        updated += left_weight[:, None] * 0.45 * attraction_strength * separation_vector
        updated -= right_weight[:, None] * 0.45 * attraction_strength * separation_vector

        bridge_weight = np.clip(left_weight + right_weight, 0.0, 1.0)[:, None]
        updated += bridge_weight * (
            0.07 + 0.10 * long_range_mixing + 0.13 * distal_bridge_strength
        ) * (midpoint - updated)
        selected_pairs.append((left_idx, right_idx, right_idx - left_idx))

    return updated, selected_pairs


def _choose_anchor_index(
    center_idx: int,
    half_width: int,
    n_monomers: int,
    rng: np.random.Generator,
) -> int:
    """Pick a distant contour location for local condensation."""
    candidate_indices = np.arange(n_monomers)
    mask = np.abs(candidate_indices - center_idx) > half_width
    valid_candidates = candidate_indices[mask]
    if len(valid_candidates) == 0:
        return int(rng.integers(0, n_monomers))
    return int(rng.choice(valid_candidates))


def _apply_multiscale_heterogeneity(
    coords: np.ndarray,
    heterogeneity: float,
    rng: np.random.Generator,
    bridge_balance: float = 0.5,
    texture_emphasis: Optional[float] = None,
) -> Tuple[np.ndarray, List[Dict[str, object]]]:
    """Create patchy multi-scale density with local clusters and looser regions."""
    n_monomers = len(coords)
    indices = np.arange(n_monomers)
    updated = coords.copy()
    regions: List[Dict[str, object]] = []
    texture_level = heterogeneity if texture_emphasis is None else float(texture_emphasis)

    region_scale = max(1.0, n_monomers / 64.0)
    scale_specs = (
        (
            "local",
            int(np.round(region_scale * (3.4 + 4.5 * heterogeneity + 2.4 * texture_level))),
            max(3, n_monomers // 28),
            max(6, n_monomers // 14),
        ),
        (
            "medium",
            int(np.round(region_scale * (1.0 + 2.4 * heterogeneity + 1.8 * bridge_balance))),
            max(6, n_monomers // 14),
            max(12, n_monomers // 8),
        ),
    )

    for scale_name, n_regions, min_width, max_width in scale_specs:
        for _ in range(n_regions):
            center_idx = int(rng.integers(0, n_monomers))
            half_width = int(rng.integers(min_width, max_width + 1))
            sigma = max(1.8, half_width / 1.8)
            weights = np.exp(-0.5 * ((indices - center_idx) / sigma) ** 2)[:, None]

            local_center = np.average(updated, axis=0, weights=weights[:, 0])
            anchor_idx = _choose_anchor_index(center_idx, half_width, n_monomers, rng)
            anchor_point = updated[anchor_idx]
            condense_region = rng.random() < (0.54 + 0.24 * heterogeneity)

            if condense_region:
                anchor_blend = 0.03 + 0.17 * bridge_balance
                if scale_name == "medium":
                    anchor_blend += 0.08 * bridge_balance
                target = (1.0 - anchor_blend) * local_center + anchor_blend * anchor_point
                strength = 0.08 + 0.18 * heterogeneity * rng.uniform(0.8, 1.25)
                updated += weights * strength * (target - updated)
                region_mode = "condense"
            else:
                strength = 0.05 + 0.08 * texture_level * rng.uniform(0.8, 1.2)
                updated += weights * strength * (updated - local_center)
                region_mode = "loosen"

            direction = rng.normal(size=2)
            direction /= np.linalg.norm(direction) + 1e-8
            perpendicular = np.array([-direction[1], direction[0]])
            axial_noise = rng.normal(scale=0.05 + 0.11 * texture_level, size=(n_monomers, 1))
            transverse_noise = rng.normal(scale=0.03 + 0.07 * texture_level, size=(n_monomers, 1))
            updated += weights * (0.65 * axial_noise * direction + 0.35 * transverse_noise * perpendicular)

            regions.append(
                {
                    "scale": scale_name,
                    "center_idx": center_idx,
                    "anchor_idx": anchor_idx,
                    "half_width": half_width,
                    "mode": region_mode,
                }
            )

    return updated, regions


def _target_radius_of_gyration(
    n_monomers: int,
    compaction: float,
    rng: np.random.Generator,
) -> float:
    """Choose a modest condition-dependent global size target."""
    base_radius = 0.97 * np.sqrt(n_monomers)
    jitter = rng.normal(scale=0.18)
    target = base_radius - 0.38 * compaction + jitter
    return float(max(0.55 * np.sqrt(n_monomers), target))


def sample_latent_parameters(
    condition: str,
    rng: Optional[np.random.Generator] = None,
) -> LatentParameters:
    """Sample one coherent shared latent state conditioned on the final class."""
    if condition not in CONDITION_TO_LABEL or condition not in SEVERITY_DISTRIBUTIONS:
        raise ValueError(f"Unknown condition: {condition!r}")

    generator = _coerce_rng(rng)
    severity_mean, severity_std = SEVERITY_DISTRIBUTIONS[condition]
    shared_severity = float(np.clip(generator.normal(loc=severity_mean, scale=severity_std), -2.5, 2.5))
    discordance = float(
        np.clip(
            generator.normal(loc=0.0, scale=0.12)
            + generator.normal(loc=0.0, scale=1.75) * float(generator.random() < 0.16),
            -2.5,
            2.5,
        )
    )
    label_probability = _sigmoid(1.75 * shared_severity)

    image_pathology = shared_severity + 1.55 * discordance
    genomic_pathology = shared_severity - 1.55 * discordance

    shared_level = _sigmoid(1.10 * shared_severity)
    image_level = _sigmoid(1.00 * image_pathology)
    genomic_level = _sigmoid(1.00 * genomic_pathology)

    compaction = float(
        np.clip(0.23 + 0.24 * shared_level + 0.03 * (image_level + genomic_level - 1.0), *LATENT_BOUNDS)
    )
    long_range_mixing = float(np.clip(0.08 + 0.30 * shared_level + 0.50 * genomic_level, *LATENT_BOUNDS))
    heterogeneity = float(np.clip(0.08 + 0.32 * shared_level + 0.60 * image_level, *LATENT_BOUNDS))
    local_texture = float(np.clip(0.05 + 0.14 * shared_level + 0.82 * image_level, *LATENT_BOUNDS))
    distal_bridge_strength = float(
        np.clip(0.04 + 0.12 * shared_level + 0.70 * genomic_level, *LATENT_BOUNDS)
    )
    bridge_balance = float(np.clip(0.06 + 0.78 * genomic_level - 0.22 * image_level, *LATENT_BOUNDS))

    return LatentParameters(
        shared_severity=shared_severity,
        discordance=discordance,
        image_pathology=image_pathology,
        genomic_pathology=genomic_pathology,
        label_probability=label_probability,
        compaction=compaction,
        long_range_mixing=long_range_mixing,
        heterogeneity=heterogeneity,
        local_texture=local_texture,
        distal_bridge_strength=distal_bridge_strength,
        bridge_balance=bridge_balance,
    )


def generate_polymer(
    n_monomers: int = 96,
    condition: str = "normal",
    latent: Optional[LatentParameters] = None,
    rng: Optional[np.random.Generator] = None,
    debug: bool = False,
):
    """Generate one stylized 2D polymer conformation."""
    if n_monomers < 8:
        raise ValueError("n_monomers must be at least 8 for meaningful long-range structure.")

    generator = _coerce_rng(rng)
    latent_parameters = latent if latent is not None else sample_latent_parameters(condition, generator)

    coords = _initialize_random_walk(n_monomers, generator)
    coords, compacted_regions = _apply_local_compaction(coords, latent_parameters.compaction, generator)
    coords = _smooth_polymer(coords, weight=0.10 + 0.05 * latent_parameters.compaction, n_steps=1)
    coords = _resample_along_contour(coords)

    coords, texture_regions = _apply_local_texture_domains(
        coords,
        latent_parameters.local_texture,
        generator,
    )
    coords = _smooth_polymer(coords, weight=0.08, n_steps=1)
    coords = _resample_along_contour(coords)

    coords, long_range_pairs = _apply_long_range_attractions(
        coords,
        latent_parameters.long_range_mixing,
        generator,
        distal_bridge_strength=latent_parameters.distal_bridge_strength,
    )
    coords = _smooth_polymer(coords, weight=0.12, n_steps=1)
    coords = _resample_along_contour(coords)

    coords, heterogeneity_regions = _apply_multiscale_heterogeneity(
        coords,
        latent_parameters.heterogeneity,
        generator,
        bridge_balance=latent_parameters.bridge_balance,
        texture_emphasis=latent_parameters.local_texture,
    )
    coords = _smooth_polymer(coords, weight=0.10, n_steps=1)
    coords = _resample_along_contour(coords)
    coords = _normalize_radius_of_gyration(
        coords,
        target=_target_radius_of_gyration(n_monomers, latent_parameters.compaction, generator),
    )
    coords = _center_coordinates(coords)

    metadata = {
        "condition": condition,
        "latent": latent_parameters.as_dict(),
        "compacted_regions": compacted_regions,
        "texture_regions": texture_regions,
        "long_range_pairs": long_range_pairs,
        "heterogeneity_regions": heterogeneity_regions,
        "radius_of_gyration_target": _radius_of_gyration(coords),
    }
    if debug:
        return coords, metadata
    return coords


def generate_ensemble(
    n_polymers: int,
    condition: str,
    n_monomers: int = 96,
    seed: Optional[int] = None,
    return_metadata: bool = False,
):
    """Generate an ensemble of polymers for one condition."""
    generator = np.random.default_rng(seed)
    polymers: List[np.ndarray] = []
    metadata: List[Dict[str, object]] = []

    for _ in range(n_polymers):
        coords, info = generate_polymer(
            n_monomers=n_monomers,
            condition=condition,
            rng=generator,
            debug=True,
        )
        polymers.append(coords)
        metadata.append(info)

    if return_metadata:
        return polymers, metadata
    return polymers
