"""V2 contradiction-regime dataset builder for shared-latent polymer experiments."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch

from ml_dataset import (
    DEFAULT_RENDER_CONFIG,
    LATENT_ARRAY_NAMES,
    MODE_CONFIGS,
    build_genomic_representation,
    train_test_split_indices,
)
from polymer_metrics import observable_contact_matrix
from polymer_generator import LATENT_BOUNDS, LatentParameters, generate_polymer
from polymer_image_renderer import render_batch


@dataclass(frozen=True)
class ContradictionRegimeConfig:
    """Configuration for one V2 contradiction regime."""

    contradiction_strength: float = 0.0
    modality_asymmetry: float = 0.0
    structural_strong_threshold: float = 0.40


V2_RENDER_CONFIG_OVERRIDES = {
    # Preserve local packing cues a bit more than the default renderer so the
    # image branch does not lose most of the new local-pathology signal.
    "sigma": 0.70,
    "blur_sigma": 0.05,
    "fit_quantile": 0.84,
    "max_segment_step": 0.45,
    "crowding_strength": 2.35,
    "contrast_gamma": 1.00,
    "normalization_gain": 5.6,
}


def _coerce_rng(seed: Optional[int]) -> np.random.Generator:
    """Create a reproducible NumPy generator."""
    return np.random.default_rng(seed)


def _sigmoid(value: float) -> float:
    """Map an unconstrained latent scalar into a smooth bounded level."""
    return float(1.0 / (1.0 + np.exp(-value)))


def _logit(value: float) -> float:
    """Map a bounded level back into an unconstrained latent scalar."""
    clipped = float(np.clip(value, 1e-6, 1.0 - 1e-6))
    return float(np.log(clipped / (1.0 - clipped)))


def _positive_part(value: float) -> float:
    """Keep only the positive part of a signed scalar."""
    return float(max(value, 0.0))


def _validate_regime_config(config: ContradictionRegimeConfig) -> ContradictionRegimeConfig:
    """Validate contradiction controls before dataset generation."""
    if not 0.0 <= float(config.contradiction_strength) <= 1.0:
        raise ValueError("contradiction_strength must be in [0, 1]")
    if not -0.5 <= float(config.modality_asymmetry) <= 0.5:
        raise ValueError("modality_asymmetry must be in [-0.5, 0.5]")
    if not 0.0 <= float(config.structural_strong_threshold) <= 1.0:
        raise ValueError("structural_strong_threshold must be in [0, 1]")
    return config


def sample_v2_latent_parameters(
    regime: ContradictionRegimeConfig,
    rng: Optional[np.random.Generator] = None,
    return_metadata: bool = False,
):
    """Sample one shared-latent polymer state under the V2 contradiction control."""
    validated_regime = _validate_regime_config(regime)
    generator = rng if rng is not None else np.random.default_rng()

    contradiction_strength = float(validated_regime.contradiction_strength)
    modality_asymmetry = float(validated_regime.modality_asymmetry)

    shared_severity = float(np.clip(generator.normal(loc=0.0, scale=1.00), -2.5, 2.5))
    local_base = float(np.clip(0.85 * shared_severity + generator.normal(loc=0.0, scale=0.42), -2.5, 2.5))
    global_base = float(np.clip(0.80 * shared_severity + generator.normal(loc=0.0, scale=0.46), -2.5, 2.5))

    contradiction_tail = float(
        generator.normal(loc=0.0, scale=1.05) * float(generator.random() < (0.10 + 0.32 * contradiction_strength))
    )
    contradiction_axis = float(
        np.clip(
            generator.normal(loc=0.0, scale=0.12 + 1.15 * contradiction_strength)
            + (0.10 + 0.85 * contradiction_strength) * contradiction_tail,
            -2.5,
            2.5,
        )
    )

    lambda_c = 1.35 * contradiction_strength
    local_effective = float(np.clip(local_base + lambda_c * contradiction_axis, -2.5, 2.5))
    global_effective = float(np.clip(global_base - lambda_c * contradiction_axis, -2.5, 2.5))

    local_observation = float(np.clip((1.0 + 0.10 * modality_asymmetry) * local_effective, -2.5, 2.5))
    global_observation = float(np.clip((1.0 - 0.10 * modality_asymmetry) * global_effective, -2.5, 2.5))

    label_noise = float(generator.normal(loc=0.0, scale=0.18))
    label_score = float(1.00 * shared_severity + 0.80 * local_effective + 0.80 * global_effective + label_noise)
    label_probability = _sigmoid(1.45 * label_score)
    label = int(label_score >= 0.0)

    shared_level = _sigmoid(1.00 * shared_severity)
    average_level = _sigmoid(0.55 * (local_effective + global_effective))
    contradiction_level = _sigmoid(0.85 * abs(local_effective - global_effective))
    image_level = _sigmoid(1.80 * local_observation + 0.24 * shared_severity)
    genomic_level = _sigmoid(1.35 * global_observation + 0.24 * shared_severity)
    signed_observation_gap = float(local_observation - global_observation)
    image_observability_drive = _sigmoid(1.10 * signed_observation_gap)
    genomic_observability_drive = _sigmoid(-1.10 * signed_observation_gap)
    contradiction_observability_level = _sigmoid(0.95 * abs(signed_observation_gap))
    image_excess = _positive_part(image_level - genomic_level)
    genomic_excess = _positive_part(genomic_level - image_level)

    # Strengthen branch-specific observability of the contradiction axis without
    # changing the shared-latent label rule.
    compaction = float(
        np.clip(
            0.13
            + 0.24 * shared_level
            + 0.05 * average_level
            + 0.10 * image_level
            + 0.03 * image_observability_drive
            + 0.02 * contradiction_observability_level
            + 0.03 * image_excess,
            *LATENT_BOUNDS,
        )
    )
    heterogeneity = float(
        np.clip(
            0.05
            + 0.08 * shared_level
            + 0.50 * image_level
            + 0.05 * average_level
            + 0.10 * image_observability_drive
            + 0.06 * contradiction_observability_level
            + 0.06 * image_excess,
            *LATENT_BOUNDS,
        )
    )
    local_texture = float(
        np.clip(
            0.04
            + 0.06 * shared_level
            + 0.56 * image_level
            + 0.12 * image_observability_drive
            + 0.10 * contradiction_level
            + 0.08 * contradiction_observability_level
            + 0.06 * image_excess,
            *LATENT_BOUNDS,
        )
    )
    long_range_mixing = float(
        np.clip(
            0.07
            + 0.09 * shared_level
            + 0.05 * average_level
            + 0.24 * genomic_level
            + 0.10 * genomic_observability_drive
            + 0.04 * contradiction_observability_level
            + 0.05 * genomic_excess,
            *LATENT_BOUNDS,
        )
    )
    distal_bridge_strength = float(
        np.clip(
            0.05
            + 0.09 * shared_level
            + 0.28 * genomic_level
            + 0.12 * genomic_observability_drive
            + 0.10 * contradiction_level
            + 0.04 * contradiction_observability_level
            + 0.05 * genomic_excess,
            *LATENT_BOUNDS,
        )
    )
    bridge_balance = float(
        np.clip(
            0.10
            + 0.22 * genomic_level
            + 0.04 * shared_level
            - 0.01 * image_level
            + 0.10 * genomic_observability_drive
            + 0.04 * genomic_excess
            - 0.03 * image_observability_drive,
            *LATENT_BOUNDS,
        )
    )

    effective_discordance = float(local_effective - global_effective)
    image_pathology = float(local_effective)
    genomic_pathology = float(global_effective)

    latent = LatentParameters(
        shared_severity=shared_severity,
        discordance=effective_discordance,
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
    regime_metadata = {
        "contradiction_strength": contradiction_strength,
        "modality_asymmetry": modality_asymmetry,
        "label": label,
        "label_score": label_score,
        "label_noise": label_noise,
        "local_pathology_base": local_base,
        "global_pathology_base": global_base,
        "local_pathology_axis": local_effective,
        "global_pathology_axis": global_effective,
        "contradiction_axis": contradiction_axis,
        "local_observation_axis": local_observation,
        "global_observation_axis": global_observation,
        "lambda_c": lambda_c,
        "raw_discordance": contradiction_axis,
        "boundary_gate": 1.0,
        "image_observability_drive": image_observability_drive,
        "genomic_observability_drive": genomic_observability_drive,
        "contradiction_observability_level": contradiction_observability_level,
        "structural_contradiction_score": float(abs(image_level - genomic_level)),
        "strong_structural_contradiction": float(
            abs(image_level - genomic_level) >= validated_regime.structural_strong_threshold
        ),
    }

    if return_metadata:
        return latent, regime_metadata
    return latent


def generate_v2_ensemble(
    n_polymers_per_label: int,
    regime: ContradictionRegimeConfig,
    n_monomers: int = 96,
    seed: Optional[int] = None,
    return_metadata: bool = False,
):
    """Generate a balanced V2 polymer ensemble under one contradiction regime."""
    generator = np.random.default_rng(seed)
    polymers: List[np.ndarray] = []
    metadata: List[Dict[str, object]] = []
    labels: List[int] = []
    target_counts = {0: int(n_polymers_per_label), 1: int(n_polymers_per_label)}
    counts = {0: 0, 1: 0}
    max_attempts = max(200, 40 * int(n_polymers_per_label))
    attempts = 0

    while min(counts.values()) < int(n_polymers_per_label):
        attempts += 1
        if attempts > max_attempts:
            raise RuntimeError("Failed to build a balanced V2 dataset from the latent score.")

        latent, regime_metadata = sample_v2_latent_parameters(
            regime=regime,
            rng=generator,
            return_metadata=True,
        )
        label = int(regime_metadata["label"])
        if counts[label] >= target_counts[label]:
            continue

        condition = "cancer" if label == 1 else "normal"
        coords, info = generate_polymer(
            n_monomers=n_monomers,
            condition=condition,
            latent=latent,
            rng=generator,
            debug=True,
        )
        info["regime"] = regime_metadata
        polymers.append(coords)
        metadata.append(info)
        labels.append(label)
        counts[label] += 1

    if return_metadata:
        return polymers, metadata, np.asarray(labels, dtype=np.int64)
    return polymers


def _stack_v2_latent_arrays(metadata: Sequence[Dict[str, object]]) -> Dict[str, np.ndarray]:
    """Collect latent and regime diagnostics into aligned arrays."""
    latent_arrays = {}
    for latent_name in LATENT_ARRAY_NAMES:
        latent_arrays[f"latent_{latent_name}"] = np.asarray(
            [item["latent"][latent_name] for item in metadata],
            dtype=np.float32,
        )
    latent_arrays["latent_pathology_gap"] = (
        latent_arrays["latent_image_pathology"] - latent_arrays["latent_genomic_pathology"]
    ).astype(np.float32)

    regime_array_names = (
        "label",
        "label_score",
        "label_noise",
        "local_pathology_base",
        "global_pathology_base",
        "local_pathology_axis",
        "global_pathology_axis",
        "contradiction_axis",
        "local_observation_axis",
        "global_observation_axis",
        "lambda_c",
        "contradiction_strength",
        "modality_asymmetry",
        "raw_discordance",
        "boundary_gate",
        "image_observability_drive",
        "genomic_observability_drive",
        "contradiction_observability_level",
        "structural_contradiction_score",
        "strong_structural_contradiction",
    )
    for regime_name in regime_array_names:
        latent_arrays[f"latent_{regime_name}"] = np.asarray(
            [item["regime"][regime_name] for item in metadata],
            dtype=np.float32,
        )
    return latent_arrays


def _build_v2_contact_matrices(polymers: Sequence[np.ndarray]) -> np.ndarray:
    """Build a lightly distal-emphasized V2 contact-map representation."""
    contact_matrices = []
    for polymer in polymers:
        matrix = observable_contact_matrix(polymer).astype(np.float32)
        indices = np.arange(matrix.shape[0], dtype=np.float32)
        normalized_separation = np.abs(indices[:, None] - indices[None, :]) / max(matrix.shape[0] - 1, 1)
        distal_boost = 1.0 + 0.30 * normalized_separation**1.25
        boosted = matrix * distal_boost
        np.fill_diagonal(boosted, 0.0)
        contact_matrices.append(boosted.astype(np.float32))
    return np.stack(contact_matrices, axis=0)


def _to_tensor(array: np.ndarray, dtype: torch.dtype) -> torch.Tensor:
    """Convert a NumPy array into a torch tensor with a fixed dtype."""
    return torch.as_tensor(array, dtype=dtype)


def _build_split_tensors(
    images: np.ndarray,
    genomics: np.ndarray,
    labels: np.ndarray,
    split_indices: Dict[str, np.ndarray],
    genomic_representation: str,
) -> Dict[str, Dict[str, torch.Tensor]]:
    """Slice paired arrays into torch tensors for train/val/test splits."""
    splits: Dict[str, Dict[str, torch.Tensor]] = {}
    for split_name, indices in split_indices.items():
        image_array = images[indices].astype(np.float32)
        if image_array.ndim == 3:
            image_array = image_array[:, None, :, :]

        genomic_array = genomics[indices].astype(np.float32)
        if genomic_representation == "matrix":
            genomic_array = genomic_array[:, None, :, :]

        splits[split_name] = {
            "image": _to_tensor(image_array, dtype=torch.float32),
            "genomic": _to_tensor(genomic_array, dtype=torch.float32),
            "label": _to_tensor(labels[indices], dtype=torch.long),
            "index": _to_tensor(indices.astype(np.int64), dtype=torch.long),
        }
    return splits


def build_v2_paired_dataset(
    regime: ContradictionRegimeConfig,
    mode: str = "full",
    image_size: int = 32,
    genomic_representation: str = "matrix",
    train_fraction: float = 0.60,
    val_fraction: float = 0.20,
    seed: int = 7,
    render_config: Optional[Dict[str, object]] = None,
) -> Dict[str, object]:
    """Build a V2 paired dataset with explicit contradiction controls."""
    _validate_regime_config(regime)
    if mode not in MODE_CONFIGS:
        raise ValueError(f"Unknown mode: {mode!r}")

    mode_config = MODE_CONFIGS[mode]
    n_monomers = int(mode_config["n_monomers"])
    n_per_condition = int(mode_config["n_polymers_per_condition"])

    seed_generator = _coerce_rng(seed)
    normal_seed = int(seed_generator.integers(0, 1_000_000_000))
    cancer_seed = int(seed_generator.integers(0, 1_000_000_000))
    render_seed = int(seed_generator.integers(0, 1_000_000_000))
    split_seed = int(seed_generator.integers(0, 1_000_000_000))

    effective_render_config = dict(DEFAULT_RENDER_CONFIG)
    effective_render_config.update(V2_RENDER_CONFIG_OVERRIDES)
    if render_config is not None:
        effective_render_config.update(render_config)
    effective_render_config["image_size"] = image_size

    polymers, metadata, labels = generate_v2_ensemble(
        n_polymers_per_label=n_per_condition,
        n_monomers=n_monomers,
        regime=regime,
        seed=normal_seed ^ cancer_seed,
        return_metadata=True,
    )
    contact_matrices = _build_v2_contact_matrices(polymers)
    images = render_batch(polymers, seed=render_seed, **effective_render_config).astype(np.float32)

    if genomic_representation == "matrix":
        genomics = contact_matrices
    else:
        genomics = build_genomic_representation(polymers, representation=genomic_representation)

    conditions = np.where(labels == 1, "cancer", "normal")
    latent_arrays = _stack_v2_latent_arrays(metadata)

    split_indices = train_test_split_indices(
        labels,
        train_fraction=train_fraction,
        val_fraction=val_fraction,
        seed=split_seed,
    )
    tensor_splits = _build_split_tensors(
        images=images,
        genomics=genomics,
        labels=labels,
        split_indices=split_indices,
        genomic_representation=genomic_representation,
    )

    return {
        "mode": mode,
        "n_monomers": n_monomers,
        "image_size": image_size,
        "genomic_representation": genomic_representation,
        "render_config": effective_render_config,
        "regime": asdict(regime),
        "splits": tensor_splits,
        "arrays": {
            "image": images,
            "genomic": genomics,
            "genomic_matrix": contact_matrices,
            "label": labels,
            "condition": conditions,
            **latent_arrays,
        },
        "split_indices": split_indices,
    }
