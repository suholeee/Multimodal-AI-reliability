"""Minimal paired dataset builder for polymer-based multimodal classification."""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np
import torch

from polymer_generator import generate_ensemble
from polymer_image_renderer import render_batch
from polymer_metrics import observable_contact_matrix


MODE_CONFIGS = {
    "debug": {"n_monomers": 48, "n_polymers_per_condition": 30},
    "full": {"n_monomers": 96, "n_polymers_per_condition": 150},
    "safety": {"n_monomers": 96, "n_polymers_per_condition": 300},
}

DEFAULT_RENDER_CONFIG = {
    "image_size": 32,
    "sigma": 1.10,
    "margin": 0.10,
    "normalize": True,
    "add_noise": False,
    "blur_sigma": 0.55,
    "backbone_weight": 0.50,
    "fit_quantile": 0.88,
    "max_segment_step": 0.55,
    "crowding_radius_scale": 0.12,
    "crowding_strength": 1.40,
    "envelope_strength": 0.04,
    "noise_scale": 0.02,
    "contrast_gamma": 1.08,
    "normalization_mode": "mass",
    "normalization_gain": 6.0,
}

LATENT_ARRAY_NAMES = (
    "shared_severity",
    "discordance",
    "image_pathology",
    "genomic_pathology",
    "label_probability",
    "compaction",
    "long_range_mixing",
    "heterogeneity",
    "local_texture",
    "distal_bridge_strength",
    "bridge_balance",
)


def _coerce_rng(seed: Optional[int]) -> np.random.Generator:
    """Create a reproducible random number generator."""
    return np.random.default_rng(seed)


def _vectorize_contact_matrix(matrix: np.ndarray) -> np.ndarray:
    """Flatten the upper triangle of a contact matrix into a feature vector."""
    upper_triangle = np.triu_indices_from(matrix, k=1)
    return np.asarray(matrix[upper_triangle], dtype=np.float32)


def _stack_latent_arrays(metadata: Sequence[Dict[str, object]]) -> Dict[str, np.ndarray]:
    """Collect latent diagnostics from generator metadata into aligned arrays."""
    latent_arrays = {}
    for latent_name in LATENT_ARRAY_NAMES:
        latent_arrays[f"latent_{latent_name}"] = np.asarray(
            [item["latent"][latent_name] for item in metadata],
            dtype=np.float32,
        )
    latent_arrays["latent_pathology_gap"] = (
        latent_arrays["latent_image_pathology"] - latent_arrays["latent_genomic_pathology"]
    ).astype(np.float32)
    return latent_arrays


def build_genomic_representation(
    polymers: Sequence[np.ndarray],
    representation: str = "matrix",
) -> np.ndarray:
    """Convert polymers into Hi-C-like genomic features."""
    if representation not in {"matrix", "vector", "summary"}:
        raise ValueError("representation must be 'matrix', 'vector', or 'summary'")

    contacts = [observable_contact_matrix(polymer).astype(np.float32) for polymer in polymers]
    if representation == "matrix":
        return np.stack(contacts, axis=0)
    if representation == "summary":
        from ml_feature_baselines import extract_genomic_features

        summary_features, _ = extract_genomic_features(contacts)
        return summary_features.astype(np.float32)
    vectors = [_vectorize_contact_matrix(matrix) for matrix in contacts]
    return np.stack(vectors, axis=0)


def _split_class_indices(
    class_indices: np.ndarray,
    rng: np.random.Generator,
    train_fraction: float,
    val_fraction: float,
) -> Dict[str, np.ndarray]:
    """Split one class into train/val/test partitions."""
    shuffled = np.array(class_indices, copy=True)
    rng.shuffle(shuffled)

    n_samples = len(shuffled)
    n_train = int(np.floor(train_fraction * n_samples))
    n_val = int(np.floor(val_fraction * n_samples))

    n_train = min(max(n_train, 1), max(n_samples - 2, 1))
    n_val = min(max(n_val, 1), max(n_samples - n_train - 1, 1))

    return {
        "train": shuffled[:n_train],
        "val": shuffled[n_train : n_train + n_val],
        "test": shuffled[n_train + n_val :],
    }


def train_test_split_indices(
    labels: Sequence[int],
    train_fraction: float = 0.60,
    val_fraction: float = 0.20,
    seed: int = 0,
) -> Dict[str, np.ndarray]:
    """Create a reproducible stratified split for binary labels."""
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must be in (0, 1)")
    if not 0.0 <= val_fraction < 1.0:
        raise ValueError("val_fraction must be in [0, 1)")
    if train_fraction + val_fraction >= 1.0:
        raise ValueError("train_fraction + val_fraction must be smaller than 1")

    labels_array = np.asarray(labels, dtype=int)
    rng = _coerce_rng(seed)
    split_indices = {"train": [], "val": [], "test": []}

    for class_value in np.unique(labels_array):
        class_indices = np.flatnonzero(labels_array == class_value)
        class_split = _split_class_indices(class_indices, rng, train_fraction, val_fraction)
        for split_name, indices in class_split.items():
            split_indices[split_name].append(indices)

    for split_name, index_groups in split_indices.items():
        merged = np.concatenate(index_groups, axis=0)
        rng.shuffle(merged)
        split_indices[split_name] = merged

    return split_indices


def _to_tensor(array: np.ndarray, dtype: torch.dtype) -> torch.Tensor:
    """Convert a numpy array into a torch tensor with an explicit dtype."""
    return torch.as_tensor(array, dtype=dtype)


def _build_split_tensors(
    images: np.ndarray,
    genomics: np.ndarray,
    labels: np.ndarray,
    split_indices: Dict[str, np.ndarray],
    genomic_representation: str,
) -> Dict[str, Dict[str, torch.Tensor]]:
    """Slice paired arrays into torch tensors for each split."""
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


def build_paired_dataset(
    mode: str = "debug",
    image_size: int = 32,
    genomic_representation: str = "matrix",
    train_fraction: float = 0.60,
    val_fraction: float = 0.20,
    seed: int = 7,
    render_config: Optional[Dict[str, object]] = None,
) -> Dict[str, object]:
    """Generate paired optical/genomic samples and stratified splits."""
    if mode not in MODE_CONFIGS:
        raise ValueError(f"Unknown mode: {mode!r}")

    mode_config = MODE_CONFIGS[mode]
    n_monomers = mode_config["n_monomers"]
    n_per_condition = mode_config["n_polymers_per_condition"]

    seed_generator = _coerce_rng(seed)
    normal_seed = int(seed_generator.integers(0, 1_000_000_000))
    cancer_seed = int(seed_generator.integers(0, 1_000_000_000))
    render_seed = int(seed_generator.integers(0, 1_000_000_000))
    split_seed = int(seed_generator.integers(0, 1_000_000_000))

    effective_render_config = dict(DEFAULT_RENDER_CONFIG)
    if render_config is not None:
        effective_render_config.update(render_config)
    effective_render_config["image_size"] = image_size

    normal_polymers, normal_metadata = generate_ensemble(
        n_polymers=n_per_condition,
        condition="normal",
        n_monomers=n_monomers,
        seed=normal_seed,
        return_metadata=True,
    )
    cancer_polymers, cancer_metadata = generate_ensemble(
        n_polymers=n_per_condition,
        condition="cancer",
        n_monomers=n_monomers,
        seed=cancer_seed,
        return_metadata=True,
    )

    normal_contact_matrices = build_genomic_representation(
        normal_polymers,
        representation="matrix",
    )
    cancer_contact_matrices = build_genomic_representation(
        cancer_polymers,
        representation="matrix",
    )

    normal_images = render_batch(normal_polymers, seed=render_seed, **effective_render_config).astype(
        np.float32
    )
    cancer_images = render_batch(
        cancer_polymers,
        seed=render_seed + 1,
        **effective_render_config,
    ).astype(np.float32)

    if genomic_representation == "matrix":
        normal_genomics = normal_contact_matrices
        cancer_genomics = cancer_contact_matrices
    else:
        normal_genomics = build_genomic_representation(
            normal_polymers,
            representation=genomic_representation,
        )
        cancer_genomics = build_genomic_representation(
            cancer_polymers,
            representation=genomic_representation,
        )

    images = np.concatenate((normal_images, cancer_images), axis=0)
    genomics = np.concatenate((normal_genomics, cancer_genomics), axis=0)
    contact_matrices = np.concatenate((normal_contact_matrices, cancer_contact_matrices), axis=0)
    labels = np.concatenate(
        (
            np.zeros(n_per_condition, dtype=np.int64),
            np.ones(n_per_condition, dtype=np.int64),
        ),
        axis=0,
    )
    conditions = np.asarray(["normal"] * n_per_condition + ["cancer"] * n_per_condition)
    metadata = list(normal_metadata) + list(cancer_metadata)
    latent_arrays = _stack_latent_arrays(metadata)

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
