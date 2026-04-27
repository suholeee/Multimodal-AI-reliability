"""Helpers for optical-image normalization and summary features."""

from __future__ import annotations

from typing import Dict

import numpy as np
import torch

from ml_feature_baselines import extract_image_features


IMAGE_NORMALIZATION_MODES = ("none", "center", "standardize")


def normalize_image_batch(images: torch.Tensor, mode: str = "none") -> torch.Tensor:
    """Apply light per-sample normalization to grayscale image batches."""
    if mode == "none":
        return images
    if mode not in IMAGE_NORMALIZATION_MODES:
        raise ValueError(f"Unknown image normalization mode: {mode!r}")

    mean = images.mean(dim=(-2, -1), keepdim=True)
    centered = images - mean
    if mode == "center":
        return centered

    std = images.std(dim=(-2, -1), keepdim=True, unbiased=False)
    return centered / (std + 1e-6)


def build_image_feature_splits(dataset: Dict[str, object]) -> Dict[str, object]:
    """Extract and train-standardize optical summary features for each split."""
    arrays = dataset["arrays"]
    split_indices = dataset["split_indices"]
    image_feature_array, feature_names = extract_image_features(arrays["image"])

    train_indices = split_indices["train"]
    train_mean = image_feature_array[train_indices].mean(axis=0, keepdims=True)
    train_std = image_feature_array[train_indices].std(axis=0, keepdims=True)
    train_std = np.where(train_std < 1e-6, 1.0, train_std)
    standardized_array = ((image_feature_array - train_mean) / train_std).astype(np.float32)

    split_tensors: Dict[str, Dict[str, torch.Tensor]] = {}
    for split_name, indices in split_indices.items():
        split_tensors[split_name] = {
            "image_feature": torch.as_tensor(standardized_array[indices], dtype=torch.float32)
        }

    return {
        "feature_array": image_feature_array.astype(np.float32),
        "standardized_array": standardized_array,
        "feature_names": feature_names,
        "split_tensors": split_tensors,
        "train_mean": train_mean.astype(np.float32),
        "train_std": train_std.astype(np.float32),
    }


def attach_image_features(dataset: Dict[str, object]) -> Dict[str, object]:
    """Return a shallow dataset copy with standardized image features in each split."""
    feature_bundle = build_image_feature_splits(dataset)
    augmented_splits: Dict[str, Dict[str, torch.Tensor]] = {}
    for split_name, split in dataset["splits"].items():
        augmented_split = dict(split)
        augmented_split["image_feature"] = feature_bundle["split_tensors"][split_name]["image_feature"]
        augmented_splits[split_name] = augmented_split

    augmented_arrays = dict(dataset["arrays"])
    augmented_arrays["image_feature"] = feature_bundle["feature_array"]
    augmented_arrays["image_feature_standardized"] = feature_bundle["standardized_array"]

    augmented_dataset = dict(dataset)
    augmented_dataset["splits"] = augmented_splits
    augmented_dataset["arrays"] = augmented_arrays
    augmented_dataset["image_feature_names"] = list(feature_bundle["feature_names"])
    augmented_dataset["image_feature_stats"] = {
        "train_mean": feature_bundle["train_mean"],
        "train_std": feature_bundle["train_std"],
    }
    return augmented_dataset
