"""Fixed non-leaky scientific tools for V3 agent benchmarks."""

from __future__ import annotations

import math
from collections import deque
from pathlib import Path
from typing import Dict, Mapping

import numpy as np
from PIL import Image

from v3_agent_dataset import V3AgentSample


SCIENTIFIC_TOOL_NAMES = {
    "measure_image_features",
    "measure_hic_features",
    "compare_modalities",
    "generate_feature_report",
}

SAFE_SCIENTIFIC_TOOL_KEYS = {
    "tool_name",
    "sample_id",
    "modality",
    "payload_type",
    "features",
    "image_features",
    "hic_features",
    "comparison",
    "notes",
}


def _round(value: float) -> float:
    if not np.isfinite(value):
        return float("nan")
    return float(round(float(value), 6))


def _load_gray(path: Path | str) -> np.ndarray:
    with Image.open(path) as image:
        values = np.asarray(image.convert("L"), dtype=np.float32) / 255.0
    if values.size == 0:
        raise ValueError(f"Empty image: {path}")
    return values


def _foreground_mask(values: np.ndarray) -> np.ndarray:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return np.zeros(values.shape, dtype=bool)
    p95 = float(np.percentile(finite, 95.0))
    p50 = float(np.percentile(finite, 50.0))
    threshold = max(0.08, p50 + 0.20 * (p95 - p50))
    mask = values > threshold
    if int(mask.sum()) < 4:
        threshold = max(0.02, float(np.percentile(finite, 85.0)))
        mask = values > threshold
    return mask


def _connected_components(mask: np.ndarray) -> int:
    height, width = mask.shape
    seen = np.zeros(mask.shape, dtype=bool)
    components = 0
    for y in range(height):
        for x in range(width):
            if not mask[y, x] or seen[y, x]:
                continue
            components += 1
            queue: deque[tuple[int, int]] = deque([(y, x)])
            seen[y, x] = True
            while queue:
                cy, cx = queue.popleft()
                for dy in (-1, 0, 1):
                    for dx in (-1, 0, 1):
                        if dy == 0 and dx == 0:
                            continue
                        ny, nx = cy + dy, cx + dx
                        if 0 <= ny < height and 0 <= nx < width and mask[ny, nx] and not seen[ny, nx]:
                            seen[ny, nx] = True
                            queue.append((ny, nx))
    return int(components)


def _entropy(values: np.ndarray, bins: int = 16) -> float:
    hist, _ = np.histogram(values[np.isfinite(values)], bins=bins, range=(0.0, 1.0))
    total = float(hist.sum())
    if total <= 0.0:
        return float("nan")
    probs = hist.astype(np.float64) / total
    probs = probs[probs > 0.0]
    return float(-(probs * np.log2(probs)).sum() / math.log2(bins))


def measure_image_features(sample: V3AgentSample) -> Dict[str, object]:
    """Measure polymer image morphology from public pixels only."""
    values = _load_gray(sample.image_path)
    mask = _foreground_mask(values)
    ys, xs = np.nonzero(mask)
    height, width = values.shape

    if len(xs) == 0:
        features = {
            "mean_intensity": _round(float(values.mean())),
            "std_intensity": _round(float(values.std())),
            "active_fraction": 0.0,
            "bbox_area_fraction": 0.0,
            "compactness": 0.0,
            "aspect_ratio": float("nan"),
            "radius_mean": float("nan"),
            "radius_std": float("nan"),
            "pca_anisotropy": float("nan"),
            "component_count": 0,
            "edge_density": 0.0,
            "intensity_entropy": _round(_entropy(values)),
        }
    else:
        weights = values[mask] + 1e-6
        x_mean = float(np.average(xs, weights=weights))
        y_mean = float(np.average(ys, weights=weights))
        x_centered = xs.astype(np.float64) - x_mean
        y_centered = ys.astype(np.float64) - y_mean
        radius = np.sqrt(x_centered**2 + y_centered**2) / max(height, width)
        covariance = np.cov(np.vstack([x_centered, y_centered]), aweights=weights)
        eigvals = np.linalg.eigvalsh(covariance) if covariance.shape == (2, 2) else np.asarray([0.0, 0.0])
        eigvals = np.maximum(eigvals, 0.0)
        major = float(np.sqrt(eigvals[-1] + 1e-9))
        minor = float(np.sqrt(eigvals[0] + 1e-9))
        x_span = int(xs.max() - xs.min() + 1)
        y_span = int(ys.max() - ys.min() + 1)
        bbox_area_fraction = (x_span * y_span) / float(height * width)
        active_fraction = float(mask.mean())
        gradient_y, gradient_x = np.gradient(values)
        features = {
            "mean_intensity": _round(float(values.mean())),
            "std_intensity": _round(float(values.std())),
            "active_fraction": _round(active_fraction),
            "bbox_area_fraction": _round(bbox_area_fraction),
            "compactness": _round(active_fraction / max(bbox_area_fraction, 1e-9)),
            "aspect_ratio": _round(max(x_span, y_span) / max(1, min(x_span, y_span))),
            "radius_mean": _round(float(radius.mean())),
            "radius_std": _round(float(radius.std())),
            "pca_anisotropy": _round(major / max(minor, 1e-9)),
            "component_count": _connected_components(mask),
            "edge_density": _round(float(np.mean(np.sqrt(gradient_x**2 + gradient_y**2)))),
            "intensity_entropy": _round(_entropy(values)),
        }

    return {
        "tool_name": "measure_image_features",
        "sample_id": sample.sample_id,
        "modality": "image",
        "payload_type": "scientific_features",
        "features": features,
        "notes": "computed from public polymer image pixels only",
    }


def _band_mask(size: int, low: int, high: int | None = None) -> np.ndarray:
    rows, cols = np.indices((size, size))
    dist = np.abs(rows - cols)
    if high is None:
        return dist >= low
    return (dist >= low) & (dist <= high)


def _distance_decay_slope(values: np.ndarray) -> float:
    size = values.shape[0]
    distances = np.arange(1, max(2, size // 2), dtype=np.float64)
    means = []
    valid_distances = []
    for distance in distances.astype(int):
        diag = np.diagonal(values, offset=distance)
        if diag.size and float(diag.mean()) > 0.0:
            valid_distances.append(distance)
            means.append(float(diag.mean()))
    if len(means) < 3:
        return float("nan")
    x = np.log1p(np.asarray(valid_distances, dtype=np.float64))
    y = np.log(np.asarray(means, dtype=np.float64) + 1e-9)
    slope, _ = np.polyfit(x, y, deg=1)
    return float(slope)


def measure_hic_features(sample: V3AgentSample) -> Dict[str, object]:
    """Measure Hi-C contact-map structure from public pixels only."""
    values = _load_gray(sample.hic_path)
    if values.shape[0] != values.shape[1]:
        side = min(values.shape)
        values = values[:side, :side]
    size = values.shape[0]
    total_signal = float(values.sum()) + 1e-9
    diag_mask = _band_mask(size, 0, 1)
    near_mask = _band_mask(size, 2, 5)
    far_mask = _band_mask(size, max(6, size // 8), None)
    half = size // 2
    same_half_mask = np.zeros_like(values, dtype=bool)
    same_half_mask[:half, :half] = True
    same_half_mask[half:, half:] = True
    cross_half_mask = ~same_half_mask
    p95 = float(np.percentile(values, 95.0))
    p99 = float(np.percentile(values, 99.0))

    diag_mean = float(values[diag_mask].mean()) if np.any(diag_mask) else float("nan")
    near_mean = float(values[near_mask].mean()) if np.any(near_mask) else float("nan")
    far_mean = float(values[far_mask].mean()) if np.any(far_mask) else float("nan")
    same_half_mean = float(values[same_half_mask].mean()) if np.any(same_half_mask) else float("nan")
    cross_half_mean = float(values[cross_half_mask].mean()) if np.any(cross_half_mask) else float("nan")

    features = {
        "mean_intensity": _round(float(values.mean())),
        "std_intensity": _round(float(values.std())),
        "diagonal_strength": _round(diag_mean),
        "near_diagonal_strength": _round(near_mean),
        "far_contact_strength": _round(far_mean),
        "long_range_fraction": _round(float(values[far_mask].sum()) / total_signal if np.any(far_mask) else float("nan")),
        "diagonal_to_far_ratio": _round(diag_mean / max(far_mean, 1e-9)),
        "symmetry_error": _round(float(np.mean(np.abs(values - values.T)))),
        "same_half_mean": _round(same_half_mean),
        "cross_half_mean": _round(cross_half_mean),
        "block_contrast": _round((same_half_mean - cross_half_mean) / max(abs(same_half_mean) + abs(cross_half_mean), 1e-9)),
        "hotspot_fraction_p95": _round(float(np.mean(values >= p95))),
        "top_signal_fraction_p99": _round(float(values[values >= p99].sum()) / total_signal),
        "distance_decay_slope": _round(_distance_decay_slope(values)),
        "intensity_entropy": _round(_entropy(values)),
    }

    return {
        "tool_name": "measure_hic_features",
        "sample_id": sample.sample_id,
        "modality": "hic",
        "payload_type": "scientific_features",
        "features": features,
        "notes": "computed from public Hi-C contact-map pixels only",
    }


def compare_modalities(sample: V3AgentSample) -> Dict[str, object]:
    """Compare public image and Hi-C feature summaries with fixed heuristics."""
    image = measure_image_features(sample)["features"]
    hic = measure_hic_features(sample)["features"]
    image_spread = float(image["radius_mean"])
    image_compactness = float(image["compactness"])
    image_fragmentation = min(1.0, float(image["component_count"]) / 20.0)
    hic_long_range = float(hic["long_range_fraction"])
    hic_blockiness = abs(float(hic["block_contrast"]))
    hic_decay = abs(float(hic["distance_decay_slope"])) if np.isfinite(float(hic["distance_decay_slope"])) else 0.0

    image_structure_index = float(np.clip(0.45 * image_spread + 0.35 * (1.0 - image_compactness) + 0.20 * image_fragmentation, 0.0, 1.0))
    hic_structure_index = float(np.clip(0.60 * hic_long_range + 0.25 * hic_blockiness + 0.15 * min(hic_decay, 2.0) / 2.0, 0.0, 1.0))
    discrepancy = abs(image_structure_index - hic_structure_index)
    if discrepancy >= 0.30:
        discrepancy_level = "high"
    elif discrepancy >= 0.15:
        discrepancy_level = "moderate"
    else:
        discrepancy_level = "low"

    comparison = {
        "image_structure_index": _round(image_structure_index),
        "hic_structure_index": _round(hic_structure_index),
        "cross_modal_discrepancy": _round(discrepancy),
        "discrepancy_level": discrepancy_level,
    }
    return {
        "tool_name": "compare_modalities",
        "sample_id": sample.sample_id,
        "modality": "image_hic",
        "payload_type": "scientific_comparison",
        "comparison": comparison,
        "notes": "fixed comparison from public pixel-derived features only",
    }


def generate_feature_report(sample: V3AgentSample) -> Dict[str, object]:
    """Return all fixed public feature summaries in one payload."""
    image_payload = measure_image_features(sample)
    hic_payload = measure_hic_features(sample)
    comparison_payload = compare_modalities(sample)
    return {
        "tool_name": "generate_feature_report",
        "sample_id": sample.sample_id,
        "modality": "image_hic",
        "payload_type": "scientific_report",
        "image_features": image_payload["features"],
        "hic_features": hic_payload["features"],
        "comparison": comparison_payload["comparison"],
        "notes": "public pixel-derived feature report only",
    }


def assert_scientific_tool_payload_is_safe(payload: Mapping[str, object]) -> None:
    """Raise if a scientific tool payload exposes evaluator-only metadata."""
    keys = set(payload.keys())
    unsafe = keys - SAFE_SCIENTIFIC_TOOL_KEYS
    if unsafe:
        raise AssertionError(f"Unexpected scientific payload keys: {sorted(unsafe)}")
    forbidden_fragments = ("label", "latent", "reliability", "contradiction", "action", "source_index", "split")

    def scan(value: object) -> None:
        if isinstance(value, Mapping):
            for nested_key, nested_value in value.items():
                tokens = str(nested_key).lower().replace("-", "_").split("_")
                if any(fragment in tokens for fragment in forbidden_fragments):
                    raise AssertionError(f"Scientific payload key appears unsafe: {nested_key!r}")
                scan(nested_value)
        elif isinstance(value, (list, tuple)):
            for item in value:
                scan(item)
        else:
            text = str(value).lower()
            if any(fragment in text for fragment in forbidden_fragments):
                raise AssertionError("Scientific payload value appears to leak hidden metadata")

    scan(payload)
