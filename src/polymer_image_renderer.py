"""Render latent polymer conformations into grayscale optical-space images."""

from __future__ import annotations

from typing import Dict, Iterable, Optional, Sequence

import numpy as np


def _validate_polymer(polymer: np.ndarray) -> np.ndarray:
    """Validate and coerce a polymer coordinate array."""
    coords = np.asarray(polymer, dtype=float)
    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError("polymer must have shape (n_monomers, 2)")
    if len(coords) < 2:
        raise ValueError("polymer must contain at least two monomers")
    if not np.all(np.isfinite(coords)):
        raise ValueError("polymer contains non-finite coordinates")
    return coords


def _coerce_rng(seed: Optional[int] = None) -> np.random.Generator:
    """Create a reproducible random generator."""
    return np.random.default_rng(seed)


def normalize_polymer_coordinates(
    polymer: np.ndarray,
    image_size: int = 32,
    margin: float = 0.10,
    fit_quantile: float = 0.92,
) -> np.ndarray:
    """Center and rescale polymer coordinates into image space.

    A robust quantile-based extent reduces the chance that a few outlier monomers
    make overall footprint size the dominant visual cue.
    """
    coords = _validate_polymer(polymer)
    if image_size < 8:
        raise ValueError("image_size must be at least 8")
    if not 0.0 <= margin < 0.45:
        raise ValueError("margin must be in [0, 0.45)")
    if not 0.50 <= fit_quantile <= 1.0:
        raise ValueError("fit_quantile must be in [0.50, 1.0]")

    centered = coords - coords.mean(axis=0, keepdims=True)
    robust_extent = max(
        float(np.quantile(np.abs(centered[:, 0]), fit_quantile)),
        float(np.quantile(np.abs(centered[:, 1]), fit_quantile)),
        1e-8,
    )
    usable_half_span = max(1.0, (image_size - 1) * (0.5 - margin))
    scaled = centered * (usable_half_span / robust_extent)
    return scaled + 0.5 * (image_size - 1)


def gaussian_stamp(sigma: float = 1.0, truncate: float = 3.0) -> np.ndarray:
    """Build a small normalized Gaussian kernel for monomer density."""
    if sigma <= 0.0:
        return np.ones((1, 1), dtype=float)

    radius = max(1, int(np.ceil(truncate * sigma)))
    offsets = np.arange(-radius, radius + 1, dtype=float)
    grid_y, grid_x = np.meshgrid(offsets, offsets, indexing="ij")
    kernel = np.exp(-0.5 * (grid_x**2 + grid_y**2) / (sigma**2))
    kernel /= kernel.sum()
    return kernel


def _interpolate_backbone_points(
    polymer_pixels: np.ndarray,
    backbone_weight: float = 0.35,
    max_segment_step: float = 0.75,
    monomer_weights: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Add light contour support so images resemble smooth density maps."""
    if backbone_weight <= 0.0 or len(polymer_pixels) < 2:
        return np.empty((0, 2), dtype=float), np.empty(0, dtype=float)
    if max_segment_step <= 0.0:
        raise ValueError("max_segment_step must be positive")

    interpolated_points = []
    weights = []

    for segment_idx, (start, end) in enumerate(zip(polymer_pixels[:-1], polymer_pixels[1:])):
        segment = end - start
        segment_length = float(np.linalg.norm(segment))
        n_points = max(1, int(np.ceil(segment_length / max_segment_step)))
        interpolation_steps = np.linspace(0.0, 1.0, n_points + 2)[1:-1]
        segment_weight = backbone_weight
        if monomer_weights is not None:
            segment_weight *= 0.5 * (monomer_weights[segment_idx] + monomer_weights[segment_idx + 1])
        interpolated_points.append(start + interpolation_steps[:, None] * segment[None, :])
        weights.append(np.full(n_points, segment_weight / n_points, dtype=float))

    return np.vstack(interpolated_points), np.concatenate(weights)


def monomer_crowding_weights(
    polymer_pixels: np.ndarray,
    radius: float,
    min_separation: int = 2,
    crowding_strength: float = 0.35,
) -> np.ndarray:
    """Boost monomers in locally crowded neighborhoods while keeping total mass stable."""
    coords = _validate_polymer(polymer_pixels)
    if crowding_strength <= 0.0:
        return np.ones(len(coords), dtype=float)
    if radius <= 0.0:
        raise ValueError("radius must be positive")

    deltas = coords[:, None, :] - coords[None, :, :]
    distances = np.linalg.norm(deltas, axis=-1)
    indices = np.arange(len(coords))
    contour_separation = np.abs(indices[:, None] - indices[None, :])
    crowded_neighbors = (distances <= radius) & (contour_separation >= min_separation)
    local_counts = crowded_neighbors.sum(axis=1).astype(float)

    centered_counts = local_counts - local_counts.mean()
    standardized = centered_counts / (local_counts.std(ddof=0) + 1e-8)
    weights = 1.0 + crowding_strength * standardized
    weights = np.clip(weights, 0.35, 2.50)
    weights /= weights.mean()
    return weights


def _accumulate_gaussian_points(
    points: np.ndarray,
    point_weights: np.ndarray,
    image_size: int,
    sigma: float,
) -> np.ndarray:
    """Rasterize weighted support points with a Gaussian stamp."""
    density = np.zeros((image_size, image_size), dtype=float)
    kernel = gaussian_stamp(sigma=sigma)
    radius = kernel.shape[0] // 2

    for (x_coord, y_coord), weight in zip(points, point_weights):
        col_center = int(np.round(x_coord))
        row_center = int(np.round(y_coord))

        if row_center < -radius or row_center > image_size - 1 + radius:
            continue
        if col_center < -radius or col_center > image_size - 1 + radius:
            continue

        row_start = max(0, row_center - radius)
        row_stop = min(image_size, row_center + radius + 1)
        col_start = max(0, col_center - radius)
        col_stop = min(image_size, col_center + radius + 1)

        if row_start >= row_stop or col_start >= col_stop:
            continue

        kernel_row_start = row_start - (row_center - radius)
        kernel_row_stop = kernel_row_start + (row_stop - row_start)
        kernel_col_start = col_start - (col_center - radius)
        kernel_col_stop = kernel_col_start + (col_stop - col_start)

        density[row_start:row_stop, col_start:col_stop] += (
            weight * kernel[kernel_row_start:kernel_row_stop, kernel_col_start:kernel_col_stop]
        )

    return density


def local_domain_centers(
    polymer_pixels: np.ndarray,
    monomer_weights: np.ndarray,
    window_size: int = 8,
    step: int = 4,
) -> tuple[np.ndarray, np.ndarray]:
    """Summarize locally compact polymer segments as broader density domains."""
    coords = _validate_polymer(polymer_pixels)
    if len(coords) == 0:
        return np.empty((0, 2), dtype=float), np.empty(0, dtype=float)

    effective_window = min(max(window_size, 3), len(coords))
    effective_step = max(step, 1)
    last_start = max(len(coords) - effective_window, 0)
    starts = list(range(0, last_start + 1, effective_step))
    if not starts or starts[-1] != last_start:
        starts.append(last_start)

    centers = []
    weights = []
    for start in starts:
        segment = coords[start : start + effective_window]
        center = segment.mean(axis=0)
        centered = segment - center
        local_rg = float(np.sqrt(np.mean(np.sum(centered**2, axis=1))))
        local_crowding = float(monomer_weights[start : start + effective_window].mean())
        compactness = 1.0 / (1.0 + local_rg)

        centers.append(center)
        weights.append(local_crowding * (0.55 + 1.45 * compactness))

    center_array = np.asarray(centers, dtype=float)
    weight_array = np.asarray(weights, dtype=float)
    weight_array /= np.clip(weight_array.mean(), 1e-8, None)
    return center_array, weight_array


def rasterize_polymer_density(
    polymer_pixels: np.ndarray,
    image_size: int = 32,
    sigma: float = 1.0,
    backbone_weight: float = 0.35,
    max_segment_step: float = 0.75,
    monomer_weights: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Accumulate Gaussian density from monomer and backbone support points."""
    coords = _validate_polymer(polymer_pixels)
    if monomer_weights is None:
        point_weights = np.ones(len(coords), dtype=float)
    else:
        point_weights = np.asarray(monomer_weights, dtype=float)
        if point_weights.shape != (len(coords),):
            raise ValueError("monomer_weights must have shape (n_monomers,)")
        if np.any(point_weights <= 0.0):
            raise ValueError("monomer_weights must be strictly positive")

    backbone_points, backbone_weights = _interpolate_backbone_points(
        coords,
        backbone_weight=backbone_weight,
        max_segment_step=max_segment_step,
        monomer_weights=point_weights,
    )
    all_points = [coords]
    all_weights = [point_weights]
    if len(backbone_points) > 0:
        all_points.append(backbone_points)
        all_weights.append(backbone_weights)

    stamp_points = np.vstack(all_points)
    stamp_weights = np.concatenate(all_weights)
    return _accumulate_gaussian_points(
        stamp_points,
        stamp_weights,
        image_size=image_size,
        sigma=sigma,
    )


def _gaussian_kernel_1d(sigma: float = 1.0, truncate: float = 3.0) -> np.ndarray:
    """Build a small 1D Gaussian kernel for image smoothing."""
    if sigma <= 0.0:
        return np.ones(1, dtype=float)

    radius = max(1, int(np.ceil(truncate * sigma)))
    offsets = np.arange(-radius, radius + 1, dtype=float)
    kernel = np.exp(-0.5 * (offsets / sigma) ** 2)
    kernel /= kernel.sum()
    return kernel


def _convolve_along_axis(image: np.ndarray, kernel: np.ndarray, axis: int) -> np.ndarray:
    """Apply a small separable convolution using reflect padding."""
    radius = len(kernel) // 2
    if radius == 0:
        return image.astype(float, copy=True)

    if axis == 0:
        padded = np.pad(image, ((radius, radius), (0, 0)), mode="reflect")
        smoothed = np.zeros_like(image, dtype=float)
        for offset, weight in enumerate(kernel):
            smoothed += weight * padded[offset : offset + image.shape[0], :]
        return smoothed

    padded = np.pad(image, ((0, 0), (radius, radius)), mode="reflect")
    smoothed = np.zeros_like(image, dtype=float)
    for offset, weight in enumerate(kernel):
        smoothed += weight * padded[:, offset : offset + image.shape[1]]
    return smoothed


def apply_gaussian_blur(image: np.ndarray, sigma: float = 0.6) -> np.ndarray:
    """Apply a light blur to mimic optical imaging."""
    kernel = _gaussian_kernel_1d(sigma=sigma)
    return _convolve_along_axis(_convolve_along_axis(image, kernel, axis=1), kernel, axis=0)


def apply_circular_envelope(
    image: np.ndarray,
    strength: float = 0.0,
    radius_fraction: float = 0.48,
) -> np.ndarray:
    """Optionally taper image corners with a weak circular envelope."""
    if strength <= 0.0:
        return image
    if not 0.0 < radius_fraction <= 0.75:
        raise ValueError("radius_fraction must be in (0, 0.75]")

    height, width = image.shape
    center_y = 0.5 * (height - 1)
    center_x = 0.5 * (width - 1)
    grid_y, grid_x = np.indices(image.shape, dtype=float)
    radius = radius_fraction * min(height, width)
    distance = np.sqrt((grid_x - center_x) ** 2 + (grid_y - center_y) ** 2)
    envelope = np.exp(-0.5 * (distance / max(radius, 1e-8)) ** 4)
    return image * ((1.0 - strength) + strength * envelope)


def normalize_image_intensity(
    image: np.ndarray,
    lower_quantile: float = 0.01,
    upper_quantile: float = 0.995,
) -> np.ndarray:
    """Map image intensity into [0, 1] using a robust percentile range."""
    clipped = np.clip(np.asarray(image, dtype=float), 0.0, None)
    lower = float(np.quantile(clipped, lower_quantile))
    upper = float(np.quantile(clipped, upper_quantile))

    if upper <= lower + 1e-8:
        return np.zeros_like(clipped)
    normalized = (clipped - lower) / (upper - lower)
    return np.clip(normalized, 0.0, 1.0)


def normalize_image_by_mass(
    image: np.ndarray,
    n_monomers: int,
    image_size: int,
    backbone_weight: float,
    gain: float = 8.0,
) -> np.ndarray:
    """Normalize intensity using expected chromatin mass per pixel.

    This keeps images on a shared scale across samples instead of stretching
    each image independently.
    """
    total_mass = float(n_monomers + max(n_monomers - 1, 0) * backbone_weight)
    reference_scale = gain * total_mass / max(image_size * image_size, 1)
    if reference_scale <= 1e-8:
        return np.zeros_like(image, dtype=float)
    return np.clip(np.asarray(image, dtype=float) / reference_scale, 0.0, 1.0)


def _add_background_noise(
    image: np.ndarray,
    noise_scale: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Inject weak low-frequency background noise."""
    if noise_scale <= 0.0:
        return image

    noise = rng.normal(size=image.shape)
    noise = apply_gaussian_blur(noise, sigma=0.8)
    noise -= noise.mean()
    amplitude = noise_scale * max(float(np.quantile(image, 0.98)), 1.0)
    return np.clip(image + amplitude * noise, 0.0, None)


def render_polymer_to_image(
    polymer: np.ndarray,
    image_size: int = 32,
    sigma: float = 1.0,
    margin: float = 0.10,
    normalize: bool = True,
    add_noise: bool = False,
    seed: Optional[int] = None,
    blur_sigma: float = 0.6,
    backbone_weight: float = 0.35,
    fit_quantile: float = 0.92,
    max_segment_step: float = 0.75,
    crowding_radius_scale: float = 0.07,
    crowding_strength: float = 0.35,
    envelope_strength: float = 0.0,
    noise_scale: float = 0.03,
    contrast_gamma: float = 1.0,
    normalization_mode: str = "mass",
    normalization_gain: float = 8.0,
) -> np.ndarray:
    """Render one polymer conformation into a grayscale chromatin-density image."""
    coords = _validate_polymer(polymer)
    rng = _coerce_rng(seed)
    polymer_pixels = normalize_polymer_coordinates(
        coords,
        image_size=image_size,
        margin=margin,
        fit_quantile=fit_quantile,
    )
    monomer_weights = monomer_crowding_weights(
        polymer_pixels,
        radius=image_size * crowding_radius_scale,
        crowding_strength=crowding_strength,
    )
    fine_density = rasterize_polymer_density(
        polymer_pixels,
        image_size=image_size,
        sigma=sigma,
        backbone_weight=backbone_weight,
        max_segment_step=max_segment_step,
        monomer_weights=0.60 + 0.40 * monomer_weights,
    )
    coarse_density = rasterize_polymer_density(
        polymer_pixels,
        image_size=image_size,
        sigma=max(1.2, 2.1 * sigma),
        backbone_weight=0.15 * backbone_weight,
        max_segment_step=1.25 * max_segment_step,
        monomer_weights=np.sqrt(monomer_weights),
    )
    domain_centers, domain_weights = local_domain_centers(
        polymer_pixels,
        monomer_weights=monomer_weights,
        window_size=max(6, len(coords) // 12),
        step=max(2, len(coords) // 24),
    )
    domain_density = _accumulate_gaussian_points(
        domain_centers,
        domain_weights,
        image_size=image_size,
        sigma=max(1.4, 2.6 * sigma),
    )
    occupancy_field = apply_gaussian_blur(
        coarse_density + 0.75 * domain_density,
        sigma=max(0.9, blur_sigma + 0.55),
    )
    image = 0.42 * fine_density + 0.24 * coarse_density + 0.72 * domain_density + 0.12 * occupancy_field
    if blur_sigma > 0.0:
        image = apply_gaussian_blur(image, sigma=blur_sigma)
    if envelope_strength > 0.0:
        image = apply_circular_envelope(image, strength=envelope_strength)
    if add_noise:
        image = _add_background_noise(image, noise_scale=noise_scale, rng=rng)
    if contrast_gamma != 1.0:
        image = np.clip(image, 0.0, None) ** contrast_gamma
    if normalize:
        if normalization_mode == "percentile":
            return normalize_image_intensity(image)
        if normalization_mode == "mass":
            return normalize_image_by_mass(
                image,
                n_monomers=len(coords),
                image_size=image_size,
                backbone_weight=backbone_weight,
                gain=normalization_gain,
            )
        raise ValueError("normalization_mode must be 'mass' or 'percentile'")
    return np.clip(image, 0.0, None)


def render_batch(
    polymers: Sequence[np.ndarray],
    seed: Optional[int] = None,
    **render_kwargs,
) -> np.ndarray:
    """Render a batch of polymers into a stack of grayscale images."""
    rng = _coerce_rng(seed)
    rendered_images = []
    for polymer in polymers:
        item_seed = int(rng.integers(0, 1_000_000_000))
        rendered_images.append(render_polymer_to_image(polymer, seed=item_seed, **render_kwargs))
    return np.stack(rendered_images, axis=0)


def image_intensity_entropy(image: np.ndarray, bins: int = 32) -> float:
    """Measure the spread of image intensities with a normalized entropy."""
    flattened = np.clip(np.asarray(image, dtype=float).ravel(), 0.0, 1.0)
    histogram, _ = np.histogram(flattened, bins=bins, range=(0.0, 1.0))
    probabilities = histogram.astype(float)
    probabilities /= np.clip(probabilities.sum(), 1e-8, None)
    nonzero = probabilities > 0.0
    entropy = -np.sum(probabilities[nonzero] * np.log(probabilities[nonzero]))
    return float(entropy / np.log(bins))


def local_texture_variation(image: np.ndarray, blur_sigma: float = 1.0) -> float:
    """Estimate local textural heterogeneity via residual energy after smoothing."""
    image_array = np.asarray(image, dtype=float)
    local_mean = apply_gaussian_blur(image_array, sigma=blur_sigma)
    residual = image_array - local_mean
    baseline = np.mean(local_mean**2)
    return float(np.mean(residual**2) / (baseline + 1e-8))


def radial_intensity_profile(image: np.ndarray, n_bins: int = 8) -> np.ndarray:
    """Average intensity as a function of radius from the image center."""
    image_array = np.asarray(image, dtype=float)
    grid_y, grid_x = np.indices(image_array.shape, dtype=float)
    center_y = 0.5 * (image_array.shape[0] - 1)
    center_x = 0.5 * (image_array.shape[1] - 1)
    distance = np.sqrt((grid_x - center_x) ** 2 + (grid_y - center_y) ** 2)
    normalized_radius = distance / np.clip(distance.max(), 1e-8, None)
    radial_edges = np.linspace(0.0, 1.0, n_bins + 1)
    profile = np.zeros(n_bins, dtype=float)

    for bin_idx in range(n_bins):
        if bin_idx == n_bins - 1:
            mask = (normalized_radius >= radial_edges[bin_idx]) & (
                normalized_radius <= radial_edges[bin_idx + 1]
            )
        else:
            mask = (normalized_radius >= radial_edges[bin_idx]) & (
                normalized_radius < radial_edges[bin_idx + 1]
            )
        profile[bin_idx] = float(image_array[mask].mean()) if np.any(mask) else 0.0

    return profile


def radial_intensity_std(image: np.ndarray, n_bins: int = 8) -> float:
    """Measure how uneven the radial intensity profile is."""
    profile = radial_intensity_profile(image, n_bins=n_bins)
    return float(profile.std(ddof=0) / (profile.mean() + 1e-8))


def patch_density_variation(image: np.ndarray, patch_size: int = 4) -> float:
    """Measure heterogeneity across coarse spatial patches."""
    image_array = np.asarray(image, dtype=float)
    usable_height = (image_array.shape[0] // patch_size) * patch_size
    usable_width = (image_array.shape[1] // patch_size) * patch_size
    if usable_height == 0 or usable_width == 0:
        return 0.0

    trimmed = image_array[:usable_height, :usable_width]
    pooled = trimmed.reshape(
        usable_height // patch_size,
        patch_size,
        usable_width // patch_size,
        patch_size,
    ).sum(axis=(1, 3))
    return float(pooled.std(ddof=0) / (pooled.mean() + 1e-8))


def occupancy_fraction(image: np.ndarray, threshold: float = 0.45) -> float:
    """Fraction of pixels above a bright-chromatin threshold."""
    image_array = np.asarray(image, dtype=float)
    return float((image_array >= threshold).mean())


def image_metrics(image: np.ndarray) -> Dict[str, float]:
    """Compute lightweight image-space summary metrics."""
    image_array = np.asarray(image, dtype=float)
    return {
        "mean_intensity": float(image_array.mean()),
        "intensity_entropy": image_intensity_entropy(image_array),
        "local_texture_variation": local_texture_variation(image_array),
        "patch_density_variation": patch_density_variation(image_array),
        "radial_intensity_std": radial_intensity_std(image_array),
        "occupancy_fraction": occupancy_fraction(image_array),
    }


def summarize_metric(metric_dicts: Iterable[Dict[str, float]], metric_name: str) -> np.ndarray:
    """Collect one image metric across a list of metric dictionaries."""
    return np.asarray([metrics[metric_name] for metrics in metric_dicts], dtype=float)
