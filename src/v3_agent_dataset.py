"""V3 visual sample export for agentic multimodal reconciliation benchmarks."""

from __future__ import annotations

import json
import hashlib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Mapping, Sequence

import numpy as np
from PIL import Image, ImageDraw

from v2_regime import ContradictionRegimeConfig, build_v2_paired_dataset


LABEL_NAMES = {0: "normal", 1: "cancer"}
CONTRADICTION_LEVELS = (0.0, 0.5, 1.0)
VALID_CONTRADICTION_STATUSES = ("none", "weak", "strong")
VALID_RECOMMENDED_ACTIONS = ("use_both", "use_image", "use_hic", "abstain")


@dataclass(frozen=True)
class V3AgentSample:
    """One visual-only benchmark sample plus hidden evaluator metadata."""

    sample_id: str
    contradiction_strength: float
    source_index: int
    split: str
    true_label: int
    true_label_name: str
    true_contradiction_status: str
    recommended_action: str
    image_reliability: float
    hic_reliability: float
    structural_contradiction_score: float
    image_path: str
    hic_path: str
    panel_path: str

    def public_dict(self) -> Dict[str, object]:
        """Return the fields that are safe to expose to an agent."""
        return {
            "sample_id": self.sample_id,
            "image_path": self.image_path,
            "hic_path": self.hic_path,
            "panel_path": self.panel_path,
        }

    def hidden_dict(self) -> Dict[str, object]:
        """Return all fields, including evaluator-only labels."""
        return asdict(self)


def _to_uint8_image(array: np.ndarray) -> Image.Image:
    """Convert a numeric array to an 8-bit grayscale PIL image."""
    values = np.asarray(array, dtype=np.float32)
    if values.ndim == 3 and values.shape[0] == 1:
        values = values[0]
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        scaled = np.zeros(values.shape, dtype=np.uint8)
    else:
        low = float(np.percentile(finite, 1.0))
        high = float(np.percentile(finite, 99.0))
        if high <= low:
            low = float(finite.min())
            high = float(finite.max())
        if high <= low:
            scaled = np.zeros(values.shape, dtype=np.uint8)
        else:
            normalized = np.clip((values - low) / (high - low), 0.0, 1.0)
            scaled = np.asarray(np.round(255.0 * normalized), dtype=np.uint8)
    return Image.fromarray(scaled, mode="L")


def _save_panel(image_path: Path, hic_path: Path, panel_path: Path) -> None:
    """Create a side-by-side visual evidence panel."""
    image = Image.open(image_path).convert("L")
    hic = Image.open(hic_path).convert("L")
    target_height = max(image.height, hic.height, 96)
    image = image.resize((target_height, target_height), Image.Resampling.NEAREST)
    hic = hic.resize((target_height, target_height), Image.Resampling.NEAREST)

    label_height = 22
    gap = 12
    panel = Image.new("RGB", (2 * target_height + gap, target_height + label_height), "white")
    draw = ImageDraw.Draw(panel)
    draw.text((4, 4), "polymer image", fill="black")
    draw.text((target_height + gap + 4, 4), "Hi-C contact map", fill="black")
    panel.paste(image.convert("RGB"), (0, label_height))
    panel.paste(hic.convert("RGB"), (target_height + gap, label_height))
    panel_path.parent.mkdir(parents=True, exist_ok=True)
    panel.save(panel_path)


def _contradiction_status(score: float, strong_threshold: float) -> str:
    """Map the latent structural contradiction score to a coarse evaluator label."""
    if score >= strong_threshold:
        return "strong"
    if score >= 0.5 * strong_threshold:
        return "weak"
    return "none"


def _recommended_action(
    contradiction_status: str,
    image_reliability: float,
    hic_reliability: float,
) -> str:
    """Choose the evaluator-only reference action from reliability proxies."""
    if contradiction_status == "none":
        return "use_both"
    if max(image_reliability, hic_reliability) < 0.55:
        return "abstain"
    if abs(image_reliability - hic_reliability) < 0.10:
        return "use_both" if contradiction_status == "weak" else "abstain"
    return "use_image" if image_reliability > hic_reliability else "use_hic"


def _safe_sample_id(strength: float, source_index: int, seed: int, local_idx: int) -> str:
    """Create an agent-visible sample ID that does not encode hidden metadata."""
    raw = f"{seed}:{strength:.6f}:{source_index}:{local_idx}".encode("utf-8")
    digest = hashlib.sha256(raw).hexdigest()[:12]
    return f"v3_sample_{digest}"


def _balanced_subset(indices: np.ndarray, labels: np.ndarray, count: int, rng: np.random.Generator) -> np.ndarray:
    """Pick an approximately label-balanced subset from split indices."""
    if count <= 0:
        return np.asarray([], dtype=np.int64)

    selected: list[int] = []
    per_label = max(1, count // 2)
    for label in (0, 1):
        candidates = np.asarray([idx for idx in indices if int(labels[idx]) == label], dtype=np.int64)
        rng.shuffle(candidates)
        selected.extend(int(idx) for idx in candidates[:per_label])

    remaining_count = count - len(selected)
    if remaining_count > 0:
        remaining = np.asarray([idx for idx in indices if int(idx) not in set(selected)], dtype=np.int64)
        rng.shuffle(remaining)
        selected.extend(int(idx) for idx in remaining[:remaining_count])

    selected_array = np.asarray(selected[:count], dtype=np.int64)
    rng.shuffle(selected_array)
    return selected_array


def _resolve_counts(
    contradiction_strengths: Sequence[float],
    samples_per_strength: int | Mapping[float, int],
) -> Dict[float, int]:
    """Normalize requested sample counts for each contradiction strength."""
    if isinstance(samples_per_strength, Mapping):
        return {
            float(strength): int(samples_per_strength.get(float(strength), 0))
            for strength in contradiction_strengths
        }
    return {float(strength): int(samples_per_strength) for strength in contradiction_strengths}


def build_v3_agent_samples(
    output_dir: Path | str,
    contradiction_strengths: Sequence[float] = CONTRADICTION_LEVELS,
    samples_per_strength: int | Mapping[float, int] = 2,
    mode: str = "debug",
    image_size: int = 64,
    split: str = "test",
    seed: int = 21,
    structural_strong_threshold: float = 0.40,
) -> list[V3AgentSample]:
    """Generate stratified V3 visual samples from the V2 synthetic regime."""
    output_path = Path(output_dir)
    asset_dir = output_path / "assets"
    asset_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    counts = _resolve_counts(contradiction_strengths, samples_per_strength)

    samples: list[V3AgentSample] = []
    for strength in contradiction_strengths:
        strength_value = float(strength)
        requested_count = int(counts[strength_value])
        if requested_count <= 0:
            continue

        dataset = build_v2_paired_dataset(
            regime=ContradictionRegimeConfig(
                contradiction_strength=strength_value,
                modality_asymmetry=0.0,
                structural_strong_threshold=structural_strong_threshold,
            ),
            mode=mode,
            image_size=image_size,
            genomic_representation="matrix",
            seed=seed + int(round(1000.0 * strength_value)),
        )
        split_indices = np.asarray(dataset["split_indices"][split], dtype=np.int64)
        labels = np.asarray(dataset["arrays"]["label"], dtype=np.int64)
        chosen_indices = _balanced_subset(split_indices, labels, requested_count, rng)

        arrays = dataset["arrays"]
        for local_idx, source_index in enumerate(chosen_indices):
            source_index_int = int(source_index)
            sample_id = _safe_sample_id(
                strength=strength_value,
                source_index=source_index_int,
                seed=seed,
                local_idx=local_idx,
            )
            sample_dir = asset_dir / sample_id
            sample_dir.mkdir(parents=True, exist_ok=True)

            image_path = sample_dir / "polymer_image.png"
            hic_path = sample_dir / "hic_contact_map.png"
            panel_path = sample_dir / "evidence_panel.png"
            _to_uint8_image(arrays["image"][source_index_int]).save(image_path)
            _to_uint8_image(arrays["genomic_matrix"][source_index_int]).save(hic_path)
            _save_panel(image_path=image_path, hic_path=hic_path, panel_path=panel_path)

            score = float(arrays["latent_structural_contradiction_score"][source_index_int])
            status = _contradiction_status(score, strong_threshold=structural_strong_threshold)
            image_reliability = float(arrays["latent_image_observability_drive"][source_index_int])
            hic_reliability = float(arrays["latent_genomic_observability_drive"][source_index_int])
            action = _recommended_action(status, image_reliability, hic_reliability)
            true_label = int(labels[source_index_int])

            samples.append(
                V3AgentSample(
                    sample_id=sample_id,
                    contradiction_strength=strength_value,
                    source_index=source_index_int,
                    split=split,
                    true_label=true_label,
                    true_label_name=LABEL_NAMES[true_label],
                    true_contradiction_status=status,
                    recommended_action=action,
                    image_reliability=image_reliability,
                    hic_reliability=hic_reliability,
                    structural_contradiction_score=score,
                    image_path=str(image_path),
                    hic_path=str(hic_path),
                    panel_path=str(panel_path),
                )
            )

    manifest_path = output_path / "sample_manifest.jsonl"
    with manifest_path.open("w", encoding="utf-8") as handle:
        for sample in samples:
            handle.write(json.dumps(sample.hidden_dict(), sort_keys=True) + "\n")
    return samples


def load_v3_agent_samples(manifest_path: Path | str) -> list[V3AgentSample]:
    """Load V3 samples from a manifest written by :func:`build_v3_agent_samples`."""
    samples = []
    with Path(manifest_path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            samples.append(V3AgentSample(**json.loads(line)))
    return samples
