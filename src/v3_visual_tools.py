"""Visual and fixed scientific tools for the V3 agent benchmark."""

from __future__ import annotations

import base64
from pathlib import Path
from typing import Dict, Mapping

from PIL import Image

from v3_agent_dataset import V3AgentSample
from v3_python_sandbox import V3PythonSandbox
from v3_scientific_tools import (
    SCIENTIFIC_TOOL_NAMES,
    compare_modalities,
    generate_feature_report,
    measure_hic_features,
    measure_image_features,
)


SAFE_TOOL_KEYS = {
    "tool_name",
    "sample_id",
    "modality",
    "image_path",
    "media_type",
    "image_base64",
    "width",
    "height",
    "payload_type",
    "features",
    "image_features",
    "hic_features",
    "comparison",
    "notes",
    "input_condition",
    "available_files",
    "returncode",
    "stdout",
    "stderr",
    "timed_out",
}


class V3VisualToolbox:
    """Toolbox that exposes only visual evidence, never evaluator labels."""

    def __init__(self, samples: list[V3AgentSample], work_dir: Path | str) -> None:
        self.samples: Dict[str, V3AgentSample] = {sample.sample_id: sample for sample in samples}
        self.work_dir = Path(work_dir)
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.python_sandbox = V3PythonSandbox(self.samples, self.work_dir / "python")

    def _sample(self, sample_id: str) -> V3AgentSample:
        if sample_id not in self.samples:
            raise KeyError(f"Unknown sample_id: {sample_id}")
        return self.samples[sample_id]

    def _path_for_modality(self, sample: V3AgentSample, modality: str) -> Path:
        if modality == "image":
            return Path(sample.image_path)
        if modality == "hic":
            return Path(sample.hic_path)
        if modality == "panel":
            return Path(sample.panel_path)
        raise ValueError("modality must be 'image', 'hic', or 'panel'")

    @staticmethod
    def _encode_png(path: Path) -> str:
        with path.open("rb") as handle:
            return base64.b64encode(handle.read()).decode("ascii")

    def _payload(self, tool_name: str, sample_id: str, modality: str, path: Path) -> Dict[str, object]:
        with Image.open(path) as image:
            width, height = image.size
        return {
            "tool_name": tool_name,
            "sample_id": sample_id,
            "modality": modality,
            "image_path": str(path),
            "media_type": "image/png",
            "image_base64": self._encode_png(path),
            "width": int(width),
            "height": int(height),
        }

    def load_sample_panel(self, sample_id: str) -> Dict[str, object]:
        """Return the side-by-side visual evidence panel for one sample."""
        sample = self._sample(sample_id)
        path = self._path_for_modality(sample, "panel")
        return self._payload("load_sample_panel", sample_id, "panel", path)

    def load_modality_image(self, sample_id: str, modality: str) -> Dict[str, object]:
        """Return one raw modality image for one sample."""
        sample = self._sample(sample_id)
        path = self._path_for_modality(sample, modality)
        return self._payload("load_modality_image", sample_id, modality, path)

    def crop_modality_image(
        self,
        sample_id: str,
        modality: str,
        x0: int,
        y0: int,
        x1: int,
        y1: int,
    ) -> Dict[str, object]:
        """Return a bounded crop from one modality image."""
        sample = self._sample(sample_id)
        source_path = self._path_for_modality(sample, modality)
        with Image.open(source_path) as image:
            width, height = image.size
            left = max(0, min(int(x0), width - 1))
            top = max(0, min(int(y0), height - 1))
            right = max(left + 1, min(int(x1), width))
            bottom = max(top + 1, min(int(y1), height))
            crop = image.crop((left, top, right, bottom))

        crop_dir = self.work_dir / sample_id
        crop_dir.mkdir(parents=True, exist_ok=True)
        crop_path = crop_dir / f"{modality}_crop_{left}_{top}_{right}_{bottom}.png"
        crop.save(crop_path)
        return self._payload("crop_modality_image", sample_id, modality, crop_path)

    def call_tool(self, name: str, arguments: Mapping[str, object]) -> Dict[str, object]:
        """Dispatch a named visual or fixed scientific tool call."""
        if name == "load_sample_panel":
            return self.load_sample_panel(sample_id=str(arguments["sample_id"]))
        if name == "load_modality_image":
            return self.load_modality_image(
                sample_id=str(arguments["sample_id"]),
                modality=str(arguments["modality"]),
            )
        if name == "crop_modality_image":
            return self.crop_modality_image(
                sample_id=str(arguments["sample_id"]),
                modality=str(arguments["modality"]),
                x0=int(arguments["x0"]),
                y0=int(arguments["y0"]),
                x1=int(arguments["x1"]),
                y1=int(arguments["y1"]),
            )
        if name in SCIENTIFIC_TOOL_NAMES:
            sample = self._sample(str(arguments["sample_id"]))
            if name == "measure_image_features":
                return measure_image_features(sample)
            if name == "measure_hic_features":
                return measure_hic_features(sample)
            if name == "compare_modalities":
                return compare_modalities(sample)
            if name == "generate_feature_report":
                return generate_feature_report(sample)
        if name == "run_python_analysis":
            return self.python_sandbox.run_python_analysis(
                sample_id=str(arguments["sample_id"]),
                code=str(arguments["code"]),
                input_condition=str(arguments.get("_input_condition", "both_modalities")),
            )
        raise ValueError(f"Unknown V3 tool: {name}")


def assert_visual_tool_payload_is_safe(payload: Mapping[str, object]) -> None:
    """Raise if a visual tool payload exposes evaluator-only metadata."""
    keys = set(payload.keys())
    unsafe = keys - SAFE_TOOL_KEYS
    if unsafe:
        raise AssertionError(f"Unexpected tool payload keys: {sorted(unsafe)}")
    forbidden_fragments = ("label", "latent", "reliability", "contradiction", "action")
    for key, value in payload.items():
        if key == "image_base64":
            continue
        text = f"{key} {value}".lower()
        if any(fragment in text for fragment in forbidden_fragments):
            raise AssertionError(f"Tool payload appears to leak hidden metadata in key {key!r}")
