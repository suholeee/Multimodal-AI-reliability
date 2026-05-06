"""Constrained Python execution tool for V3 Level 3 benchmarks."""

from __future__ import annotations

import ast
import json
import shutil
import subprocess
import sys
import tempfile
import textwrap
import uuid
from pathlib import Path
from typing import Mapping

from v3_agent_dataset import V3AgentSample


SAFE_PYTHON_TOOL_KEYS = {
    "tool_name",
    "sample_id",
    "payload_type",
    "input_condition",
    "available_files",
    "returncode",
    "stdout",
    "stderr",
    "timed_out",
    "notes",
}


DANGEROUS_IMPORT_ROOTS = {
    "ctypes",
    "ftplib",
    "glob",
    "http",
    "importlib",
    "os",
    "pathlib",
    "pickle",
    "requests",
    "shutil",
    "socket",
    "subprocess",
    "sys",
    "urllib",
}


FORBIDDEN_CODE_FRAGMENTS = (
    "..",
    "/users/",
    "/volumes/",
    "sample_manifest",
    "true_label",
    "latent",
    "reliability",
    "recommended_action",
    "source_index",
    "split_indices",
    "results/v3",
    "anthropic_api_key",
    "openai_api_key",
)


def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n...[truncated {len(text) - limit} chars]"


def _validate_code(code: str, max_chars: int) -> str | None:
    if len(code) > max_chars:
        return f"code is too long: {len(code)} chars > {max_chars}"
    lowered = code.lower()
    for fragment in FORBIDDEN_CODE_FRAGMENTS:
        if fragment in lowered:
            return f"code contains forbidden fragment: {fragment}"
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return f"syntax error: {exc}"
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".", 1)[0].lower()
                if root in DANGEROUS_IMPORT_ROOTS:
                    return f"import is not allowed in sandbox: {alias.name}"
        elif isinstance(node, ast.ImportFrom):
            root = (node.module or "").split(".", 1)[0].lower()
            if root in DANGEROUS_IMPORT_ROOTS:
                return f"import is not allowed in sandbox: {node.module}"
    return None


def _public_files_for_condition(sample: V3AgentSample, input_condition: str) -> dict[str, Path]:
    if input_condition == "image_only":
        return {"polymer_image.png": Path(sample.image_path)}
    if input_condition == "hic_only":
        return {"hic_contact_map.png": Path(sample.hic_path)}
    if input_condition == "both_modalities":
        return {
            "polymer_image.png": Path(sample.image_path),
            "hic_contact_map.png": Path(sample.hic_path),
            "evidence_panel.png": Path(sample.panel_path),
        }
    raise ValueError(f"Unknown input_condition: {input_condition}")


class V3PythonSandbox:
    """Execute model-written Python over copied public sample files."""

    def __init__(
        self,
        samples: Mapping[str, V3AgentSample],
        work_dir: Path | str,
        timeout_seconds: int = 8,
        max_code_chars: int = 6000,
        max_output_chars: int = 6000,
    ) -> None:
        self.samples = dict(samples)
        self.work_dir = Path(work_dir)
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.timeout_seconds = int(timeout_seconds)
        self.max_code_chars = int(max_code_chars)
        self.max_output_chars = int(max_output_chars)

    def run_python_analysis(self, sample_id: str, code: str, input_condition: str) -> dict[str, object]:
        """Run one Python analysis script in a temporary public-data workspace."""
        if sample_id not in self.samples:
            raise KeyError(f"Unknown sample_id: {sample_id}")
        sample = self.samples[sample_id]
        available_files = _public_files_for_condition(sample, input_condition)

        safety_error = _validate_code(str(code), max_chars=self.max_code_chars)
        if safety_error:
            return {
                "tool_name": "run_python_analysis",
                "sample_id": sample_id,
                "payload_type": "python_execution",
                "input_condition": input_condition,
                "available_files": sorted(available_files),
                "returncode": 2,
                "stdout": "",
                "stderr": safety_error,
                "timed_out": False,
                "notes": "code was rejected before execution by static sandbox checks",
            }

        run_parent = self.work_dir / sample_id
        run_parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=f"py_{uuid.uuid4().hex[:8]}_", dir=run_parent) as tmp:
            run_dir = Path(tmp)
            for public_name, source_path in available_files.items():
                shutil.copyfile(source_path, run_dir / public_name)

            code_path = run_dir / "analysis_code.py"
            code_path.write_text(str(code), encoding="utf-8")
            runner_path = run_dir / "sandbox_runner.py"
            runner_path.write_text(_runner_source(), encoding="utf-8")

            try:
                completed = subprocess.run(
                    [sys.executable, "-I", str(runner_path)],
                    cwd=run_dir,
                    capture_output=True,
                    text=True,
                    timeout=self.timeout_seconds,
                    check=False,
                )
                returncode = int(completed.returncode)
                stdout = completed.stdout
                stderr = completed.stderr
                timed_out = False
            except subprocess.TimeoutExpired as exc:
                returncode = 124
                stdout = exc.stdout or ""
                stderr = (exc.stderr or "") + f"\nTimed out after {self.timeout_seconds} seconds."
                timed_out = True

            run_dir_text = str(run_dir)
            stdout = str(stdout).replace(run_dir_text, "<sandbox>")
            stderr = str(stderr).replace(run_dir_text, "<sandbox>")
            return {
                "tool_name": "run_python_analysis",
                "sample_id": sample_id,
                "payload_type": "python_execution",
                "input_condition": input_condition,
                "available_files": sorted(available_files),
                "returncode": returncode,
                "stdout": _truncate(stdout, self.max_output_chars),
                "stderr": _truncate(stderr, self.max_output_chars),
                "timed_out": timed_out,
                "notes": "executed over copied public sample files only; hidden evaluator metadata was not copied",
            }


def _runner_source() -> str:
    """Return the isolated runner source used inside each temporary directory."""
    return textwrap.dedent(
        """
        import builtins
        import runpy
        from pathlib import Path

        SANDBOX_ROOT = Path.cwd().resolve()
        ORIGINAL_OPEN = builtins.open

        def _guard_path(path):
            resolved = (SANDBOX_ROOT / path).resolve() if not Path(path).is_absolute() else Path(path).resolve()
            if SANDBOX_ROOT not in (resolved, *resolved.parents):
                raise PermissionError(f"Access outside sandbox is blocked: {path}")
            return resolved

        def safe_open(file, mode="r", *args, **kwargs):
            return ORIGINAL_OPEN(_guard_path(file), mode, *args, **kwargs)

        builtins.open = safe_open

        PUBLIC_FILES = {
            "polymer_image": "polymer_image.png",
            "hic_contact_map": "hic_contact_map.png",
            "evidence_panel": "evidence_panel.png",
        }

        runpy.run_path(str(SANDBOX_ROOT / "analysis_code.py"), run_name="__main__")
        """
    )


def assert_python_tool_payload_is_safe(payload: Mapping[str, object]) -> None:
    """Raise if a Python tool payload exposes evaluator-only metadata keys."""
    unsafe = set(payload) - SAFE_PYTHON_TOOL_KEYS
    if unsafe:
        raise AssertionError(f"Unexpected Python tool payload keys: {sorted(unsafe)}")
    forbidden = ("true_label", "latent", "reliability", "recommended_action", "source_index")
    text = json.dumps(payload, sort_keys=True).lower()
    if any(fragment in text for fragment in forbidden):
        raise AssertionError("Python tool payload appears to leak hidden metadata")
