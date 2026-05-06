"""Claude client wrapper for the V3 agent benchmark.

Real API calls are intentionally gated by an explicit confirmation flag. Mock
mode is the default path used by tests and local smoke runs.
"""

from __future__ import annotations

import json
import os
import copy
import time
from dataclasses import dataclass
from typing import Dict, Mapping, Optional

from v3_agent_dataset import V3AgentSample
from v3_visual_tools import V3VisualToolbox


DEFAULT_CLAUDE_MODEL = "claude-sonnet-4-6"
INPUT_CONDITIONS = ("image_only", "hic_only", "both_modalities")
PROMPT_POLICIES = ("baseline", "evidence_separation")
TOOL_LEVELS = ("visual", "scientific", "python")
MODEL_CHAINS: Dict[str, tuple[str, ...]] = {
    "claude-haiku-4-5": ("claude-haiku-4-5", "claude-haiku-4-5-20251001"),
    "claude-sonnet-4-6": ("claude-sonnet-4-6",),
    "claude-opus-4-7": ("claude-opus-4-7",),
}
CLAUDE_HIGH_MODEL_SET = ("claude-haiku-4-5", "claude-sonnet-4-6", "claude-opus-4-7")


AGENT_SYSTEM_PROMPT = """You are evaluating paired experimental modalities from one synthetic polymer sample.
You may inspect only the provided visual evidence. Do not assume access to hidden labels.
Return exactly one JSON object with these fields:
predicted_label: normal or cancer
label_confidence: number from 0 to 1
contradiction_status: none, weak, strong, or unclear
trusted_evidence: image, hic, both, neither, or unclear
recommended_action: use_both, use_image, use_hic, or abstain
rationale: one concise sentence grounded in visible evidence.
"""


EVIDENCE_SEPARATION_SYSTEM_PROMPT = """You are evaluating experimental modalities from one synthetic polymer sample.
You may inspect only the provided visual evidence. Do not assume access to hidden labels.

First judge the polymer image and Hi-C contact map as separate evidence streams, then reconcile them.
For both_modalities inputs:
- assign image_predicted_label from the polymer image alone
- assign hic_predicted_label from the Hi-C contact map alone
- set modality_agreement to agree, weak_disagreement, strong_disagreement, or unclear
- use contradiction_status=strong when one modality supports normal and the other supports cancer
- use contradiction_status=weak when they point in the same label direction but visible evidence quality or structure conflicts
- use contradiction_status=none only when the modalities are visually compatible
- use recommended_action=use_both only when the modalities are compatible enough to fuse
- when they disagree, choose use_image, use_hic, or abstain based on which visible evidence is clearer
For unimodal inputs, mark the missing modality as not_visible, set modality_agreement to not_applicable, and set contradiction_status to unclear.

Return exactly one JSON object with these fields:
image_evidence: one concise sentence, or not_visible
hic_evidence: one concise sentence, or not_visible
image_predicted_label: normal, cancer, unclear, or not_visible
hic_predicted_label: normal, cancer, unclear, or not_visible
modality_agreement: agree, weak_disagreement, strong_disagreement, unclear, or not_applicable
predicted_label: normal or cancer
label_confidence: number from 0 to 1
contradiction_status: none, weak, strong, or unclear
trusted_evidence: image, hic, both, neither, or unclear
recommended_action: use_both, use_image, use_hic, or abstain
rationale: one concise sentence grounded in the separated visible evidence.
"""


SYSTEM_PROMPTS = {
    "baseline": AGENT_SYSTEM_PROMPT,
    "evidence_separation": EVIDENCE_SEPARATION_SYSTEM_PROMPT,
}


def system_prompt_for_policy(prompt_policy: str) -> str:
    """Return the system prompt for one benchmark prompt policy."""
    if prompt_policy not in SYSTEM_PROMPTS:
        raise ValueError(f"prompt_policy must be one of {PROMPT_POLICIES}")
    return SYSTEM_PROMPTS[prompt_policy]


VISUAL_TOOL_DEFINITIONS = [
    {
        "name": "load_sample_panel",
        "description": "Load the side-by-side polymer image and Hi-C contact-map panel for a sample.",
        "input_schema": {
            "type": "object",
            "properties": {"sample_id": {"type": "string"}},
            "required": ["sample_id"],
        },
    },
    {
        "name": "load_modality_image",
        "description": "Load one raw modality image for a sample. modality must be image or hic.",
        "input_schema": {
            "type": "object",
            "properties": {
                "sample_id": {"type": "string"},
                "modality": {"type": "string", "enum": ["image", "hic"]},
            },
            "required": ["sample_id", "modality"],
        },
    },
    {
        "name": "crop_modality_image",
        "description": "Crop one raw modality image using pixel coordinates.",
        "input_schema": {
            "type": "object",
            "properties": {
                "sample_id": {"type": "string"},
                "modality": {"type": "string", "enum": ["image", "hic"]},
                "x0": {"type": "integer"},
                "y0": {"type": "integer"},
                "x1": {"type": "integer"},
                "y1": {"type": "integer"},
            },
            "required": ["sample_id", "modality", "x0", "y0", "x1", "y1"],
        },
    },
]


SCIENTIFIC_TOOL_DEFINITIONS = [
    {
        "name": "measure_image_features",
        "description": "Compute fixed public morphology features from the polymer image PNG.",
        "input_schema": {
            "type": "object",
            "properties": {"sample_id": {"type": "string"}},
            "required": ["sample_id"],
        },
    },
    {
        "name": "measure_hic_features",
        "description": "Compute fixed public structural features from the Hi-C contact-map PNG.",
        "input_schema": {
            "type": "object",
            "properties": {"sample_id": {"type": "string"}},
            "required": ["sample_id"],
        },
    },
    {
        "name": "compare_modalities",
        "description": "Compare fixed public image and Hi-C feature summaries with a non-label heuristic discrepancy score.",
        "input_schema": {
            "type": "object",
            "properties": {"sample_id": {"type": "string"}},
            "required": ["sample_id"],
        },
    },
    {
        "name": "generate_feature_report",
        "description": "Return one fixed public JSON report with polymer image features, Hi-C features, and their discrepancy summary.",
        "input_schema": {
            "type": "object",
            "properties": {"sample_id": {"type": "string"}},
            "required": ["sample_id"],
        },
    },
]


PYTHON_TOOL_DEFINITION = {
    "name": "run_python_analysis",
    "description": (
        "Run model-written Python over copied public sample files only. "
        "Available local filenames depend on input_condition: polymer_image.png, "
        "hic_contact_map.png, and evidence_panel.png for both_modalities."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "sample_id": {"type": "string"},
            "code": {
                "type": "string",
                "description": "Python code to execute. Print concise JSON or text summaries to stdout.",
            },
        },
        "required": ["sample_id", "code"],
    },
}


def _visual_tool_definitions_for_condition(input_condition: str) -> list[Dict[str, object]]:
    """Return visual tool schemas that cannot expose modalities outside the condition."""
    if input_condition == "both_modalities":
        return copy.deepcopy(VISUAL_TOOL_DEFINITIONS)
    if input_condition not in {"image_only", "hic_only"}:
        raise ValueError(f"Unknown input_condition: {input_condition}")
    modality = "image" if input_condition == "image_only" else "hic"
    return [
        {
            "name": "load_modality_image",
            "description": f"Load the raw {modality} modality image for a sample.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "sample_id": {"type": "string"},
                    "modality": {"type": "string", "enum": [modality]},
                },
                "required": ["sample_id", "modality"],
            },
        },
        {
            "name": "crop_modality_image",
            "description": f"Crop the raw {modality} modality image using pixel coordinates.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "sample_id": {"type": "string"},
                    "modality": {"type": "string", "enum": [modality]},
                    "x0": {"type": "integer"},
                    "y0": {"type": "integer"},
                    "x1": {"type": "integer"},
                    "y1": {"type": "integer"},
                },
                "required": ["sample_id", "modality", "x0", "y0", "x1", "y1"],
            },
        },
    ]


def _scientific_tool_definitions_for_condition(input_condition: str) -> list[Dict[str, object]]:
    """Return scientific tool schemas compatible with the input condition."""
    if input_condition == "both_modalities":
        return copy.deepcopy(SCIENTIFIC_TOOL_DEFINITIONS)
    if input_condition == "image_only":
        return [copy.deepcopy(SCIENTIFIC_TOOL_DEFINITIONS[0])]
    if input_condition == "hic_only":
        return [copy.deepcopy(SCIENTIFIC_TOOL_DEFINITIONS[1])]
    raise ValueError(f"Unknown input_condition: {input_condition}")


def visual_tool_definitions(input_condition: str, tool_level: str = "visual") -> list[Dict[str, object]]:
    """Return tool schemas that cannot expose evidence outside the condition."""
    if tool_level not in TOOL_LEVELS:
        raise ValueError(f"tool_level must be one of {TOOL_LEVELS}")
    tools = _visual_tool_definitions_for_condition(input_condition)
    if tool_level in {"scientific", "python"}:
        tools.extend(_scientific_tool_definitions_for_condition(input_condition))
    if tool_level == "python":
        tools.append(copy.deepcopy(PYTHON_TOOL_DEFINITION))
    return tools


def _validate_tool_allowed(input_condition: str, tool_level: str, name: str, arguments: Mapping[str, object]) -> None:
    """Prevent a real API tool call from crossing input-condition boundaries."""
    allowed_names = {str(tool["name"]) for tool in visual_tool_definitions(input_condition, tool_level=tool_level)}
    if name not in allowed_names:
        raise ValueError(f"{name} is not allowed for {input_condition} at tool_level={tool_level}")
    if input_condition == "both_modalities":
        return
    allowed_modality = "image" if input_condition == "image_only" else "hic"
    if name == "load_sample_panel":
        raise ValueError(f"{name} is not allowed for {input_condition}")
    modality = str(arguments.get("modality", allowed_modality))
    if modality != allowed_modality:
        raise ValueError(f"modality {modality!r} is not allowed for {input_condition}")


@dataclass(frozen=True)
class ClaudeRunConfig:
    """Configuration for one Claude benchmark call."""

    model: str = DEFAULT_CLAUDE_MODEL
    max_tokens: int = 4096
    temperature: float = 0.0
    thinking: str = "high"
    thinking_budget_tokens: int = 2048
    max_tool_rounds: int = 4
    real_api: bool = False
    confirm_api_call: bool = False
    input_condition: str = "both_modalities"
    prompt_policy: str = "baseline"
    tool_level: str = "visual"


@dataclass(frozen=True)
class ClaudeRunResult:
    """Normalized raw result from one model call."""

    raw_output: str
    latency_seconds: float
    input_tokens: int = 0
    output_tokens: int = 0
    tool_call_count: int = 0
    stop_reason: str = "mock"
    resolved_model: str = ""


def mock_claude_response(
    sample: V3AgentSample,
    model: str,
    input_condition: str = "both_modalities",
    prompt_policy: str = "baseline",
    tool_level: str = "visual",
) -> ClaudeRunResult:
    """Produce deterministic mock output for local tests and smoke runs."""
    confidence = 0.74 if sample.true_contradiction_status == "none" else 0.61
    if input_condition == "image_only":
        contradiction_status = "unclear"
        trusted_evidence = "image"
        recommended_action = "use_image"
    elif input_condition == "hic_only":
        contradiction_status = "unclear"
        trusted_evidence = "hic"
        recommended_action = "use_hic"
    else:
        contradiction_status = sample.true_contradiction_status
        trusted_evidence = {
            "use_both": "both",
            "use_image": "image",
            "use_hic": "hic",
            "abstain": "neither",
        }[sample.recommended_action]
        recommended_action = sample.recommended_action
    payload = {
        "predicted_label": sample.true_label_name,
        "label_confidence": confidence,
        "contradiction_status": contradiction_status,
        "trusted_evidence": trusted_evidence,
        "recommended_action": recommended_action,
        "rationale": f"Mock {model} response based on {input_condition} {tool_level} evidence.",
    }
    if prompt_policy == "evidence_separation":
        if input_condition == "both_modalities":
            modality_agreement = {
                "none": "agree",
                "weak": "weak_disagreement",
                "strong": "strong_disagreement",
            }.get(sample.true_contradiction_status, "unclear")
            opposite_label = "normal" if sample.true_label_name == "cancer" else "cancer"
            if sample.true_contradiction_status == "none":
                image_label = sample.true_label_name
                hic_label = sample.true_label_name
            elif sample.recommended_action == "use_hic":
                image_label = opposite_label
                hic_label = sample.true_label_name
            else:
                image_label = sample.true_label_name
                hic_label = opposite_label
        else:
            modality_agreement = "not_applicable"
            image_label = sample.true_label_name if input_condition != "hic_only" else "not_visible"
            hic_label = sample.true_label_name if input_condition != "image_only" else "not_visible"
        payload.update(
            {
                "image_evidence": "mock separated image evidence" if input_condition != "hic_only" else "not_visible",
                "hic_evidence": "mock separated Hi-C evidence" if input_condition != "image_only" else "not_visible",
                "image_predicted_label": image_label,
                "hic_predicted_label": hic_label,
                "modality_agreement": modality_agreement,
            }
        )
    return ClaudeRunResult(
        raw_output=json.dumps(payload, sort_keys=True),
        latency_seconds=0.0,
        input_tokens=0,
        output_tokens=0,
        tool_call_count=0,
        stop_reason="mock",
        resolved_model=model,
    )


def _image_block(payload: Mapping[str, object]) -> Dict[str, object]:
    """Convert a visual-tool payload into a Claude image content block."""
    return {
        "type": "image",
        "source": {
            "type": "base64",
            "media_type": str(payload["media_type"]),
            "data": str(payload["image_base64"]),
        },
    }


def _tool_result_block(tool_use, payload: Mapping[str, object]) -> Dict[str, object]:
    """Convert a local tool payload into a Claude tool_result block."""
    if "image_base64" in payload:
        content = [
            {
                "type": "text",
                "text": (
                    f"Visual tool result: {payload['modality']} image, "
                    f"{payload['width']}x{payload['height']} pixels."
                ),
            },
            _image_block(payload),
        ]
    else:
        content = [
            {
                "type": "text",
                "text": "Scientific tool result JSON:\n" + json.dumps(payload, sort_keys=True),
            }
        ]
    return {
        "type": "tool_result",
        "tool_use_id": getattr(tool_use, "id"),
        "content": content,
    }


def _initial_user_content(
    sample: V3AgentSample,
    toolbox: V3VisualToolbox,
    input_condition: str,
    tool_level: str,
) -> list[Dict[str, object]]:
    """Build the first user turn with the sample panel image."""
    scientific_text = (
        "At tool_level=scientific, fixed public measurement tools are available: "
        "measure_image_features, measure_hic_features, compare_modalities, and generate_feature_report. "
        "These tools compute non-label quantitative summaries from visible/public pixels only. "
        "Before returning final JSON, use generate_feature_report for both_modalities, "
        "measure_image_features for image_only, or measure_hic_features for hic_only. "
        if tool_level == "scientific"
        else ""
    )
    python_text = (
        "At tool_level=python, you may use visual tools, fixed scientific tools, and run_python_analysis. "
        "Use run_python_analysis before returning final JSON to write your own analysis over copied public files only. "
        "Available filenames inside Python are polymer_image.png, hic_contact_map.png, and evidence_panel.png for both_modalities; "
        "only the visible modality file is copied for unimodal inputs. "
        "Do not try to access repo files, manifests, labels, hidden metadata, network resources, or parent directories. "
        if tool_level == "python"
        else ""
    )
    tool_text = scientific_text + python_text
    if input_condition == "image_only":
        visual_payload = toolbox.load_modality_image(sample.sample_id, "image")
        condition_text = (
            "You are given only the polymer image for this sample. "
            "Classify this sample as normal or cancer from this modality alone. "
            "Set contradiction_status to unclear because the paired Hi-C modality is not visible. "
            "Set recommended_action to use_image unless the visual evidence is insufficient, then abstain."
        )
    elif input_condition == "hic_only":
        visual_payload = toolbox.load_modality_image(sample.sample_id, "hic")
        condition_text = (
            "You are given only the Hi-C contact map for this sample. "
            "Classify this sample as normal or cancer from this modality alone. "
            "Set contradiction_status to unclear because the paired polymer image modality is not visible. "
            "Set recommended_action to use_hic unless the visual evidence is insufficient, then abstain."
        )
    elif input_condition == "both_modalities":
        visual_payload = toolbox.load_sample_panel(sample.sample_id)
        condition_text = (
            "Classify this sample as normal or cancer and reconcile the polymer image with the Hi-C contact map. "
            "Use available tools if you need to inspect, crop, or measure the public evidence."
        )
    else:
        raise ValueError(f"Unknown input_condition: {input_condition}")
    return [
        _image_block(visual_payload),
        {
            "type": "text",
            "text": (
                f"Sample ID: {sample.sample_id}\n"
                f"Input condition: {input_condition}\n"
                f"Tool level: {tool_level}\n"
                f"{tool_text}{condition_text} Return only JSON."
            ),
        },
    ]


def _thinking_payload(config: ClaudeRunConfig) -> Optional[Dict[str, object]]:
    """Map benchmark thinking setting into an Anthropic API payload."""
    if config.thinking == "none":
        return None
    return {"type": "enabled", "budget_tokens": int(config.thinking_budget_tokens)}


def _effective_temperature(config: ClaudeRunConfig) -> float:
    """Respect Anthropic's temperature constraint for extended thinking."""
    if config.thinking != "none":
        return 1.0
    return float(config.temperature)


def _supports_temperature(model: str) -> bool:
    """Return whether a Claude model accepts the temperature parameter."""
    return model != "claude-opus-4-7"


def _extract_text(content_blocks) -> str:
    """Collect text blocks from a Claude response."""
    texts = []
    for block in content_blocks:
        block_type = getattr(block, "type", None)
        if block_type == "text":
            texts.append(getattr(block, "text", ""))
    return "\n".join(texts).strip()


def _tool_uses_from_message(message) -> list[object]:
    """Collect tool-use blocks from a Claude response."""
    if message is None:
        return []
    return [
        block for block in message.content
        if getattr(block, "type", None) == "tool_use"
    ]


def _usage_counts(message) -> tuple[int, int]:
    """Read token usage from a Claude SDK message when available."""
    usage = getattr(message, "usage", None)
    if usage is None:
        return 0, 0
    return int(getattr(usage, "input_tokens", 0) or 0), int(getattr(usage, "output_tokens", 0) or 0)


def model_candidates(model: str) -> tuple[str, ...]:
    """Return the configured model fallback chain for one requested model."""
    return MODEL_CHAINS.get(model, (model,))


def _is_model_not_found_error(error: Exception) -> bool:
    """Detect model-name failures without treating transient errors as fallback-worthy."""
    class_name = error.__class__.__name__.lower()
    status_code = getattr(error, "status_code", None)
    text = str(error).lower()
    return (
        "notfound" in class_name
        or (
            status_code in {400, 404}
            and ("model" in text and ("not found" in text or "not_found" in text or "invalid" in text))
        )
    )


def run_claude_agent(
    sample: V3AgentSample,
    toolbox: V3VisualToolbox,
    config: ClaudeRunConfig,
) -> ClaudeRunResult:
    """Run one Claude benchmark sample in mock or explicitly confirmed real-API mode."""
    if config.input_condition not in INPUT_CONDITIONS:
        raise ValueError(f"input_condition must be one of {INPUT_CONDITIONS}")
    if config.prompt_policy not in PROMPT_POLICIES:
        raise ValueError(f"prompt_policy must be one of {PROMPT_POLICIES}")
    if config.tool_level not in TOOL_LEVELS:
        raise ValueError(f"tool_level must be one of {TOOL_LEVELS}")
    if not config.real_api:
        return mock_claude_response(
            sample,
            config.model,
            input_condition=config.input_condition,
            prompt_policy=config.prompt_policy,
            tool_level=config.tool_level,
        )
    if not config.confirm_api_call:
        raise RuntimeError("Real Claude API calls require --confirm-api-call.")
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise RuntimeError("ANTHROPIC_API_KEY is not set.")

    try:
        import anthropic
    except ImportError as exc:  # pragma: no cover - depends on optional package.
        raise RuntimeError("The anthropic package is required for real API calls.") from exc

    client = anthropic.Anthropic()
    messages = [{"role": "user", "content": _initial_user_content(sample, toolbox, config.input_condition, config.tool_level)}]
    kwargs: Dict[str, object] = {
        "model": config.model,
        "system": system_prompt_for_policy(config.prompt_policy),
        "messages": messages,
        "tools": visual_tool_definitions(config.input_condition, tool_level=config.tool_level),
        "max_tokens": config.max_tokens,
    }
    if _supports_temperature(config.model):
        kwargs["temperature"] = _effective_temperature(config)
    thinking = _thinking_payload(config)
    if thinking is not None:
        kwargs["thinking"] = thinking

    start_time = time.time()
    total_input_tokens = 0
    total_output_tokens = 0
    tool_call_count = 0
    last_message = None
    pending_tool_uses: list[object] = []

    resolved_model = config.model
    model_errors = []
    for candidate_model in model_candidates(config.model):
        resolved_model = candidate_model
        kwargs["model"] = candidate_model
        if _supports_temperature(candidate_model):
            kwargs["temperature"] = _effective_temperature(config)
        else:
            kwargs.pop("temperature", None)
        try:
            for _ in range(max(1, config.max_tool_rounds + 1)):
                last_message = client.messages.create(**kwargs)
                input_tokens, output_tokens = _usage_counts(last_message)
                total_input_tokens += input_tokens
                total_output_tokens += output_tokens

                tool_uses = _tool_uses_from_message(last_message)
                pending_tool_uses = tool_uses
                if not tool_uses:
                    break

                messages.append({"role": "assistant", "content": last_message.content})
                tool_results = []
                for tool_use in tool_uses:
                    tool_call_count += 1
                    tool_name = str(getattr(tool_use, "name"))
                    tool_arguments = dict(getattr(tool_use, "input") or {})
                    _validate_tool_allowed(config.input_condition, config.tool_level, tool_name, tool_arguments)
                    if tool_name == "run_python_analysis":
                        tool_arguments["_input_condition"] = config.input_condition
                    payload = toolbox.call_tool(
                        name=tool_name,
                        arguments=tool_arguments,
                    )
                    tool_results.append(_tool_result_block(tool_use, payload))
                messages.append({"role": "user", "content": tool_results})
                kwargs["messages"] = messages
            if pending_tool_uses:
                messages.append(
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": (
                                    "Tool budget is exhausted. Do not call any more tools. "
                                    "Use the evidence already gathered and return exactly one final JSON object "
                                    "matching the requested schema. Start with { and end with }. "
                                    "No markdown, no prose before JSON, no prose after JSON."
                                ),
                            }
                        ],
                    }
                )
                final_kwargs = dict(kwargs)
                final_kwargs["messages"] = messages
                final_kwargs.pop("tools", None)
                last_message = client.messages.create(**final_kwargs)
                input_tokens, output_tokens = _usage_counts(last_message)
                total_input_tokens += input_tokens
                total_output_tokens += output_tokens
            break
        except Exception as exc:  # pragma: no cover - requires real provider errors.
            if not _is_model_not_found_error(exc):
                raise
            model_errors.append(f"{candidate_model}: {exc}")
            last_message = None
            pending_tool_uses = []
            messages = [{"role": "user", "content": _initial_user_content(sample, toolbox, config.input_condition, config.tool_level)}]
            kwargs["messages"] = messages
            total_input_tokens = 0
            total_output_tokens = 0
            tool_call_count = 0
    else:
        raise RuntimeError("All Claude model candidates failed: " + " | ".join(model_errors))

    latency = time.time() - start_time
    raw_output = _extract_text(last_message.content if last_message is not None else [])
    return ClaudeRunResult(
        raw_output=raw_output,
        latency_seconds=float(latency),
        input_tokens=total_input_tokens,
        output_tokens=total_output_tokens,
        tool_call_count=tool_call_count,
        stop_reason=str(getattr(last_message, "stop_reason", "")) if last_message is not None else "unknown",
        resolved_model=resolved_model,
    )
