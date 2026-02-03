from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

from jinja2 import Template

from backend.llm.llm_handler import LLMError


@dataclass(frozen=True)
class InferencePromptSpec:
    system_instruction: str
    full_payload_template: str


@dataclass(frozen=True)
class RewritePromptSpec:
    system_instruction: str
    full_payload_template: str


@dataclass(frozen=True)
class RerankPromptSpec:
    system_instruction: str
    full_payload_template: str


@dataclass(frozen=True)
class SummaryPromptSpec:
    system_instruction: str


_CACHE: Dict[str, Tuple[float, Dict[str, Any]]] = {}


def _load_yaml(path: str) -> Dict[str, Any]:
    try:
        import yaml  # type: ignore
    except Exception as e:
        raise LLMError(
            provider="internal",
            kind="config",
            message=(
                "YAML prompt registry requires PyYAML. Install 'pyyaml' and restart the service. "
                f"Import error: {e}"
            ),
        )

    if not path or not os.path.exists(path):
        raise LLMError(
            provider="internal",
            kind="config",
            message=f"Prompt registry YAML file not found at path: {path}",
        )

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
    except Exception as e:
        raise LLMError(
            provider="internal",
            kind="config",
            message=f"Failed to read/parse prompt registry YAML at {path}: {e}",
        )

    if not isinstance(data, dict):
        raise LLMError(
            provider="internal",
            kind="config",
            message=f"Prompt registry YAML must be a mapping/object at top-level: {path}",
        )

    return data


def _get_cached_yaml(path: str) -> Dict[str, Any]:
    mtime = 0.0
    try:
        mtime = float(os.path.getmtime(path))
    except Exception:
        mtime = 0.0

    cached = _CACHE.get(path)
    if cached and cached[0] == mtime:
        return cached[1]

    data = _load_yaml(path)
    _CACHE[path] = (mtime, data)
    return data


def resolve_inference_prompt(*, registry_path: str, domain: Optional[str]) -> InferencePromptSpec:
    """Resolve stage-1 inference prompt spec.

    Rules:
    - Base prompt is global_defaults.inference
    - If domain is provided and domains.<domain>.inference.system_instruction exists, append it to the base system_instruction.
    - The full user payload template is taken from global_defaults.inference.user_messages[name=full_payload].template
      (domains may override this later, but not yet).

    Raises LLMError(kind="config") if required keys are missing.
    """

    reg = _get_cached_yaml(registry_path)

    try:
        base_inf = (reg.get("global_defaults") or {}).get("inference") or {}
    except Exception:
        base_inf = {}

    base_sys = str(base_inf.get("system_instruction") or "").strip()
    if not base_sys:
        raise LLMError(
            provider="internal",
            kind="config",
            message="Prompt registry missing required global_defaults.inference.system_instruction",
        )

    # Find full_payload template
    user_messages = base_inf.get("user_messages") or []
    full_payload_template = ""
    try:
        for item in user_messages:
            if isinstance(item, dict) and str(item.get("name") or "") == "full_payload":
                full_payload_template = str(item.get("template") or "")
                break
    except Exception:
        full_payload_template = ""

    if not full_payload_template.strip():
        raise LLMError(
            provider="internal",
            kind="config",
            message="Prompt registry missing required global_defaults.inference.user_messages[name=full_payload].template",
        )

    dom = (domain or "").strip()
    dom_sys = ""
    if dom:
        try:
            dom_inf = ((reg.get("domains") or {}).get(dom) or {}).get("inference") or {}
            dom_sys = str(dom_inf.get("system_instruction") or "").strip()
        except Exception:
            dom_sys = ""

    if dom_sys:
        system_instruction = base_sys + "\n\n" + dom_sys
    else:
        system_instruction = base_sys


    return InferencePromptSpec(
        system_instruction=system_instruction,
        full_payload_template=full_payload_template,
    )


def resolve_rewrite_prompt(*, registry_path: str, domain: Optional[str]) -> RewritePromptSpec:
    """Resolve query rewrite prompt spec.

    Rules:
    - Base prompt is global_defaults.rewrite
    - If domain is provided and domains.<domain>.rewrite.system_instruction exists, append it to the base system_instruction.
    - The full user payload template is taken from global_defaults.rewrite.user_messages[name=full_payload].template

    Raises LLMError(kind="config") if required keys are missing.
    """

    reg = _get_cached_yaml(registry_path)

    try:
        base_rw = (reg.get("global_defaults") or {}).get("rewrite") or {}
    except Exception:
        base_rw = {}

    base_sys = str(base_rw.get("system_instruction") or "").strip()
    if not base_sys:
        raise LLMError(
            provider="internal",
            kind="config",
            message="Prompt registry missing required global_defaults.rewrite.system_instruction",
        )

    user_messages = base_rw.get("user_messages") or []
    full_payload_template = ""
    try:
        for item in user_messages:
            if isinstance(item, dict) and str(item.get("name") or "") == "full_payload":
                full_payload_template = str(item.get("template") or "")
                break
    except Exception:
        full_payload_template = ""

    if not full_payload_template.strip():
        raise LLMError(
            provider="internal",
            kind="config",
            message="Prompt registry missing required global_defaults.rewrite.user_messages[name=full_payload].template",
        )

    dom = (domain or "").strip()
    dom_sys = ""
    if dom:
        try:
            dom_rw = ((reg.get("domains") or {}).get(dom) or {}).get("rewrite") or {}
            dom_sys = str(dom_rw.get("system_instruction") or "").strip()
        except Exception:
            dom_sys = ""

    if dom_sys:
        system_instruction = base_sys + "\n\n" + dom_sys
    else:
        system_instruction = base_sys


    return RewritePromptSpec(
        system_instruction=system_instruction,
        full_payload_template=full_payload_template,
    )


def resolve_rerank_prompt(*, registry_path: str, domain: Optional[str]) -> RerankPromptSpec:
    """Resolve rerank prompt spec.

    Rules:
    - Base prompt is global_defaults.rerank
    - If domain is provided and domains.<domain>.rerank.system_instruction exists, append it to the base system_instruction.
    - The full user payload template is taken from global_defaults.rerank.user_messages[name=full_payload].template
    """

    reg = _get_cached_yaml(registry_path)

    try:
        base_rr = (reg.get("global_defaults") or {}).get("rerank") or {}
    except Exception:
        base_rr = {}

    base_sys = str(base_rr.get("system_instruction") or "").strip()
    if not base_sys:
        raise LLMError(
            provider="internal",
            kind="config",
            message="Prompt registry missing required global_defaults.rerank.system_instruction",
        )

    user_messages = base_rr.get("user_messages") or []
    full_payload_template = ""
    try:
        for item in user_messages:
            if isinstance(item, dict) and str(item.get("name") or "") == "full_payload":
                full_payload_template = str(item.get("template") or "")
                break
    except Exception:
        full_payload_template = ""

    if not full_payload_template.strip():
        raise LLMError(
            provider="internal",
            kind="config",
            message="Prompt registry missing required global_defaults.rerank.user_messages[name=full_payload].template",
        )

    dom = (domain or "").strip()
    dom_sys = ""
    if dom:
        try:
            dom_rr = ((reg.get("domains") or {}).get(dom) or {}).get("rerank") or {}
            dom_sys = str(dom_rr.get("system_instruction") or "").strip()
        except Exception:
            dom_sys = ""

    if dom_sys:
        system_instruction = base_sys + "\n\n" + dom_sys
    else:
        system_instruction = base_sys


    return RerankPromptSpec(
        system_instruction=system_instruction,
        full_payload_template=full_payload_template,
    )


def resolve_summary_prompt(*, registry_path: str, domain: Optional[str]) -> SummaryPromptSpec:
    """Resolve summarizer prompt spec.

    Rules:
    - Base prompt is global_defaults.summary
    - If domain is provided and domains.<domain>.summary.system_instruction exists, append it to the base system_instruction.
    """

    reg = _get_cached_yaml(registry_path)

    try:
        base_sum = (reg.get("global_defaults") or {}).get("summary") or {}
    except Exception:
        base_sum = {}

    base_sys = str(base_sum.get("system_instruction") or "").strip()
    if not base_sys:
        raise LLMError(
            provider="internal",
            kind="config",
            message="Prompt registry missing required global_defaults.summary.system_instruction",
        )

    dom = (domain or "").strip()
    dom_sys = ""
    if dom:
        try:
            dom_sum = ((reg.get("domains") or {}).get(dom) or {}).get("summary") or {}
            dom_sys = str(dom_sum.get("system_instruction") or "").strip()
        except Exception:
            dom_sys = ""

    if dom_sys:
        system_instruction = base_sys + "\n\n" + dom_sys
    else:
        system_instruction = base_sys


    return SummaryPromptSpec(system_instruction=system_instruction)


def render_full_payload(template_str: str, *, variables: Dict[str, Any]) -> str:
    try:
        tmpl = Template(template_str)
        rendered = tmpl.render(**(variables or {}))
        out = str(rendered or "").strip()
        return out
    except Exception as e:
        raise LLMError(
            provider="internal",
            kind="config",
            message=f"Failed to render inference prompt template: {e}",
        )
