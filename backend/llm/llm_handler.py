from openai import OpenAI
import openai
import os
import logging
from typing import Any, Dict, Optional, Iterator, List, TypedDict

logger = logging.getLogger(__name__)

# ModelSpec import: support both `backend/llm/ModelSpec.py` and `llm/ModelSpec.py` layouts.
from backend.llm.ModelSpec import ModelSpec

# Model registry for parameter mapping
try:  # pragma: no cover
    from backend.llm import model_registry as _model_registry
except Exception as e:  # pragma: no cover
    # CRITICAL: Model registry import failed - this is a system failure
    import sys
    print(f"CRITICAL ERROR: Failed to import model_registry: {e}", file=sys.stderr)
    print("CRITICAL ERROR: LLM Handler cannot function without model registry. Aborting.", file=sys.stderr)
    sys.exit(1)  # Critical failure - abort execution

class LLMError(Exception):
    """Structured error raised for provider or configuration failures.

    This intentionally subclasses Exception so existing callers that catch
    generic exceptions remain backward-compatible. Callers that want richer
    handling can catch LLMError explicitly and inspect its attributes.
    """

    def __init__(
        self,
        *,
        provider: str,
        model: Optional[str] = None,
        kind: str = "llm_error",
        code: Optional[Any] = None,
        message: str = "",
        retry_after: Optional[float] = None,
    ) -> None:
        self.provider = (provider or "").lower()
        self.model = model
        self.kind = kind  # e.g. "rate_limit", "auth", "config", "model_not_found", "request"
        self.code = code
        self.retry_after = retry_after
        super().__init__(message or kind)


class LLMUsage(TypedDict, total=False):
    """Normalized token usage for LLM calls.

    Canonical fields (provider-agnostic):
      - input_tokens:      visible input tokens (includes cached)
      - cached_tokens:     subset of input_tokens served from cache (display-only)
      - output_tokens:     visible output tokens (includes reasoning, if any)
      - reasoning_tokens:  subset of output_tokens used for reasoning (display-only)
      - completion_tokens: output_tokens - reasoning_tokens (display-only)
      - total_tokens:      input_tokens + output_tokens
    """

    input_tokens: int
    cached_tokens: int
    output_tokens: int
    reasoning_tokens: int
    completion_tokens: int
    total_tokens: int


class LLMToolCall(TypedDict, total=False):
    """Normalized tool/function call descriptor.

    Mirrors the shape produced by `extract_tool_calls` in chat_manager.
    """

    name: str
    args: Any
    id: Optional[str]


class LLMResult(TypedDict, total=False):
    """Model-agnostic, normalized result for non-streaming LLM calls.

    This wraps a provider-native response (in `raw`) while exposing
    convenient, stable fields for callers. Streaming calls remain
    event-based and are summarized into this shape only after completion
    (when needed).
    """

    # Identity / metadata
    provider: str
    model: str
    id: Optional[str]
    created_at: Optional[float]

    # Content & reasoning
    text: str
    reasoning: Optional[str]
    role: str

    # Status / finish
    status: str
    finish_reason: Optional[str]

    # Token usage
    usage: LLMUsage

    # Normalized tool calls
    tool_calls: List[LLMToolCall]

    # Provider-native raw response object/dict
    raw: Any


# --- OpenAI-compatible response/event shims for non-OpenAI providers ---
class AdapterResponse:
    """Minimal OpenAI Responses-compatible response shim.

    Your existing tooling can read `output_text` (and optionally `usage`).
    `adapter_response` holds any adapter-level Responses-like wrapper (e.g.,
    a Gemini wrapper that exposes `output_text` / `output` / `usage`).
    `model_response` preserves the provider-native response for debugging.
    `finish_reason` indicates why the response stopped (e.g., 'stop', 'length', 'content_filter').
    """

    def __init__(
        self,
        *,
        output_text: str,
        model: str,
        usage: Optional[Dict[str, int]] = None,
        adapter_response: Any | None = None,
        model_response: Any | None = None,
        finish_reason: Optional[str] = None,
    ):
        self.output_text = output_text
        self.model = model
        self.usage = usage
        # New fields:
        #   - adapter_response: adapter-level Responses-like wrapper, if any
        #     (e.g., _GeminiResponsesWrapper for Gemini).
        #   - model_response: the provider-native model response, if available
        #     (e.g., adapter_response.model_response for Gemini); otherwise falls back to
        #     the adapter_response or output surface itself.
        self.adapter_response = adapter_response
        try:
            if model_response is not None:
                self.model_response = model_response
            elif adapter_response is not None and hasattr(adapter_response, "model_response"):
                # Common case for Gemini: adapter_response is a _GeminiResponsesWrapper
                # that already exposes a provider-native model_response.
                self.model_response = getattr(adapter_response, "model_response")
            else:
                # Fallback: treat adapter_response (or None) as the best-effort model_response.
                self.model_response = model_response or adapter_response
        except Exception:
            self.model_response = None
        self.finish_reason = finish_reason


class AdapterEvent:
    """Minimal OpenAI Responses-compatible streaming event shim."""

    def __init__(self, event_type: str, delta: Optional[str] = None):
        self.type = event_type
        self.delta = delta


class _ResponsesFacade:
    """Drop-in facade that mimics `client.responses.create(...)`.

    This lets existing code paths switch from `get_client_fn()` to `llm_handler` without
    refactoring every `.responses.create(...)` call.
    """

    def __init__(self, handler: "LLMHandler"):
        self._handler = handler

    def create(self, **kwargs: Any):
        # Preserve the Responses API call signature.
        stream = bool(kwargs.pop("stream", False))
        return self._handler.create(provider="openai", stream=stream, **kwargs)


class _EmbeddingsFacade:
    """Drop-in facade that mimics `client.embeddings.create(...)`.

    This is additive and allows call sites to route embeddings through llm_handler
    without changing any existing public APIs.
    """

    def __init__(self, handler: "LLMHandler"):
        self._handler = handler

    def create(self, **kwargs: Any):
        """Facade entrypoint for embeddings.

        Typical usage:

            llm_handler.embeddings.create(
                model="text-embedding-3-small",
                input="some text",
            )

        Provider defaults to "openai" if not supplied, matching LLMHandler.create.
        """
        return self._handler.create_embedding(**kwargs)


class LLMHandler:
    """Routing/adapter layer for multiple LLM providers.

    Supports both:
      1) create(provider=..., model=..., input=..., **kwargs)
      2) create(spec=ModelSpec(...), input=..., **overrides)

    Precedence (most specific wins):
      call-time overrides > spec fields/extra > defaults (none)

    NOTE: This handler must remain stateless with respect to model choice (no per-request state
    stored on the instance) so multiple users/sessions can safely use different models.
    """

    def __init__(self, *, openai_client=None, gemini_client=None):
        self._openai = openai_client
        self._gemini = gemini_client
        # Facade for compatibility with existing `client.responses.create(...)` call sites.
        self.responses = _ResponsesFacade(self)
        # New additive facade for embeddings; does not affect existing behavior.
        self.embeddings = _EmbeddingsFacade(self)

    def _get_reasoning_parameter(self, model: str) -> tuple[str, Any] | None:
        """Get reasoning parameter name and default from model registry."""
        if _model_registry is None:
            return None
        
        # First try to find by registry key (most reliable)
        if model in _model_registry.REGISTRY:
            model_info = _model_registry.REGISTRY[model]
            # Check if reasoning_parameter exists before accessing
            if hasattr(model_info, 'reasoning_parameter'):
                return model_info.reasoning_parameter
            else:
                return None
        
        # Second try to find by exact model name
        for model_info in _model_registry.REGISTRY.values():
            if model_info.model == model:
                # Check if reasoning_parameter exists before accessing
                if hasattr(model_info, 'reasoning_parameter'):
                    return model_info.reasoning_parameter
                else:
                    return None
        
        # Third try to find by registry key pattern (provider:model format)
        for key, model_info in _model_registry.REGISTRY.items():
            if key.endswith(f":{model}") or key == model:
                # Check if reasoning_parameter exists before accessing
                if hasattr(model_info, 'reasoning_parameter'):
                    return model_info.reasoning_parameter
                else:
                    return None
        
        return None

    def _get_max_tokens_parameter(self, model: str) -> str:
        """Get the correct max_tokens parameter name for a model from registry."""
        if _model_registry is None:
            # If registry is not available, return default
            return "max_output_tokens"
        
        # First try to find by registry key (most reliable)
        if model in _model_registry.REGISTRY:
            return _model_registry.REGISTRY[model].max_tokens_parameter
        
        # Second try to find by exact model name
        for model_info in _model_registry.REGISTRY.values():
            if model_info.model == model:
                return model_info.max_tokens_parameter
        
        # Third try to find by registry key pattern (provider:model format)
        for key, model_info in _model_registry.REGISTRY.items():
            if key.endswith(f":{model}") or key == model:
                return model_info.max_tokens_parameter
        
        # If model not found in registry, return default
        return "max_output_tokens"

    def _apply_max_tokens_parameter(self, model: str, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """Map the generic `max_output_tokens` (and legacy `max_tokens`) into the model's
        registry-defined max-tokens parameter name.

        Contract:
        - Call sites pass a model-agnostic `max_output_tokens`.
        - The registry defines the provider/model-specific parameter name via `max_tokens_parameter`.
        - If the registry parameter is already present, do not override it.
        - Preserve backward compatibility: if callers used `max_tokens`, map it as well.
        """
        if not isinstance(kwargs, dict) or not kwargs:
            return kwargs

        param = self._get_max_tokens_parameter(model)
        out = kwargs.copy()

        # If caller already supplied the registry-specific parameter, honor it.
        # IMPORTANT: If the registry-specific parameter IS `max_output_tokens`, do not pop it.
        if param in out and out.get(param) is not None:
            if param != "max_output_tokens":
                out.pop("max_output_tokens", None)
            out.pop("max_tokens", None)
            return out

        # Prefer the model-agnostic field.
        if "max_output_tokens" in out and out.get("max_output_tokens") is not None:
            if param == "max_output_tokens":
                # Already using the correct parameter name; keep the value in place.
                out.pop("max_tokens", None)
                return out
            out[param] = out.pop("max_output_tokens")
            out.pop("max_tokens", None)
            return out

        # Fall back to legacy `max_tokens` if provided.
        if "max_tokens" in out and out.get("max_tokens") is not None:
            out[param] = out.pop("max_tokens")
            return out

        return out

    def _lookup_model_info_from_registry(self, model: str) -> Any | None:
        """Resolve registry ModelInfo for a model identifier.

        Accepts either a registry key (preferred) or a provider-native model name.
        Returns None if the registry is unavailable or no entry matches.
        """
        if not model or _model_registry is None:
            return None
        try:
            # 1) Direct registry key match
            info = _model_registry.REGISTRY.get(model)
            if info is not None:
                return info
            # 2) Provider-native model name match
            for candidate in _model_registry.REGISTRY.values():
                if getattr(candidate, "model", None) == model:
                    return candidate
        except Exception:
            return None
        return None

    def _extract_effort_map(self, model_info: Any, spec: Any | None) -> Dict[str, float] | None:
        """Get effort->ratio map.

        Priority:
          1) model_info.thinking_tax.ratios / effort_ratios
          2) ModelSpec-provided map (effort_map / thinking_tax / extras/extra)

        Returns a dict like {"none": 0.0, "minimal": 0.0, "low": 0.25, "medium": 0.50, "high": 0.80}
        or None if no usable map is found.
        """
        # 1) Registry map (preferred)
        thinking_tax = getattr(model_info, "thinking_tax", None)
        if isinstance(thinking_tax, dict) and thinking_tax:
            effort_map = thinking_tax.get("effort_map")
            if isinstance(effort_map, dict) and effort_map:
                out: Dict[str, float] = {}
                for k, v in effort_map.items():
                    key = str(k).strip().lower()
                    if isinstance(v, dict):
                        rr = v.get("reserve_ratio")
                    else:
                        rr = v
                    try:
                        out[key] = float(rr)
                    except Exception:
                        continue
                if out:
                    return out

        # 2) Spec fallback (best-effort)
        if spec is None:
            return None

        def _get_from_mapping(obj: Any) -> Any:
            if not isinstance(obj, dict):
                return None
            return obj.get("ratios") or obj.get("effort_ratios") or obj

        # Direct attributes
        spec_map = getattr(spec, "effort_map", None) or getattr(spec, "thinking_tax", None)
        candidate = _get_from_mapping(spec_map)
        if isinstance(candidate, dict) and candidate:
            try:
                return {str(k).strip().lower(): float(v) for k, v in candidate.items()}
            except Exception:
                pass

        # spec.extra / spec.extras
        for attr_name in ("extra", "extras"):
            maybe = getattr(spec, attr_name, None)
            if isinstance(maybe, dict) and maybe:
                candidate = _get_from_mapping(maybe.get("effort_map") or maybe.get("thinking_tax"))
                if isinstance(candidate, dict) and candidate:
                    try:
                        return {str(k).strip().lower(): float(v) for k, v in candidate.items()}
                    except Exception:
                        pass

        # spec.to_kwargs()
        if hasattr(spec, "to_kwargs"):
            try:
                d = spec.to_kwargs() or {}
                if isinstance(d, dict) and d:
                    candidate = _get_from_mapping(d.get("effort_map") or d.get("thinking_tax"))
                    if isinstance(candidate, dict) and candidate:
                        return {str(k).strip().lower(): float(v) for k, v in candidate.items()}
            except Exception:
                pass

        return None

    def _normalize_effort_name(self, effort: Any) -> str:
        """Normalize reasoning effort labels to registry keys."""
        if effort is None:
            return "medium"
        eff = str(effort).strip().lower()
        if eff in ("min", "minimal"):
            return "minimal"
        if eff in ("none", "off", "0"):
            return "none"
        return eff or "medium"

    def _get_requested_effort_from_kwargs(self, model_info: Any, kwargs: Dict[str, Any]) -> Any:
        """Find the requested effort value from generic or model-specific fields."""
        if kwargs.get("reasoning_effort") is not None:
            return kwargs.get("reasoning_effort")

        # If reasoning was mapped to a provider-specific parameter, read it.
        try:
            rp = getattr(model_info, "reasoning_parameter", None)
            if isinstance(rp, tuple) and len(rp) >= 1:
                rp_name = rp[0]
                if rp_name and kwargs.get(rp_name) is not None:
                    return kwargs.get(rp_name)
        except Exception:
            pass

        return None

    def _apply_gemini_thinking_tax(self, model: str, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """Inflate Gemini max token limit to account for hidden thinking tokens.

        Contract:
        - Call sites stay model-agnostic and pass a token limit.
        - The registry defines how Gemini models should inflate that limit (effort_map/ratios).
        - If no map is found, no changes are made.
        """
        if not isinstance(kwargs, dict) or not kwargs:
            return kwargs

        # Keep the spec for effort_map fallback, but never forward it.
        spec = kwargs.get("__model_spec")
        clean_kwargs = kwargs.copy()
        clean_kwargs.pop("__model_spec", None)

        model_info = self._lookup_model_info_from_registry(model)
        if model_info is None:
            return clean_kwargs
        if getattr(model_info, "provider", None) != "gemini":
            return clean_kwargs

        # Only apply thinking-tax inflation when the registry explicitly marks the model
        # as supporting reasoning/thinking. This prevents accidental inflation for models
        # that accept token limits but do not expose thinking knobs.
        try:
            caps = getattr(model_info, "capabilities", {}) or {}
            if not bool(caps.get("reasoning_effort", False)):
                return clean_kwargs
        except Exception:
            return clean_kwargs

        effort_map = self._extract_effort_map(model_info, spec)
        if not isinstance(effort_map, dict) or not effort_map:
            return clean_kwargs

        # Determine which max-tokens parameter we should be inflating for this model.
        max_param_name = getattr(model_info, "max_tokens_parameter", None) or self._get_max_tokens_parameter(model)
        base_max = clean_kwargs.get(max_param_name)
        if base_max is None:
            return clean_kwargs
        try:
            base_max_i = int(base_max)
        except Exception:
            return clean_kwargs
        if base_max_i <= 0:
            return clean_kwargs

        requested_effort = self._get_requested_effort_from_kwargs(model_info, clean_kwargs)
        effort_name = self._normalize_effort_name(requested_effort)

        ratio = effort_map.get(effort_name)
        if ratio is None:
            ratio = effort_map.get("medium", 0.0)
        try:
            ratio_f = float(ratio)
        except Exception:
            ratio_f = 0.0

        # Ratio is an *additional* fraction of the visible max.
        # Example: base=300, ratio=0.5 => send 450.
        if ratio_f <= 0.0:
            return clean_kwargs
        
        inflated_max = int(round(base_max_i * (1.0 + ratio_f)))
        if inflated_max <= base_max_i:
            return clean_kwargs
        
        # DEBUG: Show thinking tax calculation
        logger.debug(f"[GEMINI THINKING TAX] model={model} base_max={base_max_i} effort={effort_name} ratio={ratio_f} inflated_max={inflated_max}")
        
        clean_kwargs[max_param_name] = inflated_max
        return clean_kwargs

    def _inject_gemini_thinking_config(self, model: str, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """Inject Gemini thinking configuration into `extra_body` based on the model registry.

        This keeps provider-specific payload shaping in one place, driven by registry config.

        Rules:
        - Only applies to registry models where provider == "gemini".
        - Uses `model_info.thinking_tax.kind` to decide whether to send:
            - thinking_budget ("budget" kind)
            - thinking_level ("level" kind)
        - For "budget": if budget is None, do nothing; if budget == -1 treat as dynamic (omit).
        - For "level": if a param_map exists, map generic effort labels (none/low/medium/high) to the
          provider knob values; otherwise use the provided value as-is.
        - Preserves any existing `extra_body` content by merging.
        """
        if not isinstance(kwargs, dict) or not kwargs:
            return kwargs

        model_info = self._lookup_model_info_from_registry(model)
        if model_info is None or getattr(model_info, "provider", None) != "gemini":
            return kwargs

        # Only inject Gemini thinking_config when the registry says this model supports reasoning.
        # This prevents injecting thinking_config for models/adapters that don't expose reasoning knobs.
        try:
            caps = getattr(model_info, "capabilities", {}) or {}
            if not bool(caps.get("reasoning_effort", False)):
                return kwargs
        except Exception:
            return kwargs

        thinking_tax = getattr(model_info, "thinking_tax", None)
        if not isinstance(thinking_tax, dict) or not thinking_tax:
            return kwargs

        kind = thinking_tax.get("kind")
        if kind not in ("budget", "level"):
            return kwargs

        # Determine the model-specific reasoning knob name (e.g., thinking_budget / thinking_level)
        rp = getattr(model_info, "reasoning_parameter", None)
        rp_name = None
        rp_default = None

        # Only proceed when the registry provides a tuple/list like (param_name, default_value).
        if isinstance(rp, (tuple, list)) and len(rp) >= 1:
            rp_name = rp[0]
            rp_default = rp[1] if len(rp) > 1 else None
        else:
            # No usable reasoning_parameter => no reasoning support for this model.
            return kwargs

        if not rp_name:
            return kwargs

        # Pull the requested knob value (already mapped by _map_reasoning_parameter_with_default)
        requested_value = kwargs.get(rp_name)
        if requested_value is None:
            requested_value = rp_default

        # Nothing to inject if we still don't have a value
        if requested_value is None:
            return kwargs

        # Build/merge extra_body payload
        out = dict(kwargs)
        existing = out.pop("extra_body", {})
        inner: Dict[str, Any] = {}
        if isinstance(existing, dict):
            inner = existing.get("extra_body", existing)

        inner.setdefault("google", {})

        if kind == "budget":
            # -1 means "dynamic" thinking (omit the knob so provider decides)
            try:
                budget = int(requested_value)
            except Exception:
                budget = None
            if budget is None or budget == -1:
                return out  # nothing injected
            tc = inner["google"].get("thinking_config")
            if not isinstance(tc, dict):
                tc = {}
            tc["thinking_budget"] = budget
            inner["google"]["thinking_config"] = tc

        elif kind == "level":
            level = requested_value
            # Optionally map generic effort labels to provider-specific knob values
            param_map = thinking_tax.get("param_map")
            if isinstance(param_map, dict):
                key = str(level).strip().lower()
                level = param_map.get(key, level)
            tc = inner["google"].get("thinking_config")
            if not isinstance(tc, dict):
                tc = {}
            tc["thinking_level"] = str(level)
            inner["google"]["thinking_config"] = tc

        # Double-wrap so OpenAI Python SDK merges `{ "extra_body": ... }` into request body.
        out["extra_body"] = {"extra_body": inner}
        
        # DEBUG: Show thinking config injection
        logger.debug(f"[GEMINI THINKING CONFIG] model={model} kind={kind} rp_name={rp_name} requested={requested_value} final_config={inner}")
        
        # Remove the reasoning parameter from top-level since it's now in extra_body
        if rp_name in out:
            out.pop(rp_name)
        
        return out

    def _get_model_capabilities(self, model: str) -> Dict[str, Any]:
        """Get model capabilities from registry."""
        if _model_registry is None:
            # If registry is not available, return empty capabilities (no filtering)
            return {}
        
        # First try to find by registry key (most reliable)
        if model in _model_registry.REGISTRY:
            return _model_registry.REGISTRY[model].capabilities
        
        # Second try to find by exact model name
        for model_info in _model_registry.REGISTRY.values():
            if model_info.model == model:
                return model_info.capabilities
        
        # Third try to find by registry key pattern (provider:model format)
        for key, model_info in _model_registry.REGISTRY.items():
            if key.endswith(f":{model}") or key == model:
                return model_info.capabilities
        
        # If model not found in registry, return empty capabilities (no filtering)
        return {}

    def _resolve_model_name(self, model_identifier: str) -> str:
        """Resolve a model identifier to the provider-native model name.

        Accepts either:
          - Registry key (e.g., "openai:best", "gemini:fast")
          - Provider-native model name (e.g., "gpt-4o-mini", "models/gemini-2.5-flash-lite")

        If the registry is unavailable or the identifier is not found, returns `model` unchanged.
        """
        if not model_identifier:    # If no model identifier is provided, return it unchanged.
            return model_identifier
        if _model_registry is None:  # If the model registry is not available, return the model identifier unchanged.
            return model_identifier
        try:
            if model_identifier in _model_registry.REGISTRY:
                return _model_registry.REGISTRY[model_identifier].model
        except Exception:
            return model_identifier
        return model_identifier

    def _filter_kwargs_by_capabilities(self, model: str, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """Filter kwargs based on model capabilities from registry.

        Back-compat rule:
        - Always allow token limit parameters.
        - If the registry doesn't know a param, pass it through (assume supported).
        - Only drop a param if the registry explicitly marks it unsupported.

        This avoids accidentally stripping core params like temperature/top_p when
        capabilities is sparse.
        """
        capabilities = self._get_model_capabilities(model)
        if not isinstance(kwargs, dict) or not kwargs:
            return kwargs

        # Adapter-only kwarg: `debug_thoughts` is a Gemini-only control flag.
        # Never forward it to non-Gemini providers (e.g., OpenAI Responses API).
        try:
            model_info = self._lookup_model_info_from_registry(model)
            if model_info is not None and getattr(model_info, "provider", None) != "gemini":
                if "debug_thoughts" in kwargs:
                    kwargs = dict(kwargs)
                    kwargs.pop("debug_thoughts", None)
        except Exception:
            # Best-effort only; never break filtering.
            pass

        # Always allow token limit parameters (fundamental to all models)
        token_params = {"max_output_tokens", "max_tokens", "max_completion_tokens"}

        filtered: Dict[str, Any] = {}
        for param, value in kwargs.items():
            if param in token_params:
                filtered[param] = value
                continue

            # If registry explicitly declares a param as supported/unsupported, honor it.
            if param in capabilities:
                if bool(capabilities.get(param)):
                    filtered[param] = value
                continue

            # Unknown param: keep it for backward compatibility.
            filtered[param] = value

        return filtered

    def _sanitize_tools_for_gemini_adapter(self, tools: Any) -> Any:
        """Return a Gemini-friendly tools list.

        - Accepts either flattened or nested OpenAI-style tool specs.
        - Ensures the outgoing format is nested {"type":"function","function":{...}}.
        - Recursively strips JSON schema keys that commonly trigger 400s in
          Gemini adapters (e.g., "default", "additionalProperties").
        """
        if not isinstance(tools, list):
            return tools

        def _clean_schema(obj: Any) -> Any:
            if not isinstance(obj, dict):
                return obj
            forbidden = {"default", "additionalProperties", "$schema", "title"}
            out: Dict[str, Any] = {}
            for k, v in obj.items():
                if k in forbidden:
                    continue
                out[k] = _clean_schema(v)
            return out

        cleaned: list[Any] = []
        for tool in tools:
            if not isinstance(tool, dict):
                continue

            func = tool.get("function")
            if not isinstance(func, dict):
                # Flattened form: treat the dict itself as the function spec.
                func = tool

            name = func.get("name")
            if not name:
                # Skip tools without a usable name; better to ignore than fail hard.
                continue

            params = func.get("parameters") or {"type": "object", "properties": {}}
            params = _clean_schema(params)

            cleaned.append(
                {
                    "type": tool.get("type", "function"),
                    "function": {
                        "name": name,
                        "description": func.get("description", ""),
                        "parameters": params,
                    },
                }
            )

        return cleaned or tools

    def _prepare_gemini_adapter_kwargs(self, model: str, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare kwargs for the Gemini OpenAI-compatible adapter.

        This keeps Gemini-specific normalization in one place.
        Order:
          1) capability filtering
          2) reasoning parameter mapping/defaults
          3) thinking-tax token-cap inflation (if configured in registry/spec)
          4) tools schema sanitization (only if tools provided)
        """
        # Adapter-only flag: never forward to provider.
        raw_kwargs: Dict[str, Any] = dict(kwargs or {})
        debug_thoughts = bool(raw_kwargs.pop("debug_thoughts", False))

        filtered_kwargs = self._filter_kwargs_by_capabilities(model, raw_kwargs)
        mapped_kwargs = self._map_reasoning_parameter_with_default(model, filtered_kwargs)

        prepared_kwargs: Dict[str, Any] = dict(mapped_kwargs)

        try:
            prepared_kwargs = self._apply_gemini_thinking_tax(model, prepared_kwargs)
        except Exception:
            pass

        try:
            prepared_kwargs = self._inject_gemini_thinking_config(model, prepared_kwargs)
        except Exception:
            pass

        # If requested, include Gemini thoughts ONLY when the registry says this model supports reasoning.
        if debug_thoughts:
            try:
                model_info = self._lookup_model_info_from_registry(model)
                if model_info is not None and getattr(model_info, "provider", None) == "gemini":
                    caps = getattr(model_info, "capabilities", {}) or {}
                    if bool(caps.get("reasoning_effort", False)):
                        # Merge into existing extra_body payload, preserving existing thinking_config keys.
                        existing = prepared_kwargs.get("extra_body")
                        inner: Dict[str, Any] = {}
                        if isinstance(existing, dict):
                            inner = existing.get("extra_body", existing)
                        inner.setdefault("google", {})
                        tc = inner["google"].get("thinking_config")
                        if not isinstance(tc, dict):
                            tc = {}
                        tc["include_thoughts"] = True
                        inner["google"]["thinking_config"] = tc
                        prepared_kwargs["extra_body"] = {"extra_body": inner}
            except Exception:
                # Best-effort only; never break the call.
                pass

        if "tools" in prepared_kwargs:
            try:
                prepared_kwargs["tools"] = self._sanitize_tools_for_gemini_adapter(prepared_kwargs["tools"])
            except Exception:
                pass

        return prepared_kwargs

    def _extract_finish_reason(self, resp: Any) -> Optional[str]:
        try:
            choices = getattr(resp, "choices", None)
            if isinstance(choices, list) and choices:
                return getattr(choices[0], "finish_reason", None)
        except Exception:
            pass
        return None

    def build_llm_result_from_response(self, resp: Any, *, provider: str = "openai") -> LLMResult:
        """Build an LLMResult from a non-streaming OpenAI Responses API response.

        This helper is internal for now and does not change existing call sites.
        It assumes `resp` is a completed Responses-style object (not a stream).
        """

        # ---- Identity / metadata ----
        try:
            model = getattr(resp, "model", None) or ""
        except Exception:
            model = ""

        try:
            rid = getattr(resp, "id", None)
        except Exception:
            rid = None

        try:
            created_at = getattr(resp, "created_at", None)
        except Exception:
            created_at = None

        try:
            status = getattr(resp, "status", None) or "completed"
        except Exception:
            status = "completed"

        # Prefer existing helper when it yields a value; fall back to incomplete_details.reason.
        finish_reason: Optional[str] = self._extract_finish_reason(resp)
        if finish_reason is None:
            try:
                incomplete = getattr(resp, "incomplete_details", None)
                if incomplete is not None:
                    finish_reason = getattr(incomplete, "reason", None)
            except Exception:
                finish_reason = None

        # ---- Usage mapping (canonical LLMUsage) ----
        usage_block = getattr(resp, "usage", None)
        usage: LLMUsage = {
            "input_tokens": 0,
            "cached_tokens": 0,
            "output_tokens": 0,
            "reasoning_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }

        if usage_block is not None:
            try:
                # Helper to read fields from either dict or object
                def _uget(obj: Any, name: str) -> Any:
                    if obj is None:
                        return None
                    if isinstance(obj, dict):
                        return obj.get(name)
                    return getattr(obj, name, None)

                # Provider-specific normalization
                completion_tokens = None  # explicit init; derived later if needed
                if provider == "openai":
                    # OpenAI Responses semantics
                    input_tokens = _uget(usage_block, "input_tokens")
                    if input_tokens is None:
                        input_tokens = _uget(usage_block, "prompt_tokens")

                    output_tokens = _uget(usage_block, "output_tokens")
                    if output_tokens is None:
                        output_tokens = _uget(usage_block, "completion_tokens")

                    total_tokens = _uget(usage_block, "total_tokens")

                    details_in = _uget(usage_block, "prompt_tokens_details") or _uget(usage_block, "input_tokens_details")
                    cached_tokens = (
                        getattr(details_in, "cached_tokens", None)
                        if details_in is not None and not isinstance(details_in, dict)
                        else (details_in.get("cached_tokens") if isinstance(details_in, dict) else None)
                    )

                    details_out = _uget(usage_block, "output_tokens_details")
                    reasoning_tokens = (
                        getattr(details_out, "reasoning_tokens", None)
                        if details_out is not None and not isinstance(details_out, dict)
                        else (details_out.get("reasoning_tokens") if isinstance(details_out, dict) else None)
                    )

                elif provider == "gemini":
                    # Gemini OpenAI-adapter semantics
                    prompt_tokens = _uget(usage_block, "prompt_tokens")
                    input_tokens = prompt_tokens if prompt_tokens is not None else _uget(usage_block, "input_tokens")

                    completion_tokens_raw = _uget(usage_block, "completion_tokens")
                    total_tokens = _uget(usage_block, "total_tokens")

                    # No explicit output_tokens/cached/reasoning in current adapter usage
                    output_tokens = None
                    cached_tokens = None
                    reasoning_tokens = None

                    if total_tokens is not None and input_tokens is not None:
                        try:
                            output_tokens = int(total_tokens) - int(input_tokens)
                        except Exception:
                            output_tokens = None

                    # Derive reasoning_tokens as the difference between output_tokens and completion_tokens
                    # when both are available; otherwise leave at 0.
                    if output_tokens is not None and completion_tokens_raw is not None:
                        try:
                            reasoning_tokens = int(output_tokens) - int(completion_tokens_raw)
                            if reasoning_tokens < 0:
                                reasoning_tokens = 0
                        except Exception:
                            reasoning_tokens = None

                    # For completion_tokens field in usage, prefer completion_tokens_raw; we'll normalize later.
                    completion_tokens = completion_tokens_raw

                else:
                    # Unknown provider: leave usage at zeros for safety.
                    try:
                        logger.debug(f"[LLMResult] Unknown provider '{provider}' for usage normalization; leaving usage at zeros.")
                    except Exception:
                        pass
                    input_tokens = output_tokens = cached_tokens = reasoning_tokens = total_tokens = None
                    completion_tokens = None

                # --- Canonicalization for all branches ---
                # Default missing values to 0 and compute any derived fields.
                input_tokens = input_tokens if input_tokens is not None else 0
                output_tokens = output_tokens if output_tokens is not None else 0
                cached_tokens = cached_tokens if cached_tokens is not None else 0
                reasoning_tokens = reasoning_tokens if reasoning_tokens is not None else 0

                # total_tokens: prefer provider value, otherwise sum.
                if total_tokens is None:
                    total_tokens = (input_tokens or 0) + (output_tokens or 0)

                # If completion_tokens wasn't set in provider-specific logic, derive as output - reasoning.
                if completion_tokens is None:
                    try:
                        completion_tokens = int(output_tokens or 0) - int(reasoning_tokens or 0)
                        if completion_tokens < 0:
                            completion_tokens = 0
                    except Exception:
                        completion_tokens = 0

                # Coerce to ints into canonical usage.
                try:
                    usage["input_tokens"] = int(input_tokens or 0)
                except Exception:
                    usage["input_tokens"] = 0
                try:
                    usage["output_tokens"] = int(output_tokens or 0)
                except Exception:
                    usage["output_tokens"] = 0
                try:
                    usage["total_tokens"] = int(total_tokens or 0)
                except Exception:
                    usage["total_tokens"] = usage["input_tokens"] + usage["output_tokens"]
                try:
                    usage["cached_tokens"] = int(cached_tokens or 0)
                except Exception:
                    usage["cached_tokens"] = 0
                try:
                    usage["reasoning_tokens"] = int(reasoning_tokens or 0)
                except Exception:
                    usage["reasoning_tokens"] = 0
                try:
                    usage["completion_tokens"] = int(completion_tokens or 0)
                except Exception:
                    usage["completion_tokens"] = 0
            except Exception:
                # Best-effort only; on error keep the zero-initialized usage block.
                pass

        # ---- Text extraction ----
        # We do not import chat_manager helpers here; instead we apply a
        # minimal Responses-aware extraction similar to _extract_text_from_responses.
        def _safe_get(obj: Any, name: str) -> Any:
            try:
                if isinstance(obj, dict):
                    return obj.get(name)
                return getattr(obj, name, None)
            except Exception:
                return None

        text_candidates: list[str] = []

        # 1) Direct output_text field, if present
        try:
            ot = _safe_get(resp, "output_text")
            if isinstance(ot, str) and ot.strip():
                text_candidates.append(ot.strip())
        except Exception:
            pass

        # 2) Responses-style `output` list
        try:
            output = _safe_get(resp, "output")
            if isinstance(output, list):
                for item in output:
                    it_type = _safe_get(item, "type")
                    if it_type == "text":
                        txt = _safe_get(item, "text")
                        if isinstance(txt, str) and txt.strip():
                            text_candidates.append(txt.strip())
                            continue

                    # message-style: item.type == "message" with content list
                    content = _safe_get(item, "content")
                    if isinstance(content, list):
                        for c in content:
                            txt = _safe_get(c, "text")
                            if isinstance(txt, str) and txt.strip():
                                text_candidates.append(txt.strip())
        except Exception:
            pass

        # 3) Fallback: ChatCompletions-style choices[0].message.content
        if not text_candidates:
            try:
                choices = _safe_get(resp, "choices")
                if isinstance(choices, list) and choices:
                    c0 = choices[0]
                    msg = _safe_get(c0, "message")
                    content = _safe_get(msg, "content")
                    if isinstance(content, str) and content.strip():
                        text_candidates.append(content.strip())
            except Exception:
                pass

        best_text = ""
        for c in text_candidates:
            if isinstance(c, str) and len(c) > len(best_text):
                best_text = c

        # ---- Optional Gemini debug thoughts split (<thought>...</thought>) ----
        reasoning_text = None
        try:
            if provider == "gemini" and isinstance(best_text, str):
                start_tag = "<thought>"
                end_tag = "</thought>"
                start = best_text.find(start_tag)
                end = best_text.find(end_tag)
                if start != -1 and end != -1 and end > start:
                    inner = best_text[start + len(start_tag) : end]
                    after = best_text[end + len(end_tag) :].strip()
                    if inner and isinstance(inner, str):
                        reasoning_text = inner.strip()
                    if after:
                        best_text = after
        except Exception:
            # Best-effort only; if anything goes wrong, fall back to original best_text.
            pass

        # ---- Tool-call extraction (Responses-focused subset of extract_tool_calls) ----
        tool_calls: list[LLMToolCall] = []
        try:
            output = _safe_get(resp, "output")
            if isinstance(output, list):
                for item in output:
                    it_type = _safe_get(item, "type")

                    # Direct tool_* items
                    if it_type in ("function_call", "tool_use", "tool_call"):
                        name = _safe_get(item, "name")
                        args = _safe_get(item, "arguments")
                        cid = _safe_get(item, "call_id") or _safe_get(item, "id")
                        if isinstance(name, str) and name:
                            tool_calls.append({"name": name, "args": args, "id": cid})
                        continue

                    # message wrapper: tool calls in content
                    content = _safe_get(item, "content")
                    if isinstance(content, list):
                        for c in content:
                            c_type = _safe_get(c, "type")
                            if c_type == "tool_call":
                                name = _safe_get(c, "name")
                                args = _safe_get(c, "arguments")
                                cid = _safe_get(c, "id")
                                if isinstance(name, str) and name:
                                    tool_calls.append({"name": name, "args": args, "id": cid})
        except Exception:
            # Best-effort only; if anything goes wrong, leave tool_calls empty.
            pass

        # Legacy fallback: ChatCompletions-style choices[0].message.tool_calls
        # Only used when no canonical output-derived tool calls were found.
        try:
            if not tool_calls:
                choices = _safe_get(resp, "choices")
                if isinstance(choices, list) and choices:
                    c0 = choices[0]
                    msg = _safe_get(c0, "message")
                    tc_list = _safe_get(msg, "tool_calls")
                    if isinstance(tc_list, list):
                        for t in tc_list:
                            t_type = _safe_get(t, "type")
                            # OpenAI ChatCompletions: tool_calls[].type == "function"
                            if t_type not in ("function", "tool_call"):
                                continue
                            func = _safe_get(t, "function") or t
                            name = _safe_get(func, "name")
                            args = _safe_get(func, "arguments")
                            cid = _safe_get(t, "id")
                            if isinstance(name, str) and name:
                                tool_calls.append({"name": name, "args": args, "id": cid})
        except Exception:
            # Best-effort only; do not let legacy shapes break normalization.
            pass

        # ---- Assemble LLMResult ----
        result: LLMResult = {
            "provider": provider,
            "model": model,
            "id": rid,
            "created_at": created_at,
            "text": best_text or "",
            # For Gemini debug_thoughts, this may contain the <thought>...</thought> block; otherwise None.
            "reasoning": reasoning_text,
            "role": "assistant",
            "status": status,
            "finish_reason": finish_reason,
            "usage": usage,
            "tool_calls": tool_calls,
            "raw": resp,
        }

        return result

    def _convert_reasoning_value(self, model: str, value: Any) -> Any:
        """Convert reasoning_effort value to model-specific format."""
        if _model_registry is None:
            return value

        # Resolve model_info either by registry key or by provider-native model name.
        model_info = None
        if model in _model_registry.REGISTRY:
            model_info = _model_registry.REGISTRY[model]
        else:
            for info in _model_registry.REGISTRY.values():
                if info.model == model:
                    model_info = info
                    break

        if model_info is None:
            return value

        # Check if reasoning_parameter exists before accessing
        if hasattr(model_info, "reasoning_parameter") and model_info.reasoning_parameter is not None:
            param_name, default_value = model_info.reasoning_parameter
        else:
            # No reasoning_parameter field - no conversion needed
            return value

        # Convert based on default value type
        if isinstance(default_value, (int, float)):
            # Convert string to number for token-based models
            mapping = {"low": 1000, "medium": 2000, "high": 5000}
            return mapping.get(str(value).lower(), default_value)

        # Keep as string for models that expect string
        return str(value).lower()

    def _map_reasoning_parameter_with_default(self, model: str, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """Map reasoning_effort to model-specific parameter with defaults."""
        if _model_registry is None:
            return kwargs
        
        # First try to find by registry key (most reliable)
        if model in _model_registry.REGISTRY:
            model_info = _model_registry.REGISTRY[model]
            # Check if reasoning_parameter exists before accessing
            if hasattr(model_info, 'reasoning_parameter') and model_info.reasoning_parameter is not None:
                param_name, default_value = model_info.reasoning_parameter
                mapped_kwargs = kwargs.copy()
                
                # Handle reasoning_effort parameter
                if "reasoning_effort" in kwargs:
                    # Use passed value, convert if needed
                    reasoning_value = kwargs["reasoning_effort"]
                    converted_value = self._convert_reasoning_value(model, reasoning_value)
                    mapped_kwargs[param_name] = converted_value
                    # Only pop the original key if we mapped it to a different param name.
                    if param_name != "reasoning_effort":
                        mapped_kwargs.pop("reasoning_effort", None)
                    
                    # DEBUG: Show reasoning parameter mapping
                    logger.debug(
                        f"[REASONING MAP] model={model} param_name={param_name} reasoning_value={reasoning_value} converted_value={converted_value}"
                    )
                elif default_value is not None and model_info.capabilities.get("reasoning_effort", False):
                    # Use registry default when no value passed and capability is supported
                    mapped_kwargs[param_name] = default_value
                
                return mapped_kwargs
            else:
                # No reasoning_parameter field - return kwargs unchanged
                return kwargs

        # Second try to find by exact model name
        for model_info in _model_registry.REGISTRY.values():
            if model_info.model == model:
                # Check if reasoning_parameter exists before accessing
                if hasattr(model_info, 'reasoning_parameter') and model_info.reasoning_parameter is not None:
                    param_name, default_value = model_info.reasoning_parameter
                    mapped_kwargs = kwargs.copy()
                    
                    # Handle reasoning_effort parameter
                    if "reasoning_effort" in kwargs:
                        # Use passed value, convert if needed
                        reasoning_value = kwargs["reasoning_effort"]
                        converted_value = self._convert_reasoning_value(model, reasoning_value)
                        mapped_kwargs[param_name] = converted_value
                        # Only pop the original key if we mapped it to a different param name.
                        if param_name != "reasoning_effort":
                            mapped_kwargs.pop("reasoning_effort", None)
                    elif default_value is not None and model_info.capabilities.get("reasoning_effort", False):
                        # Use registry default when no value passed and capability is supported
                        mapped_kwargs[param_name] = default_value
                    
                    return mapped_kwargs
                else:
                    # No reasoning_parameter field - return kwargs unchanged
                    return kwargs

        # Third try to find by registry key pattern (provider:model format)
        for key, model_info in _model_registry.REGISTRY.items():
            if key.endswith(f":{model}") or key == model:
                # Check if reasoning_parameter exists before accessing
                if hasattr(model_info, 'reasoning_parameter') and model_info.reasoning_parameter is not None:
                    param_name, default_value = model_info.reasoning_parameter
                    mapped_kwargs = kwargs.copy()
                    
                    # Handle reasoning_effort parameter
                    if "reasoning_effort" in kwargs:
                        # Use passed value, convert if needed
                        reasoning_value = kwargs["reasoning_effort"]
                        converted_value = self._convert_reasoning_value(model, reasoning_value)
                        mapped_kwargs[param_name] = converted_value
                        # Only pop the original key if we mapped it to a different param name.
                        if param_name != "reasoning_effort":
                            mapped_kwargs.pop("reasoning_effort", None)
                    elif default_value is not None and model_info.capabilities.get("reasoning_effort", False):
                        # Use registry default when no value passed and capability is supported
                        mapped_kwargs[param_name] = default_value
                    
                    return mapped_kwargs
                else:
                    # No reasoning_parameter field - return kwargs unchanged
                    return kwargs

        return kwargs

    # ---- lazy client getters (singletons inside the singleton) ----
    def _get_openai(self):
        if self._openai is None:
            self._openai = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        return self._openai

    def _get_gemini(self):
        """Return a Gemini client.

        By default, build an OpenAI-compatible client pointed at the Gemini OpenAI adapter
        (chat.completions-compatible) using env vars:
          - GEMINI_API_KEY
          - GEMINI_OPENAI_BASE_URL

        You can also inject a pre-constructed client via `LLMHandler(gemini_client=...)`.
        """
        if self._gemini is None:
            api_key = os.getenv("GEMINI_API_KEY")
            base_url = os.getenv("GEMINI_OPENAI_BASE_URL")
            if not api_key:
                raise LLMError(
                    provider="gemini",
                    kind="config",
                    code="missing_api_key",
                    message="Gemini client not configured: GEMINI_API_KEY is not set",
                )
            if not base_url:
                raise LLMError(
                    provider="gemini",
                    kind="config",
                    code="missing_base_url",
                    message="Gemini client not configured: GEMINI_OPENAI_BASE_URL is not set",
                )

            try:
                self._gemini = OpenAI(api_key=api_key, base_url=base_url)
            except Exception as e:  # pragma: no cover
                raise LLMError(
                    provider="gemini",
                    kind="config",
                    code="missing_openai_package",
                    message="Gemini client not configured: openai package is not installed",
                ) from e

        return self._gemini

    # ---- public API ----
    def create(
        self,
        *,
        input: Any,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        spec: Optional[ModelSpec] = None,
        stream: bool = False,
        **kwargs: Any,
    ):
        if spec is not None:
            provider = spec.provider
            model = spec.model
            merged: Dict[str, Any] = {}
            merged.update(spec.to_kwargs())
            merged.update({k: v for k, v in kwargs.items() if v is not None})
            kwargs = merged
            kwargs["__model_spec"] = spec
        else:
            provider = (provider or "").strip().lower()
            if not provider:
                provider = "openai"
            if not model:
                raise ValueError("model is required when spec is not provided")

        # Centralize model-agnostic token limit mapping here so provider-specific
        # call paths don’t need to repeat it.
        kwargs = self._apply_max_tokens_parameter(model, kwargs)

        if provider == "openai":
            kwargs.pop("__model_spec", None)
            return self._openai_call(model=model, input=input, stream=stream, **kwargs)
        if provider == "gemini":
            #kwargs.pop("__model_spec", None) 
            # Keep __model_spec for Gemini so adapter-prep can use it as a fallback.
            return self._gemini_call(model=model, input=input, stream=stream, **kwargs)

        raise LLMError(
            provider=str(provider or "unknown"),
            model=model,
            kind="config",
            code="unsupported_provider",
            message=f"Provider '{provider}' not supported",
        )

    def create_embedding(
        self,
        *,
        input: Any,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        spec: Optional[ModelSpec] = None,
        **kwargs: Any,
    ):
        """Provider-agnostic embedding creation (additive API).

        Mirrors the `create` method but targets providers' embedding endpoints.
        Existing behavior of `create` and all call sites remain unchanged.
        """
        if spec is not None:
            provider = spec.provider
            model = spec.model
            merged: Dict[str, Any] = {}
            merged.update(spec.to_kwargs())
            merged.update({k: v for k, v in kwargs.items() if v is not None})
            kwargs = merged
        else:
            provider = (provider or "").strip().lower()
            if not provider:
                provider = "openai"
            if not model:
                raise ValueError("model is required when spec is not provided")

        if provider == "openai":
            return self._openai_embedding_call(model=model, input=input, **kwargs)
        if provider == "gemini":
            return self._gemini_embedding_call(model=model, input=input, **kwargs)

        raise LLMError(
            provider=str(provider or "unknown"),
            model=model,
            kind="config",
            code="unsupported_provider_embeddings",
            message=f"Provider '{provider}' not supported for embeddings",
        )

    # ---- provider calls ----
    def _openai_call(self, *, model: str, input: Any, stream: bool, **kwargs: Any):
        client = self._get_openai()
        resolved_model = self._resolve_model_name(model)
        # Filter kwargs by capabilities, then map reasoning parameters with defaults
        filtered_kwargs = self._filter_kwargs_by_capabilities(model, kwargs)
        mapped_kwargs = self._map_reasoning_parameter_with_default(model, filtered_kwargs)
        # OpenAI Responses API (SDK v2.8.1+): reasoning effort must be nested.
        # If callers pass `reasoning_effort`, translate to `reasoning={"effort": ...}`.
        if "reasoning_effort" in mapped_kwargs:
            reasoning_value = mapped_kwargs.pop("reasoning_effort")
            # Do not overwrite an explicit reasoning object if already provided.
            mapped_kwargs.setdefault("reasoning", {"effort": reasoning_value})

        # Determine which OpenAI endpoint to use for this model. For backward
        # compatibility, default to the Responses API when the registry is
        # unavailable or the model is unknown.
        endpoint = "responses"
        model_key_dbg = None
        try:
            model_info = self._lookup_model_info_from_registry(model)
            if model_info is not None:
                endpoint = getattr(model_info, "endpoint", "responses") or "responses"
                # Best-effort: capture the registry key used for this model for debugging.
                try:
                    model_key_dbg = getattr(model_info, "key", None)
                except Exception:
                    model_key_dbg = None
        except Exception:
            endpoint = "responses"

        # DEBUG: Log the resolved endpoint for observability.
        try:
            logger.debug(
                "[OPENAI CALL] model_identifier=%s registry_key=%s resolved_model=%s endpoint=%s stream=%s kwargs_keys=%s",
                str(model),
                str(model_key_dbg or ""),
                str(resolved_model),
                str(endpoint),
                bool(stream),
                list(mapped_kwargs.keys()),
            )
        except Exception:
            pass

        # --- Standard OpenAI Responses API path (existing behavior) ---
        if endpoint == "responses":
            try:
                logger.debug("[OPENAI CALL] Using Responses API for model=%s stream=%s", resolved_model, bool(stream))
            except Exception:
                pass
            try:
                response = client.responses.create(model=resolved_model, input=input, stream=stream, **mapped_kwargs)
                return response
            except Exception:
                # Preserve existing behavior (exception type) and re-raise.
                raise

        # --- Opt-in Chat Completions path for models configured with endpoint="chat_completions" ---
        if endpoint == "chat_completions":
            # Normalize tools into the nested {"type":"function","function":{...}}
            # shape expected by the Chat Completions endpoint. This mirrors
            # the schema used for Gemini adapter tools, but is applied only
            # for OpenAI chat.completions calls so the Responses API path
            # remains unchanged.
            if "tools" in mapped_kwargs:
                try:
                    mapped_kwargs = dict(mapped_kwargs)
                    mapped_kwargs["tools"] = self._sanitize_tools_for_gemini_adapter(mapped_kwargs["tools"])
                except Exception:
                    # Best-effort only; if normalization fails, fall back to
                    # the original tools payload and let the provider surface
                    # any validation error.
                    pass

            # Normalize the input into Chat Completions messages. If callers
            # already pass a list of messages, forward it as-is; otherwise,
            # wrap the input in a single user message.
            messages = input if isinstance(input, list) else [{"role": "user", "content": str(input)}]

            # Non-streaming Chat Completions: return the raw response object.
            if not stream:
                try:
                    logger.debug("[OPENAI CALL] Using Chat Completions (non-stream) for model=%s", resolved_model)
                except Exception:
                    pass
                try:
                    resp = client.chat.completions.create(
                        model=resolved_model,
                        messages=messages,
                        stream=False,
                        **mapped_kwargs,
                    )
                    # Return the raw ChatCompletion response. Downstream helpers
                    # (build_llm_result_from_response, _extract_text_from_responses,
                    # _extract_usage_from_responses, extract_tool_calls) already
                    # understand Chat Completions shapes and will normalize text,
                    # usage, and tool calls into LLMResult.
                    return resp
                except Exception:
                    raise

            # Streaming Chat Completions: adapt chunks into AdapterEvent stream,
            # mirroring the Gemini event surface so existing streaming
            # consumers continue to work unchanged.
            def event_gen() -> Iterator[AdapterEvent]:
                try:
                    try:
                        logger.debug("[OPENAI CALL] Using Chat Completions (stream) for model=%s", resolved_model)
                    except Exception:
                        pass
                    stream_obj = client.chat.completions.create(
                        model=resolved_model,
                        messages=messages,
                        stream=True,
                        **mapped_kwargs,
                    )
                except Exception as e:
                    # Surface provider errors as-is to preserve existing
                    # exception semantics for callers.
                    raise e

                for chunk in stream_obj:
                    try:
                        if not getattr(chunk, "choices", None):
                            continue
                        delta_obj = getattr(chunk.choices[0], "delta", None)
                        delta_text = getattr(delta_obj, "content", None)
                        if delta_text:
                            yield AdapterEvent("response.output_text.delta", delta=delta_text)
                    except Exception:
                        # Best-effort only; ignore malformed chunks and
                        # continue streaming.
                        continue
                yield AdapterEvent("response.output_text.done")

            return event_gen()

        # Fallback: if an unknown endpoint is configured, treat it as Responses
        # to avoid breaking callers; this is effectively the legacy behavior.
        try:
            response = client.responses.create(model=resolved_model, input=input, stream=stream, **mapped_kwargs)
            return response
        except Exception:
            raise

    def _openai_embedding_call(self, *, model: str, input: Any, **kwargs: Any):
        """OpenAI embedding call.

        Returns the raw OpenAI embeddings response object so callers can
        access `.data[0].embedding` and `.usage` if present.
        """
        client = self._get_openai()
        resolved_model = self._resolve_model_name(model)
        return client.embeddings.create(model=resolved_model, input=input, **kwargs)

    def _gemini_call(self, *, model: str, input: Any, stream: bool, skip_reasoning: bool = False, **kwargs: Any):
        """Gemini adapter that returns OpenAI-compatible response objects/events.

        Important: when `_get_gemini()` builds an OpenAI client pointed at the Gemini OpenAI adapter
        base URL, the supported surface is typically `chat.completions.create`. In that case we must
        NOT look for native Gemini methods like `generate_content`.

        If a native Gemini SDK client is injected, we keep a best-effort fallback to
        `generate_content` / `generateContent`.
        """
        client = self._get_gemini()
        resolved_model = self._resolve_model_name(model)
        # Prepare kwargs for the Gemini OpenAI-compatible adapter in one place.
        working_kwargs = self._prepare_gemini_adapter_kwargs(model, kwargs)

        # Determine which endpoint is configured for this Gemini model. For backward
        # compatibility, default to the chat_completions adapter when the registry
        # is unavailable or the model is unknown.
        endpoint = "chat_completions"
        try:
            model_info = self._lookup_model_info_from_registry(model)
            if model_info is not None:
                endpoint = getattr(model_info, "endpoint", "chat_completions") or "chat_completions"
        except Exception:
            endpoint = "chat_completions"

        # Keep max_output_tokens consistent - let each provider handle conversion internally
        # OpenAI chat completions will use model registry to determine correct parameter name

        # --- OpenAI-adapter path (chat.completions) when endpoint is chat_completions ---
        if endpoint == "chat_completions" and hasattr(client, "chat") and hasattr(getattr(client.chat, "completions"), "create"):
            create_fn = getattr(getattr(client.chat, "completions"), "create", None)
            if callable(create_fn):
                messages = input if isinstance(input, list) else [{"role": "user", "content": str(input)}]

                if not stream:
                    # Apply capability filtering, parameter mapping, and token limit conversion
                    call_kwargs = working_kwargs.copy()
                    # DEBUG: Log the final kwargs sent to the Gemini OpenAI-compatible endpoint.
                    try:
                        token_keys = ("max_output_tokens", "max_tokens", "max_completion_tokens")
                        # Include both generic and Gemini-specific reasoning knobs so we can confirm mapping.
                        reasoning_keys = ("reasoning_effort", "thinking_budget", "thinking_level")
                        debug_keys = token_keys + reasoning_keys
                        debug_part = {k: call_kwargs.get(k) for k in debug_keys if k in call_kwargs}
                        extra_body = call_kwargs.get("extra_body")
                        has_tools = bool(call_kwargs.get("tools"))
                        logger.debug(
                            "[GEMINI DEBUG] chat.completions.create model=%s stream=%s kwargs_subset=%s has_tools=%s has_extra_body=%s extra_body=%s",
                            resolved_model,
                            False,
                            debug_part,
                            has_tools,
                            bool(extra_body),
                            extra_body,
                        )
                    except Exception:
                        pass
                    # (Token parameter mapping now handled centrally in create())
                    try:
                        resp = create_fn(model=resolved_model, messages=messages, **call_kwargs)
                    except openai.RateLimitError as e:  # type: ignore[attr-defined]
                        # Map Gemini adapter rate limits into a structured LLMError
                        # so callers (e.g., handle_chat) can surface a clear message.
                        retry_after = None
                        try:
                            # Best-effort extraction from the error structure; safe if shape changes.
                            data = getattr(e, "response", None)
                            if data and hasattr(data, "json"):
                                j = data.json()
                                # Look for google.rpc.RetryInfo style hints if present.
                                # Fallback to None if parsing fails.
                                retry_after = None  # keep as placeholder; concrete parsing can be added later.
                        except Exception:
                            retry_after = None
                        raise LLMError(
                            provider="gemini",
                            model=model,
                            kind="rate_limit",
                            code="rate_limit",
                            message=str(e),
                            retry_after=retry_after,
                        ) from e
                    except openai.NotFoundError as e:  # type: ignore[attr-defined]
                        # Map Gemini adapter 404s (e.g., unknown model) into LLMError so callers
                        # can surface a clear, structured message to users.
                        raise LLMError(
                            provider="gemini",
                            model=model,
                            kind="model_not_found",
                            code="not_found",
                            message=str(e),
                            retry_after=None,
                        ) from e

                    text = ""
                    try:
                        if resp and getattr(resp, "choices", None):
                            choice0 = resp.choices[0]
                            msg = getattr(choice0, "message", None)
                            text = getattr(msg, "content", "") or ""
                    except Exception:
                        text = ""

                    usage_dict: Optional[Dict[str, int]] = None
                    usage = getattr(resp, "usage", None)
                    if usage is not None:
                        try:
                            if isinstance(usage, dict):
                                usage_dict = {
                                    "prompt_tokens": int(usage.get("prompt_tokens") or 0),
                                    "completion_tokens": int(usage.get("completion_tokens") or 0),
                                    "total_tokens": int(usage.get("total_tokens") or 0),
                                }
                            else:
                                usage_dict = {
                                    "prompt_tokens": int(getattr(usage, "prompt_tokens", 0) or 0),
                                    "completion_tokens": int(getattr(usage, "completion_tokens", 0) or 0),
                                    "total_tokens": int(getattr(usage, "total_tokens", 0) or 0),
                                }
                        except Exception:
                            usage_dict = None

                    wrapped = self._wrap_gemini_chatcompletion_as_responses(resp=resp, output_text=text, usage=usage)
                    return AdapterResponse(
                        output_text=text,
                        model=model,
                        usage=usage_dict,
                        adapter_response=wrapped,
                        model_response=getattr(wrapped, "model_response", None),
                        finish_reason=self._extract_finish_reason(resp),
                    )

                def event_gen() -> Iterator[AdapterEvent]:
                    # Apply capability filtering, parameter mapping, and token limit conversion
                    call_kwargs = working_kwargs.copy()
                    # DEBUG: Log the final kwargs sent to the Gemini OpenAI-compatible endpoint (streaming).
                    try:
                        token_keys = ("max_output_tokens", "max_tokens", "max_completion_tokens")
                        # Include both generic and Gemini-specific reasoning knobs so we can confirm mapping.
                        reasoning_keys = ("reasoning_effort", "thinking_budget", "thinking_level")
                        debug_keys = token_keys + reasoning_keys
                        debug_part = {k: call_kwargs.get(k) for k in debug_keys if k in call_kwargs}
                        extra_body = call_kwargs.get("extra_body")
                        has_tools = bool(call_kwargs.get("tools"))
                        logger.debug(
                            "[GEMINI DEBUG] chat.completions.create model=%s stream=%s kwargs_subset=%s has_tools=%s has_extra_body=%s extra_body=%s",
                            resolved_model,
                            True,
                            debug_part,
                            has_tools,
                            bool(extra_body),
                            extra_body,
                        )
                    except Exception:
                        pass
                    # (Token parameter mapping now handled centrally in create())
                    try:
                        stream_obj = create_fn(model=resolved_model, messages=messages, stream=True, **call_kwargs)
                    except openai.RateLimitError as e:  # type: ignore[attr-defined]
                        raise LLMError(
                            provider="gemini",
                            model=model,
                            kind="rate_limit",
                            code="rate_limit",
                            message=str(e),
                            retry_after=None,
                        ) from e
                    except openai.NotFoundError as e:  # type: ignore[attr-defined]
                        raise LLMError(
                            provider="gemini",
                            model=model,
                            kind="model_not_found",
                            code="not_found",
                            message=str(e),
                            retry_after=None,
                        ) from e
                    for chunk in stream_obj:
                        try:
                            if not getattr(chunk, "choices", None):
                                continue
                            delta_obj = getattr(chunk.choices[0], "delta", None)
                            delta_text = getattr(delta_obj, "content", None)
                            if delta_text:
                                yield AdapterEvent("response.output_text.delta", delta=delta_text)
                        except Exception:
                            continue
                    yield AdapterEvent("response.output_text.done")

                return event_gen()

        # --- Native Gemini SDK fallback (only if an injected client supports it) ---
        contents = input
        # Apply capability filtering and parameter mapping for native Gemini SDK
        final_kwargs = working_kwargs.copy()
        if not stream:
            if hasattr(client, "generate_content"):
                resp = client.generate_content(model=resolved_model, contents=contents, **final_kwargs)
            elif hasattr(client, "generateContent"):
                resp = client.generateContent(model=resolved_model, contents=contents, **final_kwargs)
            else:
                raise LLMError(
                    provider="gemini",
                    model=model,
                    kind="config",
                    code="no_known_method",
                    message="Gemini client does not expose chat.completions or a known native generate method",
                )
            text = ""
            if hasattr(resp, "text") and getattr(resp, "text", None):
                text = resp.text
            else:
                try:
                    candidates = getattr(resp, "candidates", None) or []
                    if candidates:
                        content = getattr(candidates[0], "content", None)
                        parts = getattr(content, "parts", None) if content is not None else None
                        if parts:
                            text = "".join([getattr(p, "text", "") or "" for p in parts])
                except Exception:
                    pass
            # Native SDK responses may not expose ChatCompletion `choices`; still wrap for Responses-like fields.
            wrapped = self._wrap_gemini_chatcompletion_as_responses(resp=resp, output_text=text or "", usage=getattr(resp, "usage", None))
            return AdapterResponse(
                output_text=text or "",
                model=model,
                usage=None,
                adapter_response=wrapped,
                model_response=getattr(wrapped, "model_response", None),
                finish_reason=None,
            )
        def native_event_gen() -> Iterator[AdapterEvent]:
            if hasattr(client, "generate_content_stream"):
                stream_iter = client.generate_content_stream(model=resolved_model, contents=contents, **final_kwargs)
                for chunk in stream_iter:
                    delta = chunk if isinstance(chunk, str) else getattr(chunk, "text", "") or ""
                    if delta:
                        yield AdapterEvent("response.output_text.delta", delta=delta)
                yield AdapterEvent("response.output_text.done")
                return
            if hasattr(client, "generate_content"):
                stream_iter = client.generate_content(model=resolved_model, contents=contents, stream=True, **final_kwargs)
                for chunk in stream_iter:
                    delta = chunk if isinstance(chunk, str) else getattr(chunk, "text", "") or ""
                    if delta:
                        yield AdapterEvent("response.output_text.delta", delta=delta)
                yield AdapterEvent("response.output_text.done")
                return
            raise LLMError(
                provider="gemini",
                model=model,
                kind="config",
                code="no_known_streaming_method",
                message="Gemini client does not expose a known streaming method",
            )
        return native_event_gen()

    def _gemini_embedding_call(self, *, model: str, input: Any, **kwargs: Any):
        """Gemini embedding call via the OpenAI-compatible adapter client.

        Assumes `_get_gemini()` returns an OpenAI-style client pointed at a
        Gemini adapter that exposes `client.embeddings.create(...)`.

        The typical model is `gemini-embedding-001`. Dimensions are required
        and must be explicitly provided by the caller.
        """
        if "dimensions" not in kwargs:
            raise LLMError(
                provider="gemini",
                model=model,
                kind="config",
                code="missing_dimensions",
                message="Gemini embeddings require explicit 'dimensions' parameter",
            )
        client = self._get_gemini()
        resolved_model = self._resolve_model_name(model)
        return client.embeddings.create(model=resolved_model, input=input, **kwargs)

    def _wrap_gemini_chatcompletion_as_responses(self, *, resp: Any, output_text: str, usage: Any = None) -> Any:
        """Wrap a Gemini ChatCompletion-style response to look like an OpenAI Responses object.

        === CANONICAL OUTPUTS (must use for all logic) ===
        - resp.output_text → final user-visible text (PRIMARY for text extraction)
        - resp.output → tool calls + structured content (PRIMARY for tool extraction)
        - resp.usage → token usage statistics
        - resp.raw → provider-native response (debug only)

        === NON-CANONICAL OUTPUTS (compatibility/debug only) ===
        - resp.choices → legacy ChatCompletions format, DO NOT USE for logic
        - resp.choices[].message.tool_calls → legacy tool format, IGNORED

        === STRUCTURE CREATED ===
        output: [
            {"type": "text", "text": "full response text"},           # Canonical text item
            {"type": "function_call", "name": "...", "arguments": "...", "call_id": "..."}  # Canonical tool items
        ]

        === EXTRACTION CONTRACT ===
        1. All tool calls must be read from resp.output (canonical)
        2. All text must be read from resp.output_text first, then resp.output
        3. Never use resp.choices for production logic
        4. Treat resp.choices as debug/compatibility only

        This wrapper ensures Gemini responses are compatible with OpenAI Responses API extraction functions.
        """
        
        # DEBUG: Check if truncation happens before wrapper
        logger.debug(f"[GEMINI WRAPPER] Input output_text length: {len(output_text)}")
        logger.debug(f"[GEMINI WRAPPER] Input output_text preview: '{output_text[:200]}...'")
        
        # Also check original response content
        finish_reason = None
        try:
            if resp and getattr(resp, "choices", None):
                choice0 = resp.choices[0]
                finish_reason = getattr(choice0, "finish_reason", None)
                msg = getattr(choice0, "message", None)
                original_content = getattr(msg, "content", "") or ""
                logger.debug(f"[GEMINI WRAPPER] Original choices[0].message.content length: {len(original_content)}")
                logger.debug(f"[GEMINI WRAPPER] Original choices[0].message.content preview: '{original_content[:200]}...'")
        except Exception as e:
            logger.debug(f"[GEMINI WRAPPER] Could not check original content: {e}")

        # Collect tool calls from ChatCompletion message.tool_calls (NON-CANONICAL source)
        tool_items: list[dict] = []
        try:
            choices = getattr(resp, "choices", None)
            if isinstance(choices, list) and choices:
                msg = getattr(choices[0], "message", None)
                tc = getattr(msg, "tool_calls", None)
                if isinstance(tc, list):
                    for t in tc:
                        ttype = getattr(t, "type", None)
                        if ttype != "function":
                            continue
                        func = getattr(t, "function", None)
                        name = getattr(func, "name", None) if func is not None else None
                        arguments = getattr(func, "arguments", None) if func is not None else None
                        call_id = getattr(t, "id", None)
                        if name:
                            tool_items.append({
                                "type": "function_call",
                                "name": name,
                                "arguments": arguments,
                                "call_id": call_id,
                            })
        except Exception:
            tool_items = []

        # Build canonical Responses API output structure (CANONICAL output)
        output: list[dict] = []
        
        # Add text as direct canonical item (not nested in message.content)
        if isinstance(output_text, str) and output_text:
            output.append({"type": "text", "text": output_text})
        
        # Add tool calls as direct canonical items
        output.extend(tool_items)

        class _GeminiResponsesWrapper:
            def __init__(self, *, output_text: str, output: list[dict], usage: Any, choices: Any, model_response: Any):
                # === CANONICAL FIELDS (must use for all logic) ===
                self.output_text = output_text      # Canonical: final text
                self.output = output                # Canonical: tools + content
                self.usage = usage                  # Canonical: tokens
                
                # === NON-CANONICAL FIELDS (compatibility/debug only) ===
                self.choices = choices              # Legacy: DO NOT USE for logic
                # Provider-native model response (e.g., Gemini ChatCompletion or native SDK response)
                self.model_response = model_response

            # Minimal dict-like compatibility for existing debug/test code.
            def get(self, name: str, default: Any = None) -> Any:
                return getattr(self, name, default)

        return _GeminiResponsesWrapper(
            output_text=output_text or "",      # Canonical text field
            output=output,                      # Canonical structured output
            usage=usage,                        # Canonical usage field
            choices=getattr(resp, "choices", None),  # Non-canonical compatibility
            model_response=resp,                # Provider-native debug
        )


# Singleton instance (optional)
llm_handler = LLMHandler()