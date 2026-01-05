from openai import OpenAI
import openai
import os
import logging
from typing import Any, Dict, Optional, Iterator

logger = logging.getLogger(__name__)

# ModelSpec import: support both `backend/llm/ModelSpec.py` and `llm/ModelSpec.py` layouts.
from backend.llm.ModelSpec import ModelSpec

# Model registry for parameter mapping
try:  # pragma: no cover
    from backend.llm import model_registry as _model_registry
except Exception:  # pragma: no cover
    _model_registry = None  # type: ignore

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


# --- OpenAI-compatible response/event shims for non-OpenAI providers ---
class AdapterResponse:
    """Minimal OpenAI Responses-compatible response shim.

    Your existing tooling can read `output_text` (and optionally `usage`).
    `raw` preserves the provider-native response for debugging.
    """

    def __init__(
        self,
        *,
        output_text: str,
        model: str,
        usage: Optional[Dict[str, int]] = None,
        raw: Any = None,
    ):
        self.output_text = output_text
        self.model = model
        self.usage = usage
        self.raw = raw


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
            return _model_registry.REGISTRY[model].reasoning_parameter
        
        # Second try to find by exact model name
        for model_info in _model_registry.REGISTRY.values():
            if model_info.model == model:
                return model_info.reasoning_parameter
        
        # Third try to find by registry key pattern (provider:model format)
        for key, model_info in _model_registry.REGISTRY.items():
            if key.endswith(f":{model}") or key == model:
                return model_info.reasoning_parameter
        
        return None

    def _get_max_tokens_parameter(self, model: str) -> str:
        """Get the correct max_tokens parameter name for a model from registry."""
        if _model_registry is None:
            # Fallback: check if it's an o1/o3 model
            if model and (model.startswith("o1") or model.startswith("o3")):
                return "max_completion_tokens"
            return "max_tokens"
        
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
        
        # Fallback to model name detection
        if model and (model.startswith("o1") or model.startswith("o3")):
            return "max_completion_tokens"
        return "max_tokens"

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

    def _resolve_model_name(self, model: str) -> str:
        """Resolve a model identifier to the provider-native model name.

        Accepts either:
          - Registry key (e.g., "openai:best", "gemini:fast")
          - Provider-native model name (e.g., "gpt-4o-mini", "models/gemini-2.5-flash-lite")

        If the registry is unavailable or the identifier is not found, returns `model` unchanged.
        """
        if not model:
            return model
        if _model_registry is None:
            return model
        try:
            if model in _model_registry.REGISTRY:
                return _model_registry.REGISTRY[model].model
        except Exception:
            return model
        return model

    def _filter_kwargs_by_capabilities(self, model: str, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """Filter kwargs based on model capabilities from registry."""
        capabilities = self._get_model_capabilities(model)
        filtered_kwargs = {}
        
        # Special case: always allow token limit parameters (fundamental to all models)
        token_params = {"max_output_tokens", "max_tokens", "max_completion_tokens"}
        
        for param, value in kwargs.items():
            if param in token_params or capabilities.get(param, False):
                filtered_kwargs[param] = value
        
        return filtered_kwargs

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

        param_name, default_value = model_info.reasoning_parameter

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

        # Second try to find by exact model name
        for model_info in _model_registry.REGISTRY.values():
            if model_info.model == model:
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

        # Third try to find by registry key pattern (provider:model format)
        for key, model_info in _model_registry.REGISTRY.items():
            if key.endswith(f":{model}") or key == model:
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
        else:
            provider = (provider or "").strip().lower()
            if not provider:
                provider = "openai"
            if not model:
                raise ValueError("model is required when spec is not provided")

        if provider == "openai":
            return self._openai_call(model=model, input=input, stream=stream, **kwargs)
        if provider == "gemini":
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
        try:
            return client.responses.create(model=resolved_model, input=input, stream=stream, **mapped_kwargs)
        except Exception as e:
            # Preserve existing behavior (exception type) while also exposing
            # a structured LLMError for call sites that wish to distinguish
            # provider/config failures. For now we re-raise the original to
            # avoid any breaking change.
            raise

    def _openai_embedding_call(self, *, model: str, input: Any, **kwargs: Any):
        """OpenAI embedding call.

        Returns the raw OpenAI embeddings response object so callers can
        access `.data[0].embedding` and `.usage` if present.
        """
        client = self._get_openai()
        resolved_model = self._resolve_model_name(model)
        return client.embeddings.create(model=resolved_model, input=input, **kwargs)

    def _gemini_reasoning_call(self, *, model: str, input: Any, stream: bool, **kwargs: Any):
        """Call Gemini using native reasoning API formats."""
        client = self._get_gemini()

        # Get reasoning parameter from registry
        reasoning_param = self._get_reasoning_parameter(model)
        param_name, default_value = reasoning_param or (None, None)

        # Use the central resolver for model name
        actual_model = self._resolve_model_name(model)

        # Determine which format to use based on parameter type
        if param_name == "thinking_budget":
            # Use responses.create with reasoning={"budget": tokens}
            return self._gemini_budget_reasoning_call(
                client=client,
                model=actual_model,
                input=input,
                stream=stream,
                reasoning_budget=kwargs.get(param_name, default_value),
                **{k: v for k, v in kwargs.items() if k != param_name}
            )
        elif param_name == "thinking_level":
            # Use chat.completions.create with extra_body
            return self._gemini_level_reasoning_call(
                client=client,
                model=actual_model,
                input=input,
                stream=stream,
                thinking_level=kwargs.get(param_name, default_value),
                **{k: v for k, v in kwargs.items() if k != param_name}
            )
        else:
            # Fallback to regular Gemini call. Skip reasoning routing to avoid recursion.
            return self._gemini_call(model=model, input=input, stream=stream, skip_reasoning=True, **kwargs)

    def _gemini_budget_reasoning_call(self, *, client, model: str, input: Any, stream: bool, reasoning_budget: Any, **kwargs: Any):
        """Gemini reasoning using chat.completions.create with extra_body for thinking_budget."""
        logger.debug(f"[GEMINI BUDGET] Model: {model}, Budget: {reasoning_budget}, Stream: {stream}")
        # Check if client has chat completions
        if hasattr(client, "chat") and hasattr(getattr(client, "chat"), "completions"):
            create_fn = getattr(getattr(client.chat, "completions"), "create", None)
            if callable(create_fn):
                messages = input if isinstance(input, list) else [{"role": "user", "content": str(input)}]
                # Prepare call kwargs
                call_kwargs = kwargs.copy()
                # Convert reasoning_budget to integer if needed
                if reasoning_budget == -1:
                    budget = None  # Dynamic thinking
                else:
                    budget = int(reasoning_budget) if reasoning_budget else 2000
                # Add extra_body with thinking_config for budget.
                # IMPORTANT: OpenAI Python SDK treats `extra_body=...` as a merge-into-root.
                # For Google's OpenAI-compat endpoint, provider-specific payload must be
                # nested under a top-level `extra_body` field on the wire.
                if budget is not None:
                    existing = call_kwargs.pop("extra_body", {})
                    inner: Dict[str, Any] = {}
                    if isinstance(existing, dict):
                        inner = existing.get("extra_body", existing)
                    inner.setdefault("google", {})
                    inner["google"]["thinking_config"] = {"thinking_budget": budget}
                    # Double-wrap so the SDK merges `{ "extra_body": ... }` into the request body.
                    call_kwargs["extra_body"] = {"extra_body": inner}
                    logger.debug(f"[GEMINI BUDGET] API call: chat.completions.create(model={model}, thinking_budget={budget}, stream={stream})")
                else:
                    logger.debug(f"[GEMINI BUDGET] API call: chat.completions.create(model={model}, extra_body=None, stream={stream})")
                # Handle max_output_tokens conversion
                if "max_output_tokens" in call_kwargs:
                    max_tokens_param = self._get_max_tokens_parameter(model)
                    call_kwargs[max_tokens_param] = call_kwargs.pop("max_output_tokens")
                if stream:
                    stream_obj = create_fn(model=model, messages=messages, stream=True, **call_kwargs)
                    def _event_gen() -> Iterator[AdapterEvent]:
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
                    return _event_gen()
                try:
                    resp = create_fn(model=model, messages=messages, stream=False, **call_kwargs)
                except openai.RateLimitError as e:  # type: ignore[attr-defined]
                    # Map Gemini adapter rate limits into a structured LLMError
                    # so callers (e.g., handle_chat) can surface a clear message.
                    retry_after = None
                    try:
                        retry_after = getattr(e, "retry_after", None)
                        if retry_after is None:
                            # Try to extract from OpenAI error details
                            details = getattr(e, "response", {}).get("details", [])
                            for detail in details:
                                if hasattr(detail, "retryDelay"):
                                    retry_after = getattr(detail, "retryDelay", None)
                                    break
                    except Exception:
                        pass
                    
                    raise LLMError(
                        provider="gemini",
                        model=model,
                        kind="rate_limit",
                        code="rate_limit",
                        message=str(e),
                        retry_after=retry_after,
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
                return AdapterResponse(output_text=text, model=model, usage=usage_dict, raw=wrapped)
        logger.debug(f"[GEMINI BUDGET] Client missing chat completions, falling back to regular call")
        # Fallback to regular Gemini call. Skip reasoning routing to avoid recursion.
        return self._gemini_call(model=model, input=input, stream=stream, skip_reasoning=True, **kwargs)

    def _gemini_level_reasoning_call(self, *, client, model: str, input: Any, stream: bool, thinking_level: Any, **kwargs: Any):
        """Gemini reasoning using chat.completions.create with extra_body."""
        logger.debug(f"[GEMINI LEVEL] Model: {model}, Level: {thinking_level}, Stream: {stream}")
        # Check if client has chat completions
        if hasattr(client, "chat") and hasattr(getattr(client, "chat"), "completions"):
            create_fn = getattr(getattr(client.chat, "completions"), "create", None)
            if callable(create_fn):
                messages = input if isinstance(input, list) else [{"role": "user", "content": str(input)}]
                # Prepare call kwargs
                call_kwargs = kwargs.copy()
                # Add extra_body with thinking_config.
                # IMPORTANT: OpenAI Python SDK treats `extra_body=...` as a merge-into-root.
                # For Google's OpenAI-compat endpoint, provider-specific payload must be
                # nested under a top-level `extra_body` field on the wire.
                existing = call_kwargs.pop("extra_body", {})
                inner: Dict[str, Any] = {}
                if isinstance(existing, dict):
                    inner = existing.get("extra_body", existing)
                inner.setdefault("google", {})
                inner["google"]["thinking_config"] = {
                    "thinking_level": str(thinking_level) if thinking_level else "medium"
                }
                # Double-wrap so the SDK merges `{ "extra_body": ... }` into the request body.
                call_kwargs["extra_body"] = {"extra_body": inner}
                logger.debug(f"[GEMINI LEVEL] API call: chat.completions.create(model={model}, thinking_level='{thinking_level}', stream={stream})")
                # Handle max_output_tokens conversion
                if "max_output_tokens" in call_kwargs:
                    max_tokens_param = self._get_max_tokens_parameter(model)
                    call_kwargs[max_tokens_param] = call_kwargs.pop("max_output_tokens")
                if stream:
                    stream_obj = create_fn(model=model, messages=messages, stream=True, **call_kwargs)
                    def _event_gen() -> Iterator[AdapterEvent]:
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
                    return _event_gen()
                try:
                    resp = create_fn(model=model, messages=messages, stream=False, **call_kwargs)
                except openai.RateLimitError as e:  # type: ignore[attr-defined]
                    # Map Gemini adapter rate limits into a structured LLMError
                    # so callers (e.g., handle_chat) can surface a clear message.
                    retry_after = None
                    try:
                        retry_after = getattr(e, "retry_after", None)
                        if retry_after is None:
                            # Try to extract from OpenAI error details
                            details = getattr(e, "response", {}).get("details", [])
                            for detail in details:
                                if hasattr(detail, "retryDelay"):
                                    retry_after = getattr(detail, "retryDelay", None)
                                    break
                    except Exception:
                        pass
                    
                    raise LLMError(
                        provider="gemini",
                        model=model,
                        kind="rate_limit",
                        code="rate_limit",
                        message=str(e),
                        retry_after=retry_after,
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
                return AdapterResponse(output_text=text, model=model, usage=usage_dict, raw=wrapped)
        logger.debug(f"[GEMINI LEVEL] Client missing chat completions, falling back to regular call")
        # Fallback to regular Gemini call. Skip reasoning routing to avoid recursion.
        return self._gemini_call(model=model, input=input, stream=stream, skip_reasoning=True, **kwargs)

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
        # Filter kwargs by capabilities, then map reasoning parameters with defaults
        filtered_kwargs = self._filter_kwargs_by_capabilities(model, kwargs)
        mapped_kwargs = self._map_reasoning_parameter_with_default(model, filtered_kwargs)
        # Work on post-filter/post-map kwargs so sanitizer/mapping affects the wire call.
        working_kwargs = mapped_kwargs.copy()

        # Optional Gemini-specific tools sanitizer: only applied when tools are
        # present in kwargs. This keeps OpenAI behavior unchanged while
        # normalizing tools for the Gemini OpenAI adapter, which can be more
        # strict about JSON schema fields.
        def _sanitize_for_gemini_tools(tools: Any) -> Any:
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

        if "tools" in working_kwargs:
            try:
                working_kwargs["tools"] = _sanitize_for_gemini_tools(working_kwargs["tools"])
            except Exception:
                # Best-effort only; fall back to original tools on any failure.
                pass

        # Keep max_output_tokens consistent - let each provider handle conversion internally
        # OpenAI chat completions will use model registry to determine correct parameter name

        # --- Check if we should use native Gemini reasoning API formats ---
        reasoning_param = self._get_reasoning_parameter(model)
        if (not skip_reasoning) and reasoning_param and reasoning_param[0] in ("thinking_budget", "thinking_level"):
            param_name, default_value = reasoning_param
            logger.debug(f"[GEMINI REASONING] Using {param_name} format for model {model} (default: {default_value})")
            return self._gemini_reasoning_call(model=model, input=input, stream=stream, **working_kwargs)
        # --- OpenAI-adapter path (chat.completions) ---
        if hasattr(client, "chat") and hasattr(getattr(client.chat, "completions"), "create"):
            create_fn = getattr(getattr(client.chat, "completions"), "create", None)
            if callable(create_fn):
                messages = input if isinstance(input, list) else [{"role": "user", "content": str(input)}]

                if not stream:
                    # Apply capability filtering, parameter mapping, and max_tokens conversion
                    call_kwargs = working_kwargs.copy()
                    if "max_output_tokens" in call_kwargs:
                        max_tokens_param = self._get_max_tokens_parameter(model)
                        call_kwargs[max_tokens_param] = call_kwargs.pop("max_output_tokens")
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
                    return AdapterResponse(output_text=text, model=model, usage=usage_dict, raw=wrapped)

                def event_gen() -> Iterator[AdapterEvent]:
                    # Apply capability filtering, parameter mapping, and max_tokens conversion
                    call_kwargs = working_kwargs.copy()
                    if "max_output_tokens" in call_kwargs:
                        max_tokens_param = self._get_max_tokens_parameter(model)
                        call_kwargs[max_tokens_param] = call_kwargs.pop("max_output_tokens")
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
            return AdapterResponse(output_text=text or "", model=model, usage=None, raw=wrapped)
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
        try:
            if resp and getattr(resp, "choices", None):
                choice0 = resp.choices[0]
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
            def __init__(self, *, output_text: str, output: list[dict], usage: Any, choices: Any, raw: Any):
                # === CANONICAL FIELDS (must use for all logic) ===
                self.output_text = output_text      # Canonical: final text
                self.output = output                # Canonical: tools + content
                self.usage = usage                  # Canonical: tokens
                
                # === NON-CANONICAL FIELDS (compatibility/debug only) ===
                self.choices = choices              # Legacy: DO NOT USE for logic
                self.raw = raw                      # Provider-native: debug only

        return _GeminiResponsesWrapper(
            output_text=output_text or "",      # Canonical text field
            output=output,                      # Canonical structured output
            usage=usage,                        # Canonical usage field
            choices=getattr(resp, "choices", None),  # Non-canonical compatibility
            raw=resp,                           # Non-canonical debug
        )


# Singleton instance (optional)
llm_handler = LLMHandler()