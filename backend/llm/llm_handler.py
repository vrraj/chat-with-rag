from openai import OpenAI
import openai
import os
from typing import Any, Dict, Optional, Iterator

# ModelSpec import: support both `backend/llm/ModelSpec.py` and `llm/ModelSpec.py` layouts.
from backend.llm.ModelSpec import ModelSpec

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

    def __init__(self, *, openai_client=None, gemini_client=None, anthropic_client=None):
        self._openai = openai_client
        self._gemini = gemini_client
        self._anthropic = anthropic_client
        # Facade for compatibility with existing `client.responses.create(...)` call sites.
        self.responses = _ResponsesFacade(self)
        # New additive facade for embeddings; does not affect existing behavior.
        self.embeddings = _EmbeddingsFacade(self)

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

    def _get_anthropic(self):
        if self._anthropic is None:
            import anthropic
            self._anthropic = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        return self._anthropic

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
        if provider == "anthropic":
            return self._anthropic_call(model=model, input=input, stream=stream, **kwargs)
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
        if provider == "anthropic":
            return self._anthropic_embedding_call(model=model, input=input, **kwargs)
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
        try:
            return client.responses.create(model=model, input=input, stream=stream, **kwargs)
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
        return client.embeddings.create(model=model, input=input, **kwargs)

    def _anthropic_call(self, *, model: str, input: Any, stream: bool, **kwargs: Any):
        client = self._get_anthropic()
        messages = input if isinstance(input, list) else [{"role": "user", "content": str(input)}]

        max_tokens = kwargs.pop("max_tokens", None)
        if max_tokens is None:
            max_tokens = kwargs.pop("max_output_tokens", 1024)
        max_tokens = int(max_tokens)

        if not stream:
            return client.messages.create(model=model, messages=messages, max_tokens=max_tokens, **kwargs)

        def gen() -> Iterator[str]:
            with client.messages.stream(model=model, messages=messages, max_tokens=max_tokens, **kwargs) as sref:
                for chunk in sref.text_stream:
                    yield chunk

        return gen()

    def _anthropic_embedding_call(self, *, model: str, input: Any, **kwargs: Any):
        """Placeholder for Anthropic embeddings.

        Implement a concrete mapping when Anthropic exposes or you adopt
        a specific embedding endpoint.
        """
        raise LLMError(
            provider="anthropic",
            model=model,
            kind="config",
            code="embeddings_not_configured",
            message="Anthropic embeddings are not configured in this deployment",
        )

    def _gemini_call(self, *, model: str, input: Any, stream: bool, **kwargs: Any):
        """Gemini adapter that returns OpenAI-compatible response objects/events.

        Important: when `_get_gemini()` builds an OpenAI client pointed at the Gemini OpenAI adapter
        base URL, the supported surface is typically `chat.completions.create`. In that case we must
        NOT look for native Gemini methods like `generate_content`.

        If a native Gemini SDK client is injected, we keep a best-effort fallback to
        `generate_content` / `generateContent`.
        """
        client = self._get_gemini()

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

        if "tools" in kwargs:
            try:
                kwargs["tools"] = _sanitize_for_gemini_tools(kwargs["tools"])
            except Exception:
                # Best-effort only; fall back to original tools on any failure.
                pass

        # Normalize token naming.
        # - Pipeline may pass `max_output_tokens` (Responses-style)
        # - Chat Completions expects `max_tokens`
        max_output_tokens = kwargs.pop("max_output_tokens", None)
        if max_output_tokens is not None and "max_tokens" not in kwargs:
            kwargs["max_tokens"] = max_output_tokens

        # --- OpenAI-adapter path (chat.completions) ---
        if hasattr(client, "chat") and hasattr(getattr(client, "chat"), "completions"):
            create_fn = getattr(getattr(client.chat, "completions"), "create", None)
            if callable(create_fn):
                messages = input if isinstance(input, list) else [{"role": "user", "content": str(input)}]

                if not stream:
                    try:
                        resp = create_fn(model=model, messages=messages, **kwargs)
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

                    return AdapterResponse(output_text=text, model=model, usage=usage_dict, raw=resp)

                def event_gen() -> Iterator[AdapterEvent]:
                    try:
                        stream_obj = create_fn(model=model, messages=messages, stream=True, **kwargs)
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

        if max_output_tokens is not None:
            kwargs.setdefault("max_output_tokens", max_output_tokens)

        if not stream:
            if hasattr(client, "generate_content"):
                resp = client.generate_content(model=model, contents=contents, **kwargs)
            elif hasattr(client, "generateContent"):
                resp = client.generateContent(model=model, contents=contents, **kwargs)
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

            return AdapterResponse(output_text=text or "", model=model, usage=None, raw=resp)

        def native_event_gen() -> Iterator[AdapterEvent]:
            if hasattr(client, "generate_content_stream"):
                stream_iter = client.generate_content_stream(model=model, contents=contents, **kwargs)
                for chunk in stream_iter:
                    delta = chunk if isinstance(chunk, str) else getattr(chunk, "text", "") or ""
                    if delta:
                        yield AdapterEvent("response.output_text.delta", delta=delta)
                yield AdapterEvent("response.output_text.done")
                return

            if hasattr(client, "generate_content"):
                stream_iter = client.generate_content(model=model, contents=contents, stream=True, **kwargs)
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
        return client.embeddings.create(model=model, input=input, **kwargs)


# Singleton instance (optional)
llm_handler = LLMHandler()