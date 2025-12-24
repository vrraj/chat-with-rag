# LLM Handler Overview

This module provides a provider-agnostic façade for LLM calls used by the chat pipeline.
All callers (such as `backend/chat/chat_manager.py`) should route LLM work through
`llm_handler` instead of talking directly to provider SDKs.

## Responsibilities

- Normalize configuration and routing across supported providers.
- Expose a unified Responses-like surface for text generation and embeddings.
- Translate common provider / configuration errors into a structured `LLMError`.
- Hide provider-specific quirks (e.g., Gemini adapter behavior) from the rest
  of the codebase.

## Providers

Today the handler supports:

- **OpenAI**
  - Text: via `client.responses.create(...)`.
  - Embeddings: via `client.embeddings.create(...)`.
- **Gemini (via OpenAI-compatible adapter)**
  - Text: primarily via `chat.completions.create(...)` on an OpenAI-style client
    pointed at a Gemini adapter base URL.
  - Embeddings: via `client.embeddings.create(...)` if configured.

(Anthropic is wired for text but embeddings are intentionally not configured.)

### `LLMError`

`LLMError` is a lightweight, structured exception raised by `llm_handler` for
known configuration and provider failures. It carries:

- `provider`: e.g. `"openai"`, `"gemini"`.
- `model`: model identifier string.
- `kind`: coarse error category, such as `"config"`.
- `code`: short machine-readable code.
- `message`: human-readable message suitable for logs / UI.
- `retry_after`: optional numeric hint (seconds) when rate-limited.

Callers that wish to surface provider issues to the user (for example,
`handle_chat`) can catch `LLMError` specifically and include the
`message`/`provider`/`model` in their response payload. All other callers
can continue to rely on generic `Exception` handling; `LLMError` still
inherits from `Exception`, so this is non-breaking.

## Gemini Tools Sanitizer

When the pipeline enables tools, it passes a list of OpenAI-style tool
definitions in `kwargs["tools"]`. OpenAI and Gemini (via an adapter) do
not accept exactly the same JSON shapes and schema metadata, so the
handler normalizes *only the Gemini path*.

Inside `_gemini_call`, when `"tools"` is present in `kwargs`, the handler
applies a small sanitizer before calling the Gemini client:

- Accepts either flattened or nested OpenAI-style tool specs.
- Produces a nested structure of the form:

  ```json
  {
    "type": "function",
    "function": {
      "name": "...",
      "description": "...",
      "parameters": { ... }
    }
  }
  ```

- Recursively strips JSON schema keys known to cause 400 errors in some
  Gemini adapter deployments, such as:

  - `default`
  - `additionalProperties`
  - `$schema`
  - `title`

If sanitization fails for any reason, the handler falls back to the
original `tools` value. This keeps behavior non-breaking and confines all
Gemini-specific logic to the provider adapter layer.

## Non-breaking Design

Key design choices to avoid surprising callers:

- All provider routing is internal to `llm_handler`; call signatures stay
  stable for upstream code.
- `LLMError` is additive and still derives from `Exception`.
- The Gemini tools sanitizer runs **only** when `kwargs` contains
  `"tools"`; calls without tools are unaffected.
- OpenAI calls are not modified by Gemini-specific logic.

Future provider-specific tweaks should follow the same pattern:
localized inside the handler, opt-in based on provider, and safe to
ignore when not applicable.
