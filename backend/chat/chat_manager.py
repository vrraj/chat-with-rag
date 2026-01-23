"""Chat Manager Module

Entry Point:
- handle_chat(payload: Dict) -> Dict: Main entry point for "statless" chat requests

Pipeline Initialization:
1. Initialize dependencies (DB, clients, tools)
2. Parse request payload
3. Set up metrics and logging

Chat Pipeline Stages (with History Integration):
1. Query Rewrite (optional) --> 2. Context (Document) Retrieval --> 3. Reranking --> 4. Context Summarization --> 5. Prompt Construction --> 6. LLM Inference --> 7. Tool Execution (if needed) --> 8. Final Response Generation


Conversation State:
- Full history maintained in memory
- Each turn appends both user message and assistant response
- Configurable history window size controls context length
- Automatic summarization of older messages
"""

from typing import List, Dict, Any, Set, TypedDict, Optional
class StageSpec(TypedDict):
    """Type definition for stage configuration dictionaries.

    Each stage has a provider, model, and additional keyword arguments.
    """
    provider: str
    model: str
    kwargs: Dict[str, Any]
import logging

logger = logging.getLogger(__name__)
import json
import re
import uuid
import tiktoken
import time
from collections import defaultdict
# NOTE: SSE stage emission is centralized in backend/stream_emit.py so chat_manager stays agnostic of registry details.
# Stream emission helpers (centralized in backend/stream_emit.py)
from backend.stream_emit import emit_stage, close_stream
from backend.core.config import settings
from backend.db import QdrantDB
from backend.chat.web_search import WebSearchClient
from backend.embeddings.specs import resolve_embedding_spec
from backend.tools import list_tools, get_executor

from backend.llm.llm_handler import llm_handler, LLMError

_SUMMARY_CACHE: Dict[str, str] = {}
# Option A support: index of namespace -> set of cache keys for precise clearing
_SUMMARY_NS_INDEX: Dict[str, Set[str]] = defaultdict(set)
# Option A support: last-seen timestamp per namespace for idle eviction
_SUMMARY_NS_LAST_SEEN: Dict[str, float] = {}


# --- LLM call helper (OpenAI-first, via llm_handler facade) ---
# NOTE: This is a compatibility shim only. It preserves existing behavior by routing to
# `llm_handler.responses.create(**kwargs)`. This avoids relying on environment variables
# inside llm_handler.


def _responses_create(provider: str | None = None, **kwargs: Any):
    """Compatibility shim for LLM calls.

    Default behavior (no provider provided):
      - Preserve existing behavior by routing to `llm_handler.responses.create(**kwargs)`

    Optional behavior (provider provided):
      - Route via `llm_handler.create(provider=..., model=..., input=..., stream=..., **kwargs)`
      - This enables stage-level provider selection later without changing call sites.

    """

    prov = (provider or "openai").strip().lower() # default llm provider to openai

    # Preserve existing Responses API path when provider is not explicitly set (or is openai).
    if provider is None or prov == "openai":
        return llm_handler.responses.create(**kwargs)

    # Provider-aware path (used in later steps when stage selection is enabled).
    model = kwargs.pop("model", None)
    inp = kwargs.pop("input", None)
    stream = bool(kwargs.pop("stream", False))
    return llm_handler.create(provider=prov, model=model, input=inp, stream=stream, **kwargs)


# --- Stage resolver (read-only; mirrors existing fields as-is) ---
# Produces per-stage provider/model/kwargs, with provider defaulting to "openai".
# Frontend params can override these later; for now we only read existing settings.

def resolve_stage_specs(
    *,
    settings_obj: Any,
    params: Dict[str, Any] | None,
    enable_tools: bool,
    prompt_input: Any,
    message: str,
    list_tools_fn: Any,
) -> Dict[str, StageSpec]:
    """Return stage specs using current flat settings (no behavior change).

    Output shape:
        {
          "rewrite":   {"provider": "openai", "model": "...", "kwargs": {...}},
          "summary":   {"provider": "openai", "model": "...", "kwargs": {...}},
          "rerank":    {"provider": "openai", "model": "...", "kwargs": {...}},
          "inference": {"provider": "openai", "model": "...", "kwargs": {...}},
          "tools_synth": {"provider": "openai", "model": "...", "kwargs": {...}},
          "embedding": {"provider": "openai", "model": "...", "kwargs": {...}},
        }

    """
    p = params or {}

    # Optional per-request overrides (from the frontend / API)
    rerank_provider_override = str(p.get("rerank_provider") or "").strip()
    rerank_model_override = str(p.get("rerank_model") or "").strip()
    rewrite_provider_override = str(p.get("rewrite_provider") or "").strip()
    rewrite_model_override = str(p.get("rewrite_model") or "").strip()
    summary_provider_override = str(p.get("summary_provider") or "").strip()
    summary_model_override = str(p.get("summary_model") or "").strip()
    inference_provider_override = str(p.get("inference_provider") or "").strip()
    inference_model_override = str(p.get("inference_model") or "").strip()

    # Optional per-request model_keys map (stage -> registry key). When
    # present, these override the model name for that stage so callers can
    # select registry entries like "openai:chat_fast" while keeping
    # provider settings/backwards compatibility intact.
    model_keys = p.get("model_keys") or {}
    try:
        inference_model_key_override = str(model_keys.get("inference") or "").strip()
        rewrite_model_key_override = str(model_keys.get("rewrite") or "").strip()
        summary_model_key_override = str(model_keys.get("summary") or "").strip()
        rerank_model_key_override = str(model_keys.get("rerank") or "").strip()
    except Exception:
        inference_model_key_override = ""
        rewrite_model_key_override = ""
        summary_model_key_override = ""
        rerank_model_key_override = ""

    # Base models from settings (stage defaults)
    rewrite_model = getattr(settings_obj, "rewrite_model", getattr(settings_obj, "inference_model", ""))
    summarizer_model = getattr(settings_obj, "summarizer_model", getattr(settings_obj, "inference_model", ""))
    rerank_model = getattr(settings_obj, "re_ranker_model", getattr(settings_obj, "inference_model", ""))

    # Inference model: allow per-request override to affect downstream defaults.
    inference_model = getattr(settings_obj, "inference_model", "")
    # Prefer explicit model_key override when present, then legacy name override,
    # then settings default. This lets callers opt into registry keys such as
    # "openai:chat_fast" without breaking existing configs.
    effective_inference_model = (
        inference_model_key_override
        or inference_model_override
        or inference_model
    )
    
    # Inference provider: allow per-request override to affect downstream defaults.
    inference_provider = getattr(settings_obj, "inference_provider", "openai")
    effective_inference_provider = (inference_provider_override or inference_provider)

    # Tools synthesis model uses the same model as inference
    tools_synth_model = effective_inference_model

    # Embedding spec: settings.embedding_model is a provider key (openai/gemini).
    # Resolve it into a provider-specific embedding model name for stage_specs consistency.
    try:
        _emb = resolve_embedding_spec(settings_obj)  # {provider, model, dimensions}
    except Exception:
        _emb = {"provider": "openai", "model": getattr(settings_obj, "openai_embedding_model", ""), "dimensions": None}

    emb_provider = str((_emb or {}).get("provider") or "openai").strip() or "openai"
    emb_model = str((_emb or {}).get("model") or "").strip()

    # Existing flat temps/limits (read as-is)
    rewrite_temp = float(getattr(settings_obj, "rewrite_temperature", 0.2))
    rewrite_max_out = int(getattr(settings_obj, "rewrite_max_output_tokens", 128))

    summarizer_temp = float(getattr(settings_obj, "summarizer_temperature", 0.3))
    summarizer_max_in = int(getattr(settings_obj, "summarizer_max_input_tokens", 512))
    summarizer_max_out = int(getattr(settings_obj, "summarizer_max_output_tokens", 128))

    try:
        logger.debug(
            "[STAGE SPECS] summary provider=%s model=%s temp=%.3f max_in=%d max_out=%d",
            (summary_provider_override or "openai"),
            (summary_model_override or summarizer_model),
            summarizer_temp,
            summarizer_max_in,
            summarizer_max_out,
        )
    except Exception:
        pass

    rerank_temp = float(getattr(settings_obj, "re_ranker_temperature", 0.0))
    rerank_max_out = int(getattr(settings_obj, "re_ranker_max_output_tokens", 64))

    inference_temp = float(getattr(settings_obj, "inference_temperature", 0.2))
    inference_top_p = float(getattr(settings_obj, "inference_top_p", 1.0))
    inference_max_out = int(getattr(settings_obj, "max_inference_output_tokens", 800))
    # Tools synthesis can optionally use its own output token budget.
    # Back-compat default: fall back to inference_max_out when no dedicated setting exists.
    tools_synth_max_out = int(getattr(settings_obj, "tools_synth_max_output_tokens", inference_max_out))

    # Tools kwargs are attached at the inference call site today; mirror the current logic here.
    tools_kwargs: Dict[str, Any] = {}
    if enable_tools and isinstance(prompt_input, list):
        try:
            tools = list_tools_fn()

            def _is_web_search_requested(latest_user_msg: str) -> bool:
                if not latest_user_msg:
                    return False
                txt = latest_user_msg.lower()
                keys = [
                    "use web search",
                    "search the web",
                    "web search",
                    "search online",
                    "browse the web",
                    "do a web search",
                    "google this",
                    "bing this",
                ]
                return any(k in txt for k in keys)

            if not _is_web_search_requested(message):
                tools = [t for t in tools if (t.get("name") or t.get("function", {}).get("name")) != "web_search"]
            tools_kwargs["tools"] = tools
        except Exception:
            tools_kwargs["tools"] = []

    try:
        logger.debug(
            "[STAGE SPECS] inference provider=%s model=%s max_out=%d | tools_synth max_out=%d",
            effective_inference_provider,
            effective_inference_model,
            inference_max_out,
            tools_synth_max_out,
        )
    except Exception:
        pass

    return {
        "embedding": {
            "provider": emb_provider,
            "model": emb_model,
            "kwargs": {},
        },
        "rewrite": {
            "provider": (rewrite_provider_override or "openai"),
            "model": (rewrite_model_key_override or rewrite_model_override or rewrite_model),
            "kwargs": {
                "temperature": rewrite_temp,
                "max_output_tokens": rewrite_max_out
            },
        },
        "summary": {
            "provider": (summary_provider_override or "openai"),
            "model": (summary_model_key_override or summary_model_override or summarizer_model),
            "kwargs": {
                "temperature": summarizer_temp,
                "max_output_tokens": summarizer_max_out,
                # NOTE: summarizer input budgeting is handled by prompt construction, not API args.
                "_max_input_tokens": summarizer_max_in,
            },
        },
        "rerank": {
            "provider": (rerank_provider_override or "openai"),
            "model": (rerank_model_key_override or rerank_model_override or rerank_model),
            "kwargs": {
                "temperature": rerank_temp,
                "max_output_tokens": rerank_max_out,
            },
        },
        "inference": {
            "provider": effective_inference_provider,
            # Prefer the effective_inference_model, which already folds in model_keys.inference (registry key) when provided. 
            "model": effective_inference_model,
            "kwargs": {
                "temperature": inference_temp,
                "top_p": inference_top_p,
                "max_output_tokens": inference_max_out,
                "reasoning_effort": getattr(settings, "inference_reasoning_effort", "low"),
                "debug_thoughts": getattr(settings, "debug_thoughts", False),
                **tools_kwargs,
            },
        },
        "tools_synth": {
            # Tools synthesis uses the same provider as inference, but may have its own model.
            "provider": effective_inference_provider,
            "model": tools_synth_model,
            "kwargs": {
                "temperature": inference_temp,
                "max_output_tokens": tools_synth_max_out,
                "reasoning_effort": getattr(settings, "inference_reasoning_effort", "low"),
                "debug_thoughts": getattr(settings, "debug_thoughts", False),
            },
        },
    }

# ---- Conversation totals accumulator (per-namespace, module-level) ----
COST_BASIS = float(getattr(settings, "cost_basis_tokens", 1_000_000))

# Back-compat default (used when namespace is empty)
CONVO_TOTALS = {
    "tokens": {
        "embedding": 0,
        "llm_input": 0,      # prompt + cached tokens across stages
        "llm_output": 0,     # completion tokens across stages
        "conversation_total": 0,
    },
    "costs": {
        "embedding": 0.0,
        "llm_input": 0.0,
        "llm_output": 0.0,
        "total": 0.0,
        "conversation_total": 0.0,
    },
}

# Per-conversation/session totals, keyed by namespace (typically user_id:conversation_id or conversation_id)
_CONVO_TOTALS_BY_NS: Dict[str, Dict[str, Any]] = {}

def _new_convo_totals() -> Dict[str, Any]:
    return {
        "tokens": {
            "embedding": 0,
            "llm_input": 0,
            "llm_output": 0,
            "conversation_total": 0,
        },
        "costs": {
            "embedding": 0.0,
            "llm_input": 0.0,
            "llm_output": 0.0,
            "total": 0.0,
            "conversation_total": 0.0,
        },
    }


def _zero_totals_dict(totals: Dict[str, Any]) -> None:
    """Best-effort reset of a totals dict to zeros."""
    try:
        totals["tokens"].update({
            "embedding": 0,
            "llm_input": 0,
            "llm_output": 0,
            "conversation_total": 0,
        })
        totals["costs"].update({
            "embedding": 0.0,
            "llm_input": 0.0,
            "llm_output": 0.0,
            "total": 0.0,
            "conversation_total": 0.0,
        })
    except Exception:
        # Best-effort only; never break the pipeline.
        pass


def _get_convo_totals_for_namespace(namespace: str) -> Dict[str, Any]:
    """Return a mutable totals dict scoped to `namespace` (conversation/session)."""
    ns = str(namespace or "").strip()
    if not ns:
        return CONVO_TOTALS
    existing = _CONVO_TOTALS_BY_NS.get(ns)
    if existing is None:
        existing = _new_convo_totals()
        _CONVO_TOTALS_BY_NS[ns] = existing
    return existing


def _zero_convo_totals() -> None:
    """Back-compat: reset the default (empty-namespace) accumulator."""
    _zero_totals_dict(CONVO_TOTALS)


def clear_convo_totals_for_namespace(namespace: str) -> Dict[str, Any]:
    """Clear totals for a specific namespace (conversation/session). Returns stats."""
    ns = str(namespace or "").strip()
    if not ns:
        _zero_convo_totals()
        return {"cleared": True, "namespace": "", "active_namespaces": len(_CONVO_TOTALS_BY_NS)}
    existed = ns in _CONVO_TOTALS_BY_NS
    if existed:
        _CONVO_TOTALS_BY_NS.pop(ns, None)
    return {"cleared": bool(existed), "namespace": ns, "active_namespaces": len(_CONVO_TOTALS_BY_NS)}
 # ---- end accumulator ----


def _extract_text_from_responses(resp: Any) -> str:
    """Return response text from a Responses-like object.

    Delegates to llm_handler.build_llm_result_from_response so that 
    provider-specific parsing (Responses vs ChatCompletions vs adapters)
    is centralized in one place.
    """

    # Prefer adapter_response surface when present (e.g., Gemini
    # _GeminiResponsesWrapper); otherwise, use the response as-is.
    base = getattr(resp, "adapter_response", resp)

    try:
        llm_result = llm_handler.build_llm_result_from_response(base)
    except Exception:
        try:
            logger.exception("[RESP DEBUG] failed to build LLMResult for text extraction")
        except Exception:
            pass
        return ""

    text = ""
    try:
        text = str(llm_result.get("text") or "")
    except Exception:
        text = ""

    try:
        logger.debug("[RESP DEBUG] extracted text length=%d", len(text))
    except Exception:
        pass

    return text


def _extract_usage_from_responses(resp, provider: str = "openai") -> Dict[str, int] | None:
    """Extract canonical usage fields from a response object.

    Delegates to LLMHandler for provider-specific normalization.
    Returns canonical fields: {input_tokens, cached_tokens, output_tokens,
                               reasoning_tokens, completion_tokens, total_tokens}
    All fields default to 0 if missing.
    """
    if resp is None:
        return None

    # If already an LLMResult-like dict with canonical usage, return it directly.
    if isinstance(resp, dict) and "usage" in resp:
        u = resp["usage"]
        if isinstance(u, dict) and "input_tokens" in u:
            return u

    # Prefer adapter_response surface when present (e.g., Gemini
    # _GeminiResponsesWrapper); otherwise, use the response as-is.
    base = getattr(resp, "adapter_response", resp)

    # Delegate to LLMHandler for provider-specific normalization.
    try:
        result = llm_handler.build_llm_result_from_response(base, provider=provider)
        usage = result.get("usage") or {}
    except Exception:
        try:
            logger.exception("[USAGE DEBUG] failed to build LLMResult for usage extraction")
        except Exception:
            pass
        usage = {}

    # Normalize and ensure all canonical fields are present and numeric.
    norm = {
        "input_tokens": 0,
        "cached_tokens": 0,
        "output_tokens": 0,
        "reasoning_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
    }
    if isinstance(usage, dict):
        for k in norm.keys():
            v = usage.get(k)
            try:
                if isinstance(v, (int, float)):
                    norm[k] = int(v)
            except Exception:
                continue

    return norm


# --- Small shared helpers for chat ---

def _candidate_texts(pool: List[Dict[str, Any]]) -> List[str]:
    """Return plain text snippets from a rerank pool (payload->text/snippet/content)."""
    out: List[str] = []
    for res in pool or []:
        pl = res.get("payload") or {}
        txt = pl.get("text") or pl.get("snippet") or pl.get("content") or ""
        out.append(txt)
    return out


def _make_rerank_prompt(query: str, cand_texts: List[str], chunk_size: int) -> str:
    """Build a compact rerank prompt identical to existing inline versions."""
    return (
        "You are a reranker. Given a user query and N candidate snippets, return the indices "
        "of the snippets in strictly decreasing relevance order. Crucially, ensure the top results cover distinct facets of the topic and minimize redundancy. Return ONLY a JSON array of integers. "
        "No prose, no code fences, no extra text. Example: [3,0,1].\n\n"
        f"Query: {query}\n\nCandidates (index: text excerpt):\n"
        + "\n".join([f"[{i}] {t[:chunk_size]}" for i, t in enumerate(cand_texts)])
    )


def _strip_code_fences(text: str) -> str:
    """Remove ```json ... ``` or ``` ... ``` fences if present."""
    if not isinstance(text, str):
        return ""
    s = text.strip()
    if s.startswith("```json") and s.endswith("```"):
        return s[7:-3].strip()
    if s.startswith("```") and s.endswith("```"):
        return s[3:-3].strip()
    return s



def _parse_json_array_in_text(content: str, pool_n: int) -> List[int]:
    """Robustly parse a JSON array of ints from model output; fallback to original order."""
    try_content = _strip_code_fences(content or "")
    try:
        order = json.loads(try_content)
    except json.JSONDecodeError:
        start = try_content.find("[")
        end = try_content.rfind("]")
        if start != -1 and end != -1 and start < end:
            try:
                order = json.loads(try_content[start:end+1])
            except json.JSONDecodeError:
                order = list(range(pool_n))
        else:
            order = list(range(pool_n))
    # Keep only valid indices
    return [i for i in (order or []) if isinstance(i, int) and 0 <= i < pool_n] or list(range(pool_n))

# Inserted helper: _pick

def _pick(params: Dict[str, Any] | None, keys: List[str], default=None):
    """Pick the first present, non-None key from params; else default."""
    p = params or {}
    for k in keys:
        if k in p and p[k] is not None:
            return p[k]
    return default

# Inserted helper: _get_param_int
def _get_param_int(params: Dict[str, Any] | None, keys: List[str], default: int, minimum: int | None = None, maximum: int | None = None) -> tuple[int, str]:
    """
    Return (value, source) reading the first available key in `keys` from params, else the `default`.
    Coerces to int and clamps to [minimum, maximum] if provided.
    Source is 'param:<key>' or 'settings'.
    """
    p = params or {}
    for k in keys:
        if k in p and p[k] is not None:
            try:
                v = int(p[k])
                if minimum is not None:
                    v = max(minimum, v)
                if maximum is not None:
                    v = min(maximum, v)
                return v, f"param:{k}"
            except Exception:
                continue
    try:
        v = int(default)
    except Exception:
        v = default if isinstance(default, int) else 0
    if minimum is not None:
        v = max(minimum, v)
    if maximum is not None:
        v = min(maximum, v)
    return v, "settings"

# Inserted helper: _get_param_float
def _get_param_float(params: Dict[str, Any] | None, keys: List[str], default: float,
                     minimum: float | None = None, maximum: float | None = None) -> tuple[float, str]:
    """
    Return (value, source) reading the first available key in `keys` from params, else the `default`.
    Coerces to float and clamps to [minimum, maximum] if provided.
    Source is 'param:<key>' or 'settings'.
    """
    p = params or {}
    for k in keys:
        if k in p and p[k] is not None:
            try:
                v = float(p[k])
                if minimum is not None:
                    v = max(minimum, v)
                if maximum is not None:
                    v = min(maximum, v)
                return v, f"param:{k}"
            except Exception:
                continue
    try:
        v = float(default)
    except Exception:
        v = default if isinstance(default, float) else 0.0
    if minimum is not None:
        v = max(minimum, v)
    if maximum is not None:
        v = min(maximum, v)
    return v, "settings"



def _format_context_lines(items: List[Dict[str, Any]]) -> str:
    """Format context lines with numeric indices and section/subsection info."""
    lines: List[str] = []
    for i, c in enumerate(items or []):
        pl = c.get("payload") or {}
        text = (
            pl.get("text")
            or pl.get("snippet")
            or pl.get("content")
            or c.get("text", "")
            or ""
        ).strip()
        section = pl.get("section") or c.get("section", "N/A")
        subsection = pl.get("subsection") or c.get("subsection", "N/A")
        lines.append(f"[{i+1}] {text} (Section: {section} > {subsection})")
    return "\n".join(lines)


# --- Local token-budget helpers for summarizer ---
def _get_encoder_for_model(model_name: str):
    """Best-effort tiktoken encoder for a model; falls back to cl100k_base."""
    try:
        return tiktoken.encoding_for_model(model_name)
    except Exception:
        try:
            return tiktoken.get_encoding("cl100k_base")
        except Exception:
            # As a last resort, return a tiny shim with encode = lambda s: list(s)
            class _Shim:
                def encode(self, s): return list(s or "")
            return _Shim()

def _build_summary_prompt_with_budget(messages: List[Dict[str, str]], max_input_tokens: int | None, model_name: str) -> str:
    """
    Build a summary prompt that fits within `max_input_tokens` by trimming older lines first.
    Guarantees the most recent line is always included (clipped if necessary).
    """
    header = "Summarize the following conversation in a few sentences:\n\n"
    if not messages:
        return header

    # If no budget is set or <=0, include all lines verbatim.
    if max_input_tokens is None or int(max_input_tokens) <= 0:
        lines = [f"{m.get('role','user')}: {m.get('content','')}" for m in messages]
        return header + "\n".join(lines)

    enc = _get_encoder_for_model(model_name)

    def tok_len(s: str) -> int:
        try:
            return len(enc.encode(s))
        except Exception:
            return len(s or "")

    budget = int(max_input_tokens)
    used = tok_len(header)
    kept_rev: List[str] = []

    # Walk from newest to oldest so we never drop the freshest content.
    for m in reversed(messages):
        role = m.get("role", "user")
        content = m.get("content", "") or ""
        prefix = f"{role}: "
        line = prefix + content
        line_tokens = tok_len(line) + 1  # +1 for newline

        if not kept_rev:
            # Ensure newest line is included; clip content if it doesn't fit.
            if used + line_tokens > budget:
                # Compute remaining room for the line (excluding newline).
                remaining = max(0, budget - used - 1)
                if remaining <= tok_len(prefix):
                    # No room for content; keep a truncated prefix-only line.
                    line = prefix.strip()
                else:
                    room_for_content = remaining - tok_len(prefix)
                    try:
                        content_tokens = enc.encode(content)
                        if room_for_content < len(content_tokens):
                            # Keep leading tokens only and add ellipsis.
                            clipped_tokens = content_tokens[:max(0, room_for_content)]
                            clipped = enc.decode(clipped_tokens) if clipped_tokens else ""
                            line = f"{prefix}{clipped}…"
                    except Exception:
                        # Fallback: naive slice
                        line = prefix + (content[:max(0, room_for_content)] + "…")
                line_tokens = tok_len(line) + 1
            kept_rev.append(line)
            used += line_tokens
        else:
            # For older lines, only include if they fully fit; otherwise stop.
            if used + line_tokens <= budget:
                kept_rev.append(line)
                used += line_tokens
            else:
                break

    kept = list(reversed(kept_rev))
    # One-line debug counter: kept lines vs total, plus token usage vs budget
    try:
        logger.debug("[SUMMARY] input_budget kept_lines=%d/%d used_tokens≈%d budget=%d", len(kept), len(messages), used, budget)
    except Exception:
        pass
    return header + "\n".join(kept)
# --- end local token-budget helpers ---

def _summarize_messages_with_cache(
    messages: List[Dict[str, str]],
    cache: Dict[str, str],
    *,
    tag: str,
    model: str,
    temperature: float,
    max_output_tokens: int | None = None,
    max_input_tokens: int | None = None,
    log_prefix: str = "[SUMMARY]",
    stage_spec: Dict[str, Any] | None = None,
) -> tuple[str, bool, Dict[str, int] | None]:
    """Summarize a slice of messages with a tiny prompt, caching by (messages, tag).

    Returns: (summary_text, from_cache, usage_dict_or_none)
    """
    # Log current cache size for observability
    if logger.isEnabledFor(logging.DEBUG):
        try:
            total_bytes = 0
            try:
                total_bytes = sum(len(v.encode('utf-8')) for v in cache.values())
            except Exception:
                # Fallback to character count if encoding fails
                total_bytes = sum(len(v) for v in cache.values())
            logger.debug("%s Cache size: %d entries, %d bytes", log_prefix, len(cache), total_bytes)
        except Exception:
            pass
    try:
        if not messages:
            return "", True, None

        # Build a cleaned copy of messages so we do not mutate the original history.
        # For assistant messages, strip any trailing 'Sources:' block before summarizing.
        cleaned_messages: List[Dict[str, str]] = []
        stripped = 0
        try:
            for m in messages:
                role = m.get("role", "user")
                content = m.get("content", "") or ""
                if role == "assistant":
                    new_content = _strip_trailing_sources_block(content)
                    if new_content != content:
                        stripped += 1
                    content = new_content
                cleaned_messages.append({"role": role, "content": content})
            if stripped:
                logger.debug("%s stripped trailing Sources: blocks from %d assistant messages before summary", log_prefix, stripped)
        except Exception:
            # If anything goes wrong during cleanup, fall back to the original messages.
            cleaned_messages = [{"role": m.get("role", "user"), "content": m.get("content", "") or ""} for m in messages]

        key = _summary_cache_key(cleaned_messages, tag=tag)
        cached = cache.get(key)
        if cached is not None:
            logger.debug(f"{log_prefix} summary cache HIT ({tag}); len=%d", len(cached))
            return cached, True, None

        _ss = stage_spec or {}
        _provider = str(_ss.get("provider") or "openai")
        _model = str(_ss.get("model") or model)

        # Build the prompt using the effective model so token budgeting matches the selected provider/model.
        sum_prompt = _build_summary_prompt_with_budget(cleaned_messages, max_input_tokens, _model)
        logger.debug(f"{log_prefix} applied local input budget; prompt_len_chars=%d", len(sum_prompt))

        _call_kwargs: Dict[str, Any] = dict(_ss.get("kwargs") or {})
        if not _call_kwargs:
            _call_kwargs = {"temperature": float(temperature)}
            if max_output_tokens is not None:
                _call_kwargs["max_output_tokens"] = int(max_output_tokens)

        # Strip internal-only keys (e.g., _max_input_tokens) so they are not sent to providers.
        try:
            _call_kwargs = {k: v for k, v in _call_kwargs.items() if not str(k).startswith("_")}
        except Exception:
            # Best-effort; if filtering fails, fall back to original kwargs.
            pass

        resp = _responses_create(
            provider=_provider,
            model=_model,
            input=sum_prompt,
            **_call_kwargs,
        )
        summary_text = _extract_text_from_responses(resp).strip()
        cache[key] = summary_text
        # Option A: record key under namespace (if tag includes namespace|...)
        try:
            if isinstance(tag, str) and '|' in tag:
                ns = tag.split('|', 1)[0].strip()
                if ns:
                    _SUMMARY_NS_INDEX[ns].add(key)
        except Exception:
            pass
        logger.debug(f"{log_prefix} summary cache MISS -> stored; len=%d", len(summary_text))
        usage = _extract_usage_from_responses(resp, provider=_provider)
        return summary_text, False, usage
    except LLMError:
        # Let LLMError (including rate_limit) propagate so outer callers can
        # apply consistent quota handling and surface clear messages.
        raise
    except Exception as e:
        logger.warning(f"{log_prefix} summary failed: %s", e, exc_info=True)
        return "", False, None
# --- end shared helpers ---


# --- Helper to clear summaries for a namespace ---
def clear_summaries_for_namespace(namespace: str) -> Dict[str, int]:
    """Clear cached summary entries for a given namespace.

    Returns a dict with counts for observability: {removed: int, remaining: int, reclaimed_bytes: int}
    """
    ns = str(namespace or "").strip()
    if not ns:
        return {"removed": 0, "remaining": len(_SUMMARY_CACHE), "reclaimed_bytes": 0}
    keys = _SUMMARY_NS_INDEX.pop(ns, set())
    removed = 0
    reclaimed = 0
    try:
        for k in list(keys):
            v = _SUMMARY_CACHE.pop(k, None)
            if isinstance(v, str):
                try:
                    reclaimed += len(v.encode('utf-8'))
                except Exception:
                    reclaimed += len(v)
            removed += 1
    except Exception:
        pass
    # Best-effort: also prune empty sets that may linger
    try:
        if ns in _SUMMARY_NS_INDEX and not _SUMMARY_NS_INDEX[ns]:
            _SUMMARY_NS_INDEX.pop(ns, None)
    except Exception:
        pass
    # Also drop last-seen entry for this namespace so it doesn't linger
    try:
        _SUMMARY_NS_LAST_SEEN.pop(ns, None)
    except Exception:
        pass
    return {"removed": removed, "remaining": len(_SUMMARY_CACHE), "reclaimed_bytes": reclaimed}


# --- Namespace last-seen tracking and idle eviction ---
def _touch_namespace(namespace: str) -> None:
    """Record the last-seen time for a namespace in the module-level cache."""
    try:
        ns = str(namespace or "").strip()
        if not ns:
            return
        _SUMMARY_NS_LAST_SEEN[ns] = time.time()
    except Exception:
        # Best-effort only; never break the pipeline
        pass

def _evict_idle_namespaces(now: float | None = None, max_idle_seconds: int | None = None) -> Dict[str, int]:
    """
    Best-effort eviction of idle namespaces from the module-level summary cache.

    A namespace is considered idle if it has not been seen for more than `max_idle_seconds`.
    Defaults to 3600 seconds (1 hour) or `settings.summary_cache_idle_ttl_seconds` if present.

    Returns:
        {"namespaces_cleared": int, "summaries_removed": int, "reclaimed_bytes": int}
    """
    try:
        # Resolve TTL from settings with a safe default
        if max_idle_seconds is None:
            try:
                max_idle_seconds = int(getattr(settings, "summary_cache_idle_ttl_seconds", 3600))
            except Exception:
                max_idle_seconds = 3600
        if max_idle_seconds is None or max_idle_seconds <= 0:
            return {"namespaces_cleared": 0, "summaries_removed": 0, "reclaimed_bytes": 0}
        if now is None:
            now = time.time()
        cleared = 0
        removed_total = 0
        reclaimed_total = 0
        # Work on a snapshot so we can modify the dict while iterating
        for ns, last in list(_SUMMARY_NS_LAST_SEEN.items()):
            try:
                idle_for = now - float(last)
            except Exception:
                idle_for = max_idle_seconds + 1
            if idle_for > max_idle_seconds:
                stats = clear_summaries_for_namespace(ns)
                cleared += 1
                removed_total += int(stats.get("removed", 0) or 0)
                reclaimed_total += int(stats.get("reclaimed_bytes", 0) or 0)
                _SUMMARY_NS_LAST_SEEN.pop(ns, None)
        if cleared:
            try:
                logger.info(
                    "[SUMMARY] idle eviction cleared %d namespaces; removed=%d summaries reclaimed=%d bytes",
                    cleared,
                    removed_total,
                    reclaimed_total,
                )
            except Exception:
                pass
        return {"namespaces_cleared": cleared, "summaries_removed": removed_total, "reclaimed_bytes": reclaimed_total}
    except Exception:
        # Never let cache eviction break the main pipeline
        return {"namespaces_cleared": 0, "summaries_removed": 0, "reclaimed_bytes": 0}


# --- Cost breakdown utility ---
def _compute_stage_cost(
    stage: str,
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
    cached_tokens: int = 0,
    model: str | None = None,
    provider: str | None = None,
    model_key: str | None = None,
) -> Dict[str, float]:
    """Return cost breakdown for a stage using per-million rates and COST_BASIS.

    Notes:
      * ``prompt_tokens`` is the canonical ``input_tokens`` (includes cached).
      * ``completion_tokens`` is the canonical ``output_tokens`` (includes reasoning).
      * ``cached_tokens`` is a subset of ``prompt_tokens`` and priced separately when a
        distinct cached-input rate is available in the model registry.

    Pricing is resolved **exclusively** from ``backend.llm.model_registry``:

      * When a matching ``ModelInfo`` is found, per-million rates are taken from its
        ``pricing`` field and costs are computed by splitting input into non-cached and
        cached portions.
      * If the model cannot be resolved from the registry, this function returns zeros
        for all cost fields. In this deployment that should be treated as a
        configuration error (missing ModelInfo or pricing), not as a valid "free" run.

    NOTE: This function only affects cost math. It does NOT change any pipeline
    control flow or LLM behavior.
    """

    # --- 1) Model-registry pricing (preferred) ---
    pricing = llm_handler.get_pricing_for_model(provider=provider, model=model, model_key=model_key)

    if pricing is not None:
        try:
            in_rate = float(getattr(pricing, "input_per_mm", 0.0) or 0.0)
            out_rate = float(getattr(pricing, "output_per_mm", 0.0) or 0.0)
            cached_rate = float(getattr(pricing, "cached_input_per_mm", 0.0) or 0.0)
        except Exception:
            in_rate = out_rate = cached_rate = 0.0

        # Split canonical input_tokens into non-cached and cached portions so that
        # only the non-cached portion is billed at the primary input rate.
        non_cached = max(int(prompt_tokens) - int(cached_tokens), 0)

        cost_prompt = (non_cached / COST_BASIS) * in_rate
        cost_cached = (cached_tokens / COST_BASIS) * cached_rate
        # `completion_tokens` here is the canonical `output_tokens` (includes reasoning).
        cost_completion = (completion_tokens / COST_BASIS) * out_rate
        total = cost_prompt + cost_cached + cost_completion
        return {
            "cost_prompt": round(cost_prompt, 8),
            "cost_cached": round(cost_cached, 8),
            "cost_completion": round(cost_completion, 8),
            "cost_total": round(total, 8),
        }

    # If model registry cannot be resolved, return zero costs rather than
    # guessing. In this deployment, this should be treated as a configuration
    # error (missing ModelInfo or pricing) and will be logged explicitly.
    try:
        _prov = str(provider or "").strip()
        _model_str = str(model or "").strip()
        _mk = str(model_key or "").strip()
        logger.error(
            "[METRICS] Missing pricing in model_registry for provider=%s model=%s model_key=%s; "
            "returning zero costs.",
            _prov,
            _model_str,
            _mk,
        )
    except Exception:
        # Never break the pipeline due to logging.
        pass

    return {
        "cost_prompt": 0.0,
        "cost_cached": 0.0,
        "cost_completion": 0.0,
        "cost_total": 0.0,
    }


# --- Centralized metrics helper (no integration yet) ---
class Metrics:
    """Centralizes stage usage parsing, cost math, and totals.

    Usage pattern (later steps):
        m = Metrics(settings, CONVO_TOTALS)
        m.record_stage("inference", model=settings.inference_model, usage=resp.usage)
        m.finalize_turn()
        turn_metrics, convo = m.snapshot()
    """
    def __init__(self, settings_obj, convo_totals_ref: Dict[str, Any]):
        self.settings = settings_obj
        # Resolve embedding spec once so we can report the concrete model name
        try:
            _emb_spec = resolve_embedding_spec(settings_obj)
            _emb_model_name = str(_emb_spec.get("model") or getattr(settings_obj, "embedding_model", "embedding"))
        except Exception:
            _emb_model_name = getattr(settings_obj, "embedding_model", "embedding")
        # Exact shape expected by the UI
        self.turn: Dict[str, Any] = {
            "embedding": {"model": _emb_model_name, "input_tokens": 0, "cost": 0.0},
            "rerank": {"model": settings_obj.re_ranker_model, "input_tokens": 0, "output_tokens": 0, "candidates_reranked": 0, "cost": 0.0},
            "summary": {"model": settings_obj.summarizer_model, "applied": False, "reason": "", "input_tokens": 0, "output_tokens": 0, "cost": 0.0},
            "rewrite": {"model": getattr(settings_obj, "rewrite_model", settings_obj.inference_model), "applied": False, "reason": "", "input_tokens": 0, "output_tokens": 0, "cost": 0.0},
            # Inference pass #1 (initial answer / tool-planning)
            "inference": {
                "model": settings_obj.inference_model,
                "input_tokens": 0,
                "cached_tokens": 0,
                "output_tokens": 0,
                "reasoning_tokens": 0,
                "cost_input": 0.0,
                "cost_cached": 0.0,
                "cost_output": 0.0,
                "cost_total": 0.0,
            },
            # Inference pass #2 (tool synthesis). Uses same model as inference for consistency.
            "inference_tools_synth": {
                "model": settings_obj.inference_model,
                "input_tokens": 0,
                "cached_tokens": 0,
                "output_tokens": 0,
                "reasoning_tokens": 0,
                "cost_input": 0.0,
                "cost_cached": 0.0,
                "cost_output": 0.0,
                "cost_total": 0.0,
            },
            "totals": {"tokens": {"turn_total": 0}, "cost": {"turn_total": 0.0}},
        }
        # Module-level accumulator reference (shared per process)
        self.convo: Dict[str, Any] = convo_totals_ref

    # --- Helpers ---
    def _normalize_usage(self, resp_or_usage: Any, provider: str = "openai") -> Dict[str, int]:
        """Return dict with canonical usage fields (zeros if missing).

        Canonical fields: input_tokens, cached_tokens, output_tokens,
                          reasoning_tokens, completion_tokens, total_tokens.
        """
        try:
            # Accept either a full response object, a dict with nested usage, or a plain usage dict
            if hasattr(resp_or_usage, "usage"):
                # Full Responses API object
                u = _extract_usage_from_responses(resp_or_usage, provider=provider)
            elif isinstance(resp_or_usage, dict) and ("usage" in resp_or_usage):
                # Dict wrapping usage -> extract
                u = _extract_usage_from_responses(resp_or_usage, provider=provider)
            elif isinstance(resp_or_usage, dict) and (
                "input_tokens" in resp_or_usage
                or "output_tokens" in resp_or_usage
            ):
                # Already a canonical usage dict
                u = resp_or_usage
            else:
                u = None
        except Exception:
            u = None
        u = u or {}
        return {
            "input_tokens": int(u.get("input_tokens", 0) or 0),
            "cached_tokens": int(u.get("cached_tokens", 0) or 0),
            "output_tokens": int(u.get("output_tokens", 0) or 0),
            "reasoning_tokens": int(u.get("reasoning_tokens", 0) or 0),
            "completion_tokens": int(u.get("completion_tokens", 0) or 0),
            "total_tokens": int(u.get("total_tokens", 0) or 0),
        }

    def _cost(self, stage: str, model: str, pt: int, ct: int, cached: int, model_key: str | None = None) -> Dict[str, float]:
        # Delegate to existing utility for a single source of truth
        # NOTE: Costs are resolved via model_registry when possible.
        return _compute_stage_cost(
            stage,
            prompt_tokens=pt,
            completion_tokens=ct,
            cached_tokens=cached,
            model=model,
            model_key=model_key,
        )

    # --- Public API ---
    def record_stage(
        self,
        stage: str,
        *,
        model: str,
        usage: Any | None = None,
        pt: int | None = None,
        ct: int | None = None,
        cached: int | None = None,
        model_key: str | None = None,
        extra: Dict[str, Any] | None = None,
    ) -> None:
        """Record metrics for a pipeline stage.
        Either pass a `usage` (response or usage dict) or explicit pt/ct/cached counts.
        `extra` lets callers set fields like candidates_reranked/applied/reason.
        """
        if stage not in self.turn:
            return
        # Always stamp the model that ran
        self.turn[stage]["model"] = model

        # Extract canonical usage fields
        reasoning = 0
        if usage is not None and pt is None and ct is None and cached is None:
            u = self._normalize_usage(usage)
            pt, ct, cached, reasoning = u["input_tokens"], u["output_tokens"], u["cached_tokens"], u["reasoning_tokens"]
        pt = int(pt or 0)
        ct = int(ct or 0)
        cached = int(cached or 0)
        reasoning = int(reasoning or 0)

        if stage == "embedding":
            # input-only; we treat provided pt as input_tokens
            self.turn[stage]["input_tokens"] = pt
            c = self._cost("embedding", model, pt, 0, 0, model_key=model_key)
            self.turn[stage]["cost"] = c["cost_prompt"]
        elif stage == "rerank":
            # Use canonical input_tokens; cached is a subset and tracked separately via cost math.
            self.turn[stage]["input_tokens"] = pt
            self.turn[stage]["output_tokens"] = ct
            c = self._cost("rerank", model, pt, ct, cached, model_key=model_key)
            self.turn[stage]["cost"] = c["cost_total"]
        elif stage == "summary":
            self.turn[stage]["input_tokens"] = pt
            self.turn[stage]["output_tokens"] = ct
            c = self._cost("summary", model, pt, ct, cached, model_key=model_key)
            self.turn[stage]["cost"] = c["cost_total"]
        elif stage == "rewrite":
            self.turn[stage]["input_tokens"] = pt
            self.turn[stage]["output_tokens"] = ct
            c = self._cost("rewrite", model, pt, ct, cached, model_key=model_key)
            self.turn[stage]["cost"] = c["cost_total"]
        elif stage in ("inference", "inference_tools_synth"):
            # Accumulate tokens and costs across multiple inference calls in a single turn.
            prev_in = int(self.turn[stage].get("input_tokens") or 0)
            prev_ck = int(self.turn[stage].get("cached_tokens") or 0)
            prev_out = int(self.turn[stage].get("output_tokens") or 0)
            prev_reason = int(self.turn[stage].get("reasoning_tokens") or 0)

            in_total = prev_in + pt
            ck_total = prev_ck + cached
            out_total = prev_out + ct
            reason_total = prev_reason + reasoning

            self.turn[stage]["input_tokens"] = in_total
            self.turn[stage]["cached_tokens"] = ck_total
            self.turn[stage]["output_tokens"] = out_total
            self.turn[stage]["reasoning_tokens"] = reason_total

            # Cost for this specific call (still priced under the "inference" stage)
            c = self._cost("inference", model, pt, ct, cached, model_key=model_key)
            self.turn[stage]["cost_input"] = float(self.turn[stage].get("cost_input", 0.0)) + c["cost_prompt"]
            self.turn[stage]["cost_cached"] = float(self.turn[stage].get("cost_cached", 0.0)) + c["cost_cached"]
            self.turn[stage]["cost_output"] = float(self.turn[stage].get("cost_output", 0.0)) + c["cost_completion"]
            self.turn[stage]["cost_total"] = float(self.turn[stage].get("cost_total", 0.0)) + c["cost_total"]

        if extra:
            try:
                self.turn[stage].update(extra)
            except Exception:
                pass

    def finalize_turn(self) -> None:
        """Compute turn rollups and accumulate into conversation totals."""
        emb = int(self.turn["embedding"].get("input_tokens") or 0)
        rin = int(self.turn["rerank"].get("input_tokens") or 0)
        rout = int(self.turn["rerank"].get("output_tokens") or 0)
        sin = int(self.turn["summary"].get("input_tokens") or 0)
        sout = int(self.turn["summary"].get("output_tokens") or 0)
        rwin = int(self.turn["rewrite"].get("input_tokens") or 0)
        rwout = int(self.turn["rewrite"].get("output_tokens") or 0)

        # Inference pass #1
        ip1 = int(self.turn["inference"].get("input_tokens") or 0)
        ik1 = int(self.turn["inference"].get("cached_tokens") or 0)
        ic1 = int(self.turn["inference"].get("output_tokens") or 0)

        # Inference pass #2 (tool synthesis)
        ip2 = int(self.turn["inference_tools_synth"].get("input_tokens") or 0)
        ik2 = int(self.turn["inference_tools_synth"].get("cached_tokens") or 0)
        ic2 = int(self.turn["inference_tools_synth"].get("output_tokens") or 0)

        # Combined for totals/conversation metrics
        ip = ip1 + ip2
        ik = ik1 + ik2
        ic = ic1 + ic2

        # NOTE: cached tokens are a subset of prompt/input tokens; do NOT add them again to totals.
        total_tokens = emb + rin + rout + sin + sout + rwin + rwout + ip + ic
        self.turn["totals"]["tokens"]["turn_total"] = total_tokens

        total_cost = (
            float(self.turn["embedding"].get("cost") or 0.0)
            + float(self.turn["rerank"].get("cost") or 0.0)
            + float(self.turn["summary"].get("cost") or 0.0)
            + float(self.turn["rewrite"].get("cost") or 0.0)
            + float(self.turn["inference"].get("cost_total") or 0.0)
            + float(self.turn["inference_tools_synth"].get("cost_total") or 0.0)
        )
        self.turn["totals"]["cost"]["turn_total"] = round(total_cost, 8)

        # Accumulate into shared conversation totals (robust to 'cost' vs 'costs')
        try:
            self.convo["tokens"]["embedding"] += emb
            # NOTE: cached tokens are already included in stage input/prompt token counts; track them separately but don't double-count.
            self.convo["tokens"]["llm_input"] += (rin + sin + rwin + ip)
            self.convo["tokens"]["llm_output"] += (rout + sout + rwout + ic)
            self.convo["tokens"]["conversation_total"] += total_tokens
            if "cost" in self.convo:
                self.convo["cost"]["conversation_total"] = round(float(self.convo["cost"].get("conversation_total", 0.0)) + total_cost, 8)
            elif "costs" in self.convo:
                self.convo["costs"]["conversation_total"] = round(float(self.convo["costs"].get("conversation_total", 0.0)) + total_cost, 8)
            logger.debug("[TOTALS] Metrics Finalize Turn turn_total=%d convo_total_now=%d" % (self.turn["totals"]["tokens"]["turn_total"], self.convo["tokens"]["conversation_total"]))
        except Exception:
            # Never let metrics break the answer path
            logger.error("[TOTALS] Metrics Finalize Turn Failure")
            pass

    def snapshot(self) -> tuple[Dict[str, Any], Dict[str, Any]]:
        """Return the current turn metrics and a frontend-aligned conversation totals snapshot."""
        convo_cost = 0.0
        if isinstance(self.convo, dict):
            if "cost" in self.convo:
                convo_cost = float(self.convo["cost"].get("conversation_total", 0.0))
            elif "costs" in self.convo:
                convo_cost = float(self.convo["costs"].get("conversation_total", 0.0))
        convo_snapshot = {
            "tokens": self.convo.get("tokens", {"embedding": 0, "llm_input": 0, "llm_output": 0, "conversation_total": 0}),
            "cost": {"conversation_total": convo_cost},
        }
        return self.turn, convo_snapshot

    def reset_convo(self) -> None:
        try:
            _zero_convo_totals()
        except Exception:
            pass
# --- end Metrics helper ---

# --- Simple utilities to collapse duplicate sources by URL + payload fields ---

def _render_source_line(indices: list[int], url: str, section: str, subsection: str) -> str:
    idx_text = ", ".join(str(i) for i in sorted(set(indices)))
    return f"[{idx_text}] {url} (Section: {section} > {subsection})"


def _collapse_sources(indexed_items: List[Dict[str, Any]]) -> str:
    """Group by (url, section, subsection) and collapse indices.
    `indexed_items` items look like: {index:int, url:str, section:str, subsection:str}
    Returns a single string with one line per unique group.
    """
    groups: Dict[tuple, Dict[str, Any]] = {}
    for it in indexed_items:
        url = (it.get("url") or "unknown").strip()
        section = (it.get("section") or "N/A").strip()
        subsection = (it.get("subsection") or "N/A").strip()
        key = (url, section, subsection)
        if key not in groups:
            groups[key] = {"indices": [], "url": url, "section": section, "subsection": subsection}
        idx = int(it.get("index", 0) or 0)
        if idx > 0:
            groups[key]["indices"].append(idx)

    lines: List[str] = []
    for (_url, _section, _subsection), data in groups.items():
        if data["indices"]:
            lines.append(_render_source_line(data["indices"], data["url"], data["section"], data["subsection"]))
    return "\n".join(lines)


# --- end utilities ---

# --- Summary cache key helper ---

def _summary_cache_key(msgs: List[Dict[str, str]] | None, tag: str = "") -> str:
    """Create a stable cache key for a list of {role, content} messages.
    `tag` distinguishes different uses (e.g., 'rewrite' vs 'inference').
    """
    import hashlib as _hl
    items = msgs or []
    m = _hl.sha1()
    m.update(tag.encode("utf-8"))
    for it in items:
        role = (it.get("role") or "").strip()
        content = (it.get("content") or "").strip()
        m.update(role.encode("utf-8"))
        m.update(b"\x1f")
        m.update(content.encode("utf-8"))
        m.update(b"\x1e")
    return m.hexdigest()

# --- Query rewrite helpers (not wired; no behavior change) ---
_REWRITE_DEICTIC_RE = re.compile(r"\b(it|this|that|these|those|here|there|they|them|their|its|he|she|his|her|also|then)\b", re.I)
_REWRITE_SHORT_Q_RE = re.compile(r"\b(where|when|how|why|which|what)\b", re.I)

def should_rewrite(message: str) -> bool:
    """Heuristic: return True if the message is likely underspecified (coreference or very short).
    Safe default: if this returns False, we skip rewrite and use the original message.
    Diagnostic logging included.
    """
    if not message:
        logger.debug("[REWRITE] heuristic=empty_message -> False")
        return False
    txt = message.strip()
    # Short messages are often follow-ups (<= 7 words)
    if len(txt.split()) <= 7:
        logger.debug("[REWRITE] heuristic=short_message words=%d -> True", len(txt.split()))
        return True
    # Contains deictic pronouns or bare question words (without explicit entities)
    if _REWRITE_DEICTIC_RE.search(txt) and _REWRITE_SHORT_Q_RE.search(txt):
        logger.debug("[REWRITE] heuristic=deictic+wh -> True")
        return True
    logger.debug("[REWRITE] heuristic=none -> False")
    return False


def build_rewrite_prompt(
    tail_messages: List[Dict[str, str]] | None,
    summary_text: str,
    message: str,
) -> str:
    """Build a structured prompt for the rewrite model using conversation history.

    Uses:
      - Optional long-term summary (summary_text)
      - Recent verbatim turns (tail_messages)
      - Current user message (message)

    The model is instructed to output ONLY a JSON object with the shape:
      {"rewritten":"...","changed":true|false,"confidence":0.0,"ambiguous":true|false,"reason":"..."}
    """
    parts: List[str] = []

    # 1. SYSTEM IDENTITY (Static/Cacheable)
    parts.append("### ROLE\n")
    parts.append(
        "You are a Search Query Optimizer. Your task is to rewrite the user's latest message "
        "into a single, standalone search query that can be sent to a vector database or search engine, "
        "using only the provided conversation context.\n\n"
    )

    # 2. HIERARCHICAL INSTRUCTIONS (Static/Cacheable)
    parts.append("### INSTRUCTIONS\n")
    parts.append(
        "1. PRIORITY: Use the 'RECENT CONVERSATION' to resolve immediate pronouns and vague references "
        "(e.g., it, this, that, they, those, their, its).\n"
    )
    parts.append(
        "2. BACKGROUND: Use the 'CONVERSATION SUMMARY' only to understand the overall topic when the recent turns "
        "are not sufficient by themselves.\n"
    )
    parts.append(
        "3. STANDALONE QUERY: The 'rewritten' field must be a clear, self-contained search query. "
        "If the user's latest message is already a clear standalone question, return it unchanged and set "
        "'changed': false.\n"
    )
    parts.append(
        "4. NO CHAT / NO ANSWERS: Do not answer the question. Do not add explanations, opinions, or chit-chat. "
        "Your only job is to rewrite the query for retrieval.\n"
    )
    parts.append(
        "5. AMBIGUITY HANDLING: If you cannot confidently resolve what a pronoun or vague reference refers to, "
        "keep the original question unchanged, set 'ambiguous': true, and briefly explain why in 'reason'.\n\n"
    )

    # 3. OUTPUT SCHEMA (Static/Cacheable)
    parts.append("### OUTPUT SCHEMA\n")
    parts.append(
        "Return STRICTLY a single JSON object with this shape and field meanings (no extra text, no code fences):\n"
    )
    parts.append(
        '{"rewritten":"...","changed":true|false,"confidence":0.0,'
        '"ambiguous":true|false,"reason":"..."}\n'
    )
    parts.append(
        "- rewritten: the final standalone search query.\n"
        "- changed: true if you modified the user question, false if you kept it as-is.\n"
        "- confidence: a float from 0.0 to 1.0 indicating how confident you are in the rewrite.\n"
        "- ambiguous: true if the context is too unclear to safely rewrite; otherwise false.\n"
        "- reason: a short phrase (max ~15-20 tokens) explaining your decision.\n\n"
    )

    # 4. FEW-SHOT EXAMPLES (Static/Cacheable)
    parts.append("### EXAMPLES\n")

    # EXAMPLE 1: Mount Whitney (domain-specific, location + weather + airport)
    parts.append("EXAMPLE 1\n")
    parts.append("SUMMARY: user is interested in mount whitney.\n")
    parts.append(
        'RECENT: USER: "what is the elevtion ." | ASSISTANT: "The elevation is 14505 ft."\n'
    )
    parts.append('CURRENT: "Current weather and closest airport."\n')
    parts.append(
        '{"rewritten": "what is the current weather in Mount Whitney, California and the closest airport to it", '
        '"changed": true, "confidence": 0.98, "ambiguous": false, '
        '"reason": "added full context for mount whitney"}\n\n'
    )

    # EXAMPLE 2: SDK / Linux compatibility (resolving "it" to Python SDK)
    parts.append("EXAMPLE 2\n")
    parts.append("SUMMARY: user is installing and using a Python SDK.\n")
    parts.append(
        'RECENT: USER: "How do I install the Python SDK?" | ASSISTANT: "You can install it with pip using `pip install my-sdk`."\n'
    )
    parts.append('CURRENT: "Does it work on Linux?"\n')
    parts.append(
        '{"rewritten": "Linux compatibility and system requirements for the my-sdk Python SDK", '
        '"changed": true, "confidence": 0.96, "ambiguous": false, '
        '"reason": "resolved it to Python SDK and added linux compatibility context"}\n\n'
    )

    # EXAMPLE 3: Q3 revenue / projections (topic continuation)
    parts.append("EXAMPLE 3\n")
    parts.append("SUMMARY: user is asking about company financial performance for Q3.\n")
    parts.append(
        'RECENT: USER: "Tell me about the Q3 revenue." | ASSISTANT: "Q3 revenue was $45M, up 12% year-over-year."\n'
    )
    parts.append('CURRENT: "What about the projections?"\n')
    parts.append(
        '{"rewritten": "Q3 revenue projections and future financial outlook for the company", '
        '"changed": true, "confidence": 0.94, "ambiguous": false, '
        '"reason": "used prior Q3 revenue topic to expand vague projections into standalone query"}\n\n'
    )

    # EXAMPLE 4: Car clicking sound (entity clarification)
    parts.append("EXAMPLE 4\n")
    parts.append("SUMMARY: user has a car with a clicking sound and wants to understand the issue.\n")
    parts.append(
        'RECENT: USER: "My car is making a clicking sound when I accelerate." | ASSISTANT: "That can have several causes, typically related to CV joints or engine components."\n'
    )
    parts.append('CURRENT: "How much to fix it?"\n')
    parts.append(
        '{"rewritten": "cost and repair estimates to fix a car that makes a clicking sound when accelerating", '
        '"changed": true, "confidence": 0.92, "ambiguous": false, '
        '"reason": "expanded it into explicit car clicking sound repair cost query"}\n\n'
    )

    # EXAMPLE 5: Already clear airports question (no semantic change, but more context)
    parts.append("EXAMPLE 5\n")
    parts.append("SUMMARY: user is researching airports near Mount Whitney.\n")
    parts.append(
        'RECENT: USER: "What are the closest airports to Mount Whitney?" | ASSISTANT: "There are several nearby, including Bishop and Fresno Yosemite."\n'
    )
    parts.append('CURRENT: "What are the closest airports to Mount Whitney?"\n')
    parts.append(
        '{"rewritten": "What are the closest airports to Mount Whitney?", '
        '"changed": false, "confidence": 0.99, "ambiguous": false, '
        '"reason": "question already clear and self-contained"}\n\n'
    )

    # EXAMPLE 6: Truly ambiguous follow-up (keep original, mark ambiguous)
    parts.append("EXAMPLE 6\n")
    parts.append("SUMMARY: user is asking many unrelated questions about different products and topics.\n")
    parts.append(
        'RECENT: USER: "How do I reset my router?" | ASSISTANT: "Press and hold the reset button for 10 seconds."\n'
    )
    parts.append('CURRENT: "What about that one?"\n')
    parts.append(
        '{"rewritten": "What about that one?", '
        '"changed": false, "confidence": 0.2, "ambiguous": true, '
        '"reason": "cannot determine what that one refers to from context"}\n\n'
    )

    # 5. DYNAMIC DATA (Changes every turn)
    if summary_text:
        parts.append("### CONVERSATION SUMMARY (Long-term Context)\n")
        parts.append(summary_text.strip())
        parts.append("\n\n")

    if tail_messages:
        parts.append("### RECENT CONVERSATION (Immediate Context)\n")
        for m in tail_messages:
            role = (m.get("role") or "user").upper()
            content = m.get("content", "") or ""
            parts.append(f"{role}: {content}\n")
        parts.append("\n")

    parts.append("### CURRENT USER QUESTION\n")
    parts.append(message.strip())
    parts.append("\n")

    return "".join(parts)


def rewrite_query(
    tail_messages: List[Dict[str, str]] | None,
    summary_text: str,
    message: str,
    log_prefix: str = "[REWRITE]",
    stage_spec: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """Call the rewrite model to produce a self-contained query.
    Returns a dict with keys: rewritten, changed, confidence, ambiguous, reason.
    On any failure, returns the original unmodified with changed=False.
    """
    try:
        prompt = build_rewrite_prompt(tail_messages, summary_text, message)
        # Log an estimated prompt token count for rewrite
        try:
            _rw = stage_spec or {}
            _model_for_est = str(_rw.get("model") or settings.rewrite_model)
            enc = _get_encoder_for_model(_model_for_est)
            pt_est = len(enc.encode(prompt))
            #logger.debug(f"{log_prefix} prompt_token_est≈%d model=%s", pt_est, _model_for_est)
        except Exception:
            pass
        # Invoke the rewrite model with the prompt for the user's latest message for it to rewrite it
        _rw = stage_spec or {}
        _provider = str(_rw.get("provider") or "openai")
        _model = str(_rw.get("model") or settings.rewrite_model)
        _kwargs = dict(_rw.get("kwargs") or {})
        if not _kwargs:
            _kwargs = {
                "max_output_tokens": int(settings.rewrite_max_output_tokens),
                "temperature": float(settings.rewrite_temperature),
            }

        resp = _responses_create(
            provider=_provider,
            model=_model,
            input=prompt,
            **_kwargs,
        )
        usage = _extract_usage_from_responses(resp, provider=_provider)
        raw = _extract_text_from_responses(resp).strip()
        # Log the raw JSON candidate before parsing so we can debug provider outputs.
        #logger.debug("[REWRITE] JSON candidate before parsing=%s", raw)
        try:
            if isinstance(usage, dict):
                pt = int(usage.get("input_tokens") or 0)
                ct = int(usage.get("output_tokens") or 0)
                ck = int(usage.get("cached_tokens") or 0)
                tt = int(usage.get("total_tokens") or (pt + ct + ck))
                logger.debug(f"{log_prefix} usage input=%d cached=%d output=%d total=%d", pt, ck, ct, tt)
        except Exception:
            pass
        # Tolerate fenced code blocks
        if raw.startswith("```json") and raw.endswith("```"):
            raw = raw[7:-3].strip()
        elif raw.startswith("```") and raw.endswith("```"):
            raw = raw[3:-3].strip()
        
        # Validate JSON format and retry if invalid
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as e:
            logger.warning(f"{log_prefix} Invalid JSON from LLM: {e}. Response was: '{raw[:200]}...'")
            logger.warning(f"{log_prefix} This often happens when model ignores JSON schema instruction")
            logger.warning(f"{log_prefix} Response may be truncated - checking max_output_tokens setting")
            
            # Retry with stronger emphasis on JSON format
            retry_prompt = prompt + "\n\nIMPORTANT: You MUST return ONLY a complete JSON object. No conversational text. Ensure all fields are included."
            #logger.debug(f"{log_prefix} Retrying with stronger JSON instruction")
            
            resp_retry = _responses_create(
                provider=_provider,
                model=_model,
                input=retry_prompt,
                **_kwargs,
            )
            raw_retry = _extract_text_from_responses(resp_retry).strip()
            
            # Clean retry response
            if raw_retry.startswith("```json") and raw_retry.endswith("```"):
                raw_retry = raw_retry[7:-3].strip()
            elif raw_retry.startswith("```") and raw_retry.endswith("```"):
                raw_retry = raw_retry[3:-3].strip()
            
            try:
                data = json.loads(raw_retry)
                logger.info(f"{log_prefix} Retry successful, got valid JSON")
            except json.JSONDecodeError:
                logger.error(f"{log_prefix} Retry also failed, using original query")
                # Fallback to original with changed=False
                data = {
                    "rewritten": message,
                    "changed": False,
                    "confidence": 0.0,
                    "ambiguous": True,
                    "reason": "JSON parsing failed, using original"
                }
        
        # Debug: log parsed JSON (truncated)
        #try:
            #logger.debug(f"{log_prefix} Parsed JSON: {str(data)[:100]}...")
        #except Exception:
            #pass
        try:
            _t = int(getattr(settings, "debug_log_truncate_chars", 4000))
        except Exception:
            _t = 400
        try:
            _js = json.dumps(data, ensure_ascii=False)
            #logger.debug(f"{log_prefix} json=%s", _js if len(_js) <= _t else (_js[:_t] + "…"))
        except Exception:
            pass
        # Normalize/validate fields
        rewritten = str(data.get("rewritten", message) or message)
        changed = bool(data.get("changed", False))
        confidence = float(data.get("confidence", 0.0) or 0.0)
        ambiguous = bool(data.get("ambiguous", False))
        reason = str(data.get("reason", "") or "")
        return {
            "rewritten": rewritten,
            "changed": changed,
            "confidence": confidence,
            "ambiguous": ambiguous,
            "reason": reason,
            "_usage": usage,
        }
    except LLMError as e:
        # Special-case provider rate limits so callers can surface a clear message
        # instead of treating this as an ambiguous query.
        try:
            kind = getattr(e, "kind", "") or ""
            provider = getattr(e, "provider", "") or ""
            model = getattr(e, "model", "") or ""
        except Exception:
            kind = ""
            provider = ""
            model = ""

        if kind == "rate_limit":
            logger.warning(
                "%s provider rate limit in rewrite: provider=%s model=%s error=%s",
                log_prefix,
                provider,
                model,
                e,
                exc_info=True,
            )
            # Mark as a rate-limit condition without flagging ambiguity so the
            # pipeline can handle this distinctly (e.g., by emitting a final
            # quota-exceeded message instead of entering Clarify).
            return {
                "rewritten": message,
                "changed": False,
                "confidence": 0.0,
                "ambiguous": False,
                "reason": "llm_rate_limit",
                "_usage": None,
                "_provider": provider,
                "_model": model,
            }

        # Non-rate-limit LLMErrors are treated as generic rewrite failures.
        logger.warning("[REWRITE] failed with LLMError; using original: %s", e, exc_info=True)
        return {
            "rewritten": message,
            "changed": False,
            "confidence": 0.0,
            "ambiguous": True,
            "reason": "rewrite_error_or_ambiguous",
            "_usage": None,
        }
    except Exception as e:
        # Never let rewrite failures affect the main flow
        logger.warning("[REWRITE] failed to parse/produce JSON: %s", e, exc_info=True)
        return {
            "rewritten": message,
            "changed": False,
            "confidence": 0.0,
            "ambiguous": True,
            "reason": "rewrite_error_or_ambiguous",
            "_usage": None,
        }
# --- end query rewrite helpers ---

class ChatManager:
    def reset_metrics(self):
        """Reset conversation totals and all cached chat state. Used by /chat/reset."""
        _zero_convo_totals()
        # Clear in-memory conversation state so "Clear chat" truly resets context.
        try:
            self.chat_history = []
        except Exception:
            pass
        # Clear per-instance summary cache used by ChatManager.chat()
        try:
            self._summary_cache.clear()
        except Exception:
            pass

    def __init__(self):
        self.qdrant_db = QdrantDB(
            host=settings.qdrant_host,
            port=settings.qdrant_port,
            collection_name=settings.collection_name
        )
        self.web_search = WebSearchClient()
        self.chat_history = []
        # Per-instance lightweight cache for history summaries reused within a turn
        self._summary_cache: Dict[str, str] = {}

    def _get_context(self, query: str, limit: int | None = None, score_threshold: float | None = None) -> List[Dict]:
        """Get relevant context from QdrantDB"""
        #logger.debug("Searching Qdrant for query: %s", query)
        try:
            limit = limit or int(settings.top_k)
            score_threshold = float(score_threshold if score_threshold is not None else settings.score_threshold)
            logger.debug("Qdrant search using limit=%s, score_threshold=%s", limit, score_threshold)
            # Use HNSW for faster search and Exact = False for faster search
            results = self.qdrant_db.search_similar(
                query=query,
                limit=int(limit),
                score_threshold=float(score_threshold),
                with_vectors=False,
                with_payload=True,
                exact=getattr(settings, "exact_match", False),
            )
            logger.debug("Qdrant search returned %d results", len(results))
            if results:
                logger.debug("First result score: %s", results[0].get('score', 'N/A'))
            return results
        except Exception as e:
            logger.exception("Error in _get_context: %s", e)
            return []

    def _get_web_context(self, query: str, existing_context: List[Dict]) -> List[Dict]:
        """Get additional context from web search"""
        return self.web_search.get_additional_context(query, existing_context)

    def chat(self, message: str, context: List[Dict], use_web_search: bool = False, params: Dict[str, Any] | None = None) -> Dict:
        """
        Thin wrapper: delegate to run_pipeline (Option A).
        Maintains stateful history and returns answer + sources.
        """

        # Prefer caller-provided query_id (so SSE subscriber can pre-open /chat/stream/stages?query_id=...)
        _p = params or {}
        req_id = str(_p.get("query_id") or _p.get("request_id") or uuid.uuid4().hex[:8])
        logger.info("Starting chat in chat_manager.chat() [req_id=%s]", req_id, extra={"message": message})
        logger.debug("Context length=%d use_web_search=%s", len(context), use_web_search)

        # Always use orchestrator; legacy inlined flow removed (kept in git history).
        try:
            deps = {
                "db": self.qdrant_db,
                "cache": self._summary_cache,
                "settings": settings,
                "list_tools": list_tools,
                "get_executor": get_executor,
                "get_web_context": (lambda q, existing: self._get_web_context(q, existing)) if use_web_search else (lambda q, existing: []),
                # Preserve previous stateful behavior: build a single prompt string (no tools)
                "style": "messages",
                "enable_tools": False,
                "enable_query_rewrite": bool(getattr(settings, "enable_query_rewrite", False)),
                "use_web_search": bool(use_web_search),
                "log_origin": "chat_manager.chat[orchestrator]",
                "request_id": req_id,
            }
            req = {"message": message, "history": self.chat_history, "params": (params or {})}

            out = run_pipeline(deps=deps, req=req)
            answer_text = out.get("answer", "") or ""

            # Update stateful history to preserve conversation context
            self.chat_history.extend([
                {"role": "user", "content": message},
                {"role": "assistant", "content": answer_text},
            ])

            return {
                "response": answer_text,
                "sources": out.get("sources", []),
            }
        except Exception as e:
            logger.exception("Exception in chat: %s", e)
            err_text = f"I'm sorry, I encountered an error while processing your request: {str(e)}"
            # Best-effort: terminate SSE stage stream on errors so UI doesn't hang when using this stateful path.
            try:
                emit_stage(req_id, "Final Answer", final=True, finalContent=err_text)
            except Exception:
                pass
            try:
                emit_stage(req_id, "Done", final=True)
            except Exception:
                pass
            try:
                close_stream(req_id)
            except Exception:
                pass
            return {
                "response": err_text,
                "sources": []
            }
# --- debug helper ---

def _dbg(label: str, text: str) -> None:
    """Guarded debug logging with truncation, controlled by config flags.
    Only logs when settings.debug_verbose is True.
    """
    try:
        if settings.debug_verbose:
            maxc = int(settings.debug_log_truncate_chars)
            snippet = text if len(text) <= maxc else (text[:maxc] + "…")
            logger.debug("%s %s", label, snippet)
    except Exception:
        # Never let logging break flow
        pass


# --- Tool-call parsing helpers (module-level, pure) ---
def extract_tool_calls(resp: Any) -> List[Dict[str, Any]]:
    """Extract tool/function calls from a Responses API object or dict.
    Returns a list of {name, args, id}.
    """
    # Unwrap adapter-style responses first (e.g., AdapterResponse from llm_handler)
    # so we always inspect the provider-native object for tool_calls.
    base = getattr(resp, "adapter_response", resp)
    try:
        logger.debug("[TOOLS] extract_tool_calls: base type=%s repr=%r", type(base), base)
    except Exception:
        # Never let logging break tool-call extraction
        pass

    def _dedup(_calls: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Best-effort deduplication across providers/wrappers.
        Dedup key prefers call id when present, otherwise falls back to (name, args).
        """
        out: List[Dict[str, Any]] = []
        seen: set[tuple] = set()
        for c in _calls:
            try:
                name = (c.get("name") or "").strip()
                cid = (c.get("id") or "").strip()
                args = c.get("args")
                if isinstance(args, str):
                    akey = args
                elif isinstance(args, dict):
                    try:
                        akey = json.dumps(args, sort_keys=True, ensure_ascii=False)
                    except Exception:
                        akey = str(args)
                else:
                    akey = str(args)
                key = (cid,) if cid else (name, akey)
                if name and key not in seen:
                    seen.add(key)
                    out.append({"name": name, "args": args, "id": c.get("id")})
            except Exception:
                # If anything goes wrong, keep the call to avoid dropping tool execution.
                out.append(c)
        return out

    try:
        # Prefer adapter_response surface when present (e.g., Gemini
        # _GeminiResponsesWrapper); otherwise, pass the response as-is.
        llm_base = getattr(resp, "adapter_response", resp)
        llm_result = llm_handler.build_llm_result_from_response(llm_base)
        calls = list(llm_result.get("tool_calls") or [])
    except Exception:
        calls = []

    calls = [c for c in calls if c.get("name")]
    try:
        logger.debug("[TOOLS] extract_tool_calls: raw calls from LLMResult before dedup: %r", calls)
    except Exception:
        pass
    deduped = _dedup(calls)
    try:
        logger.debug("[TOOLS] extract_tool_calls: deduped calls from LLMResult: %r", deduped)
    except Exception:
        pass

    return deduped


def parse_tool_args(raw: Any) -> Dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except Exception:
            return {}
    return {}

#
# --- History slicing helper  ---
def split_history_for_prompt(history_msgs: List[Dict[str, str]] | None, raw_tail_turns: int, window_turns: int):
    """
    Split a flat message list into (to_summarize, verbatim_tail).

    Definitions:
      - 1 turn = 2 messages (user + assistant).
      - verbatim_tail: the last `raw_tail_turns` turns (up to available), kept verbatim in the prompt.
      - to_summarize: `window_turns` turns immediately before the tail, used to build a short summary.

    This function is pure and does not modify the input list.
    """
    msgs = history_msgs or []
    msgs_per_turn = 2
    tail_msg_count = max(0, int(raw_tail_turns)) * msgs_per_turn
    window_msg_count = max(0, int(window_turns)) * msgs_per_turn

    total = len(msgs)
    # Verbatim tail = last K turns (2*K messages)
    verbatim_tail = msgs[-tail_msg_count:] if tail_msg_count > 0 else []

    # Summary window = the turns immediately before the tail (2*window_turns messages)
    end = max(0, total - tail_msg_count)
    start = max(0, end - window_msg_count)
    to_summarize = msgs[start:end]

    return to_summarize, verbatim_tail

# --- end helper ---


# --- Tail cleanup helper (optional, cosmetic) ---
def _strip_trailing_sources_block(text: str) -> str:
    """
    Remove a trailing 'Sources:' block from an assistant message, if present.
    Only strips when the block appears at the end to avoid cutting inline mentions.
    """
    try:
        s = (text or "").rstrip()
        # Match a final block that starts with 'Sources:' on its own line to the end of string
        m = re.search(r"(?:\r?\n)Sources:\s*\r?\n[\s\S]*\Z", s)
        if m:
            s = s[:m.start()]
        return s.rstrip()
    except Exception:
        return text or ""

# --- end tail cleanup helper - when tool results are present but no sources---

def _has_tool_results(tools_used: Any) -> bool:
    """True if tools were executed / tool results exist."""
    try:
        if not tools_used:
            return False
        if isinstance(tools_used, list):
            return len(tools_used) > 0
        if isinstance(tools_used, dict):
            return len(tools_used.keys()) > 0
        return True
    except Exception:
        return False


def _should_emit_no_supported_sources(sources: Any, tools_used: Any) -> bool:
    """Only emit NO_SUPPORTED_SOURCES when we have neither doc sources nor tool results."""
    try:
        has_sources = isinstance(sources, list) and len(sources) > 0
    except Exception:
        has_sources = bool(sources)
    return (not has_sources) and (not _has_tool_results(tools_used))

# --- Unified pipeline orchestrator (Option A) ---

def run_pipeline(*, deps: Dict[str, Any], req: Dict[str, Any]) -> Dict[str, Any]:
    """
    Unified pipeline:
    retrieve -> maybe_rerank -> summarize -> build prompt -> inference -> optional tools -> sources -> metrics

    deps:
      - db: QdrantDB-like (must support search_similar)
      - cache: dict-like for summaries
      - settings: Settings object
      - list_tools: callable() -> list of tools
      - get_executor: callable(name) -> tool executor
      - get_web_context: callable(query, existing_context) -> list (optional)
      - style: 'messages' | 'flat'
      - enable_tools: bool
      - enable_query_rewrite: bool
      - use_web_search: bool
      - log_origin: str (for logs)
    req:
      - message: str
      - history: list[{role, content}]
      - params: dict
    """
    settings_obj = deps["settings"]
    db = deps["db"]
    cache = deps.get("cache", {})
    style = deps.get("style", "flat")
    enable_tools = bool(deps.get("enable_tools", False))
    enable_query_rewrite = bool(deps.get("enable_query_rewrite", False))
    use_web_search = bool(deps.get("use_web_search", False))
    get_web_context_fn = deps.get("get_web_context") or (lambda q, existing: [])
    list_tools_fn = deps.get("list_tools", list_tools)
    get_executor_fn = deps.get("get_executor", get_executor)
    log_origin = str(deps.get("log_origin", "pipeline"))
    req_id = deps.get("request_id") or uuid.uuid4().hex[:8]
    log_origin = f"{log_origin}#{req_id}"
    # Option A: namespace for cache keying (may be empty for backward compat)
    namespace = str(deps.get("namespace", "") or "").strip()
    # For stateless paths that use the module-level summary cache, track last-seen
    # and evict idle namespaces based on a TTL. This is best-effort and does not
    # affect correctness; it only bounds process memory.
    try:
        if namespace and (cache is _SUMMARY_CACHE):
            _touch_namespace(namespace)
            _evict_idle_namespaces()
    except Exception:
        # Never let cache housekeeping break the main flow
        pass

    message: str = (req or {}).get("message") or ""
    history: List[Dict[str, str]] = (req or {}).get("history") or []
    params: Dict[str, Any] = (req or {}).get("params") or {}

    # --- Model registry keys (cost-only) ---
    # Optional stable model aliases from params, used ONLY for accurate cost lookup.
    _mk = lambda k: (str(params.get(k)).strip() or None) if params.get(k) is not None else None
    _stage_model_keys = {s: _mk(f"{s}_model_key") for s in ("embedding", "rewrite", "summary", "rerank", "inference", "tools_synth")}
    _stage_model_keys["tools_synth"] = _stage_model_keys.get("tools_synth") or _stage_model_keys.get("inference")

    # Per-UI control for whether to append Sources: blocks and structured sources.
    # Mode is set by frontends (e.g. chat.js, chat-embed.js) via params.mode.
    try:
        mode = str(params.get("mode", "")).strip().lower() or "chat"
    except Exception:
        mode = "chat"
    try:
        if mode == "embed":
            display_sources = bool(getattr(settings_obj, "display_sources_for_embed", False))
        else:
            display_sources = bool(getattr(settings_obj, "display_sources_for_chat", True))
    except Exception:
        display_sources = True

    # Per-turn control for emitting intermediate processing stages to SSE.
    # Precedence: params.show_processing_steps (per turn) overrides settings_obj.show_processing_steps (global default).
    try:
        if "show_processing_steps" in params:
            show_processing_steps = bool(params.get("show_processing_steps"))
        else:
            show_processing_steps = bool(getattr(settings_obj, "show_processing_steps", True))
    except Exception:
        show_processing_steps = True

    # --- Stage resolver (Step 1): compute provider/model/kwargs per stage from existing settings ---
    # NOTE: This is read-only in this step (no behavior change). We compute it early so later
    # steps can pull from a single source instead of scattered getattr(...) calls.
    try:
        _prompt_input_hint: Any = [] if str(style) == "messages" else ""
        stage_specs = resolve_stage_specs(
            settings_obj=settings_obj,
            params=params,
            enable_tools=enable_tools,
            prompt_input=_prompt_input_hint,
            message=message,
            list_tools_fn=list_tools_fn,
        )
    except Exception:
        stage_specs = {}

    # UI-friendly summary of rewrite decision (always returned)
    rewrite_display: Dict[str, Any] = {
        "enabled": bool(enable_query_rewrite),
        "triggered": False,
        "accepted": False,
        "original": message,
    }

    # Conversation totals should be scoped per namespace (conversation_id/tab/session)
    _totals_ref = _get_convo_totals_for_namespace(namespace)
    m = Metrics(settings_obj, _totals_ref)
    # Diagnostics: show whether we're using the default accumulator vs a namespace-scoped one
    try:
        if namespace:
            logger.debug("[TOTALS] (%s) using namespace-scoped totals ns='%s'", log_origin, namespace)
        else:
            logger.warning("[TOTALS] (%s) namespace is empty -> using default totals accumulator", log_origin)
    except Exception:
        pass

    # --- Resolve retrieval knobs
    try:
        top_k = int(params.get("top_k") or getattr(settings_obj, "top_k", 8))
        score_threshold = float(params.get("score_threshold") or getattr(settings_obj, "score_threshold", 0.0))
    except Exception:
        top_k = int(getattr(settings_obj, "top_k", 8))
        score_threshold = float(getattr(settings_obj, "score_threshold", 0.0))

    # Stage: Query Rewrite 
    # --- Optional query rewrite (kept simple; only if both flag + heuristic)
    effective_query = message
    # Clarification control (may be set during rewrite decision)
    need_clarify = False
    clarify_reason = ""
    clarify_options: List[str] = []
    # Rewrite only after first turn by checking history
    if enable_query_rewrite and history:
        try:
            logger.info("[PIPELINE] emit stage: Query Rewrite")
            if show_processing_steps:
                emit_stage(req_id, "Query Rewrite")
        except Exception:
            pass
        try:
            heur = should_rewrite(message)
            if not heur:
                try:
                    rewrite_display.update({"triggered": False, "accepted": False, "reason": "heuristic_false"})
                except Exception:
                    pass
            if heur:
                # Per-turn overrides for rewrite behavior (handle_chat passes these via params)
                rw_tail, src_rw_tail = _get_param_int(params, ["rewrite_tail_turns"], getattr(settings_obj, "rewrite_tail_turns", 1), minimum=0)
                rw_tail = int(rw_tail)
                thr, src_thr = _get_param_float(params, ["rewrite_confidence_threshold"], getattr(settings_obj, "rewrite_confidence_threshold", 0.6), minimum=0.0, maximum=0.99)
                logger.debug("[REWRITE PARAMS] (%s) enable=%s tail_turns=%d (%s) threshold=%.2f (%s)", log_origin, True, rw_tail, src_rw_tail, thr, src_thr)
                raw_tail = max(0, int(rw_tail))
                window_turns = max(1, int(getattr(settings_obj, "chat_history_window_turns", 3)))
                to_sum_rw, tail_rw = split_history_for_prompt(history, raw_tail, window_turns)
                summary_rw = ""
                if to_sum_rw:
                    # Prefix tag with namespace if provided to isolate cache entries by conversation
                    _tag_rw = (f"{namespace}|rewrite" if namespace else "rewrite")
                    sum_spec = (stage_specs or {}).get("summary") or {}
                    try:
                        summary_rw, _from_cache_rw, _u_rw = _summarize_messages_with_cache(
                            to_sum_rw,
                            cache,
                            tag=_tag_rw,
                            model=getattr(settings_obj, "summarizer_model", settings_obj.inference_model),
                            temperature=float(getattr(settings_obj, "summarizer_temperature", 0.3)),
                            max_input_tokens=int(getattr(settings_obj, "summarizer_max_input_tokens", 512)),
                            max_output_tokens=int(getattr(settings_obj, "summarizer_max_output_tokens", 128)),
                            log_prefix=f"[REWRITE] {log_origin}",
                            stage_spec=sum_spec,
                        )
                    except LLMError as e:
                        # Surface rate limits from the summarizer used during rewrite pre-summary.
                        kind = getattr(e, "kind", "") or ""
                        if kind == "rate_limit":
                            try:
                                _prov = str(getattr(e, "provider", "") or "").strip() or "the summarizer provider"
                                _model = str(getattr(e, "model", "") or "").strip() or "(unspecified summarizer model)"
                                quota_msg = (
                                    f"Our summarizer model (provider={_prov}, model={_model}) "
                                    "is currently over its rate-limit or quota. I couldn't "
                                    "prepare the query rewrite safely, so this turn has been "
                                    "stopped. Please try again later or contact the "
                                    "administrator to increase the quota."
                                )
                            except Exception:
                                quota_msg = (
                                    "The summarizer model is currently over its rate limit or quota. "
                                    "Please try again later."
                                )

                            try:
                                emit_stage(
                                    req_id,
                                    "Final Answer",
                                    final=True,
                                    finalContent=quota_msg,
                                )
                            except Exception:
                                pass
                            try:
                                close_stream(req_id)
                            except Exception:
                                pass
                            try:
                                m.finalize_turn()
                                turn_metrics, convo_snapshot = m.snapshot()
                            except Exception:
                                turn_metrics = m.turn
                                convo_snapshot = {
                                    "tokens": {
                                        "embedding": 0,
                                        "llm_input": 0,
                                        "llm_output": 0,
                                        "conversation_total": 0,
                                    },
                                    "cost": {"conversation_total": 0.0},
                                }
                            return {
                                "answer": quota_msg,
                                "sources": [],
                                "turn_metrics": turn_metrics,
                                "conversation_totals": convo_snapshot,
                                "metrics": {"vectors_retrieved": 0},
                                "tools_used": [],
                                "rewrite_display": rewrite_display,
                            }

                        # Non-rate-limit LLMErrors fall back to the outer rewrite error handler.
                        raise

                    # Record rewrite pre-summary usage as part of the summary bucket (cache misses only).
                    if (not _from_cache_rw) and _u_rw:
                        _summary_model_used = str(
                            (sum_spec or {}).get("model")
                            or getattr(settings_obj, "summarizer_model", settings_obj.inference_model)
                        )
                        m.record_stage(
                            "summary",
                            model=_summary_model_used,
                            usage=_u_rw,
                            extra={"applied": False, "reason": "rewrite_pre_summary"},
                            model_key=(_stage_model_keys or {}).get("summary"),
                        )
                if tail_rw or summary_rw:
                    rw_spec = (stage_specs or {}).get("rewrite") or {}
                    rw = rewrite_query(tail_rw, summary_rw, message, log_prefix=f"[REWRITE] {log_origin}", stage_spec=rw_spec)
                    threshold = float(thr)
                    usage_rw = rw.get("_usage") if isinstance(rw, dict) else None

                    # If the rewrite step hit a provider rate limit (e.g., Gemini quota),
                    # short-circuit the turn with a clear, user-visible message rather
                    # than entering the Clarify path.
                    if isinstance(rw, dict) and rw.get("reason") == "llm_rate_limit":
                        try:
                            _prov = str(rw.get("_provider") or "").strip() or "the rewrite provider"
                            _model = str(rw.get("_model") or "").strip() or "(unspecified model)"
                            quota_msg = (
                                f"Our query rewrite model (provider={_prov}, model={_model}) "
                                "is currently over its rate-limit or quota. I couldn't safely rewrite your "
                                "question, so this turn has been stopped. Please try again later "
                                "or contact the administrator to increase the quota."
                            )
                        except Exception:
                            quota_msg = (
                                "The rewrite model is currently over its rate limit or quota. "
                                "Please try again later."
                            )

                        try:
                            emit_stage(
                                req_id,
                                "Final Answer",
                                final=True,
                                finalContent=quota_msg,
                            )
                        except Exception:
                            pass
                        try:
                            close_stream(req_id)
                        except Exception:
                            pass
                        try:
                            m.finalize_turn()
                            turn_metrics, convo_snapshot = m.snapshot()
                        except Exception:
                            turn_metrics = m.turn
                            convo_snapshot = {
                                "tokens": {
                                    "embedding": 0,
                                    "llm_input": 0,
                                    "llm_output": 0,
                                    "conversation_total": 0,
                                },
                                "cost": {"conversation_total": 0.0},
                            }
                        return {
                            "answer": quota_msg,
                            "sources": [],
                            "turn_metrics": turn_metrics,
                            "conversation_totals": convo_snapshot,
                            "metrics": {"vectors_retrieved": 0},
                            "tools_used": [],
                            "rewrite_display": rewrite_display,
                        }

                    accepted = bool(rw.get("changed")) and (not rw.get("ambiguous")) and (float(rw.get("confidence", 0.0) or 0.0) >= threshold)
                    if usage_rw:
                        _rw_m = str(((rw_spec or {}).get("model")) or getattr(settings_obj, "rewrite_model", settings_obj.inference_model))
                        m.record_stage(
                            "rewrite", 
                            model=_rw_m, 
                            usage=usage_rw, 
                            extra={"applied": True, "reason": ("accepted" if accepted else "rejected")},
                            model_key=(_stage_model_keys or {}).get("rewrite"),
                            )
                    if accepted:
                        effective_query = rw.get("rewritten") or message
                        logger.info("[REWRITE] (%s) accepted >=%s", log_origin, threshold)
                        try:
                            _t = int(getattr(settings_obj, "debug_log_truncate_chars", 4000))
                        except Exception:
                            _t = 400
                        logger.debug("[REWRITE] (%s) original='%s' rewritten='%s'", log_origin, (message or "")[:_t], (effective_query or "")[:_t])
                        try:
                            rewrite_display.update({
                                "triggered": True,
                                "accepted": True,
                                "rewritten": effective_query,
                                "confidence": float(rw.get("confidence", 0.0) or 0.0),
                                "threshold": float(thr),
                                "ambiguous": bool(rw.get("ambiguous", False)),
                                "reason": str(rw.get("reason", "") or ""),
                                "changed": bool(rw.get("changed", False)),
                            })
                        except Exception:
                            pass
                    else:
                        try:
                            _t = int(getattr(settings_obj, "debug_log_truncate_chars", 4000))
                        except Exception:
                            _t = 400
                        logger.debug("[REWRITE] (%s) original='%s' candidate='%s' (REJECTED)", log_origin, (message or "")[:_t], (rw.get("rewritten") or "")[:_t])
                        # Rejected rewrite; decide if we should ask a quick clarification instead of retrieving.
                        try:
                            _conf = float(rw.get("confidence", 0.0) or 0.0)
                            _amb = bool(rw.get("ambiguous", False))
                            if _amb or (_conf < threshold):
                                need_clarify = True
                                clarify_reason = "ambiguous" if _amb else f"low_conf({_conf:.2f}<{threshold})"
                                # Extract up to 3 likely referents, prioritizing USER turns only (avoid assistant prose).
                                try:
                                    # Build a small pool of recent USER messages (most recent first)
                                    user_msgs = [m for m in (tail_rw or []) if (m.get("role") or "").lower() == "user"]
                                    # Append the current message if present (usually short, but keeps ordering consistent)
                                    if message:
                                        user_msgs.append({"role": "user", "content": message})
                                    # Consider only the last 2 user messages to keep it tight
                                    user_msgs = user_msgs[-2:]

                                    # Proper-noun bigrams/trigrams; allow alphanumerics and symbols common in models/SKUs
                                    pat = re.compile(r"\b([A-Z][A-Za-z0-9+/-]*(?:\s+[A-Z][A-Za-z0-9+/-]*){1,2})\b")

                                    STOP = {
                                        "the","this","that","it","weather","climate","sources","section","lead","edit","n/a",
                                        "today","now","summer","winter","spring","autumn","fall",
                                        "jan","feb","mar","apr","may","jun","jul","aug","sep","sept","oct","nov","dec",
                                        "january","february","march","april","may","june","july","august","september","october","november","december"
                                    }

                                    seen_norm = set()
                                    tmp_opts = []

                                    def _normalize(s: str) -> tuple[str, str, int]:
                                        # strip leading 'the ' and trailing punctuation; return (display, key, token_count)
                                        s2 = s.strip()
                                        s2 = re.sub(r"^[Tt]he\s+", "", s2)
                                        s2 = re.sub(r"[\s\-–—,:;.!?]+$", "", s2)
                                        key = s2.lower()
                                        tcount = len(s2.split())
                                        return s2, key, tcount

                                    # Harvest candidates from user text
                                    for mm in reversed(user_msgs):
                                        txt = (mm.get("content") or "")
                                        for m_ in pat.findall(txt):
                                            disp, key, tcount = _normalize(m_)
                                            if key in seen_norm:
                                                continue
                                            if tcount < 2:
                                                continue
                                            if key in STOP:
                                                continue
                                            # Also skip if first token is a stopword
                                            first_tok = disp.split()[0].lower() if disp else ""
                                            if first_tok in STOP:
                                                continue
                                            seen_norm.add(key)
                                            tmp_opts.append(disp)
                                            if len(tmp_opts) >= 5:
                                                break
                                        if len(tmp_opts) >= 5:
                                            break

                                    # Keep at most 3, prefer longer phrases (3 words > 2 words)
                                    tmp_opts.sort(key=lambda s: (-len(s.split()), s))
                                    clarify_options.extend(tmp_opts[:3])

                                    logger.debug("[CLARIFY] (%s) pool: from_user=%s", log_origin, clarify_options)
                                except Exception:
                                    pass
                            # Record the rejected rewrite in the display block regardless
                            try:
                                rewrite_display.update({
                                    "triggered": True,
                                    "accepted": False,
                                    "candidate": str(rw.get("rewritten", "") or ""),
                                    "confidence": float(rw.get("confidence", 0.0) or 0.0),
                                    "threshold": float(thr),
                                    "ambiguous": bool(rw.get("ambiguous", False)),
                                    "reason": str(rw.get("reason", "") or ""),
                                    "changed": bool(rw.get("changed", False)),
                                })
                            except Exception:
                                pass
                        except Exception:
                            logger.info("[REWRITE] (%s) rejected", log_origin)
                else:
                    try:
                        rewrite_display.update({"triggered": False, "accepted": False, "reason": "no_history"})
                    except Exception:
                        pass
        except Exception as e:
            logger.warning("[REWRITE] (%s) failed; using original: %s", log_origin, e, exc_info=True)
            try:
                rewrite_display.update({"triggered": False, "accepted": False, "reason": "error"})
            except Exception:
                pass

    # If rewrite indicated ambiguity/low confidence, ask a brief clarification instead of retrieving.
    if need_clarify:
        if len(clarify_options) >= 2:
            if len(clarify_options) == 2:
                answer = f"Quick clarifier — do you mean {clarify_options[0]} or {clarify_options[1]}?"
            else:
                answer = f"Quick clarifier — which do you mean: {clarify_options[0]}, {clarify_options[1]}, or {clarify_options[2]}?"
        else:
            answer = "Quick clarifier — which specific place/product do you mean?"
        logger.info("[CLARIFY] (%s) reason=%s options=%s", log_origin, clarify_reason, clarify_options)
        # Emit a clarifier stage to SSE and immediately close this stream for the current turn
        try:
                emit_stage(
                    req_id,
                    "Clarification Needed",
                    prompt=answer,
                    options=clarify_options,
                    reason=clarify_reason
                )
        except Exception:
            pass
        try:
            close_stream(req_id)
        except Exception:
            pass
        try:
            m.finalize_turn()
            turn_metrics, convo_snapshot = m.snapshot()
        except Exception:
            turn_metrics = m.turn
            convo_snapshot = {"tokens": {"embedding": 0, "llm_input": 0, "llm_output": 0, "conversation_total": 0}, "cost": {"conversation_total": 0.0}}
        return {
            "answer": answer,
            "sources": [],
            "turn_metrics": turn_metrics,
            "conversation_totals": convo_snapshot,
            "metrics": {"vectors_retrieved": 0},
            "tools_used": [],
            "rewrite_display": rewrite_display,
        }

    logger.debug("[RETRIEVE] (%s) query=%s top_k=%s thr=%.3f", log_origin, effective_query, top_k, score_threshold)
    # Best-effort debug log of the embedding spec used for retrieval (provider/model/dimensions).
    try:
        _emb_spec_dbg = resolve_embedding_spec(settings_obj)
        logger.debug("[EMB] (%s) db.last_embedding_usage=%r", log_origin, getattr(db, "last_embedding_usage", None))
    except Exception:
        pass

    # Stage: Retrieve Vectors
    try:
        logger.info("[PIPELINE] emit stage: Retrieve Vectors")
        if show_processing_steps:
            emit_stage(req_id, "Retrieve Vectors")
    except Exception:
        pass
    # --- Retrieve
    try:
        results = db.search_similar(
            query=effective_query,
            limit=int(top_k),
            score_threshold=float(score_threshold),
            with_vectors=False,
            with_payload=True,
            exact=True,
        )
    except LLMError as e:
        # Surface embedding/provider rate limits that occur during retrieval.
        kind = getattr(e, "kind", "") or ""
        if kind == "rate_limit":
            try:
                _prov = str(getattr(e, "provider", "") or "").strip() or "the embedding provider"
                _model = str(getattr(e, "model", "") or "").strip() or "(unspecified embedding model)"
                quota_msg = (
                    f"Our embedding model (provider={_prov}, model={_model}) "
                    "is currently over its rate-limit or quota. I couldn't "
                    "retrieve context safely, so this turn has been stopped. "
                    "Please try again later or contact the administrator to "
                    "increase the quota."
                )
            except Exception:
                quota_msg = (
                    "The embedding model is currently over its rate limit or quota. "
                    "Please try again later."
                )

            try:
                emit_stage(
                    req_id,
                    "Final Answer",
                    final=True,
                    finalContent=quota_msg,
                )
            except Exception:
                pass
            try:
                close_stream(req_id)
            except Exception:
                pass
            try:
                m.finalize_turn()
                turn_metrics, convo_snapshot = m.snapshot()
            except Exception:
                turn_metrics = m.turn
                convo_snapshot = {
                    "tokens": {
                        "embedding": 0,
                        "llm_input": 0,
                        "llm_output": 0,
                        "conversation_total": 0,
                    },
                    "cost": {"conversation_total": 0.0},
                }
            return {
                "answer": quota_msg,
                "sources": [],
                "turn_metrics": turn_metrics,
                "conversation_totals": convo_snapshot,
                "metrics": {"vectors_retrieved": 0},
                "tools_used": [],
                "rewrite_display": rewrite_display,
            }

        # Non-rate-limit LLMErrors fall back to the outer handler.
        raise

    n = len(results) if results else 0
    logger.debug("[RETRIEVE] (%s) Qdrant returned %d", log_origin, n)

    # Embedding stage metrics
    try:
        raw_last = getattr(db, "last_embedding_usage", None)
        logger.debug("[EMB] (%s) db.last_embedding_usage=%r", log_origin, raw_last)
        last = raw_last or {}
        embed_tokens = int((last.get("input_tokens") or last.get("total_tokens") or 0))
        logger.debug("[EMB] (%s) parsed embed_tokens=%d", log_origin, embed_tokens)
    except Exception:
        embed_tokens = 0
    # Use the concrete embedding model name resolved from settings so that
    # pricing in model_registry can be applied correctly for cost metrics.
    try:
        _emb_spec_cost = resolve_embedding_spec(settings_obj) or {}
        _emb_model_for_cost = str(
            (_emb_spec_cost.get("model") or getattr(settings_obj, "embedding_model", "embedding"))
        )
    except Exception:
        _emb_model_for_cost = getattr(settings_obj, "embedding_model", "embedding")

    m.record_stage(
        "embedding",
        model=_emb_model_for_cost,
        pt=embed_tokens,
        model_key=(_stage_model_keys or {}).get("embedding"),
    )

# Stage: Rerank Retrieval Results
    
    # --- Rerank Decision Policy ---
    # 
    # Determines whether to apply reranking to search results based on several heuristics.
    # The policy aims to skip expensive reranking when it's unlikely to improve results.
    #
    # Parameters:
    #   - settings_obj: Configuration object containing reranking parameters
    #   - results: List of search results with scores and metadata
    #   - n: Total number of results available
    #
    # Returns:
    #   - need_rerank: Boolean indicating if reranking should be performed
    #   - skip_reason: String explaining why reranking was skipped (if applicable)
    #   - kept: Number of top results to consider for reranking
    #   - reranked: Initially set to input results, modified later if reranking is applied
    #
    # Decision Logic:
    # 1. Skip if there's only 1 or fewer results (nothing to rerank)
    # 2. Skip if results are fewer than re_ranker_input_rows (default 5)
    # 3. Check for exact matches in top 5 results (fast path)
    # 4. Check if top result is a clear winner based on:
    #    - Score above rerank_clear_winner_min_top1 (default 0.65)
    #    - Margin above 5th result > rerank_clear_winner_min_delta (default 0.15)
    # 5. If any condition is met, skip reranking; otherwise, perform reranking
    #
    # Note: All thresholds are configurable via settings with sensible defaults.
    kept = min(int(getattr(settings_obj, "re_ranker_input_rows", 5)), n)
    reranked = results
    need_rerank = False
    skip_reason = ""

    if n <= 1:
        need_rerank = False
        skip_reason = "<=1 candidate"
    elif n < int(getattr(settings_obj, "re_ranker_input_rows", 5)):
        need_rerank = False
        skip_reason = f"fewer than re_ranker_input_rows ({n} < {getattr(settings_obj, 're_ranker_input_rows', 5)})"
    else:
        try:
            scores = [float(r.get("score", 0.0) or 0.0) for r in results]
            top1 = scores[0]
            top5 = scores[4] if n >= 5 else scores[-1]
            margin = top1 - top5
            min_top1 = float(getattr(settings_obj, "rerank_clear_winner_min_top1", 0.65))
            min_delta = float(getattr(settings_obj, "rerank_clear_winner_min_delta", 0.15))

            # exact-match fast path in payload
            has_exact = False
            try:
                for r in results[:5]:
                    pl = r.get("payload") or {}
                    if pl.get("exact_match") or pl.get("is_exact_match") or pl.get("id_match"):
                        if float(r.get("score", 0.0) or 0.0) >= float(getattr(settings_obj, "rerank_exact_match_min_score", 0.80)):
                            has_exact = True
                            break
            except Exception:
                has_exact = False

            if has_exact:
                need_rerank = False
                skip_reason = "exact-match fast path"
            elif (top1 >= min_top1) and (margin >= min_delta):
                need_rerank = False
                skip_reason = f"clear winner (top1={top1:.2f}, Δ={margin:.2f})"
            else:
                need_rerank = True
        except Exception as e:
            logger.warning("[RERANK] (%s) score analysis failed; defaulting to rerank: %s", log_origin, e, exc_info=True)
            need_rerank = True
    if need_rerank:
        try:
            logger.info("[PIPELINE] emit stage: Rerank Retrieval Results")
            if show_processing_steps:
                emit_stage(req_id, "Rerank Retrieval Results")
        except Exception:
            pass

    if not need_rerank:
        _dbg(f"[RERANK] {log_origin}", f"skipping rerank: {skip_reason}")
        if show_processing_steps:
            emit_stage(req_id, "Skipping Rerank")
        reranked = results[:kept]
    else:
        _dbg(f"[RERANK] {log_origin}", f"applying rerank over {n} candidates; pool capped to {kept}")
        logger.debug("Rerank pool stats", extra={"candidates": n, "kept": kept})
        pool = results[:kept]
        pool_n = len(pool)
        logger.debug("[RERANK] (%s) Pool size=%d of %d", log_origin, pool_n, n)
        try:
            cand_text = _candidate_texts(pool)
            prompt_text = _make_rerank_prompt(
                effective_query,
                cand_text,
                int(getattr(settings_obj, "reranker_chunk_size", 600)),
            )
            _dbg(f"[RERANK] {log_origin} prompt:", prompt_text)

            # Provider-aware rerank call via stage_specs (behavior-identical defaults).
            _rs = (stage_specs or {}).get("rerank") or {}
            _provider = str(_rs.get("provider") or "openai")
            _model = str(
                _rs.get("model")
                or getattr(settings_obj, "re_ranker_model", settings_obj.inference_model)
            )
            _kwargs = dict(_rs.get("kwargs") or {})

            logger.debug(
                "[RERANK] (%s) provider=%s model=%s kwargs=%r",
                log_origin,
                _provider,
                _model,
                _kwargs,
            )

            resp_rerank = _responses_create(
                provider=_provider,
                model=_model,
                input=prompt_text.strip(),
                **_kwargs,
            )
            content = _extract_text_from_responses(resp_rerank).strip()
            _dbg(f"[RERANK] {log_origin} raw:", content)
            order = _parse_json_array_in_text(content, pool_n)
            reranked = [pool[i] for i in order] or pool
            reranked = reranked[:kept]

            usage_rr = _extract_usage_from_responses(resp_rerank, provider=_provider) or {}
            # Record rerank metrics against the actual model used (from stage_specs).
            m.record_stage(
                "rerank",
                model=_model,
                usage=usage_rr,
                extra={"candidates_reranked": n},
                model_key=(_stage_model_keys or {}).get("rerank"),
            )
        except LLMError as e:
            # Surface provider rate limits with a clear final answer instead of silently
            # falling back when the rerank model is over its rate limit or quota.
            kind = getattr(e, "kind", "") or ""
            if kind == "rate_limit":
                try:
                    _prov = str(getattr(e, "provider", "") or "").strip() or "the rerank provider"
                    _model = str(getattr(e, "model", "") or "").strip() or "(unspecified model)"
                    quota_msg = (
                        f"Our rerank model (provider={_prov}, model={_model}) "
                        "is currently over its rate-limit or quota. I couldn't rerank "
                        "your results safely, so this turn has been stopped. Please try "
                        "again later or contact the administrator to increase the quota."
                    )
                except Exception:
                    quota_msg = (
                        "The rerank model is currently over its rate limit or quota. "
                        "Please try again later."
                    )

                try:
                    emit_stage(
                        req_id,
                        "Final Answer",
                        final=True,
                        finalContent=quota_msg,
                    )
                except Exception:
                    pass
                try:
                    close_stream(req_id)
                except Exception:
                    pass
                try:
                    m.finalize_turn()
                    turn_metrics, convo_snapshot = m.snapshot()
                except Exception:
                    turn_metrics = m.turn
                    convo_snapshot = {
                        "tokens": {
                            "embedding": 0,
                            "llm_input": 0,
                            "llm_output": 0,
                            "conversation_total": 0,
                        },
                        "cost": {"conversation_total": 0.0},
                    }
                return {
                    "answer": quota_msg,
                    "sources": [],
                    "turn_metrics": turn_metrics,
                    "conversation_totals": convo_snapshot,
                    "metrics": {"vectors_retrieved": 0},
                    "tools_used": [],
                    "rewrite_display": rewrite_display,
                }

            logger.error("[RERANK] (%s) failed; falling back: %s", log_origin, e, exc_info=True)
            reranked = results[:kept]
        except Exception as e:
            logger.error("[RERANK] (%s) failed; falling back: %s", log_origin, e, exc_info=True)
            reranked = results[:kept]

# Stage: History Summary
    try:
        logger.info("[PIPELINE] emit stage: Summarize Chat History")
        if show_processing_steps:
            emit_stage(req_id, "Summarize Chat History")
    except Exception:
        pass
    # --- History summary slices
    summary_text = ""
    recent_block_str = ""
    try:
        raw_tail, src_raw_tail = _get_param_int(params, ["raw_tail_turns", "raw-tail_turns"], getattr(settings_obj, "raw_tail_turns", 2), minimum=0)
        window_turns, src_window = _get_param_int(params, ["chat_history_window_turns", "chat_history_max_turns"], getattr(settings_obj, "chat_history_window_turns", 3), minimum=1)
        to_summarize, verbatim_tail = split_history_for_prompt(history, raw_tail, window_turns)
        if to_summarize:
            sum_in, src_sum_in = _get_param_int(params, ["summarizer_max_input_tokens"], getattr(settings_obj, "summarizer_max_input_tokens", 512), minimum=0)
            sum_out, src_sum_out = _get_param_int(params, ["summarizer_max_output_tokens"], getattr(settings_obj, "summarizer_max_output_tokens", 128), minimum=1)
            logger.debug("[PARAMS] (%s) raw_tail_turns=%d (%s), chat_history_window_turns=%d (%s), summarizer_max_input_tokens=%d (%s), summarizer_max_output_tokens=%d (%s)",
                         log_origin, raw_tail, src_raw_tail, window_turns, src_window, sum_in, src_sum_in, sum_out, src_sum_out)
            # Prefix tag with namespace if provided to isolate cache entries by conversation
            _tag_inf = (f"{namespace}|inference" if namespace else "inference")
            sum_spec = (stage_specs or {}).get("summary") or {}
            try:
                summary_text, _from_cache_inf, _u_inf = _summarize_messages_with_cache(
                    to_summarize,
                    cache,
                    tag=_tag_inf,
                    model=getattr(settings_obj, "summarizer_model", settings_obj.inference_model),
                    temperature=float(getattr(settings_obj, "summarizer_temperature", 0.3)),
                    max_input_tokens=sum_in,
                    max_output_tokens=sum_out,
                    log_prefix=f"[SUMMARY] {log_origin}",
                    stage_spec=sum_spec,
                )
            except LLMError as e:
                # Surface rate limits from the summarizer model as a clear final answer.
                kind = getattr(e, "kind", "") or ""
                if kind == "rate_limit":
                    try:
                        _prov = str(getattr(e, "provider", "") or "").strip() or "the summarizer provider"
                        _model = str(getattr(e, "model", "") or "").strip() or "(unspecified summarizer model)"
                        quota_msg = (
                            f"Our summarizer model (provider={_prov}, model={_model}) "
                            "is currently over its rate-limit or quota. I couldn't "
                            "summarize the chat history safely, so this turn has been "
                            "stopped. Please try again later or contact the "
                            "administrator to increase the quota."
                        )
                    except Exception:
                        quota_msg = (
                            "The summarizer model is currently over its rate limit or quota. "
                            "Please try again later."
                        )

                    try:
                        emit_stage(
                            req_id,
                            "Final Answer",
                            final=True,
                            finalContent=quota_msg,
                        )
                    except Exception:
                        pass
                    try:
                        close_stream(req_id)
                    except Exception:
                        pass
                    try:
                        m.finalize_turn()
                        turn_metrics, convo_snapshot = m.snapshot()
                    except Exception:
                        turn_metrics = m.turn
                        convo_snapshot = {
                            "tokens": {
                                "embedding": 0,
                                "llm_input": 0,
                                "llm_output": 0,
                                "conversation_total": 0,
                            },
                            "cost": {"conversation_total": 0.0},
                        }
                    return {
                        "answer": quota_msg,
                        "sources": [],
                        "turn_metrics": turn_metrics,
                        "conversation_totals": convo_snapshot,
                        "metrics": {"vectors_retrieved": 0},
                        "tools_used": [],
                        "rewrite_display": rewrite_display,
                    }

                # Non-rate-limit LLMErrors fall through to the generic summary error handler.
                raise

            if not _from_cache_inf and _u_inf:
                _summary_model_used = str((sum_spec or {}).get("model") or getattr(settings_obj, "summarizer_model", settings_obj.inference_model))
                m.record_stage(
                    "summary",
                    model=_summary_model_used,
                    usage=_u_inf,
                    extra={"applied": True, "reason": f"prev {window_turns} turns (before last {raw_tail} turns)"},
                    model_key=(_stage_model_keys or {}).get("summary"),
                    )
        if verbatim_tail:
            tail_lines: List[str] = []
            trimmed = 0
            for msg in verbatim_tail:
                role = msg.get("role", "user")
                content = msg.get("content", "") or ""
                if role == "assistant":
                    cleaned = _strip_trailing_sources_block(content)
                    if cleaned != content:
                        trimmed += 1
                    content = cleaned
                tail_lines.append(f"{role}: {content}")
            recent_block_str = "Recent conversation:\n" + "\n".join(tail_lines) + "\n\n"
            if trimmed:
                logger.debug("[TAIL] (%s) stripped trailing Sources: blocks from %d assistant messages", log_origin, trimmed)
    except Exception as e:
        logger.warning("[SUMMARY] (%s) failed; proceeding without summary/tail: %s", log_origin, e, exc_info=True)

    # Stage: Establish Web Context
    # --- (Optional) web context
   
    web_context: List[Dict[str, Any]] = []
    try:
        if use_web_search:
            if show_processing_steps:
                emit_stage(req_id, "Establish Web Context")
            web_context = get_web_context_fn(effective_query, results or [])
    except Exception as e:
        logger.debug("[WEB] (%s) ignored web context due to error: %s", log_origin, e)

    # --- Context + sources 
    # Limit the number of context rows (from ) to the number of retrieved items (or kept) and the inference_context_rows setting
    inference_rows = int(getattr(settings_obj, "inference_context_rows", kept) or kept)
    inference_rows = min(max(1, inference_rows), kept)
    _dbg(f"[CONTEXT] {log_origin}", f"using {inference_rows} of {kept} retrieved items")
    context_items = (reranked or [])[:inference_rows]
    context_text = _format_context_lines(context_items)
   
    indexed_for_collapse = [
        {
            "index": i + 1,
            "url": ((item.get('payload') or {}).get('url_lower', (item.get('payload') or {}).get('url', 'unknown'))),
            "section": (item.get('payload') or {}).get('section', 'N/A'),
            "subsection": (item.get('payload') or {}).get('subsection', 'N/A'),
        }
        for i, item in enumerate(context_items)
    ]
    sources_section = "\nSources:\n" + _collapse_sources(indexed_for_collapse)
    if web_context:
        web_notes = "\n" + "\n".join([f"[web-{i+1}] {item.get('url', 'Web result')}" for i, item in enumerate(web_context)])
        sources_section += web_notes

    # Stage: Inference Prompt Build
    # --- Prompt build
    if show_processing_steps:
        emit_stage(req_id, "Inference Prompt Build")
    strict_rag_prompt = (
        "You are a question-answering assistant for a retrieval-augmented system.\n"
        "STRICT RULES:\n"
        "1. Base your answer ONLY on information in the Context section (and Web search results if present).\n"
        "2. Do NOT use any outside knowledge, general world knowledge, training data, or assumptions beyond that context.\n"
        "3. If the context does not contain enough information to answer the question, USE THE AVAILABLE TOOLS to gather the information you need.\n"
        "4. Only if tools cannot help and you still cannot answer, then reply with: I couldn't find any information to answer this question. NO_SUPPORTED_SOURCES\n"
        "5. If any context chunk has a citation like [1], [2], etc., retain it in your response.\n"
        "6. Do not fabricate sources or facts.\n"
        "7. If a source URL is available (shown in the final 'Sources' section), you may reference it by its tag like [1]."
    )
    system_prompt = strict_rag_prompt

    
    prompt_input = None  # what we pass as `input` to Responses
    if style == "messages":
        messages = [{"role": "system", "content": system_prompt}]
        if summary_text:
            messages.append({"role": "system", "content": f"Previous conversation summary: {summary_text}"})
        if recent_block_str:
            messages.append({"role": "system", "content": recent_block_str.strip()})
        messages.append({"role": "system", "content": f"Context:\n{context_text}"})
        if web_context:
            web_text = "\n".join([f"{i+1}. {item.get('title','')}\n{item.get('snippet','')}\nURL: {item.get('url','')}" for i, item in enumerate(web_context, start=1)])
            messages.append({"role": "system", "content": f"Web search results:\n{web_text}"})
        # Current user query
        messages.append({"role": "user", "content": message})
        #logger.debug(f"[DEBUG] Before flattening Added user message: {message}")
        #logger.debug(f"[DEBUG] Total messages count: {len(messages)}")
        #logger.debug(f"[DEBUG] Final messages: {messages}")

        # Convert to a single prompt string unless tools are enabled
        try:
            from backend.utils.prompt_utils import convert_messages_to_prompt
            prompt_str = convert_messages_to_prompt(messages) # flatten messages to a single string
        except Exception:
            # Fall back to naive join if util unavailable
            prompt_str = "\n".join([f"{m['role']}: {m['content']}" for m in messages])
        #_dbg(f"[FULL PROMPT] {log_origin}", prompt_str)
        
        prompt_input = messages
        # Current user query
        #logger.debug(f"[DEBUG] After flattening Added user message: {message}")
        #logger.debug(f"[DEBUG] Total messages count: {len(messages)}")
        #logger.debug(f"[DEBUG] Final messages: {messages}")

    else:
        # flat string (stateless path)
        summary_block = (f"Previous conversation summary: {summary_text}\n\n" if summary_text else "")
        prompt_str = (
            strict_rag_prompt + "\n\n"
            + summary_block
            + recent_block_str
            + f"Context:\n{context_text}\n\n"
            + f"Question: {message}\n"
        )
        
        #_dbg(f"[FULL INFERENCE PROMPT] {log_origin}", prompt_str)
        prompt_input = [{"role": "user", "content": prompt_str}] if enable_tools else prompt_str

    # --- Inference decode params
    temperature = _pick(params, ["temperature", "inference_temperature", "INFERENCE_TEMPERATURE"], getattr(settings_obj, "inference_temperature", 0.7))
    max_out = _pick(params, ["max_output_tokens", "max_inference_output_tokens", "MAX_INFERENCE_OUTPUT_TOKENS"], getattr(settings_obj, "max_inference_output_tokens", 300))
    top_p = _pick(params, ["top_p", "inference_top_p", "INFERENCE_TOP_P"], getattr(settings_obj, "inference_top_p", None))
    
    # Stage: Inference API call
    # --- Inference API call - Orchestrater stage with Tool Calls
    logger.info("[PIPELINE] emit stage: Generating Response")
    if show_processing_steps:
        emit_stage(req_id, "Generating Response")

    # Resolve provider/model/kwargs for inference from stage_specs (no behavior change to temps/limits).
    inf_spec = (stage_specs or {}).get("inference") or {}
    _inf_provider = str(inf_spec.get("provider") or "openai")
    _inf_model = str(inf_spec.get("model") or getattr(settings_obj, "inference_model", "gpt-4o"))
    
    # Auto-detect provider from model name to prevent mismatch
    if _inf_provider == "openai" and (_inf_model.startswith("models/gemini") or _inf_model.startswith("gemini")):
        _inf_provider = "gemini"
        logger.debug(f"[INFERENCE AUTO-DETECT] Corrected provider to 'gemini' for model '{_inf_model}'")
    
    # DEBUG: Check provider/model mismatch
    logger.debug(f"[INFERENCE DEBUG] provider={_inf_provider} model={_inf_model}")
    if _inf_model.startswith("models/gemini") and _inf_provider == "openai":
        logger.warning(f"[INFERENCE MISMATCH] Gemini model '{_inf_model}' with OpenAI provider '{_inf_provider}' - this will fail!")

    _kwargs_inf: Dict[str, Any] = dict(inf_spec.get("kwargs") or {})

    # Per-request overrides / additions (preserve existing semantics)
    _kwargs_inf["input"] = prompt_input
    _kwargs_inf["temperature"] = float(temperature)
    _kwargs_inf["max_output_tokens"] = int(max_out)
    if top_p is not None:
        _kwargs_inf["top_p"] = float(top_p)
   
    if enable_tools and isinstance(prompt_input, list):
        try:
            logger.debug("[PIPELINE] Before Tools list function")
            tools = list_tools_fn()
            logger.debug("[PIPELINE] After Tools list function %s ", tools[:100])
            # Avoid web_search unless explicitly requested in the message
            def _is_web_search_requested(latest_user_msg: str) -> bool:
                if not latest_user_msg:
                    return False
                txt = latest_user_msg.lower()
                keys = ["use web search", "search the web", "web search", "search online", "browse the web", "do a web search", "google this", "bing this"]
                return any(k in txt for k in keys)
            if not _is_web_search_requested(message):
                tools = [t for t in tools if (t.get("name") or t.get("function", {}).get("name")) != "web_search"]
            _kwargs_inf["tools"] = tools
        except Exception:
            _kwargs_inf["tools"] = []

    logger.info("[INFERENCE] %s: Attempting Responses with Inference model: %s", log_origin, _inf_model)
    #logger.debug("[%s] Call to Inference API with Prompt: %s", log_origin, _kwargs_inf["input"])
    
    # DEBUG: Add debug before _responses_create call
    #logger.debug(f"[INFERENCE] About to call _responses_create: provider={_inf_provider} model={_inf_model}")
    #logger.debug(f"[INFERENCE] _kwargs_inf keys: {list(_kwargs_inf.keys())}")
    
    try:
        resp_inf = _responses_create(
            provider=_inf_provider,
            model=_inf_model,
            **_kwargs_inf,
        )
        logger.debug(f"[INFERENCE] _responses_create succeeded: type={type(resp_inf)}")
    except Exception as e:
        logger.error(f"[INFERENCE] _responses_create failed with {type(e).__name__}: {str(e)[:200]}...")
        logger.debug(f"[INFERENCE] Exception details: type={type(e)} args={getattr(e, 'args', None)}")
        
        # Check if it's an LLMError (especially rate limit) and handle it appropriately
        if isinstance(e, LLMError):
            logger.error(f"[INFERENCE] Caught LLMError: kind={getattr(e, 'kind', 'None')} provider={getattr(e, 'provider', 'None')} message={str(e)[:100]}...")
            kind = getattr(e, "kind", "") or ""
            if kind == "rate_limit":
                logger.info(f"[INFERENCE] Processing rate limit error for user")
                try:
                    _prov = str(getattr(e, "provider", "") or "").strip() or "the inference provider"
                    _model = str(getattr(e, "model", "") or "").strip() or "(unspecified model)"
                    quota_msg = (
                        f"Our inference model (provider={_prov}, model={_model}) "
                        "is currently over its rate-limit or quota. I couldn't "
                        "generate a response safely, so this turn has been stopped. "
                        "Please try again later or contact the administrator to "
                        "increase the quota."
                    )
                except Exception:
                    quota_msg = (
                        "The inference model is currently over its rate limit or quota. "
                        "Please try again later."
                    )

                logger.info(f"[INFERENCE] Created rate limit message: {quota_msg[:100]}...")
                try:
                    emit_stage(
                        req_id,
                        "Final Answer",
                        final=True,
                        finalContent=quota_msg,
                    )
                    logger.info(f"[INFERENCE] Rate limit message emitted successfully")
                except Exception as emit_err:
                    logger.error(f"[INFERENCE] Failed to emit rate limit message: {emit_err}")
                    pass
                try:
                    close_stream(req_id)
                except Exception:
                    pass
                try:
                    m.finalize_turn()
                    turn_metrics, convo_snapshot = m.snapshot()
                except Exception:
                    turn_metrics = m.turn
                    convo_snapshot = {
                        "tokens": {
                            "embedding": 0,
                            "llm_input": 0,
                            "llm_output": 0,
                            "conversation_total": 0,
                        },
                        "cost": {"conversation_total": 0.0},
                    }
                return {
                    "answer": quota_msg,
                    "sources": [],
                    "turn_metrics": turn_metrics,
                    "conversation_totals": convo_snapshot,
                    "metrics": {"vectors_retrieved": 0},
                    "tools_used": [],
                    "rewrite_display": rewrite_display,
                }
        # Non-LLMError exceptions, re-raise to be handled by outer pipeline
            raise

    # Note: LLMError exceptions are now handled in the first except block above
    try:
        # Log the provider-native response object for debugging tool-calls behavior.
        _raw = getattr(resp_inf, "raw", resp_inf)
        logger.debug("[INFERENCE] (%s) raw response: %r", log_origin, _raw)
    except Exception:
        pass
    _dbg(f"[INFERENCE] Inference 1 response {log_origin}", str(resp_inf))
    usage_inf = _extract_usage_from_responses(resp_inf, provider=_inf_provider)
    # Record Inference Usage - 1st Inference (will determine if we need to call tool calls)
    if usage_inf:
        m.record_stage(
            "inference",
            model=_inf_model,
            usage=usage_inf,
            model_key=(_stage_model_keys or {}).get("inference"),
        )

    # Stage: Tool Calls
    # --- Tool Calls - Single pass thru all tools required
    # Optional tool loop (bounded for safety)

    answer_override: str | None = None
    tool_answer_text: str = ""
    used_tools: List[str] = []
    
    if enable_tools and isinstance(_kwargs_inf.get("input"), list):
        # NOTE: Single-pass tool execution.
        # The previous bounded while-loop was ineffective because `resp_inf` is not updated in-loop,
        # and we already synthesize once per turn. Keep behavior identical via a single pass.
        if show_processing_steps:
            emit_stage(req_id, "Tool Calls")
        try:
            # Extract tool calls from the first inference response
            tool_calls = extract_tool_calls(resp_inf)
            logger.debug("[TOOLS] Found %d tool calls", len(tool_calls))
            
            if not tool_calls:
                raise StopIteration  # handled by outer try/except; leaves answer_override=None

            tool_outputs_list: List[Dict[str, Any]] = []
            chat_context = list(history or []) + [{"role": "user", "content": message}]

            # Hoist doc-context allowlist outside the per-tool loop
            tools_with_doc_ctx = set(getattr(settings_obj, "tools_with_document_context", []) or [])

            # Helper: format a safe tool-results fallback without duplicating the same message twice.
            def _format_tool_fallback(_tool_answer_text: str, _tools_text: str) -> str:
                _tt = (_tool_answer_text or "").strip()
                _tools = (_tools_text or "").strip()
                if _tt and _tools and (_tt not in _tools):
                    return _tt + "\n\n" + "--- External Tool Results ---\n" + _tools
                return "--- External Tool Results ---\n" + _tools

            # DEBUG: Print extracted tool calls before execution
            logger.debug(f"[DEBUG] Extracted tool calls ({len(tool_calls)}):")
            for i, call in enumerate(tool_calls):
                name = call.get("name", "")
                call_id = call.get("id", "")
                args = call.get("args", {})
                logger.debug(f"[DEBUG]   Tool {i+1}: name='{name}', id='{call_id}', args={args}")

            for call in tool_calls:
                name = call.get("name") or ""
                call_id = call.get("id") or call.get("tool_call_id")
                args = parse_tool_args(call.get("args"))
                logger.debug("[TOOLS] Tool call: name=%s id=%s args=%s", name, call_id, args)

                if show_processing_steps:
                    emit_stage(req_id, f"Calling Tool: {name}")
                executor = get_executor_fn(name)
                logger.debug("[TOOLS] Found executor for %s: %s", name, "Yes" if executor else "No")

                if not executor:
                    result_text: Any = f"Tool '{name}' is not available."
                    logger.warning("[TOOLS] Tool not found: %s", name)
                else:
                    try:
                        exec_combined_context = None
                        if name in tools_with_doc_ctx:
                            exec_combined_context = [
                                {
                                    "url": (it.get("payload") or {}).get("url")
                                    or (it.get("payload") or {}).get("url_lower", ""),
                                    "title": (it.get("payload") or {}).get("title") or "",
                                    "snippet": (it.get("payload") or {}).get("text")
                                    or (it.get("payload") or {}).get("snippet")
                                    or "",
                                }
                                for it in (reranked or [])
                            ]

                        result_text = executor(args, chat_context, existing_context=exec_combined_context)
                        logger.debug("[TOOLS] Executed tool %s returned: %r", name, result_text)
                    except Exception as ex:
                        result_text = f"Tool '{name}' failed: {ex}"

                # Normalize empty tool outputs so the user can see a clear outcome.
                try:
                    if result_text is None:
                        result_text = ""
                    if isinstance(result_text, str) and not result_text.strip():
                        result_text = f"Tool '{name}' executed but returned no results."
                except Exception:
                    pass

                if name:
                    used_tools.append(name)

                tool_outputs_list.append({"tool_call_id": call_id or "", "output": str(result_text)})
                logger.debug("[TOOLS] Tool output added - ID: %s, Output: %r", call_id or "N/A", result_text)
                logger.debug("[TOOLS] Current tool_outputs_list: %s", tool_outputs_list)

                # Preserve first non-empty tool message for final fallback rendering
                try:
                    txt = (result_text.strip() if isinstance(result_text, str) else str(result_text).strip())
                    if txt and not tool_answer_text:
                        tool_answer_text = txt
                except Exception:
                    pass

            if not tool_outputs_list:
                raise StopIteration

            tools_text = "\n\n".join([str(t.get("output", "")) for t in tool_outputs_list]).strip()
            logger.debug("[TOOLS] tools_text before synthesis: %r", tools_text)
            if not tools_text:
                tools_text = "Tool(s) executed but returned no results."

            # Build messages for tool synthesis (consistent with main inference)
            synth_messages = [
                {"role": "system", "content": "You are a question-answering assistant for a retrieval-augmented system.\nSTRICT RULES:\n1. Base your answer ONLY on information in the provided Context and Tool results.\n2. Do NOT use any outside knowledge.\n3. If the context does not contain enough information to answer the question, USE THE AVAILABLE TOOLS to gather the information you need.\n4. Only if tools cannot help and you still cannot answer, then reply with: I couldn't find any information to answer this question. NO_SUPPORTED_SOURCES\n5. Retain any numeric citations like [1], [2] from the Context.\n6. Do not fabricate sources of facts.\n7. Use citations like [1], [2] when using Context.\n8. Integrate Tool results where relevant (do not invent citations for tool facts).\n9. Be concise.\n10. Do not add any extra text beyond what is in the Context or Tool results."}
            ]

            if summary_text:
                synth_messages.append({"role": "system", "content": f"Previous conversation summary:\n{summary_text}"})

            if recent_block_str:
                synth_messages.append({"role": "system", "content": recent_block_str.strip()})

            synth_messages.append({"role": "system", "content": f"Context:\n{context_text}"})
            synth_messages.append({"role": "system", "content": f"Tool results:\n{tools_text}"})
            synth_messages.append({"role": "user", "content": f"Question: {message}\n\nTask: Produce the final answer to the Question using the Context and Tool results."})

            ts_spec = (stage_specs or {}).get("tools_synth") or {}
            _ts_provider = str(ts_spec.get("provider") or "openai")
            _ts_model = str(ts_spec.get("model"))

            _kwargs_synth: Dict[str, Any] = dict(ts_spec.get("kwargs") or {})
            _kwargs_synth["input"] = synth_messages
            if "max_output_tokens" not in _kwargs_synth and max_out is not None:
                _kwargs_synth["max_output_tokens"] = int(max_out)
            if "temperature" not in _kwargs_synth:
                _kwargs_synth["temperature"] = float(temperature)

            try:
                if show_processing_steps:
                    emit_stage(req_id, "Generating Responses with Tools")
                _dbg(f"[TOOLS] {log_origin} Final Inference with Tools Synthesis synth messages : %s", str(synth_messages))
                resp_synth = _responses_create(
                    provider=_ts_provider,
                    model=_ts_model,
                    **_kwargs_synth,
                )
                _dbg(f"INFERENCE 2 response {log_origin} Generating responses with tools", str(resp_synth))
                combined = _extract_text_from_responses(resp_synth).strip()
                logger.debug(f"[TOOLS] {log_origin} tools synthesis combined before override : %s", combined)

                if combined and ("NO_SUPPORTED_SOURCES" not in combined):
                    answer_override = combined
                else:
                    answer_override = _format_tool_fallback(tool_answer_text, tools_text)
                # --- Cosmetic cleanup: do not leak tool citation label into user answer ---
                try:
                    answer_override = re.sub(r"\s*\[Tool results\]\s*", "", answer_override  or "").strip()
                except Exception:
                    pass

                logger.debug(f"[TOOLS] {log_origin} tools synthesis combined: %s  and answer_override %s ", combined, answer_override)
                usage_synth = _extract_usage_from_responses(resp_synth, provider=_ts_provider)
                if usage_synth:
                    m.record_stage(
                        "inference_tools_synth",
                        model=_ts_model,
                        usage=usage_synth,
                        model_key=(_stage_model_keys or {}).get("tools_synth"),
                    )
            except LLMError as e:
                kind = getattr(e, "kind", "") or ""
                if kind == "rate_limit":
                    try:
                        _prov = str(getattr(e, "provider", "") or "").strip() or "the tools synthesis provider"
                        _model = str(getattr(e, "model", "") or "").strip() or "(unspecified model)"
                        quota_msg = (
                            f"Our tools synthesis model (provider={_prov}, model={_model}) "
                            "is currently over its rate-limit or quota. I couldn't "
                            "combine the tool results safely, so this turn has been "
                            "stopped. Please try again later or contact the "
                            "administrator to increase the quota."
                        )
                    except Exception:
                        quota_msg = (
                            "The tools synthesis model is currently over its rate limit or quota. "
                            "Please try again later."
                        )

                    try:
                        emit_stage(
                            req_id,
                            "Final Answer",
                            final=True,
                            finalContent=quota_msg,
                        )
                    except Exception:
                        pass
                    try:
                        close_stream(req_id)
                    except Exception:
                        pass
                    try:
                        m.finalize_turn()
                        turn_metrics, convo_snapshot = m.snapshot()
                    except Exception:
                        turn_metrics = m.turn
                        convo_snapshot = {
                            "tokens": {
                                "embedding": 0,
                                "llm_input": 0,
                                "llm_output": 0,
                                "conversation_total": 0,
                            },
                            "cost": {"conversation_total": 0.0},
                        }
                    return {
                        "answer": quota_msg,
                        "sources": [],
                        "turn_metrics": turn_metrics,
                        "conversation_totals": convo_snapshot,
                        "metrics": {"vectors_retrieved": 0},
                        # Normalize directly from used_tools here because tools_out
                        # is computed later in the happy path.
                        "tools_used": sorted({t for t in used_tools if t}) if used_tools else [],
                        "rewrite_display": rewrite_display,
                    }

                logger.debug("[TOOLS] (%s) tools synthesis failed: %s", log_origin, e, exc_info=True)
                answer_override = _format_tool_fallback(tool_answer_text, tools_text)
            except Exception as ex:
                logger.debug("[TOOLS] (%s) tools synthesis failed: %s", log_origin, ex, exc_info=True)
                answer_override = _format_tool_fallback(tool_answer_text, tools_text)
        except StopIteration:
            # No tool calls / no outputs; treat as "no tools" path.
            pass
        except Exception as e:
            logger.debug("[TOOLS] (%s) tool loop failed: %s", log_origin, e, exc_info=True)
        # Final safety: if tools ran but synthesis produced no answer, use tool output directly
        if (not answer_override) and tool_answer_text:
            logger.debug(f"[TOOLS] {log_origin} Falling back to tool answer text %s ", tool_answer_text[:100])
            answer_override = tool_answer_text

    # Normalize tool usage for downstream rendering/metadata
    tools_out = sorted({t for t in used_tools if t}) if used_tools else []

    # Stage: Final answer and packing
    # Principle:
    # - If tools ran, `answer_override` is considered the authoritative final answer.
    # - If tools did not run, the first inference answer may carry NO_SUPPORTED_SOURCES and we suppress sources accordingly.

    # Default sources are the retrieved + optional web context
    sources = (reranked or []) + (web_context or [])

    if answer_override is not None:
        # Tools path: pass through exactly what the tools/synthesis logic produced.
        answer = answer_override or ""

        # If we are returning tool-only output (or we have no doc/web sources), suppress the Sources block
        # to avoid implying citations for tool facts.
        try:
            if ("--- External Tool Results ---" in (answer or "")) or (not sources):
                sources = []
                sources_section = ""
        except Exception:
            pass

    else:
        # No-tools path: use the original inference output.
        answer = (_extract_text_from_responses(resp_inf) or "")

        # Sentinel / unsupported-source handling (no-tools only)
        _ans_raw = (answer or "").rstrip()
        _ans_norm_end = _ans_raw.rstrip(" \t\r\n'\"")

        # If the model indicates no supported sources AND we did not use tools, suppress sources.
        if _ans_norm_end.endswith("NO_SUPPORTED_SOURCES"):
            # Remove sentinel from final answer text
            if "\n" in _ans_raw:
                _ans_raw = _ans_raw.rsplit("\n", 1)[0].rstrip()
            else:
                _ans_raw = _ans_raw.replace("NO_SUPPORTED_SOURCES", "").rstrip()
            answer = _ans_raw
            sources = []  # JSON: no sources returned
            sources_section = ""  # No sources block appended

        # Heuristic: if the model indicates lack of supporting context, even without sentinel.
        lower_ans = (_ans_raw or "").lower()
        if (
            "the provided context does not contain" in lower_ans
            or "the context provided does not contain" in lower_ans
            or "provided context does not" in lower_ans
            or "context does not" in lower_ans
        ):
            answer = _ans_raw
            sources = []
            sources_section = ""

    # Final per-UI toggle: hide sources for this response if configured.
    # This runs after tool/sentinel logic so it can override visibility per mode.
    try:
        if not display_sources:
            sources = []
            sources_section = ""
    except Exception:
        pass

    try:
        m.finalize_turn()
        turn_metrics, convo_snapshot = m.snapshot()
    except Exception:
        turn_metrics = m.turn
        convo_snapshot = {
            "tokens": {"embedding": 0, "llm_input": 0, "llm_output": 0, "conversation_total": 0},
            "cost": {"conversation_total": 0.0},
        }

    legacy_metrics = {"vectors_retrieved": n}
    return {
        "answer": (answer.rstrip("\n") + sources_section),
        "sources": sources,
        "turn_metrics": turn_metrics,
        "conversation_totals": convo_snapshot,
        "metrics": legacy_metrics,
        "tools_used": tools_out,
        "rewrite_display": rewrite_display,
    }

# --- end orchestrator ---

def handle_chat(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Thin handler: stateless path using orchestrator only.

    Expects payload with keys: message, params.top_k, params.score_threshold.
    Returns orchestrator answer and metrics; no legacy/flag paths.
    """
    # Extract request fields
    message: str = (payload or {}).get("message") or ""
    history: List[Dict[str, str]] = (payload or {}).get("history") or []
    params: Dict[str, Any] = (payload or {}).get("params") or {}

    if not message:
        return {"answer": "", "metrics": {"vectors_retrieved": 0}}
    logger.info("Before generating req_id display query_id: %s", params.get("query_id"))
    req_id = params.get("query_id") or uuid.uuid4().hex[:8]
    logger.info("[REQ] handle_chat start stateless [req_id=%s]", req_id)

    # Determine rewrite toggle (param overrides settings)
    rewrite_enabled = bool((params or {}).get("enable_query_rewrite", getattr(settings, "enable_query_rewrite", False)))
    logger.debug("[REWRITE] (handle_chat#%s) enabled=%s", req_id, rewrite_enabled)
    try:
        _thr = params.get("rewrite_confidence_threshold")
        _tail = params.get("rewrite_tail_turns")
        if _thr is not None or _tail is not None:
            logger.debug("[REWRITE] (handle_chat#%s) overrides: threshold=%s tail_turns=%s", req_id, _thr, _tail)
    except Exception:
        pass

    # Fresh Qdrant client for stateless path
    db = QdrantDB(
        host=settings.qdrant_host,
        port=settings.qdrant_port,
        collection_name=settings.collection_name,
    )

    # Determine tools flag (preserve prior behavior)
    enable_tools = False
    if isinstance(params, dict):
        if params.get("use_tools") is not None:
            enable_tools = bool(params.get("use_tools"))
        elif params.get("enable_tools") is not None:
            enable_tools = bool(params.get("enable_tools"))
        else:
            enable_tools = bool(getattr(settings, "enable_tools", False))
    else:
        enable_tools = bool(getattr(settings, "enable_tools", False))

    # --- Derive a namespace for cache keying (non-breaking) ---
    try:
        _uid = str((params or {}).get("user_id") or "").strip()
        _cid = str((params or {}).get("conversation_id") or "").strip()
        _ns = (f"{_uid}:{_cid}" if _uid and _cid else (_cid or ""))
    except Exception:
        _ns = ""

    # Diagnostics: confirm namespace / conversation_id on every stateless request
    try:
        logger.info(
            "[REQ %s] ns='%s' user_id='%s' conversation_id='%s' query_id='%s'",
        req_id,
        _ns,
        (_uid or ""),
        (_cid or ""),
        (params.get("query_id") if isinstance(params, dict) else None),
    )
        if not _ns:
         logger.warning(
            "[REQ %s] EMPTY namespace -> using default CONVO_TOTALS (totals may appear to reset/collide)",
            req_id,
        )
    except Exception:
        pass

    deps = {
        "db": db,
        "cache": _SUMMARY_CACHE,
        "settings": settings,
        "list_tools": list_tools,
        "get_executor": get_executor,
        "get_web_context": (lambda q, existing: []),  # stateless: no auto web
        "style": "messages", # flat or messages (use messages for clear systemvs user roles separation)
        "enable_tools": bool(enable_tools),
        "enable_query_rewrite": bool(rewrite_enabled),
        "use_web_search": False,
        "log_origin": "handle_chat",
        "request_id": req_id,
        "namespace": _ns,
    }
    req = {"message": message, "history": history, "params": params}

    try:
        # Log cache size before running the pipeline
        try:
            _pre_bytes = 0
            try:
                _pre_bytes = sum(len(v.encode('utf-8')) for v in _SUMMARY_CACHE.values())
            except Exception:
                _pre_bytes = sum(len(v) for v in _SUMMARY_CACHE.values())
            logger.info(
                "[REQ %s] _SUMMARY_CACHE size (pre): %d entries, %d bytes | user_id=%s conversation_id=%s",
                req_id, len(_SUMMARY_CACHE), _pre_bytes, (_uid or ""), (_cid or ""))
        except Exception:
            pass
        # Run the orchestrator (chat pipeline)
        logger.info("[PIPELINE] handle_chat running pipeline orchestrator")

        out = run_pipeline(deps=deps, req=req)

        # Log cache size after the pipeline completes
        try:
            _post_bytes = 0
            try:
                _post_bytes = sum(len(v.encode('utf-8')) for v in _SUMMARY_CACHE.values())
            except Exception:
                _post_bytes = sum(len(v) for v in _SUMMARY_CACHE.values())
            logger.info(
                "[REQ %s] _SUMMARY_CACHE size (post): %d entries, %d bytes | user_id=%s conversation_id=%s",
                req_id, len(_SUMMARY_CACHE), _post_bytes, (_uid or ""), (_cid or ""))
        except Exception:
            pass
        logger.info("[PIPELINE] handle_chat returning orchestrator output: %s", out.get("answer", ""))
        # Ensure the final message is properly formatted for the frontend
        emit_stage(req_id, "Final Answer", final=True, finalContent=out.get("answer", ""))
        # Send an explicit close message
        emit_stage(req_id, "Done", final=True)

        # Base response: preserve existing shape/keys for compatibility.
        resp: Dict[str, Any] = {
            "answer": out.get("answer", ""),
            "response": out.get("answer", ""),  # legacy compatibility for frontend expecting 'response'
            "metrics": out.get("metrics", {"vectors_retrieved": 0}),
            "turn_metrics": out.get("turn_metrics", {}),
            "conversation_totals": out.get("conversation_totals", {}),
            "tools_used": out.get("tools_used", []),
            "rewrite_display": out.get("rewrite_display", {}),
        }

        # Non-breaking: only surface reasoning when present and non-empty.
        try:
            reasoning = out.get("reasoning") if isinstance(out, dict) else None
        except Exception:
            reasoning = None
        if reasoning:
            resp["reasoning"] = reasoning

        return resp
    except LLMError as e:
        # Fatal provider/config/LLM failure (e.g., missing API key, unsupported provider).
        # Surface a clear, structured error back to the caller while preserving
        # the existing SSE shutdown behavior.
        logger.error(
            "[PIPELINE] handle_chat fatal LLMError: provider=%s model=%s kind=%s code=%s msg=%s",
            getattr(e, "provider", None),
            getattr(e, "model", None),
            getattr(e, "kind", None),
            getattr(e, "code", None),
            str(e),
            exc_info=True,
        )
        err_text = str(e) or "LLM error during inference."
        try:
            emit_stage(req_id, "Final Answer", final=True, finalContent=err_text)
        except Exception:
            pass
        try:
            emit_stage(req_id, "Done", final=True)
        except Exception:
            pass
        try:
            close_stream(req_id)
        except Exception:
            pass
        return {
            "answer": err_text,
            "response": err_text,
            "metrics": {"vectors_retrieved": 0},
            "error": {
                "stage": "inference",
                "provider": getattr(e, "provider", None),
                "model": getattr(e, "model", None),
                "kind": getattr(e, "kind", None),
                "code": getattr(e, "code", None),
                "message": str(e) or "LLM error during inference.",
            },
        }
    except Exception as e:
        logger.exception("[PIPELINE] handle_chat orchestrator failed: %s", e)
        err_text = "Sorry, something went wrong."
        # Ensure SSE stage stream terminates on errors so the UI doesn't hang.
        try:
            emit_stage(req_id, "Final Answer", final=True, finalContent=err_text)
        except Exception:
            pass
        try:
            emit_stage(req_id, "Done", final=True)
        except Exception:
            pass
        try:
            close_stream(req_id)
        except Exception:
            pass
        return {
            "answer": err_text,
            "response": err_text,
            "metrics": {"vectors_retrieved": 0},
        }
