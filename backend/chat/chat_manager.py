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
from pathlib import Path
from collections import defaultdict
import yaml
import bleach
# NOTE: SSE stage emission is centralized in backend/stream_emit.py so chat_manager stays agnostic of registry details.
# Stream emission helpers (centralized in backend/stream_emit.py)
from backend.stream_emit import emit_stage, close_stream
from backend.core.config import settings
from backend.db import QdrantDB
from backend.chat.web_search import WebSearchClient
from backend.embeddings.specs import resolve_embedding_spec
from backend.tools import list_tools, get_executor

from backend.llm.llm_client import generate, embed, get_pricing_for_model, LLMError
from backend.chat.simple_history_processor import SimpleHistoryProcessor
from backend.chat.utils import _get_param_int, split_history_for_prompt
from backend.chat.chunked_history_manager import ChunkedHistoryManager
from backend.chat.prompt_registry import (
    resolve_inference_prompt,
    resolve_rewrite_prompt,
    resolve_rerank_prompt,
    resolve_summary_prompt,
    resolve_tools_synth_prompt,
    render_full_payload,
)
from backend.retrieval.config import resolve_retrieval_specs
from backend.retrieval.schemas import EmbeddingSpec
from backend.retrieval.providers.fastembed_embedding_provider import FastEmbedEmbeddingProvider
from backend.retrieval.retrieval_eval_service import RetrievalEvalService
from backend.markdown_render import render_markdown_to_html

_SUMMARY_CACHE: Dict[str, str] = {}
_SUMMARY_CACHE_LAST_SEEN: Dict[str, float] = {}

_TOOL_REGISTRY_CACHE: Dict[str, Dict[str, Any]] = {}

_CHUNK_MANAGERS_BY_NS: Dict[str, ChunkedHistoryManager] = {}
_CHUNK_MANAGERS_LAST_SEEN: Dict[str, float] = {}
_FASTEMBED_EMBEDDING_PROVIDER = FastEmbedEmbeddingProvider()


def _evict_idle_chunk_managers(now: float | None = None, max_idle_seconds: int | None = None) -> Dict[str, int]:
    try:
        _now = float(now if now is not None else time.time())
    except Exception:
        _now = time.time()
    try:
        _ttl = int(max_idle_seconds) if max_idle_seconds is not None else int(getattr(settings, "chunk_manager_idle_ttl_seconds", 3600) or 3600)
    except Exception:
        _ttl = 3600

    cleared = 0
    try:
        idle_keys = [k for k, ts in _CHUNK_MANAGERS_LAST_SEEN.items() if (_now - float(ts or 0.0)) > _ttl]
        for k in idle_keys:
            _CHUNK_MANAGERS_LAST_SEEN.pop(k, None)
            if k in _CHUNK_MANAGERS_BY_NS:
                _CHUNK_MANAGERS_BY_NS.pop(k, None)
                cleared += 1
    except Exception:
        pass
    return {"cleared": cleared, "active_namespaces": len(_CHUNK_MANAGERS_BY_NS)}


def _get_chunk_manager_for_namespace(namespace: str, settings_obj: Any) -> ChunkedHistoryManager:
    ns = str(namespace or "").strip()
    key = ns or ""
    try:
        _evict_idle_chunk_managers()
    except Exception:
        pass

    mgr = _CHUNK_MANAGERS_BY_NS.get(key)
    if mgr is None:
        try:
            chunk_size_limit = int(getattr(settings_obj, "raw_tail_turns", 10) or 10)
        except Exception:
            chunk_size_limit = 10
        mgr = ChunkedHistoryManager(chunk_size_limit=chunk_size_limit, session_id=(ns or "default"))
        _CHUNK_MANAGERS_BY_NS[key] = mgr

    try:
        _CHUNK_MANAGERS_LAST_SEEN[key] = time.time()
    except Exception:
        pass
    return mgr


def clear_chunk_manager_for_namespace(namespace: str) -> Dict[str, Any]:
    ns = str(namespace or "").strip()
    key = ns or ""
    existed = key in _CHUNK_MANAGERS_BY_NS
    if existed:
        _CHUNK_MANAGERS_BY_NS.pop(key, None)
    _CHUNK_MANAGERS_LAST_SEEN.pop(key, None)
    return {"cleared": bool(existed), "namespace": key, "active_namespaces": len(_CHUNK_MANAGERS_BY_NS)}


# Option A support: index of namespace -> set of cache keys for precise clearing
_SUMMARY_NS_INDEX: Dict[str, Set[str]] = defaultdict(set)
# Option A support: last-seen timestamp per namespace for idle eviction
_SUMMARY_NS_LAST_SEEN: Dict[str, float] = {}


# --- LLM call helper ---


def _responses_create(provider: str | None = None, **kwargs: Any):
    """Compatibility shim for LLM calls."""
    # Extract model from kwargs to use as model_key
    model = kwargs.get("model")
    if not model:
        raise ValueError("model is required for LLM calls")
    
    # DEBUG: Log what model is being used
    logger.info(f"[DEBUG] _responses_create called with model={model}, provider={provider}")
    
    # Filter out conflicting parameters
    filtered_kwargs = {k: v for k, v in kwargs.items() if k not in ['model', 'model_key']}
    
    # Use generate() - provider inferred from model
    return generate(model_key=model, **filtered_kwargs)


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
    
    # DEBUG: Log model_keys received from frontend
    logger.info(f"[DEBUG] model_keys from frontend: {model_keys}")
    
    try:
        inference_model_key_override = str(model_keys.get("inference") or "").strip()
        rewrite_model_key_override = str(model_keys.get("rewrite") or "").strip()
        summary_model_key_override = str(model_keys.get("summary") or "").strip()
        rerank_model_key_override = str(model_keys.get("rerank") or "").strip()
        
        # DEBUG: Log extracted overrides
        logger.info(f"[DEBUG] extracted overrides: inference={inference_model_key_override}, rewrite={rewrite_model_key_override}, summary={summary_model_key_override}, rerank={rerank_model_key_override}")
        
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
    inference_model_key = getattr(settings_obj, "inference_model_key", "llm-adapter")
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

    # Resolve retrieval runtime specs (defaults + optional domain override).
    # Domain precedence: params.active_domain -> params.prompt_domain -> settings.active_domain.
    _active_domain = str(
        p.get("active_domain")
        or p.get("prompt_domain")
        or getattr(settings_obj, "active_domain", "")
        or ""
    ).strip()
    _retrieval_specs = resolve_retrieval_specs(
        domain=_active_domain,
        config_path=str(getattr(settings_obj, "retrieval_config_path", "") or "").strip() or None,
    )

    # Embedding spec fallback: resolve from model registry.
    try:
        _emb = resolve_embedding_spec(settings_obj)  # {provider, model, dimensions}
    except Exception:
        _emb = {"provider": "openai", "model": "text-embedding-3-small", "dimensions": 1536}

    _emb_cfg = (_retrieval_specs or {}).get("embedding") or {}
    emb_runtime = str(_emb_cfg.get("runtime") or "hosted").strip() or "hosted"
    emb_provider = str(_emb_cfg.get("provider") or (_emb or {}).get("provider") or "openai").strip() or "openai"
    emb_model = str(_emb_cfg.get("model") or (_emb or {}).get("model") or "").strip()
    emb_dimensions = _emb_cfg.get("dimensions", (_emb or {}).get("dimensions"))
    emb_normalize = bool(_emb_cfg.get("normalize", True))
    try:
        emb_batch_size = int(_emb_cfg.get("batch_size", 32))
    except Exception:
        emb_batch_size = 32
    emb_device = _emb_cfg.get("device")
    emb_extra = _emb_cfg.get("extra") if isinstance(_emb_cfg.get("extra"), dict) else {}

    # Existing flat temps/limits (read as-is)
    rewrite_temp = float(getattr(settings_obj, "rewrite_temperature", 0.2))
    rewrite_max_out = int(getattr(settings_obj, "rewrite_max_output_tokens", getattr(settings_obj, "rewrite_max_tokens", 128)))

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

    _rr_cfg = (_retrieval_specs or {}).get("rerank") or {}
    rerank_runtime = str(_rr_cfg.get("runtime") or "llm").strip() or "llm"
    rerank_enabled_cfg = bool(_rr_cfg.get("enabled", True))
    rerank_provider_cfg = str(_rr_cfg.get("provider") or "").strip()
    rerank_model_cfg = str(_rr_cfg.get("model") or "").strip()
    rerank_top_n_cfg = _rr_cfg.get("top_n")
    rerank_device_cfg = _rr_cfg.get("device")
    try:
        rerank_batch_size_cfg = int(_rr_cfg.get("batch_size", 16))
    except Exception:
        rerank_batch_size_cfg = 16
    rerank_extra_cfg = _rr_cfg.get("extra") if isinstance(_rr_cfg.get("extra"), dict) else {}

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
            "runtime": emb_runtime,
            "kwargs": {
                "dimensions": emb_dimensions,
                "normalize": emb_normalize,
                "batch_size": emb_batch_size,
                "device": emb_device,
                **emb_extra,
            },
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
            "runtime": rerank_runtime,
            "provider": (rerank_provider_override or rerank_provider_cfg or "openai"),
            "model": (rerank_model_key_override or rerank_model_override or rerank_model_cfg or rerank_model),
            "kwargs": {
                "enabled": rerank_enabled_cfg,
                "temperature": rerank_temp,
                "max_output_tokens": rerank_max_out,
                "top_n": rerank_top_n_cfg,
                "batch_size": rerank_batch_size_cfg,
                "device": rerank_device_cfg,
                **rerank_extra_cfg,
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
                "debug_thoughts": getattr(settings, "debug_thoughts", True),
                **tools_kwargs,
            },
        },
        "tools_synth": {
            # Tools synthesis uses the same provider as inference, but may have its own model.
            # IMPORTANT: This is the final synthesis step. Do NOT add tool-calling params here
            # (e.g., tools/tool_choice/parallel_tool_calls). Tool calls should only happen in
            # the primary inference stage.
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
    """Return response text from a Responses-like object."""

    # If it's already a normalized response (has 'text' field), extract directly
    if isinstance(resp, dict) and resp.get("text"):
        try:
            return str(resp.get("text") or "")
        except Exception:
            pass
    
    # Handle raw responses - prefer adapter_response surface when present
    base = getattr(resp, "adapter_response", resp)

    # Try to extract text from the response
    try:
        if hasattr(base, "output_text"):
            return str(base.output_text or "")
        elif hasattr(base, "text"):
            return str(base.text or "")
        elif isinstance(base, dict):
            return str(base.get("text") or base.get("output_text") or "")
    except Exception:
        pass

    # Fallback - try common response attributes
    for attr in ["output_text", "text", "content"]:
        try:
            val = getattr(resp, attr, None)
            if val:
                return str(val)
        except Exception:
            pass
    
    return ""


def _extract_reasoning_from_responses(resp: Any) -> str | None:
    """Return reasoning text from a Responses-like object."""
    
    # If it's already a normalized response (has 'reasoning' field), extract directly
    if isinstance(resp, dict) and resp.get("reasoning"):
        try:
            reasoning = str(resp.get("reasoning") or "")
            return reasoning if reasoning.strip() else None
        except Exception:
            pass
    
    # Handle raw responses - prefer adapter_response surface when present
    base = getattr(resp, "adapter_response", resp)

    # Try to extract reasoning from the response
    try:
        if hasattr(base, "reasoning"):
            reasoning = str(base.reasoning or "")
            return reasoning if reasoning.strip() else None
        elif isinstance(base, dict):
            reasoning = str(base.get("reasoning") or "")
            return reasoning if reasoning.strip() else None
    except Exception:
        pass

    return None


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

    # Try to extract usage from normalized response or raw response
    try:
        # If it's already a normalized response
        if isinstance(resp, dict) and resp.get("usage"):
            usage = resp.get("usage") or {}
        else:
            # For raw responses, try to extract usage directly
            if hasattr(base, "usage"):
                usage = base.usage or {}
            elif isinstance(base, dict):
                usage = base.get("usage") or {}
            else:
                usage = {}
    except Exception:
        try:
            logger.exception("[USAGE DEBUG] failed to extract usage")
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


def _retrieve_with_rerank(
    *,
    query: str,
    active_domain: str,
    search_mode: str,
    top_k: int,
    score_threshold: float | None,
    use_colbert: bool,
    colbert_top_n: int,
    enable_cross_encoder_rerank: bool,
    cross_encoder_top_n: int,
) -> Dict[str, Any]:
    """
    Retrieve and optionally rerank using RetrievalEvalService.
    This is a modular wrapper that reuses the retrieval-evals pattern for chat.
    """
    service = RetrievalEvalService(active_domain=active_domain)
    
    result = service.run_pipeline(
        query=query,
        search_mode=search_mode,
        top_k=top_k,
        score_threshold=score_threshold,
        query_filter=None,
        with_payload=True,
        exact=False,
        use_colbert=use_colbert,
        colbert_top_n=colbert_top_n,
        enable_cross_encoder_rerank=enable_cross_encoder_rerank,
        cross_encoder_top_n=cross_encoder_top_n,
    )
    
    return result


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

def _build_summary_prompt_with_budget(messages: List[Dict[str, str]], max_input_tokens: int | None, model_name: str, header: str | None = None) -> str:
    """
    Build a summary prompt that fits within `max_input_tokens` by trimming older lines first.
    Guarantees the most recent line is always included (clipped if necessary).
    
    NOTE: This function is currently ONLY used for rewrite stage pre-summarization.
    Other stages (inference, rerank) do not use this token budgeting mechanism.
    Chunked history mode bypasses this entirely and uses ChunkedHistoryManager.
    """
    header = (header if isinstance(header, str) and header else "Summarize the following conversation in a few sentences:\n\n")
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
    tag: str = "",
    model: str | None = None,
    temperature: float | None = None,
    max_input_tokens: int | None = None,
    max_output_tokens: int | None = None,
    log_prefix: str = "[SUMMARY]",
    stage_spec: Dict[str, Any] | None = None,
    provider: str | None = None,
    prompt_domain: str = "",
) -> tuple[str, bool, Dict[str, int] | None]:
    """
    Summarize a slice of messages with a tiny prompt, caching by (messages, tag).

    Returns: (summary_text, from_cache, usage_dict_or_none)
    
    NOTE: This function is currently ONLY used for rewrite stage pre-summarization.
    The tag parameter is used to distinguish cache keys (e.g., 'rewrite' vs 'namespace|rewrite').
    Other stages do not use this summarization mechanism.
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

        registry_path = str(getattr(settings, "inference_prompt_registry_path", "") or "").strip()
        sum_spec = resolve_summary_prompt(registry_path=registry_path, domain=(prompt_domain or "").strip())
        header = (sum_spec.system_instruction or "").strip()
        if header:
            header = header + "\n\n"
        # Build the prompt using the effective model so token budgeting matches the selected provider/model.
        sum_prompt = _build_summary_prompt_with_budget(cleaned_messages, max_input_tokens, _model, header=header)
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

        # DEBUG: Log summary stage details
        logger.info(f"[DEBUG] SUMMARY stage: provider={_provider}, model={_model}")

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


      * When a matching ``ModelInfo`` is found, per-million rates are taken from its
        ``pricing`` field and costs are computed by splitting input into non-cached and
        cached portions.
      * If the model cannot be resolved from the registry, this function returns zeros
        for all cost fields. In this deployment that should be treated as a
        configuration error (missing ModelInfo or pricing), not as a valid "free" run.

    NOTE: This function only affects cost math. It does NOT change any pipeline
    control flow or LLM behavior.
    """

    # Use model_key as primary identifier, provider is inferred
    pricing = get_pricing_for_model(model_key=model_key or model)

    if pricing is not None:
        try:
            # pricing is returned as a dict, use dict access instead of getattr
            in_rate = float(pricing.get("input_per_mm", 0.0) or 0.0)
            out_rate = float(pricing.get("output_per_mm", 0.0) or 0.0)
            cached_rate = float(pricing.get("cached_input_per_mm", 0.0) or 0.0)
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
            "cost_prompt": round(cost_prompt, 10),
            "cost_cached": round(cost_cached, 10),
            "cost_completion": round(cost_completion, 10),
            "cost_total": round(total, 10),
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
            _emb_model_name = str(_emb_spec.get("model") or "text-embedding-3-small")
        except Exception:
            _emb_model_name = "text-embedding-3-small"
        # Exact shape expected by the UI
        self.turn: Dict[str, Any] = {
            "embedding": {"model": _emb_model_name, "input_tokens": 0, "costs": 0.0},
            "rerank": {"model": settings_obj.re_ranker_model, "input_tokens": 0, "output_tokens": 0, "candidates_reranked": 0, "costs": 0.0},
            "summary": {"model": settings_obj.summarizer_model, "applied": False, "reason": "", "input_tokens": 0, "output_tokens": 0, "costs": 0.0},
            "rewrite": {"model": getattr(settings_obj, "rewrite_model", settings_obj.inference_model), "applied": False, "reason": "", "input_tokens": 0, "output_tokens": 0, "costs": 0.0},
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
            "totals": {"tokens": {"turn_total": 0}, "costs": {"turn_total": 0.0}},
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
        pt: int | None = None,
        ct: int | None = None,
        cached: int | None = None,
        model: str | None = None,
        usage: Any | None = None,
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
            self.turn[stage]["costs"] = c["cost_prompt"]
        elif stage == "rerank":
            # Use canonical input_tokens; cached is a subset and tracked separately via cost math.
            self.turn[stage]["input_tokens"] = pt
            self.turn[stage]["output_tokens"] = ct
            c = self._cost("rerank", model, pt, ct, cached, model_key=model_key)
            self.turn[stage]["costs"] = c["cost_total"]
        elif stage == "summary":
            self.turn[stage]["input_tokens"] = pt
            self.turn[stage]["output_tokens"] = ct
            c = self._cost("summary", model, pt, ct, cached, model_key=model_key)
            self.turn[stage]["costs"] = c["cost_total"]
        elif stage == "rewrite":
            self.turn[stage]["input_tokens"] = pt
            self.turn[stage]["output_tokens"] = ct
            c = self._cost("rewrite", model, pt, ct, cached, model_key=model_key)
            self.turn[stage]["costs"] = c["cost_total"]
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
            float(self.turn["embedding"].get("costs") or 0.0)
            + float(self.turn["rerank"].get("costs") or 0.0)
            + float(self.turn["summary"].get("costs") or 0.0)
            + float(self.turn["rewrite"].get("costs") or 0.0)
            + float(self.turn["inference"].get("cost_total") or 0.0)
            + float(self.turn["inference_tools_synth"].get("cost_total") or 0.0)
        )
        self.turn["totals"]["costs"]["turn_total"] = round(total_cost, 10)

        # Accumulate into shared conversation totals
        try:
            self.convo["tokens"]["embedding"] += emb
            # NOTE: cached tokens are already included in stage input/prompt token counts; track them separately but don't double-count.
            self.convo["tokens"]["llm_input"] += (rin + sin + rwin + ip)
            self.convo["tokens"]["llm_output"] += (rout + sout + rwout + ic)
            self.convo["tokens"]["conversation_total"] += total_tokens
            self.convo["costs"]["conversation_total"] = round(float(self.convo["costs"].get("conversation_total", 0.0)) + total_cost, 10)
            logger.debug("[TOTALS] Metrics Finalize Turn turn_total=%d convo_total_now=%d" % (self.turn["totals"]["tokens"]["turn_total"], self.convo["tokens"]["conversation_total"]))
        except Exception:
            # Never let metrics break the answer path
            logger.error("[TOTALS] Metrics Finalize Turn Failure")
            pass

    def snapshot(self) -> tuple[Dict[str, Any], Dict[str, Any]]:
        """Return the current turn metrics and a frontend-aligned conversation totals snapshot."""
        convo_cost = 0.0
        if isinstance(self.convo, dict):
            convo_cost = float(self.convo["costs"].get("conversation_total", 0.0))
        convo_snapshot = {
            "tokens": self.convo.get("tokens", {"embedding": 0, "llm_input": 0, "llm_output": 0, "conversation_total": 0}),
            "costs": {"conversation_total": convo_cost},
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


def _format_web_context_as_text(web_context: Any) -> str:
    try:
        return "\n".join(
            [
                f"{i+1}. {item.get('title','')}\n{item.get('snippet','')}\nURL: {item.get('url','')}"
                for i, item in enumerate(web_context, start=1)
            ]
        )
    except Exception:
        return ""


def _build_inference_messages(
    *,
    system_prompt: str,
    summary_text: str,
    recent_block_str: str,
    context_text: str,
    web_context: Any,
    message: str,
) -> List[Dict[str, str]]:
    messages: List[Dict[str, str]] = [{"role": "system", "content": system_prompt}]
    if summary_text:
        messages.append({"role": "user", "content": f"CONVERSATION SUMMARY: {summary_text}"})
    if recent_block_str:
        messages.append({"role": "user", "content": "RECENT CONVERSATION:\n" + recent_block_str.strip()})
    if context_text:
        messages.append({"role": "user", "content": f"CONTEXT:\n{context_text}"})
    if web_context:
        web_text = _format_web_context_as_text(web_context)
        messages.append({"role": "user", "content": f"WEB SEARCH RESULTS:\n{web_text}"})
    messages.append({"role": "user", "content": message})
    return messages


def _build_tools_synth_messages(
    *,
    system_prompt: str,
    summary_text: str,
    recent_block_str: str,
    context_text: str,
    tool_outputs_list: List[Dict[str, Any]],
    used_tools: List[str],
    tools_text: str,
    message: str,
) -> List[Dict[str, str]]:
    synth_messages: List[Dict[str, str]] = [{"role": "system", "content": system_prompt}]
    if summary_text:
        synth_messages.append({"role": "user", "content": f"CONVERSATION SUMMARY:\n{summary_text}"})
    if recent_block_str:
        synth_messages.append({"role": "user", "content": "RECENT CONVERSATION:\n" + recent_block_str.strip()})
    synth_messages.append({"role": "user", "content": f"[SOURCE: KNOWLEDGE_BASE]\nCONTEXT:\n{context_text}"})
    synth_messages.append({"role": "user", "content": f"TOOLS USED:\n{', '.join(used_tools) if used_tools else ''}"})
    for t in tool_outputs_list:
        synth_messages.append(
            {
                "role": "user",
                "content": f"[SOURCE: TOOL - {t.get('name') or 'unknown'}]\n{str(t.get('output', ''))}",
            }
        )
    synth_messages.append(
        {
            "role": "user",
            "content": f"Question: {message}\n\nTask: Produce the final answer to the Question using the Context and Tool results.",
        }
    )
    return synth_messages

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
    prompt_domain: str = "",
    log_prefix: str = "[REWRITE]",
    stage_spec: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """Call the rewrite model to produce a self-contained query.
    Returns a dict with keys: rewritten, changed, confidence, ambiguous, reason.
    On any failure, returns the original unmodified with changed=False.
    """
    try:
        registry_path = str(getattr(settings, "inference_prompt_registry_path", "") or "").strip()
        rw_spec = resolve_rewrite_prompt(registry_path=registry_path, domain=(prompt_domain or "").strip())

        tail_lines: List[str] = []
        try:
            for m in (tail_messages or []):
                role = str(m.get("role") or "user")
                content = str(m.get("content") or "")
                if role == "assistant":
                    try:
                        content = _strip_trailing_sources_block(content)
                    except Exception:
                        pass
                tail_lines.append(f"{role}: {content}")
        except Exception:
            tail_lines = []
        recent_block_str = "\n".join(tail_lines)

        payload = render_full_payload(
            rw_spec.full_payload_template,
            variables={
                "summary_text": summary_text or "",
                "recent_block_str": (recent_block_str or "").strip(),
                "message": message or "",
            },
        )

        prompt = rw_spec.system_instruction + "\n\n" + payload
        # Log an estimated prompt token count for rewrite
        try:
            _rw = stage_spec or {}
            _model_for_est = str(_rw.get("model") or getattr(settings, 'rewrite_model_key', 'openai:gpt-4o-mini'))
            enc = _get_encoder_for_model(_model_for_est)
            pt_est = len(enc.encode(prompt))
            #logger.debug(f"{log_prefix} prompt_token_est≈%d model=%s", pt_est, _model_for_est)
        except Exception:
            pass
        # Invoke the rewrite model with the prompt for the user's latest message for it to rewrite it
        _rw = stage_spec or {}
        _provider = str(_rw.get("provider") or "openai")
        _model = str(_rw.get("model") or getattr(settings, 'rewrite_model_key', 'openai:gpt-4o-mini'))
        _kwargs = dict(_rw.get("kwargs") or {})
        if not _kwargs:
            _kwargs = {
                "max_output_tokens": int(getattr(settings, 'rewrite_max_output_tokens', 300)),
                "temperature": float(getattr(settings, 'rewrite_temperature', 0.3)),
            }

        # DEBUG: Log rewrite stage details with endpoint info
        try:
            from backend.llm.llm_client import get_model_info
            model_info = get_model_info(model_key=_model)
            endpoint = getattr(model_info, 'endpoint', 'unknown')
            logger.info(f"[DEBUG] REWRITE stage: provider={_provider}, model={_model}, endpoint={endpoint}, stage_spec={stage_spec}")
        except Exception as e:
            logger.info(f"[DEBUG] REWRITE stage: provider={_provider}, model={_model}, endpoint=unknown (error: {e}), stage_spec={stage_spec}")

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
        
        # Chunked history managers for different sessions
        self.chunk_managers: Dict[str, ChunkedHistoryManager] = {}

    def get_or_create_chunk_manager(self, session_id: str = "default") -> ChunkedHistoryManager:
        """Get or create a chunk manager for the given session."""
        if session_id not in self.chunk_managers:
            chunk_size_limit = getattr(settings, 'raw_tail_turns', 10)  # Use existing config
            self.chunk_managers[session_id] = ChunkedHistoryManager(
                chunk_size_limit=chunk_size_limit,
                session_id=session_id
            )
            logger.debug(f"[CHUNKED] Created new chunk manager for session {session_id}, chunk_size={chunk_size_limit}")
        return self.chunk_managers[session_id]

    def _get_session_id(self, params: Dict[str, Any] | None) -> str:
        """Extract session ID from params or use default."""
        if not params:
            return "default"
        return str(params.get("session_id", "default"))

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

    def chat(self, message: str, context: List[Dict], use_web_search: bool | None = None, params: Dict[str, Any] | None = None) -> Dict:
        """
        Thin wrapper: delegate to run_pipeline (Option A).
        Maintains stateful history and returns answer + sources.
        """

        # Prefer caller-provided query_id (so SSE subscriber can pre-open /chat/stream/stages?query_id=...)
        _p = params or {}
        req_id = str(_p.get("query_id") or _p.get("request_id") or uuid.uuid4().hex[:8])
        logger.info("Starting chat in chat_manager.chat() [req_id=%s] [msg=%s]", req_id, message[:50])
        if use_web_search is None:
            use_web_search = bool(getattr(settings, "use_web_search", False))
        try:
            if isinstance(params, dict) and params.get("use_web_search") is not None:
                use_web_search = bool(params.get("use_web_search"))
        except Exception:
            pass
        logger.debug("Context length=%d use_web_search=%s", len(context), use_web_search)
        
        # Debug: Show what history we're using
        history_to_use = context if context is not None else self.chat_history
        logger.info("Using history: %d messages from %s", len(history_to_use), 
                   "session context" if context is not None else "ChatManager history")
        if history_to_use:
            logger.debug("History preview: %s", 
                        [{"role": msg.get("role", "unknown"), "content": msg.get("content", "")[:50] + "..."} 
                         for msg in history_to_use[-3:]])  # Show last 3 messages

        # Always use orchestrator; legacy inlined flow removed (kept in git history).
        # Derive namespace for token accounting (use session_id as conversation_id)
        try:
            _uid = str((params or {}).get("user_id") or "").strip()
            _session_id = str((params or {}).get("session_id") or "").strip()
            # For session-based chat, use session_id as namespace for proper token accounting
            session_namespace = f"session:{_session_id}" if _session_id else ""
        except Exception:
            session_namespace = ""

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
                "namespace": session_namespace,  # Add namespace for token accounting
            }
            # Use context parameter if provided, otherwise use self.chat_history
            # This allows session-based chat to work properly while maintaining backward compatibility
            history_to_use = context if context is not None else self.chat_history
            req = {"message": message, "history": history_to_use, "params": (params or {})}

            out = run_pipeline(deps=deps, req=req)
            answer_text = out.get("answer", "") or ""
            
            # Debug: Log what we got from orchestrator
            logger.info("DEBUG: orchestrator out keys: %s", list(out.keys()) if isinstance(out, dict) else "not a dict")
            logger.info("DEBUG: orchestrator out: %s", out)

            # Update stateful history to preserve conversation context
            self.chat_history.extend([
                {"role": "user", "content": message},
                {"role": "assistant", "content": answer_text},
            ])

            # Prepare response with all metrics
            response_dict = {
                "response": answer_text,
                "answer": answer_text,  # Add for consistency with handle_chat
                "sources": out.get("sources", []),
                "metrics": out.get("metrics", {"vectors_retrieved": 0}),
                "turn_metrics": out.get("turn_metrics", {}),
                "conversation_totals": out.get("conversation_totals", {}),
                "tools_used": out.get("tools_used", []),
                "rewrite_display": out.get("rewrite_display", {}),
            }
            
            logger.info("DEBUG: response_dict keys: %s", list(response_dict.keys()))
            logger.info("DEBUG: response_dict: %s", response_dict)
            
            return response_dict
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
    # Unwrap adapter-style responses first (e.g., AdapterResponse from llm_adapter)
    # so we always inspect the provider-native object for tool_calls.
    base = getattr(resp, "adapter_response", resp)

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
        # If it's already a normalized response, extract tool_calls directly
        if isinstance(resp, dict) and resp.get("tool_calls"):
            calls = list(resp.get("tool_calls") or [])
        else:
            # For raw responses, try to extract tool_calls directly
            if hasattr(base, "tool_calls"):
                calls = list(base.tool_calls or [])
            elif isinstance(base, dict):
                calls = list(base.get("tool_calls") or [])
            else:
                calls = []
    except Exception:
        calls = []

    calls = [c for c in calls if c.get("name")]
    # Deduplicate calls
    deduped = _dedup(calls)

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


def _tool_registry_path(settings_obj: Any) -> Path:
    raw_path = str(getattr(settings_obj, "tool_registry_path", "") or "").strip() or "prompts/tool_registry.yaml"
    candidate = Path(raw_path)
    if candidate.is_absolute():
        return candidate
    repo_root = Path(__file__).resolve().parents[2]
    return repo_root / candidate


def clear_tool_registry_cache(path: str | None = None) -> int:
    """Clear cached tool registry entries and return number removed."""
    if path:
        key = str(path)
        return 1 if _TOOL_REGISTRY_CACHE.pop(key, None) is not None else 0
    removed = len(_TOOL_REGISTRY_CACHE)
    _TOOL_REGISTRY_CACHE.clear()
    return removed


def _load_tool_registry(settings_obj: Any) -> Dict[str, Dict[str, Any]]:
    path = _tool_registry_path(settings_obj)
    cache_key = str(path)
    try:
        mtime = float(path.stat().st_mtime) if path.exists() else -1.0
    except Exception:
        mtime = -1.0

    cached = _TOOL_REGISTRY_CACHE.get(cache_key)
    if isinstance(cached, dict):
        try:
            if float(cached.get("mtime", -2.0)) == mtime and isinstance(cached.get("value"), dict):
                return dict(cached.get("value") or {})
        except Exception:
            pass

    try:
        if not path.exists():
            return {}
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
    except Exception:
        return {}

    tools = data.get("tools") if isinstance(data, dict) else []
    if not isinstance(tools, list):
        return {}

    by_name: Dict[str, Dict[str, Any]] = {}
    for item in tools:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or "").strip()
        if not name:
            continue
        runtime = item.get("runtime") if isinstance(item.get("runtime"), dict) else {}
        endpoint = runtime.get("endpoint") if isinstance(runtime.get("endpoint"), dict) else {}
        endpoint_type = str(endpoint.get("type") or "").strip()
        endpoint_url = str(endpoint.get("url") or "").strip()
        if not endpoint_type or not endpoint_url:
            raise ValueError(
                f"Tool '{name}' in tool registry must define runtime.endpoint.type and runtime.endpoint.url"
            )
        by_name[name] = item

    artifact_injection = data.get("artifact_injection") if isinstance(data, dict) else {}
    if not isinstance(artifact_injection, dict):
        artifact_injection = {}

    loaded = {
        "tools_by_name": by_name,
        "artifact_injection": artifact_injection,
    }
    _TOOL_REGISTRY_CACHE[cache_key] = {"mtime": mtime, "value": loaded}
    return loaded


def _extract_artifacts_from_tool_outputs(
    tool_outputs_list: List[Dict[str, Any]],
    tool_registry: Dict[str, Any],
) -> List[Dict[str, str]]:
    artifacts: List[Dict[str, str]] = []
    default_max_artifact_chars = 120000
    placeholder_re = re.compile(r"^\{\{ARTIFACT:[A-Za-z0-9_.:-]{1,64}\}\}$")
    artifact_injection = tool_registry.get("artifact_injection") if isinstance(tool_registry, dict) else {}
    if not isinstance(artifact_injection, dict):
        artifact_injection = {}
    if not bool(artifact_injection.get("enabled", True)):
        return artifacts

    security_cfg = artifact_injection.get("security") if isinstance(artifact_injection, dict) else {}
    if not isinstance(security_cfg, dict):
        security_cfg = {}

    try:
        max_artifact_chars = int(security_cfg.get("max_artifact_chars", default_max_artifact_chars) or default_max_artifact_chars)
    except Exception:
        max_artifact_chars = default_max_artifact_chars
    if max_artifact_chars <= 0:
        max_artifact_chars = default_max_artifact_chars

    allowed_artifact_types = {
        str(v).strip().lower()
        for v in (security_cfg.get("allowed_artifact_types") or ["svg"])
        if isinstance(v, str) and str(v).strip()
    }
    if not allowed_artifact_types:
        allowed_artifact_types = {"svg"}

    allowed_injection_modes = {
        str(v).strip().lower()
        for v in (security_cfg.get("allowed_injection_modes") or ["verbatim"])
        if isinstance(v, str) and str(v).strip()
    }
    if not allowed_injection_modes:
        allowed_injection_modes = {"verbatim"}

    enforce_placeholder_format = bool(security_cfg.get("enforce_placeholder_format", True))

    raw_allowed = artifact_injection.get("allowed_tools")
    allowed_artifact_tools = {
        str(n).strip()
        for n in (raw_allowed or [])
        if isinstance(n, str) and str(n).strip()
    }

    tools_by_name = tool_registry.get("tools_by_name") if isinstance(tool_registry, dict) else {}
    if not isinstance(tools_by_name, dict):
        tools_by_name = {}

    for t in (tool_outputs_list or []):
        tool_name = str(t.get("name") or "").strip()
        if not tool_name:
            continue
        if allowed_artifact_tools and tool_name not in allowed_artifact_tools:
            continue
        cfg = tools_by_name.get(tool_name) or {}
        artifact_cfg = cfg.get("artifact") if isinstance(cfg, dict) else None
        if not isinstance(artifact_cfg, dict):
            continue
        if not bool(artifact_cfg.get("produces_artifact")):
            continue

        artifact_key = str(artifact_cfg.get("artifact_key") or "").strip()
        placeholder = str(artifact_cfg.get("placeholder") or "").strip()
        artifact_type = str(artifact_cfg.get("artifact_type") or "").strip().lower()
        injection_mode = str(artifact_cfg.get("injection_mode") or "").strip().lower()
        if not artifact_key:
            continue
        if artifact_type not in allowed_artifact_types:
            logger.warning("[ARTIFACT] tool=%s skipped: unsupported artifact_type=%s", tool_name, artifact_type)
            continue
        if injection_mode not in allowed_injection_modes:
            logger.warning("[ARTIFACT] tool=%s skipped: unsupported injection_mode=%s", tool_name, injection_mode)
            continue
        if enforce_placeholder_format and placeholder and not placeholder_re.match(placeholder):
            logger.warning("[ARTIFACT] tool=%s skipped: invalid placeholder format", tool_name)
            continue

        payload = ""
        out = str(t.get("output") or "")
        try:
            parsed = json.loads(out)
            if isinstance(parsed, dict):
                v = parsed.get(artifact_key)
                if isinstance(v, str) and v.strip():
                    payload = v.strip()
        except Exception:
            payload = ""

        if not payload and artifact_type == "svg":
            try:
                m = re.search(r"<svg\\b[\\s\\S]*?</svg>", out, flags=re.IGNORECASE)
                if m and m.group(0).strip():
                    payload = m.group(0).strip()
            except Exception:
                payload = ""

        if payload and len(payload) > max_artifact_chars:
            logger.warning(
                "[ARTIFACT] tool=%s skipped: payload too large chars=%d max=%d",
                tool_name,
                len(payload),
                max_artifact_chars,
            )
            payload = ""

        if payload and artifact_type == "svg":
            # Runtime hardening: reject known-unsafe patterns and sanitize SVG before injection.
            lower_payload = payload.lower()
            if (
                "<script" in lower_payload
                or "javascript:" in lower_payload
                or "<foreignobject" in lower_payload
                or re.search(r"\son[a-z]+\s*=", lower_payload) is not None
            ):
                logger.warning("[ARTIFACT] tool=%s skipped: unsafe svg pattern detected", tool_name)
                payload = ""
            else:
                try:
                    sanitized = bleach.clean(
                        payload,
                        tags=["svg", "polyline", "line", "rect", "path", "text", "g"],
                        attributes={
                            "svg": ["width", "height", "viewBox", "role", "aria-hidden", "xmlns"],
                            "polyline": ["fill", "stroke", "stroke-width", "stroke-linecap", "stroke-linejoin", "points"],
                            "line": ["x1", "y1", "x2", "y2", "stroke", "stroke-width", "stroke-linecap"],
                            "rect": ["x", "y", "width", "height", "fill", "rx", "ry", "stroke", "stroke-width"],
                            "path": ["d", "fill", "stroke", "stroke-width", "stroke-linecap", "stroke-linejoin"],
                            "text": ["x", "y", "text-anchor", "fill", "font-size", "font-weight"],
                            "g": ["transform", "fill", "stroke", "stroke-width"],
                        },
                        protocols=["http", "https"],
                        strip=True,
                    ).strip()
                    if "<svg" not in sanitized.lower() or "</svg>" not in sanitized.lower():
                        logger.warning("[ARTIFACT] tool=%s skipped: sanitized svg invalid", tool_name)
                        payload = ""
                    else:
                        payload = sanitized
                except Exception:
                    logger.warning("[ARTIFACT] tool=%s skipped: svg sanitization failed", tool_name)
                    payload = ""

        if payload:
            artifacts.append(
                {
                    "tool": tool_name,
                    "payload": payload,
                    "placeholder": placeholder,
                    "injection_mode": injection_mode,
                    "artifact_type": artifact_type,
                }
            )
    return artifacts


def _inject_registered_artifacts(text: str, artifacts: List[Dict[str, str]]) -> str:
    combined = str(text or "")
    for a in artifacts:
        tool_name = str(a.get("tool") or "unknown")
        payload = str(a.get("payload") or "").strip()
        placeholder = str(a.get("placeholder") or "").strip()
        injection_mode = str(a.get("injection_mode") or "").strip().lower()
        artifact_type = str(a.get("artifact_type") or "").strip().lower()
        if not payload:
            logger.debug("[ARTIFACT] tool=%s skipped: empty payload", tool_name)
            continue

        if placeholder and placeholder in combined:
            combined = combined.replace(placeholder, payload)
            logger.debug("[ARTIFACT] tool=%s placeholder_hit=true placeholder=%s", tool_name, placeholder)
            continue
        if placeholder:
            logger.debug("[ARTIFACT] tool=%s placeholder_hit=false placeholder=%s", tool_name, placeholder)
            try:
                token_match = re.match(r"^\{\{(ARTIFACT:[A-Za-z0-9_.:-]{1,64})\}\}$", placeholder)
                token = token_match.group(1) if token_match else ""
                if token:
                    loose_pattern = re.compile(r"\{+\s*" + re.escape(token) + r"\s*\}+")
                    combined, loose_count = loose_pattern.subn(payload, combined)
                    logger.debug(
                        "[ARTIFACT] tool=%s placeholder_loose_match_replacements=%d token=%s",
                        tool_name,
                        int(loose_count),
                        token,
                    )
                    if loose_count > 0:
                        continue
            except Exception:
                logger.debug("[ARTIFACT] tool=%s placeholder_loose_match_failed", tool_name, exc_info=True)

        # If model output contains a truncated SVG fragment, remove the broken tail before
        # injecting canonical SVG. This prevents malformed nested markup in finalHtml.
        if artifact_type == "svg":
            has_svg_open = "<svg" in combined.lower()
            has_svg_close = "</svg>" in combined.lower()
            if has_svg_open and not has_svg_close:
                try:
                    prefix = combined.split("<svg", 1)[0].rstrip()
                    combined = prefix
                    logger.debug("[ARTIFACT] tool=%s removed_truncated_svg_prefix=true", tool_name)
                except Exception:
                    pass

        if injection_mode == "verbatim" and payload not in combined:
            if combined.strip():
                combined = f"{combined.strip()}\n\n{payload}"
            else:
                combined = payload
            logger.debug("[ARTIFACT] tool=%s appended_verbatim=true", tool_name)
        elif injection_mode == "verbatim":
            logger.debug("[ARTIFACT] tool=%s appended_verbatim=false reason=already_present", tool_name)
    return combined


def _redact_tool_outputs_for_synth(tool_outputs_list: List[Dict[str, Any]], tool_registry: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Return tool outputs safe for tools_synth prompt by removing artifact payload fields.

    The second LLM synthesis pass should not receive large binary-like artifacts (e.g., SVG).
    We keep compact metadata so the model can still reference tool results in prose.
    """
    redacted: List[Dict[str, Any]] = []
    tools_by_name = tool_registry.get("tools_by_name") if isinstance(tool_registry, dict) else {}
    if not isinstance(tools_by_name, dict):
        tools_by_name = {}

    for t in (tool_outputs_list or []):
        item = dict(t or {})
        tool_name = str(item.get("name") or "").strip()
        output_text = str(item.get("output") or "")

        cfg = tools_by_name.get(tool_name) if tool_name else None
        artifact_cfg = cfg.get("artifact") if isinstance(cfg, dict) else None
        produces_artifact = bool(isinstance(artifact_cfg, dict) and artifact_cfg.get("produces_artifact"))
        artifact_key = str((artifact_cfg or {}).get("artifact_key") or "").strip()
        placeholder = str((artifact_cfg or {}).get("placeholder") or "").strip()

        if not produces_artifact or not artifact_key:
            redacted.append(item)
            continue

        compact = ""
        try:
            parsed = json.loads(output_text)
            if isinstance(parsed, dict):
                if artifact_key in parsed:
                    parsed.pop(artifact_key, None)
                if placeholder:
                    parsed["artifact_placeholder"] = placeholder
                parsed["artifact_payload_omitted"] = True
                compact = json.dumps(parsed, ensure_ascii=False)
        except Exception:
            compact = ""

        if not compact:
            compact = f"Artifact payload omitted for synthesis. placeholder={placeholder or '(none)'}"

        item["output"] = compact
        redacted.append(item)

    return redacted


def _strip_svg_from_messages(messages: List[Dict[str, Any]]) -> tuple[List[Dict[str, Any]], int]:
    """Best-effort removal of raw SVG blocks from model input messages.

    This is a last-mile safety guard to ensure tools_synth never receives raw
    artifact payloads, even if an upstream path accidentally includes them.
    """
    stripped_count = 0
    out: List[Dict[str, Any]] = []
    for m in (messages or []):
        item = dict(m or {})
        content = item.get("content")
        if isinstance(content, str):
            new_content, n = re.subn(r"<svg\b[\s\S]*?</svg>", "[SVG_ARTIFACT_OMITTED]", content, flags=re.IGNORECASE)
            if n > 0:
                stripped_count += int(n)
            item["content"] = new_content
        out.append(item)
    return out, stripped_count

#
# --- History slicing helper moved to utils.py ---


# --- Tail cleanup helper (optional, cosmetic) ---
def _strip_trailing_sources_block(text: str) -> str:
    """
    Remove a trailing 'Sources:' block from an assistant message, if present.
    Only strips when the block appears at the end to avoid cutting inline mentions.
    """
    try:
        s = (text or "").rstrip()
        # Handle both old Sources: and new <sources>Sources</sources>: patterns for backward compatibility
        m = re.search(r"(?:\r?\n)(?:<sources>Sources</sources>|Sources|sources):\s*\r?\n[\s\S]*\Z", s)
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
    # Support both flat format (embedding_model_key) and nested format (model_keys.embedding)
    # Fallback to settings model keys if not provided in params
    _mk = lambda k: (str(params.get(k)).strip() or None) if params.get(k) is not None else None
    _stage_model_keys = {s: _mk(f"{s}_model_key") for s in ("embedding", "rewrite", "summary", "rerank", "inference", "tools_synth")}
    # Also check nested model_keys format from frontend
    model_keys_nested = params.get("model_keys") or {}
    for stage in ("embedding", "rewrite", "summary", "rerank", "inference", "tools_synth"):
        if model_keys_nested.get(stage) and not _stage_model_keys.get(stage):
            _stage_model_keys[stage] = str(model_keys_nested.get(stage)).strip() or None
    # Fallback to settings model keys if still None
    if not _stage_model_keys.get("embedding"):
        _stage_model_keys["embedding"] = str(getattr(settings_obj, "embedding_model_key", "openai:embed_small"))
    if not _stage_model_keys.get("inference"):
        _stage_model_keys["inference"] = str(getattr(settings_obj, "inference_model_key", "openai:gpt-4o-mini"))
    if not _stage_model_keys.get("rewrite"):
        _stage_model_keys["rewrite"] = str(getattr(settings_obj, "rewrite_model_key", "openai:gpt-4o-mini"))
    if not _stage_model_keys.get("rerank"):
        _stage_model_keys["rerank"] = str(getattr(settings_obj, "rerank_model_key", "openai:gpt-4o-mini"))
    if not _stage_model_keys.get("summary"):
        _stage_model_keys["summary"] = str(getattr(settings_obj, "summarizer_model_key", "openai:gpt-4o-mini"))
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

    # Optional per-request override
    try:
        if isinstance(params, dict) and "show_sources" in params:
            _v = params.get("show_sources")
            if _v is not None:
                display_sources = bool(_v)
    except Exception:
        pass

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
                rw_summary, src_rw_summary = _get_param_int(params, ["rewrite_summary_turns"], getattr(settings_obj, "rewrite_summary_turns", 3), minimum=0)
                thr, src_thr = _get_param_float(params, ["rewrite_confidence_threshold"], getattr(settings_obj, "rewrite_confidence_threshold", 0.6), minimum=0.0, maximum=0.99)
                logger.debug("[REWRITE PARAMS] (%s) enable=%s tail_turns=%d (%s) summary_turns=%d (%s) threshold=%.2f (%s)", log_origin, True, rw_tail, src_rw_tail, rw_summary, src_rw_summary, thr, src_thr)
                raw_tail = max(0, int(rw_tail))
                window_turns = max(0, int(rw_summary))
                to_sum_rw, tail_rw = split_history_for_prompt(history, raw_tail, window_turns)
                summary_rw = ""
                # Skip pre-summary if rewrite_summary_turns is 0
                if window_turns > 0 and to_sum_rw:
                    # Prefix tag with namespace if provided to isolate cache entries by conversation
                    _tag_rw = (f"{namespace}|rewrite" if namespace else "rewrite")
                    sum_spec = (stage_specs or {}).get("summary") or {}
                    try:
                        # NOTE: This is the ONLY place where _summarize_messages_with_cache is called
                        # Used exclusively for rewrite stage pre-summarization when rewrite_summary_turns > 0
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
                                    "costs": {"conversation_total": 0.0},
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
                    try:
                        _pd = str((params or {}).get("prompt_domain") or "").strip()
                    except Exception:
                        _pd = ""
                    if not _pd:
                        try:
                            _pd = str(getattr(settings_obj, "prompt_domain_default", "") or "").strip()
                        except Exception:
                            _pd = ""
                    rw = rewrite_query(
                        tail_rw,
                        summary_rw,
                        message,
                        prompt_domain=_pd,
                        log_prefix=f"[REWRITE] {log_origin}",
                        stage_spec=rw_spec,
                    )
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
                                "costs": {"conversation_total": 0.0},
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
            convo_snapshot = {"tokens": {"embedding": 0, "llm_input": 0, "llm_output": 0, "conversation_total": 0}, "costs": {"conversation_total": 0.0}}
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
    
    # Extract new retrieval/rerank parameters from request
    _active_domain = str(params.get("active_domain") or params.get("prompt_domain") or getattr(settings_obj, "active_domain", "") or "").strip()
    _search_mode = str(params.get("search_mode") or "dense").strip().lower()
    _use_colbert = bool(params.get("use_colbert", False))
    _colbert_top_n = int(params.get("colbert_top_n", 8))
    _enable_cross_encoder_rerank = bool(params.get("enable_cross_encoder_rerank", True))
    _cross_encoder_top_n = int(params.get("cross_encoder_top_n", 5))
    
    # Use the new retrieval/rerank service if the parameters are provided
    # Otherwise fall back to the legacy path
    use_new_retrieval = bool(params.get("search_mode") or params.get("use_colbert") or params.get("enable_cross_encoder_rerank"))
    skip_rerank = False  # Will be set to True if we use the new service
    
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
    
    # --- Retrieve with new service or legacy path ---
    if use_new_retrieval:
        # Use the new retrieval/rerank service
        try:
            logger.info("[RETRIEVE] (%s) using new retrieval service with search_mode=%s, use_colbert=%s, enable_cross_encoder=%s", 
                       log_origin, _search_mode, _use_colbert, _enable_cross_encoder_rerank)
            retrieval_result = _retrieve_with_rerank(
                query=effective_query,
                active_domain=_active_domain,
                search_mode=_search_mode,
                top_k=int(top_k),
                score_threshold=float(score_threshold) if score_threshold is not None else None,
                use_colbert=_use_colbert,
                colbert_top_n=_colbert_top_n,
                enable_cross_encoder_rerank=_enable_cross_encoder_rerank,
                cross_encoder_top_n=_cross_encoder_top_n,
            )
            
            # Extract results from the retrieval response
            if _enable_cross_encoder_rerank and retrieval_result.get("reranked"):
                # Use cross-encoder reranked results
                reranked = retrieval_result["reranked"]
                results = [item["item"] for item in reranked.get("items", [])]
            else:
                # Use retrieval results (with or without ColBERT)
                retrieval = retrieval_result.get("retrieval", {})
                results = retrieval.get("results", [])
            
            # Skip reranking stage since we already did it in the service
            skip_rerank = True
        except Exception as e:
            logger.error("[RETRIEVE] (%s) new retrieval service failed, falling back to legacy: %s", log_origin, e)
            # Fall back to legacy path
            use_new_retrieval = False
            skip_rerank = False
    
    if not use_new_retrieval:
        # Legacy retrieval path
        _es = (stage_specs or {}).get("embedding") or {}
        _emb_runtime = str(_es.get("runtime") or "hosted").strip().lower() or "hosted"
        _emb_provider_stage = str(_es.get("provider") or "openai").strip() or "openai"
        _emb_model_stage = str(_es.get("model") or "").strip()
        _emb_kwargs_stage = dict(_es.get("kwargs") or {})
        _emb_extra_stage = {
            k: v
            for k, v in _emb_kwargs_stage.items()
            if k not in {"dimensions", "normalize", "batch_size", "device"}
        }
        _embed_model_for_metrics = _emb_model_stage

        try:
            if _emb_runtime == "fastembed":
                try:
                    _dims = _emb_kwargs_stage.get("dimensions")
                    _dims_i = int(_dims) if _dims is not None else None
                except Exception:
                    _dims_i = None
                try:
                    _batch_size = int(_emb_kwargs_stage.get("batch_size", 32))
                except Exception:
                    _batch_size = 32
                _device = _emb_kwargs_stage.get("device")

                _emb_spec = EmbeddingSpec(
                    task="embedding",
                    runtime="fastembed",
                    provider=_emb_provider_stage,
                    model=_emb_model_stage,
                    dimensions=_dims_i,
                    normalize=bool(_emb_kwargs_stage.get("normalize", True)),
                    batch_size=max(1, _batch_size),
                    device=(str(_device).strip() if _device is not None else None),
                    extra=_emb_extra_stage,
                )
                _emb_res = _FASTEMBED_EMBEDDING_PROVIDER.embed([effective_query], _emb_spec)
                _qv = (_emb_res.vectors or [[]])[0]
                results = db.search_similar_by_embedding(
                    query_embedding=_qv,
                    limit=int(top_k),
                    score_threshold=float(score_threshold),
                    with_payload=True,
                    exact=True,
                )
            else:
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
                        "costs": {"conversation_total": 0.0},
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
    embed_tokens = 0
    if not use_new_retrieval and _emb_runtime == "fastembed":
        # Local FastEmbed path does not currently expose token accounting in db.last_embedding_usage.
        embed_tokens = 0
    else:
        try:
            raw_last = getattr(db, "last_embedding_usage", None)
            logger.debug("[EMB] (%s) db.last_embedding_usage=%r", log_origin, raw_last)
            last = raw_last or {}
            embed_tokens = int((last.get("input_tokens") or last.get("total_tokens") or 0))
            logger.debug("[EMB] (%s) parsed embed_tokens=%d", log_origin, embed_tokens)
        except Exception:
            embed_tokens = 0

    # Use stage-selected embedding model when provided; otherwise fall back to resolved hosted spec.
    if _embed_model_for_metrics:
        _emb_model_for_cost = _embed_model_for_metrics
    else:
        try:
            _emb_spec_cost = resolve_embedding_spec(settings_obj) or {}
            _emb_model_for_cost = str(
                (_emb_spec_cost.get("model") or "text-embedding-3-small")
            )
        except Exception:
            _emb_model_for_cost = "text-embedding-3-small"

    m.record_stage(
        "embedding",
        model=_emb_model_for_cost,
        pt=embed_tokens,
        model_key=(_stage_model_keys or {}).get("embedding"),
    )

# Stage: Rerank Retrieval Results
    
    # Skip rerank stage if we already used the new retrieval/rerank service
    if skip_rerank:
        logger.debug("[RERANK] (%s) skipping rerank stage (already handled by retrieval service)", log_origin)
        reranked = results
    else:
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
        _rs_cfg = (stage_specs or {}).get("rerank") or {}
        _rs_kwargs = dict(_rs_cfg.get("kwargs") or {})
        kept = min(int(getattr(settings_obj, "re_ranker_input_rows", 5)), n)
        reranked = results
        rerank_enabled = bool(_rs_kwargs.get("enabled", True))
        skip_reason = ""
        need_rerank = False

        if not rerank_enabled:
            need_rerank = False
            skip_reason = "disabled by retrieval config"
        elif n <= 1:
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
                try:
                    _pd_rr = str((params or {}).get("prompt_domain") or "").strip()
                except Exception:
                    _pd_rr = ""
                if not _pd_rr:
                    try:
                        _pd_rr = str(getattr(settings_obj, "prompt_domain_default", "") or "").strip()
                    except Exception:
                        _pd_rr = ""

                chunk_size = int(getattr(settings_obj, "reranker_chunk_size", 600))
                candidates_block = "\n".join([f"[{i}] {t[:chunk_size]}" for i, t in enumerate(cand_text or [])])

                registry_path = str(getattr(settings_obj, "inference_prompt_registry_path", "") or "").strip()
                rr_spec = resolve_rerank_prompt(registry_path=registry_path, domain=_pd_rr)
                rr_payload = render_full_payload(
                    rr_spec.full_payload_template,
                    variables={
                        "query": effective_query,
                        "candidates_block": candidates_block,
                    },
                )
                prompt_text = rr_spec.system_instruction + "\n\n" + rr_payload
                _dbg(f"[RERANK] {log_origin} prompt:", prompt_text)

                # Provider-aware rerank call via stage_specs (behavior-identical defaults).
                _rs = (stage_specs or {}).get("rerank") or {}
                _provider = str(_rs.get("provider") or "openai")
                _model = str(
                    _rs.get("model")
                    or getattr(settings_obj, "re_ranker_model", settings_obj.inference_model)
                )
                _runtime = str(_rs.get("runtime") or "llm").strip() or "llm"
                _kwargs = dict(_rs.get("kwargs") or {})

                logger.debug(
                    "[RERANK] (%s) runtime=%s provider=%s model=%s kwargs=%r",
                    log_origin,
                    _runtime,
                    _provider,
                    _model,
                    _kwargs,
                )

                usage_rr: Dict[str, Any] = {}
                if _runtime == "llm":
                    logger.info(f"[DEBUG] RERANK stage: provider={_provider}, model={_model}")
                    _kwargs_llm = {
                        k: v
                        for k, v in _kwargs.items()
                        if k not in {"enabled", "top_n", "batch_size", "device"}
                    }
                    resp_rerank = _responses_create(
                        provider=_provider,
                        model=_model,
                        input=prompt_text.strip(),
                        **_kwargs_llm,
                    )
                    content = _extract_text_from_responses(resp_rerank).strip()
                    _dbg(f"[RERANK] {log_origin} raw:", content)
                    order = _parse_json_array_in_text(content, pool_n)
                    reranked = [pool[i] for i in order] or pool
                    reranked = reranked[:kept]
                    usage_rr = _extract_usage_from_responses(resp_rerank, provider=_provider) or {}
                else:
                    raise ValueError(f"Unsupported rerank runtime: {_runtime}")

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
                            "costs": {"conversation_total": 0.0},
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
    # Initialize simple history processor for consistent formatting
    history_processor = SimpleHistoryProcessor(settings_obj)
    
    # Always use chunked history approach
    chunk_manager = _get_chunk_manager_for_namespace(namespace, settings_obj)
    try:
        _effective_chunk_turns, _chunk_turns_src = _get_param_int(
            params,
            ["raw_tail_turns"],
            int(getattr(settings_obj, "raw_tail_turns", 10) or 10),
        )
    except Exception:
        _effective_chunk_turns = int(getattr(settings_obj, "raw_tail_turns", 10) or 10)
    try:
        if int(_effective_chunk_turns or 0) > 0 and int(getattr(chunk_manager, "chunk_size_limit", 0) or 0) != int(_effective_chunk_turns):
            chunk_manager.chunk_size_limit = int(_effective_chunk_turns)
    except Exception:
        pass
    try:
        logger.debug(f"[CHUNKED] Using chunked history for namespace '{namespace or ''}'")
    except Exception:
        pass
    
    # Check if we should use token-based chunks
    enable_token_based = getattr(settings_obj, 'enable_token_based_chunks', False)
    
    # Check if we need to create a new chunk
    should_create_chunk = False
    if enable_token_based:
        # Use token-based chunk detection
        token_limit = getattr(settings_obj, 'raw_tail_token_limit', 4000)
        current_chunk = chunk_manager.get_current_chunk_messages(history)
        should_create_chunk = chunk_manager.should_create_new_chunk_by_tokens(current_chunk, token_limit)
        if should_create_chunk:
            logger.info(f"[CHUNKED] Creating new chunk for namespace '{namespace or ''}' (token limit reached: {token_limit})")
    else:
        # Use turn-based chunk detection
        should_create_chunk = chunk_manager.should_create_new_chunk()
        if should_create_chunk:
            logger.info(f"[CHUNKED] Creating new chunk for namespace '{namespace or ''}' (turn limit reached)")
    
    if should_create_chunk:
        success = chunk_manager.create_new_chunk(history, settings_obj, cache, namespace)
        if not success:
            logger.warning(f"[CHUNKED] Failed to create new chunk for namespace '{namespace or ''}', falling back to current chunk")
    
    # Get history for prompt from chunk manager
    recent_conversation, summary_text = chunk_manager.get_history_for_prompt(history)
    to_summarize = []  # No separate summarization in chunked mode
    verbatim_tail = recent_conversation
    
    # Increment turn count for current chunk
    chunk_manager.increment_turn_count()
    
    # Handle recent conversation formatting with SimpleHistoryProcessor for byte-level consistency
    recent_block_str = ""
    if verbatim_tail:
        recent_block_str = history_processor.format_recent_conversation(
            verbatim_tail, params, log_origin
        )

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
    # Use <sources> marker for reliable stripping from history while maintaining clean user display
    sources_section = "\n<sources>Sources</sources>:\n" + _collapse_sources(indexed_for_collapse)
    if web_context:
        web_notes = "\n" + "\n".join([f"[web-{i+1}] {item.get('url', 'Web result')}" for i, item in enumerate(web_context)])
        sources_section += web_notes

    # Stage: Inference Pass 1: Inference Context Assembly + Tools Output (if enabled and needed)
    if show_processing_steps:
        emit_stage(req_id, "Inference Context Assembly")
    # Inference Prompt built from the Prompt Registry YAML file
    # Resolve prompt domain for this turn. Infer
    try:
        prompt_domain = str((params or {}).get("prompt_domain") or "").strip()
    except Exception:
        prompt_domain = ""
    if not prompt_domain:
        try:
            prompt_domain = str(getattr(settings_obj, "prompt_domain_default", "") or "").strip()
        except Exception:
            prompt_domain = ""

    # Require YAML registry to exist to prevent prompt drift across code paths.
    registry_path = str(getattr(settings_obj, "inference_prompt_registry_path", "") or "").strip()
    spec = resolve_inference_prompt(registry_path=registry_path, domain=prompt_domain)

    # Build the full inference prompt payload from YAML.
    web_text = ""
    try:
        if web_context:
            web_text = _format_web_context_as_text(web_context)
    except Exception:
        web_text = ""

    payload = render_full_payload(
        spec.full_payload_template,
        variables={
            "recent_block_str": (recent_block_str or "").strip(),
            "summary_text": summary_text or "",
            "context_text": context_text or "",
            "web_context": web_text or "",
            "message": message or "",
        },
    )

    prompt_input = None  # what we pass as `input` to Responses
    if style == "messages":
        prompt_input = [
            {"role": "system", "content": spec.system_instruction},
            {"role": "user", "content": payload},
        ]
    else:
        prompt_str = spec.system_instruction + "\n\n" + payload
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
   
   # Tools evaluation
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
            try:
                offered_tool_names = [
                    str(t.get("name") or t.get("function", {}).get("name") or "")
                    for t in (tools or [])
                ]
                logger.info(
                    "[TOOLS] (%s) tools_offered_to_inference count=%d names=%s",
                    log_origin,
                    len([n for n in offered_tool_names if n]),
                    [n for n in offered_tool_names if n],
                )
            except Exception:
                pass
        except Exception:
            _kwargs_inf["tools"] = []
            logger.info("[TOOLS] (%s) tools_offered_to_inference count=0 (list_tools failed)", log_origin)

    logger.info("[INFERENCE] %s: Attempting Responses with Inference model: %s", log_origin, _inf_model)
    
    # DEBUG: Log inference stage details with endpoint info
    try:
        from backend.llm.llm_client import get_model_info
        model_info = get_model_info(model_key=_inf_model)
        endpoint = getattr(model_info, 'endpoint', 'unknown')
        logger.info(f"[DEBUG] INFERENCE stage: provider={_inf_provider}, model={_inf_model}, endpoint={endpoint}")
    except Exception as e:
        logger.info(f"[DEBUG] INFERENCE stage: provider={_inf_provider}, model={_inf_model}, endpoint=unknown (error: {e})")
    
    resp_inf = None  # Initialize to fix variable scope issue
    
    try:
        resp_inf = _responses_create(
            provider=_inf_provider,
            model=_inf_model,
            **_kwargs_inf,
        )
        logger.debug(f"[INFERENCE] Response from _responses_create: type={type(resp_inf)}")
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
                        "costs": {"conversation_total": 0.0},
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
        if resp_inf is not None:
            _raw = getattr(resp_inf, "raw", resp_inf)
            logger.debug("[INFERENCE] (%s) raw response: %r", log_origin, _raw)
    except Exception:
        pass
    usage_inf = _extract_usage_from_responses(resp_inf, provider=_inf_provider) if resp_inf is not None else None
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

    answer_override: str | None = None
    tool_answer_text: str = ""
    used_tools: List[str] = []
    
    if enable_tools and isinstance(_kwargs_inf.get("input"), list):
        # NOTE: Single-pass tool execution.
        try:
            tool_registry = _load_tool_registry(settings_obj)
            tools_by_name = tool_registry.get("tools_by_name") if isinstance(tool_registry, dict) else {}
            if not isinstance(tools_by_name, dict):
                tools_by_name = {}

            # Extract tool calls from the first inference response
            tool_calls = extract_tool_calls(resp_inf)
            try:
                logger.debug(
                    "[TOOLS] (%s) extracted_tool_calls count=%d names=%s",
                    log_origin,
                    len(tool_calls or []),
                    [str((c or {}).get("name") or "") for c in (tool_calls or [])],
                )
            except Exception:
                pass
            
            if not tool_calls:
                logger.debug("[TOOLS] (%s) no_tool_calls_from_inference", log_origin)
                raise StopIteration  # handled by outer try/except; leaves answer_override=None

            # Only show "Tool Calls" stage when actually executing tools
            if show_processing_steps:
                emit_stage(req_id, "Tool Calls")

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

            for i, call in enumerate(tool_calls):
                name = call.get("name", "")
                call_id = call.get("id", "")
                args = call.get("args", {})

            for call in tool_calls:
                name = call.get("name") or ""
                call_id = call.get("id") or call.get("tool_call_id")
                args = parse_tool_args(call.get("args"))

                if show_processing_steps:
                    emit_stage(req_id, f"Calling Tool: {name}")
                executor = get_executor_fn(name)
                logger.debug(
                    "[TOOLS] (%s) tool_dispatch name=%s call_id=%s executor_found=%s",
                    log_origin,
                    name,
                    call_id,
                    bool(executor),
                )

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

                        tool_registry_entry = tools_by_name.get(name) if isinstance(tools_by_name, dict) else None
                        runtime_cfg = (
                            tool_registry_entry.get("runtime")
                            if isinstance(tool_registry_entry, dict) and isinstance(tool_registry_entry.get("runtime"), dict)
                            else {}
                        )
                        endpoint_cfg = runtime_cfg.get("endpoint") if isinstance(runtime_cfg.get("endpoint"), dict) else {}
                        logger.debug(
                            "[TOOLS] (%s) tool_execute_start name=%s args=%s endpoint_type=%s endpoint_url=%s",
                            log_origin,
                            name,
                            args,
                            str(endpoint_cfg.get("type") or ""),
                            str(endpoint_cfg.get("url") or ""),
                        )
                        result_text = executor(
                            args,
                            chat_context,
                            existing_context=exec_combined_context,
                            tool_runtime=runtime_cfg,
                            tool_registry_entry=tool_registry_entry,
                        )
                        logger.debug(
                            "[TOOLS] (%s) tool_execute_done name=%s output_type=%s output_len=%d",
                            log_origin,
                            name,
                            type(result_text).__name__,
                            len(str(result_text or "")),
                        )
                    except Exception as ex:
                        logger.debug("[TOOLS] (%s) tool_execute_error name=%s err=%s", log_origin, name, ex, exc_info=True)
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

                try:
                    if isinstance(result_text, (dict, list)):
                        normalized_tool_output = json.dumps(result_text, ensure_ascii=False)
                    else:
                        normalized_tool_output = str(result_text)
                except Exception:
                    normalized_tool_output = str(result_text)

                tool_outputs_list.append({"tool_call_id": call_id or "", "name": name, "output": normalized_tool_output})
                
                # Preserve first non-empty tool message for final fallback rendering
                try:
                    txt = (result_text.strip() if isinstance(result_text, str) else str(result_text).strip())
                    if txt and not tool_answer_text:
                        tool_answer_text = txt
                except Exception:
                    pass

            if not tool_outputs_list:
                logger.debug("[TOOLS] (%s) no_tool_outputs_after_execution", log_origin)
                raise StopIteration

            artifacts_from_registry = _extract_artifacts_from_tool_outputs(tool_outputs_list, tool_registry)
            redacted_tool_outputs_for_synth = _redact_tool_outputs_for_synth(tool_outputs_list, tool_registry)

            tools_text = "\n\n".join(
                [
                    f"[SOURCE: TOOL - {t.get('name') or 'unknown'}]\n{str(t.get('output', ''))}"
                    for t in redacted_tool_outputs_for_synth
                ]
            ).strip()
            if not tools_text:
                tools_text = "Tool(s) executed but returned no results."
            logger.debug(
                "[ARTIFACT] extracted_count=%d tools=%s",
                len(artifacts_from_registry),
                [a.get("tool") for a in artifacts_from_registry],
            )

            ts_prompt_domain = (prompt_domain or "").strip()
            ts_registry_path = str(getattr(settings_obj, "inference_prompt_registry_path", "") or "").strip()
            ts_prompt_spec = resolve_tools_synth_prompt(
                registry_path=ts_registry_path,
                domain=ts_prompt_domain,
            )
            tools_synth_system_prompt = (ts_prompt_spec.system_instruction or "").strip()

            synth_messages = _build_tools_synth_messages(
                system_prompt=tools_synth_system_prompt,
                summary_text=summary_text,
                recent_block_str=recent_block_str,
                context_text=context_text,
                tool_outputs_list=redacted_tool_outputs_for_synth,
                used_tools=used_tools,
                tools_text=tools_text,
                message=message,
            )
            synth_messages, stripped_svg_blocks = _strip_svg_from_messages(synth_messages)
            logger.debug("[ARTIFACT] tools_synth_svg_blocks_stripped=%d", stripped_svg_blocks)

            ts_spec = (stage_specs or {}).get("tools_synth") or {}
            _ts_provider = str(ts_spec.get("provider") or "openai")
            _ts_model = str(ts_spec.get("model"))

            # NOTE: tools_synth is the *final synthesis* step. Ensure stage_specs['tools_synth']['kwargs']
            # does NOT include tool-calling params (e.g., tools/tool_choice), otherwise the model may
            # attempt additional tool calls during synthesis.
            _kwargs_synth: Dict[str, Any] = dict(ts_spec.get("kwargs") or {})
            _kwargs_synth["input"] = synth_messages
            if "max_output_tokens" not in _kwargs_synth and max_out is not None:
                _kwargs_synth["max_output_tokens"] = int(max_out)
            if "temperature" not in _kwargs_synth:
                _kwargs_synth["temperature"] = float(temperature)

            try:
                if show_processing_steps:
                    emit_stage(req_id, "Generating Responses with Tools")
                
                # DEBUG: Log tools synthesis stage details
                logger.info(f"[DEBUG] TOOLS SYNTHESIS stage: provider={_ts_provider}, model={_ts_model}")
                
                resp_synth = _responses_create(
                    provider=_ts_provider,
                    model=_ts_model,
                    **_kwargs_synth,
                )
                combined = _extract_text_from_responses(resp_synth).strip()
                combined = _inject_registered_artifacts(combined, artifacts_from_registry)

                # Prefer canonical SVG from tool output when synthesis text is truncated.
                def _extract_svg_from_tool_outputs(_tool_outputs: List[Dict[str, Any]]) -> str:
                    for _t in (_tool_outputs or []):
                        _out = str(_t.get("output") or "")
                        try:
                            _parsed = json.loads(_out)
                            if isinstance(_parsed, dict):
                                _svg_val = _parsed.get("svg")
                                if isinstance(_svg_val, str) and _svg_val.strip():
                                    return _svg_val.strip()
                        except Exception:
                            pass

                        try:
                            _m = re.search(r"<svg\\b[\\s\\S]*?</svg>", _out, flags=re.IGNORECASE)
                            if _m and _m.group(0).strip():
                                return _m.group(0).strip()
                        except Exception:
                            pass
                    return ""

                _svg_from_tool = _extract_svg_from_tool_outputs(tool_outputs_list)
                _has_artifact_token = bool(re.search(r"\{+\s*ARTIFACT:[A-Za-z0-9_.:-]{1,64}\s*\}+", combined or ""))
                if _svg_from_tool and _has_artifact_token:
                    try:
                        combined, _n_repl = re.subn(
                            r"\{+\s*ARTIFACT:[A-Za-z0-9_.:-]{1,64}\s*\}+",
                            _svg_from_tool,
                            combined,
                        )
                        logger.info(
                            "[ARTIFACT] (%s) unresolved_artifact_token_replacements=%d",
                            log_origin,
                            int(_n_repl),
                        )
                    except Exception:
                        logger.debug("[ARTIFACT] (%s) unresolved_artifact_token_replace_failed", log_origin, exc_info=True)
                _chart_requested = bool(re.search(r"\b(chart|sparkline|trend|line\s*chart|bar\s*chart|time[-\s]?series|visual)\b", str(message or ""), flags=re.IGNORECASE))
                if _svg_from_tool and _chart_requested:
                    _has_svg = "<svg" in combined.lower()
                    _has_svg_close = "</svg>" in combined.lower()

                    # If synthesis missed/trimmed SVG, keep narrative prefix and inject full tool SVG.
                    if not _has_svg or not _has_svg_close:
                        _prefix = ""
                        if _has_svg:
                            try:
                                _prefix = combined.split("<svg", 1)[0].strip()
                            except Exception:
                                _prefix = ""
                        else:
                            _prefix = combined.strip()

                        if _prefix:
                            combined = f"{_prefix}\n\n{_svg_from_tool}"
                        else:
                            combined = _svg_from_tool

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
                            "costs": {"conversation_total": 0.0},
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
            # Remove sentinel "NO_SUPPORTED_SOURCES" from final answer text. This is only used to remove sources from the final answer text.
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

    # Filter Sources to ONLY those cited in the final answer.
    # Supported citation formats:
    # - Knowledge base chunks: [1], [2], ...
    # - Web results: [web-1], [web-2], ...
    try:
        _ans_for_citations = str(answer or "")
        cited_doc_idxs = set(int(x) for x in re.findall(r"\[(\d+)\]", _ans_for_citations))
        cited_web_idxs = set(int(x) for x in re.findall(r"\[web-(\d+)\]", _ans_for_citations, flags=re.I))

        if not cited_doc_idxs and not cited_web_idxs:
            sources = []
            sources_section = ""
        else:
            # KB sources: keep only cited indices from the prompt-provided context_items
            if cited_doc_idxs:
                filtered_indexed = [d for d in indexed_for_collapse if int(d.get("index") or 0) in cited_doc_idxs]
                # Use <sources> marker for reliable stripping from history while maintaining clean user display
                sources_section = "\n<sources>Sources</sources>:\n" + _collapse_sources(filtered_indexed) if filtered_indexed else ""
                try:
                    # Return only the cited reranked items (in original order)
                    sources = [it for i, it in enumerate(context_items, start=1) if i in cited_doc_idxs]
                except Exception:
                    sources = []
            else:
                sources_section = ""
                sources = []

            # Web sources: include only cited web indices
            if cited_web_idxs and web_context:
                web_notes = "\n" + "\n".join(
                    [
                        f"[web-{i}] {web_context[i-1].get('url', 'Web result')}"
                        for i in sorted(cited_web_idxs)
                        if 1 <= i <= len(web_context)
                    ]
                )
                # Use <sources> marker for reliable stripping from history while maintaining clean user display
                sources_section = (sources_section or "\n<sources>Sources</sources>:\n") + web_notes
                try:
                    sources.extend([web_context[i-1] for i in sorted(cited_web_idxs) if 1 <= i <= len(web_context)])
                except Exception:
                    pass
    except Exception:
        pass

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
            "costs": {"conversation_total": 0.0},
        }

    legacy_metrics = {"vectors_retrieved": n}
    
    # Strip <sources> tags before returning to user for clean display
    # History processing strips the entire sources block, but users see clean "Sources:" text
    final_answer = answer.rstrip("\n") + sources_section
    final_answer = re.sub(r"<sources>Sources</sources>:", "Sources:", final_answer)
    
    # Extract reasoning from the inference response if available
    reasoning = None
    try:
        if resp_inf is not None:
            reasoning = _extract_reasoning_from_responses(resp_inf)
    except Exception:
        reasoning = None
    
    return {
        "answer": final_answer,
        "sources": sources,
        "turn_metrics": turn_metrics,
        "conversation_totals": convo_snapshot,
        "metrics": legacy_metrics,
        "tools_used": tools_out,
        "rewrite_display": rewrite_display,
        "reasoning": reasoning,
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

    # Optional server-side Markdown -> sanitized HTML rendering (feature-flagged)
    try:
        _render_html = bool((params or {}).get("render_html", False))
    except Exception:
        _render_html = False

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

    # Resolve effective retrieval domain for this request.
    # Priority: params.active_domain -> params.prompt_domain -> settings.active_domain
    available_domains = getattr(settings, "DOMAIN_EMBEDDING_CONFIG", {}) or {}
    configured_default_domain = str(getattr(settings, "active_domain", "") or "").strip() or "default"
    requested_domain = str(
        (params or {}).get("active_domain")
        or (params or {}).get("prompt_domain")
        or configured_default_domain
    ).strip()
    effective_domain = requested_domain if requested_domain in available_domains else configured_default_domain
    domain_cfg = available_domains.get(effective_domain) or available_domains.get(configured_default_domain) or {}
    domain_collection = str(domain_cfg.get("collection_name") or settings.collection_name)
    domain_embedding_model_key = str(domain_cfg.get("embedding_model_key") or settings.embedding_model_key)

    # Fresh Qdrant client for stateless path using per-request domain routing
    db = QdrantDB(
        host=settings.qdrant_host,
        port=settings.qdrant_port,
        collection_name=domain_collection,
        embedding_model_key=domain_embedding_model_key,
    )

    try:
        logger.info(
            "[REQ %s] domain routing requested=%s effective=%s collection=%s embedding_model_key=%s",
            req_id,
            requested_domain,
            effective_domain,
            domain_collection,
            domain_embedding_model_key,
        )
    except Exception:
        pass

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

    # Determine web search toggle (request overrides settings)
    try:
        _req_web = (payload or {}).get("use_web_search")
    except Exception:
        _req_web = None
    if _req_web is None:
        use_web_search = bool((params or {}).get("use_web_search", getattr(settings, "use_web_search", False)))
    else:
        use_web_search = bool(_req_web)

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
        "get_web_context": (lambda q, existing: WebSearchClient().get_additional_context(q, existing)) if use_web_search else (lambda q, existing: []),
        "style": "messages", # flat or messages (use messages for clear systemvs user roles separation)
        "enable_tools": bool(enable_tools),
        "enable_query_rewrite": bool(rewrite_enabled),
        "use_web_search": bool(use_web_search),
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

        _answer_text = out.get("answer", "")
        _answer_html = ""
        logger.debug("RENDER HTML: %s", _render_html)
        if _render_html:
            try:
                _answer_html = render_markdown_to_html(_answer_text)
                logger.debug("RENDER HTML: %s", _answer_html)
                logger.debug("RENDER SUCCESS: %s", "SUCCESS")
            except Exception as e:
                logger.debug("Exception in render_markdown_to_html: %s", str(e))
                logger.debug("RENDER FAILED: %s", "FAILED")
                _answer_html = ""

        # Ensure the final message is properly formatted for the frontend
        if _render_html and _answer_html:
            emit_stage(req_id, "Final Answer", final=True, finalContent=_answer_text, finalHtml=_answer_html)
        else:
            emit_stage(req_id, "Final Answer", final=True, finalContent=_answer_text)
        # Send an explicit close message
        emit_stage(req_id, "Done", final=True)

        # Base response: preserve existing shape/keys for compatibility.
        resp: Dict[str, Any] = {
            "answer": _answer_text,
            "response": _answer_text,  # legacy compatibility for frontend expecting 'response'
            "metrics": out.get("metrics", {"vectors_retrieved": 0}),
            "turn_metrics": out.get("turn_metrics", {}),
            "conversation_totals": out.get("conversation_totals", {}),
            "tools_used": out.get("tools_used", []),
            "rewrite_display": out.get("rewrite_display", {}),
        }

        if _render_html and _answer_html:
            resp["answer_html"] = _answer_html

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
