"""Chat Manager Module

Entry Point:
- handle_chat(payload: Dict) -> Dict: Main entry point for chat requests

Pipeline Initialization:
1. Initialize dependencies (DB, clients, tools)
2. Parse request payload
3. Set up metrics and logging

Chat Pipeline Stages (with History Integration):
1. Query Rewrite (optional)
   - Uses history to resolve ambiguous references (pronouns, etc.)
   - Helps disambiguate queries based on conversation context

2. Context Retrieval
   - Qdrant vector database search
   - Score and filter results
   - Current query combined with history for better context

3. Reranking
   - Reranks results using both query and conversation history
   - Improves relevance based on the full conversation context

4. Web Search (optional)
   - Augments context with web results when needed
   - Uses history to maintain search context

5. Summarization
   - Processes conversation history:
     - Keeps recent messages verbatim
     - Condenses older messages to save tokens
   - Uses `split_history_for_prompt()` to manage history window

6. Prompt Construction
   - Combines:
     - System instructions
     - Summarized history
     - Verbatim recent messages
     - Retrieved context
     - Current query

7. LLM Inference
   - Processes the full prompt with history
   - Generates response using conversation context

8. Tool Execution (if needed)
   - May use history for tool parameter resolution
   - Maintains tool state across turns

9. Response Generation
   - Formats final response
   - Updates conversation history with new exchange

Conversation State:
- Full history maintained in memory
- Each turn appends both user message and assistant response
- Configurable history window size controls context length
- Automatic summarization of older messages
"""

from typing import List, Dict, Any, Set
import logging

logger = logging.getLogger(__name__)
import json
import re
import uuid
import tiktoken
import time
from openai import OpenAI
from collections import defaultdict
# NOTE: SSE stage emission is centralized in backend/stream_emit.py so chat_manager stays agnostic of registry details.
# Stream emission helpers (centralized in backend/stream_emit.py)
from backend.stream_emit import emit_stage, close_stream
from backend.core.config import settings
from backend.db import QdrantDB
from backend.chat.web_search import WebSearchClient
from backend.tools import list_tools, get_executor

# Lazy OpenAI client (initialized on first use)
_client = None

# Module-level cache for summaries
_SUMMARY_CACHE: Dict[str, str] = {}
# Option A support: index of namespace -> set of cache keys for precise clearing
_SUMMARY_NS_INDEX: Dict[str, Set[str]] = defaultdict(set)
# Option A support: last-seen timestamp per namespace for idle eviction
_SUMMARY_NS_LAST_SEEN: Dict[str, float] = {}


def get_client():
    global _client
    if _client is None:
        logger.debug("Initializing OpenAI client")
        try:
            _client = OpenAI(api_key=settings.openai_api_key)
        except Exception as e:
            logger.error("Failed to create OpenAI client: %s", e)
            raise
    return _client

# ---- Conversation totals accumulator (module-level)  ----
COST_BASIS = float(getattr(settings, "cost_basis_tokens", 1_000_000))

CONVO_TOTALS = {
    "tokens": {
        "embedding": 0,
        "llm_input": 0,      # prompt + cached tokens across stages
        "llm_output": 0,     # completion tokens across stages
        "conversation_total": 0
    },
    "costs": {
        "embedding": 0.0,
        "llm_input": 0.0,
        "llm_output": 0.0,
        "total": 0.0,
        "conversation_total": 0.0
    },
}

def _zero_convo_totals():
    CONVO_TOTALS["tokens"].update({
        "embedding": 0,
        "llm_input": 0,
        "llm_output": 0,
        "conversation_total": 0,
    })
    CONVO_TOTALS["costs"]["conversation_total"] = 0.0
    # ---- end accumulator ----

def _extract_text_from_responses(resp) -> str:
    """Return response text from Responses API object.

    Prefers `resp.output_text`. If absent, concatenates any `.text` parts
    from `resp.output[...].content[...]` entries. Falls back to empty string.
    """
    #logger.debug(f"Full response object: {resp}")
    # Prefer direct output_text if available
    text = getattr(resp, "output_text", None)
    if isinstance(text, str) and text:
        return text

    # Try to read from output -> content -> text
    output = getattr(resp, "output", None)
    if output is None and isinstance(resp, dict):
        output = resp.get("output")

    parts: List[str] = []
    if output and isinstance(output, list):
        for item in output:
            content = getattr(item, "content", None)
            if content is None and isinstance(item, dict):
                content = item.get("content")
            if not content or not isinstance(content, list):
                continue
            for c in content:
                txt = getattr(c, "text", None)
                if txt is None and isinstance(c, dict):
                    txt = c.get("text")
                if isinstance(txt, str) and txt:
                    parts.append(txt)

    return "".join(parts) if parts else ""



def _extract_usage_from_responses(resp) -> Dict[str, int] | None:
    """Extract usage fields. Supports both old (prompt/completion) and new (input/output) names.
       Returns: {prompt_tokens, completion_tokens, total_tokens, cached_tokens?}
    """
    usage = getattr(resp, "usage", None)
    if usage is None and isinstance(resp, dict):
        usage = resp.get("usage")
    if usage is None:
        return None

    def _get(u, name):
        return getattr(u, name, None) if not isinstance(u, dict) else u.get(name)

    # Accept both naming schemes
    p = _get(usage, "prompt_tokens")
    if p is None:
        p = _get(usage, "input_tokens")

    c = _get(usage, "completion_tokens")
    if c is None:
        c = _get(usage, "output_tokens")

    t = _get(usage, "total_tokens")

    # cached tokens can be in either prompt_tokens_details or input_tokens_details
    details = _get(usage, "prompt_tokens_details") or _get(usage, "input_tokens_details")
    cached = None
    if details is not None:
        cached = getattr(details, "cached_tokens", None) if not isinstance(details, dict) else details.get("cached_tokens")

    if p is None and c is None and t is None and cached is None:
        return None

    out: Dict[str, int] = {}
    if p is not None: out["prompt_tokens"] = int(p)
    if c is not None: out["completion_tokens"] = int(c)
    if t is not None: out["total_tokens"] = int(t)
    if cached is not None: out["cached_tokens"] = int(cached)
    return out


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

        sum_prompt = _build_summary_prompt_with_budget(cleaned_messages, max_input_tokens, model)
        logger.debug(f"{log_prefix} applied local input budget; prompt_len_chars=%d", len(sum_prompt))

        kwargs: Dict[str, Any] = {
            "model": model,
            "input": sum_prompt,
            "temperature": float(temperature),
        }
        if max_output_tokens is not None:
            kwargs["max_output_tokens"] = int(max_output_tokens)

        resp = get_client().responses.create(**kwargs)
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
        usage = _extract_usage_from_responses(resp)
        return summary_text, False, usage
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
def _compute_stage_cost(stage: str, prompt_tokens: int = 0, completion_tokens: int = 0, cached_tokens: int = 0) -> Dict[str, float]:
    """Return cost breakdown for a stage using per-million rates and COST_BASIS."""
    if stage == "inference":
        in_rate = float(settings.inference_cost_per_MM_tokens_input)
        out_rate = float(settings.inference_cost_per_MM_tokens_output)
        cached_rate = float(getattr(settings, "inference_cost_per_MM_tokens_cached_input", in_rate / 2.0))
    elif stage == "rerank":
        in_rate = float(settings.re_ranker_cost_per_MM_tokens_input)
        out_rate = float(settings.re_ranker_cost_per_MM_tokens_output)
        cached_rate = float(getattr(settings, "re_ranker_cost_per_MM_tokens_cached_input", in_rate / 2.0))
    elif stage == "summary":
        in_rate = float(settings.summarizer_cost_per_MM_tokens_input)
        out_rate = float(settings.summarizer_cost_per_MM_tokens_output)
        cached_rate = float(getattr(settings, "summarizer_cost_per_MM_tokens_cached_input", in_rate / 2.0))
    elif stage == "rewrite":
        # Use dedicated rewrite pricing when available; fallback to summarizer rates
        in_rate = float(getattr(settings, "rewrite_cost_per_MM_tokens_input", getattr(settings, "summarizer_cost_per_MM_tokens_input", 0.0)))
        out_rate = float(getattr(settings, "rewrite_cost_per_MM_tokens_output", getattr(settings, "summarizer_cost_per_MM_tokens_output", 0.0)))
        cached_rate = float(getattr(settings, "rewrite_cost_per_MM_tokens_cached_input", getattr(settings, "summarizer_cost_per_MM_tokens_cached_input", in_rate / 2.0)))
    elif stage == "embedding":
        # embeddings are input-only
        in_rate = float(settings.embedding_cost_per_MM_tokens)
        out_rate = 0.0
        cached_rate = 0.0
    else:
        in_rate = out_rate = cached_rate = 0.0

    cost_prompt = (prompt_tokens / COST_BASIS) * in_rate
    cost_cached = (cached_tokens / COST_BASIS) * cached_rate
    cost_completion = (completion_tokens / COST_BASIS) * out_rate
    total = cost_prompt + cost_cached + cost_completion
    return {
        "cost_prompt": round(cost_prompt, 8),
        "cost_cached": round(cost_cached, 8),
        "cost_completion": round(cost_completion, 8),
        "cost_total": round(total, 8),
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
        # Exact shape expected by the UI
        self.turn: Dict[str, Any] = {
            "embedding": {"model": getattr(settings_obj, "embedding_model", "embedding"), "input_tokens": 0, "cost": 0.0},
            "rerank": {"model": settings_obj.re_ranker_model, "input_tokens": 0, "output_tokens": 0, "candidates_reranked": 0, "cost": 0.0},
            "summary": {"model": settings_obj.summarizer_model, "applied": False, "reason": "", "input_tokens": 0, "output_tokens": 0, "cost": 0.0},
            "rewrite": {"model": getattr(settings_obj, "rewrite_model", settings_obj.inference_model), "applied": False, "reason": "", "input_tokens": 0, "output_tokens": 0, "cost": 0.0},
            "inference": {"model": settings_obj.inference_model, "prompt_tokens": 0, "prompt_cached_tokens": 0, "completion_tokens": 0, "cost_prompt": 0.0, "cost_cached": 0.0, "cost_completion": 0.0, "cost_total": 0.0},
            "totals": {"tokens": {"turn_total": 0}, "cost": {"turn_total": 0.0}},
        }
        # Module-level accumulator reference (shared per process)
        self.convo: Dict[str, Any] = convo_totals_ref

    # --- Helpers ---
    def _normalize_usage(self, resp_or_usage: Any) -> Dict[str, int]:
        """Return dict with prompt_tokens, completion_tokens, cached_tokens, total_tokens (zeros if missing)."""
        try:
            # Accept either a full response object, a dict with nested usage, or a plain usage dict
            if hasattr(resp_or_usage, "usage"):
                # Full Responses API object
                u = _extract_usage_from_responses(resp_or_usage)
            elif isinstance(resp_or_usage, dict) and ("usage" in resp_or_usage):
                # Dict wrapping usage -> extract
                u = _extract_usage_from_responses(resp_or_usage)
            elif isinstance(resp_or_usage, dict) and (
                "input_tokens" in resp_or_usage
                or "output_tokens" in resp_or_usage
                or "prompt_tokens" in resp_or_usage
                or "completion_tokens" in resp_or_usage
            ):
                # Already a normalized/usage-like dict
                u = resp_or_usage
            else:
                u = None
        except Exception:
            u = None
        u = u or {}
        return {
            "prompt_tokens": int(u.get("prompt_tokens", 0) or 0),
            "completion_tokens": int(u.get("completion_tokens", 0) or 0),
            "cached_tokens": int(u.get("cached_tokens", 0) or 0),
            "total_tokens": int(u.get("total_tokens", 0) or 0),
        }

    def _cost(self, stage: str, pt: int, ct: int, cached: int) -> Dict[str, float]:
        # Delegate to existing utility for a single source of truth
        return _compute_stage_cost(stage, prompt_tokens=pt, completion_tokens=ct, cached_tokens=cached)

    # --- Public API ---
    def record_stage(self, stage: str, *, model: str, usage: Any | None = None,
                     pt: int | None = None, ct: int | None = None, cached: int | None = None,
                     extra: Dict[str, Any] | None = None) -> None:
        """Record metrics for a pipeline stage.
        Either pass a `usage` (response or usage dict) or explicit pt/ct/cached counts.
        `extra` lets callers set fields like candidates_reranked/applied/reason.
        """
        if stage not in self.turn:
            return
        # Always stamp the model that ran
        self.turn[stage]["model"] = model

        if usage is not None and pt is None and ct is None and cached is None:
            u = self._normalize_usage(usage)
            pt, ct, cached = u["prompt_tokens"], u["completion_tokens"], u["cached_tokens"]
        pt = int(pt or 0)
        ct = int(ct or 0)
        cached = int(cached or 0)

        if stage == "embedding":
            # input-only; we treat provided pt as input_tokens
            self.turn[stage]["input_tokens"] = pt
            c = self._cost("embedding", pt, 0, 0)
            self.turn[stage]["cost"] = c["cost_prompt"]
        elif stage == "rerank":
            self.turn[stage]["input_tokens"] = pt + cached
            self.turn[stage]["output_tokens"] = ct
            c = self._cost("rerank", pt, ct, cached)
            self.turn[stage]["cost"] = c["cost_total"]
        elif stage == "summary":
            self.turn[stage]["input_tokens"] = pt + cached
            self.turn[stage]["output_tokens"] = ct
            c = self._cost("summary", pt, ct, cached)
            self.turn[stage]["cost"] = c["cost_total"]
        elif stage == "rewrite":
            self.turn[stage]["input_tokens"] = pt + cached
            self.turn[stage]["output_tokens"] = ct
            c = self._cost("rewrite", pt, ct, cached)
            self.turn[stage]["cost"] = c["cost_total"]
        elif stage == "inference":
            # Accumulate tokens and costs across multiple inference calls in a single turn.
            prev_pt = int(self.turn[stage].get("prompt_tokens") or 0)
            prev_ck = int(self.turn[stage].get("prompt_cached_tokens") or 0)
            prev_ct = int(self.turn[stage].get("completion_tokens") or 0)

            pt_total = prev_pt + pt
            ck_total = prev_ck + cached
            ct_total = prev_ct + ct

            self.turn[stage]["prompt_tokens"] = pt_total
            self.turn[stage]["prompt_cached_tokens"] = ck_total
            self.turn[stage]["completion_tokens"] = ct_total

            # Cost for this specific call
            c = self._cost("inference", pt, ct, cached)
            self.turn[stage]["cost_prompt"] = float(self.turn[stage].get("cost_prompt", 0.0)) + c["cost_prompt"]
            self.turn[stage]["cost_cached"] = float(self.turn[stage].get("cost_cached", 0.0)) + c["cost_cached"]
            self.turn[stage]["cost_completion"] = float(self.turn[stage].get("cost_completion", 0.0)) + c["cost_completion"]
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
        ip = int(self.turn["inference"].get("prompt_tokens") or 0)
        ik = int(self.turn["inference"].get("prompt_cached_tokens") or 0)
        ic = int(self.turn["inference"].get("completion_tokens") or 0)

        # NOTE: cached tokens are a subset of prompt/input tokens; do NOT add them again to totals.
        total_tokens = emb + rin + rout + sin + sout + rwin + rwout + ip + ic
        self.turn["totals"]["tokens"]["turn_total"] = total_tokens

        total_cost = (
            float(self.turn["embedding"].get("cost") or 0.0)
            + float(self.turn["rerank"].get("cost") or 0.0)
            + float(self.turn["summary"].get("cost") or 0.0)
            + float(self.turn["rewrite"].get("cost") or 0.0)
            + float(self.turn["inference"].get("cost_total") or 0.0)
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


def build_rewrite_prompt(tail_messages: List[Dict[str, str]] | None, summary_text: str, message: str) -> str:
    """Build a compact prompt for the rewrite model using only a tiny context pack.
    We include: (optional) summary of earlier turns and the verbatim recent tail, plus the user's latest message.
    The model is instructed to return JSON only and to keep the original if ambiguous.
    """
    parts: List[str] = []
    parts.append("Rewrite the user's latest question into a self-contained question so it is self-contained using only the recent conversation. \n")
    parts.append("If you cannot resolve the reference to the conversation, return the original question unchanged.\n")
    parts.append("Return strictly the following JSON (no extra text, no explanations):\n")
    parts.append("Keep reason to a short phrase (max 15-20 tokens). \n")
    parts.append('{"rewritten":"...","changed":true|false,"confidence":0.0,"ambiguous":true|false,"reason":"..."}\n\n')
    if summary_text:
        parts.append("Previous conversation summary:\n" + summary_text.strip() + "\n\n")
    if tail_messages:
        parts.append("Recent conversation:\n")
        for m in tail_messages:
            role = m.get("role", "user")
            content = m.get("content", "")
            parts.append(f"{role}: {content}\n")
        parts.append("\n")
    parts.append(f"User question: {message.strip()}\n")
    return "".join(parts)


def rewrite_query(
    tail_messages: List[Dict[str, str]] | None,
    summary_text: str,
    message: str,
    log_prefix: str = "[REWRITE]",
) -> Dict[str, Any]:
    """Call the rewrite model to produce a self-contained query.
    Returns a dict with keys: rewritten, changed, confidence, ambiguous, reason.
    On any failure, returns the original unmodified with changed=False.
    """
    try:
        prompt = build_rewrite_prompt(tail_messages, summary_text, message)
        # Log an estimated prompt token count for rewrite
        try:
            enc = _get_encoder_for_model(settings.rewrite_model)
            pt_est = len(enc.encode(prompt))
            logger.debug(f"{log_prefix} prompt_token_est≈%d model=%s", pt_est, settings.rewrite_model)
        except Exception:
            pass
        # Invoke the rewrite model with the prompt for the user's latest message for it to rewrite it
        resp = get_client().responses.create(
            model=settings.rewrite_model,
            input=prompt,
            max_output_tokens=int(settings.rewrite_max_output_tokens),
            temperature=float(settings.rewrite_temperature),
        )
        usage = _extract_usage_from_responses(resp)
        raw = _extract_text_from_responses(resp).strip()
        logger.debug("[REWRITE] response output raw=%s", raw[:400])
        try:
            if isinstance(usage, dict):
                pt = int(usage.get("prompt_tokens") or 0)
                ct = int(usage.get("completion_tokens") or 0)
                ck = int(usage.get("cached_tokens") or 0)
                tt = int(usage.get("total_tokens") or (pt + ct + ck))
                logger.debug(f"{log_prefix} usage pt=%d cached=%d ct=%d total=%d", pt, ck, ct, tt)
        except Exception:
            pass
        # Tolerate fenced code blocks
        if raw.startswith("```json") and raw.endswith("```"):
            raw = raw[7:-3].strip()
        elif raw.startswith("```") and raw.endswith("```"):
            raw = raw[3:-3].strip()
        data = json.loads(raw)
        # Debug: log the parsed JSON (truncated)
        try:
            _t = int(getattr(settings, "debug_log_truncate_chars", 4000))
        except Exception:
            _t = 400
        try:
            _js = json.dumps(data, ensure_ascii=False)
            logger.debug(f"{log_prefix} json=%s", _js if len(_js) <= _t else (_js[:_t] + "…"))
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
                "get_client": get_client,
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
            return {
                "response": f"I'm sorry, I encountered an error while processing your request: {str(e)}",
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
    calls: List[Dict[str, Any]] = []
    try:
        output = getattr(resp, "output", None) if not isinstance(resp, dict) else resp.get("output")
        if isinstance(output, list):
            for item in output:
                it_type = getattr(item, "type", None) if not isinstance(item, dict) else item.get("type")
                if it_type in ("function_call", "tool_use", "tool_call"):
                    name = getattr(item, "name", None) if not isinstance(item, dict) else item.get("name")
                    args = getattr(item, "arguments", None) if not isinstance(item, dict) else item.get("arguments")
                    cid = getattr(item, "call_id", None) if not isinstance(item, dict) else (item.get("call_id") or item.get("id"))
                    calls.append({"name": name, "args": args, "id": cid})
                else:
                    content = getattr(item, "content", None) if not isinstance(item, dict) else item.get("content")
                    if isinstance(content, list):
                        for c in content:
                            c_type = getattr(c, "type", None) if not isinstance(c, dict) else c.get("type")
                            if c_type in ("tool_use", "tool_call"):
                                name = getattr(c, "name", None) if not isinstance(c, dict) else c.get("name")
                                cid = getattr(c, "id", None) if not isinstance(c, dict) else c.get("id")
                                a = None
                                if isinstance(c, dict):
                                    if "input" in c:
                                        a = c.get("input")
                                    elif "arguments" in c:
                                        a = c.get("arguments")
                                else:
                                    a = getattr(c, "input", None) or getattr(c, "arguments", None)
                                calls.append({"name": name, "args": a, "id": cid})
    except Exception:
        pass
    try:
        choices = getattr(resp, "choices", None) if not isinstance(resp, dict) else resp.get("choices")
        if isinstance(choices, list) and choices:
            msg = getattr(choices[0], "message", None) if not isinstance(choices[0], dict) else choices[0].get("message")
            tc = getattr(msg, "tool_calls", None) if not isinstance(msg, dict) else msg.get("tool_calls")
            if isinstance(tc, list):
                for t in tc:
                    ttype = getattr(t, "type", None) if not isinstance(t, dict) else t.get("type")
                    if ttype == "function":
                        func = getattr(t, "function", None) if not isinstance(t, dict) else t.get("function")
                        name = getattr(func, "name", None) if not isinstance(func, dict) else (func or {}).get("name")
                        arguments = getattr(func, "arguments", None) if not isinstance(func, dict) else (func or {}).get("arguments")
                        tc_id = getattr(t, "id", None) if not isinstance(t, dict) else t.get("id")
                        calls.append({"name": name, "args": arguments, "id": tc_id})
    except Exception:
        pass
    return [c for c in calls if c.get("name")]


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
# --- end tail cleanup helper ---

# --- Unified pipeline orchestrator (Option A) ---

def run_pipeline(*, deps: Dict[str, Any], req: Dict[str, Any]) -> Dict[str, Any]:
    """
    Unified pipeline:
    retrieve -> maybe_rerank -> summarize -> build prompt -> inference -> optional tools -> sources -> metrics

    deps:
      - db: QdrantDB-like (must support search_similar)
      - cache: dict-like for summaries
      - settings: Settings object
      - get_client: callable() -> OpenAI client
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
    get_client_fn = deps.get("get_client", get_client)
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

    # UI-friendly summary of rewrite decision (always returned)
    rewrite_display: Dict[str, Any] = {
        "enabled": bool(enable_query_rewrite),
        "triggered": False,
        "accepted": False,
        "original": message,
    }

    # Metrics helper
    m = Metrics(settings_obj, CONVO_TOTALS)

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
                    summary_rw, _from_cache_rw, _u_rw = _summarize_messages_with_cache(
                        to_sum_rw,
                        cache,
                        tag=_tag_rw,
                        model=getattr(settings_obj, "summarizer_model", settings_obj.inference_model),
                        temperature=float(getattr(settings_obj, "summarizer_temperature", 0.3)),
                        max_input_tokens=int(getattr(settings_obj, "summarizer_max_input_tokens", 512)),
                        max_output_tokens=int(getattr(settings_obj, "summarizer_max_output_tokens", 128)),
                        log_prefix=f"[REWRITE] {log_origin}"
                    )
                if tail_rw or summary_rw:
                    rw = rewrite_query(tail_rw, summary_rw, message, log_prefix=f"[REWRITE] {log_origin}")
                    threshold = float(thr)
                    usage_rw = rw.get("_usage") if isinstance(rw, dict) else None
                    accepted = bool(rw.get("changed")) and (not rw.get("ambiguous")) and (float(rw.get("confidence", 0.0) or 0.0) >= threshold)
                    if usage_rw:
                        m.record_stage("rewrite", model=getattr(settings_obj, "rewrite_model", settings_obj.inference_model), usage=usage_rw, extra={"applied": True, "reason": ("accepted" if accepted else "rejected")})
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

    # Stage: Retrieve Vectors
    try:
        logger.info("[PIPELINE] emit stage: Retrieve Vectors")
        emit_stage(req_id, "Retrieve Vectors")
    except Exception:
        pass
    # --- Retrieve
    results = db.search_similar(
        query=effective_query,
        limit=int(top_k),
        score_threshold=float(score_threshold),
        with_vectors=False,
        with_payload=True,
        exact=True,
    )
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
    m.record_stage("embedding", model=getattr(settings_obj, "embedding_model", "embedding"), pt=embed_tokens)

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
            emit_stage(req_id, "Rerank Retrieval Results")
        except Exception:
            pass

    if not need_rerank:
        _dbg(f"[RERANK] {log_origin}", f"skipping rerank: {skip_reason}")
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
            prompt_text = _make_rerank_prompt(effective_query, cand_text, int(getattr(settings_obj, "reranker_chunk_size", 600)))
            _dbg(f"[RERANK] {log_origin} prompt:", prompt_text)
            resp_rerank = get_client_fn().responses.create(
                model=getattr(settings_obj, "re_ranker_model", settings_obj.inference_model),
                input=prompt_text.strip(),
                max_output_tokens=int(getattr(settings_obj, "re_ranker_max_output_tokens", 128)),
                temperature=float(getattr(settings_obj, "re_ranker_temperature", 0.0)),
            )
            content = _extract_text_from_responses(resp_rerank).strip()
            _dbg(f"[RERANK] {log_origin} raw:", content)
            order = _parse_json_array_in_text(content, pool_n)
            reranked = [pool[i] for i in order] or pool
            reranked = reranked[:kept]

            usage_rr = _extract_usage_from_responses(resp_rerank) or {}
            m.record_stage("rerank", model=getattr(settings_obj, "re_ranker_model", "rerank"), usage=usage_rr, extra={"candidates_reranked": n})
        except Exception as e:
            logger.error("[RERANK] (%s) failed; falling back: %s", log_origin, e, exc_info=True)
            reranked = results[:kept]

# Stage: History Summary
    try:
        logger.info("[PIPELINE] emit stage: Summarize Chat History")
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
            summary_text, _from_cache_inf, _u_inf = _summarize_messages_with_cache(
                to_summarize,
                cache,
                tag=_tag_inf,
                model=getattr(settings_obj, "summarizer_model", settings_obj.inference_model),
                temperature=float(getattr(settings_obj, "summarizer_temperature", 0.3)),
                max_input_tokens=sum_in,
                max_output_tokens=sum_out,
                log_prefix=f"[SUMMARY] {log_origin}"
            )
            if not _from_cache_inf and _u_inf:
                m.record_stage("summary", model=getattr(settings_obj, "summarizer_model", "summary"), usage=_u_inf, extra={"applied": True, "reason": f"prev {window_turns} turns (before last {raw_tail} turns)"})
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
    emit_stage(req_id, "Inference Prompt Build")
    strict_rag_prompt = (
        "You are a question-answering assistant for a retrieval-augmented system.\n"
        "STRICT RULES:\n"
        "1. Base your answer ONLY on information in the Context section (and Web search results if present).\n"
        "2. Do NOT use any outside knowledge, general world knowledge, training data, or assumptions beyond that context.\n"
        "3. If the context does not contain enough information to answer the question, reply exactly with: I couldn't find any information to answer this question. NO_SUPPORTED_SOURCES\n"
        "4. If any context chunk has a citation like [1], [2], etc., retain it in your response.\n"
        "5. Do not fabricate sources or facts.\n"
        "6. If a source URL is available (shown in the final 'Sources' section), you may reference it by its tag like [1]."
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
        messages.append({"role": "user", "content": message})

        # Convert to a single prompt string unless tools are enabled
        try:
            from backend.utils.prompt_utils import convert_messages_to_prompt
            prompt_str = convert_messages_to_prompt(messages)
        except Exception:
            # Fall back to naive join if util unavailable
            prompt_str = "\n".join([f"{m['role']}: {m['content']}" for m in messages])
        _dbg(f"[FULL PROMPT] {log_origin}", prompt_str)
        prompt_input = prompt_str if not enable_tools else messages
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
        
        _dbg(f"[FULL INFERENCE PROMPT] {log_origin}", prompt_str)
        prompt_input = [{"role": "user", "content": prompt_str}] if enable_tools else prompt_str

    # --- Inference decode params
    temperature = _pick(params, ["temperature", "inference_temperature", "INFERENCE_TEMPERATURE"], getattr(settings_obj, "inference_temperature", 0.7))
    max_out = _pick(params, ["max_output_tokens", "max_inference_output_tokens", "MAX_INFERENCE_OUTPUT_TOKENS"], getattr(settings_obj, "max_inference_output_tokens", 300))
    top_p = _pick(params, ["top_p", "inference_top_p", "INFERENCE_TOP_P"], getattr(settings_obj, "inference_top_p", None))

    # Stage: Inference API call
    # --- Inference API call - Orchestrater stage with Tool Calls
    logger.info("[PIPELINE] emit stage: Generating Response")
    emit_stage(req_id, "Generating Response")
    _kwargs_inf: Dict[str, Any] = {
        "model": getattr(settings_obj, "inference_model", "gpt-4o"),
        "input": prompt_input,
        "temperature": float(temperature),
        "max_output_tokens": int(max_out),
    }
    if top_p is not None:
        _kwargs_inf["top_p"] = float(top_p)
    if getattr(settings_obj, "inference_reasoning_model", False):
        _kwargs_inf["reasoning"] = {"effort": getattr(settings_obj, "inference_reasoning_effort", "low")}

    if enable_tools and isinstance(prompt_input, list):
        try:
            tools = list_tools_fn()
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

    logger.info("[INFERENCE] %s: Attempting Responses API with Inference model: %s", log_origin, _kwargs_inf["model"])
    #logger.debug("[%s] Call to Inference API with Prompt: %s", log_origin, _kwargs_inf["input"])
    resp_inf = get_client_fn().responses.create(**_kwargs_inf)
    _dbg(f"[INFERENCE] Inference 1 response {log_origin}", str(resp_inf))
    usage_inf = _extract_usage_from_responses(resp_inf)
    # Record Inference Usage - 1st Inference (will determine if we need to call tool calls)
    if usage_inf:
        m.record_stage("inference", model=_kwargs_inf["model"], usage=usage_inf)

    # Stage: Tool Calls
    # --- Tool Calls - Single pass thru all tools required
    # Optional tool loop (bounded for safety)
    
    answer_override: str | None = None
    used_tools: List[str] = []
    if enable_tools and isinstance(_kwargs_inf.get("input"), list):
        # Tool Calls - Single pass thru all tools required
        emit_stage(req_id, "Tool Calls")
        try:
            max_loops = getattr(settings_obj, "max_tool_passes", 2)
            loops = 0
            while loops < max_loops:
                # Tool Calls - Extract tool calls from inference response
                tool_calls = extract_tool_calls(resp_inf)
                if not tool_calls:
                    break
                tool_outputs_list = []
                for call in tool_calls:
                    name = call.get("name") or ""
                    call_id = call.get("id") or call.get("tool_call_id")
                    args = parse_tool_args(call.get("args"))
                    logger.debug("[PIPELINE] emit stage: Tool Calls %s", name)
                    emit_stage(req_id, f"Calling Tool: {name}")
                    executor = get_executor_fn(name)
                    if not executor:
                        result_text = f"Tool '{name}' is not available."
                    else:
                        try:
                            chat_context = list(history or []) + [{"role": "user", "content": message}]

                            # Only pass document snippets to tools that explicitly need them.
                            tools_with_doc_ctx = set(getattr(settings_obj, "tools_with_document_context", []) or [])
                            combined_context = None
                            if name in tools_with_doc_ctx:
                                combined_context = [
                                    {
                                        "url": (it.get("payload") or {}).get("url") or (it.get("payload") or {}).get("url_lower", ""),
                                        "title": (it.get("payload") or {}).get("title") or "",
                                        "snippet": (it.get("payload") or {}).get("text") or (it.get("payload") or {}).get("snippet") or "",
                                    }
                                    for it in (reranked or [])
                                ]

                            result_text = executor(args, chat_context, existing_context=combined_context)
                        except Exception as ex:
                            result_text = f"Tool '{name}' failed: {ex}"
                    if name:
                        used_tools.append(name)
                    tool_outputs_list.append({"tool_call_id": call_id or "", "output": str(result_text)})
                    _dbg(f"[TOOLS] {log_origin} tool outputs : %s", str(tool_outputs_list))
                if tool_outputs_list:
                    tools_text = "\n\n".join([t.get("output", "") for t in tool_outputs_list if t.get("output")])
                    rag_draft = _extract_text_from_responses(resp_inf) or ""
                    _dbg(f"[TOOLS] RAG DRAFT {log_origin} rag draft : %s", str(rag_draft))
                    synth_prompt = (
                        "You are a question-answering assistant for a retrieval-augmented system.\n"
                        "STRICT RULES:\n"
                        "1. Base your answer ONLY on information in the Context section and Tool results.\n"
                        "2. Do NOT use any outside knowledge.\n"
                        "3. If the context does not contain enough information to answer the question, reply exactly with: "
                        "I couldn't find any information to answer this question. NO_SUPPORTED_SOURCES\n"
                        "4. Retain any numeric citations like [1], [2] from the Context.\n"
                        "5. Do not fabricate sources or facts.\n\n"
                        f"Question:\n{message}\n\n"
                        + (f"Previous conversation summary:\n{summary_text}\n\n" if summary_text else "")
                        + (f"{recent_block_str}\n" if recent_block_str else "")
                        + f"Context:\n{context_text}\n\n"
                        + f"Tool results:\n{tools_text}\n\n"
                        + f"Draft answer (may be empty):\n{rag_draft}\n\n"
                        + "Task:\n"
                        "- Produce the final answer to the Question.\n"
                        "- Use citations like [1], [2] when using Context.\n"
                        "- Integrate Tool results where relevant (do not invent citations for tool facts).\n"
                        "- Be concise.\n"
                    )
                    try:
                        _kwargs_synth = {
                            "model": getattr(settings_obj, "tools_synthesis_model", _kwargs_inf["model"]),
                            "input": synth_prompt,
                            "max_output_tokens": int(max_out),
                            "temperature": float(temperature),
                        }
                        # Final Inference with Tools Synthesis (if tools are required for response)
                        emit_stage(req_id, "Generating Responses with Tools")
                        _dbg(f"[TOOLS] {log_origin} Final Inference with Tools Synthesis synth prompt : %s", str(synth_prompt))
                        resp_synth = get_client_fn().responses.create(**_kwargs_synth)
                        _dbg(f"INFERENCE 2 response {log_origin} Generating responses with tools", str(resp_synth))
                        combined = _extract_text_from_responses(resp_synth).strip()
                        answer_override = combined if combined else (rag_draft + "\n\n--- Live data ---\n" + tools_text)
                        # Record Inference Usage - 2nd Inference (tools synthesis)
                        usage_synth = _extract_usage_from_responses(resp_synth)
                        if usage_synth:
                            m.record_stage("inference", model=_kwargs_synth["model"], usage=usage_synth)
                    except Exception:
                        answer_override = rag_draft + "\n\n--- Live data ---\n" + tools_text
                break
        except Exception as e:
            logger.debug("[TOOLS] (%s) tool loop failed: %s", log_origin, e, exc_info=True)

    
    # Stage: Final answer and packing
    # --- Final answer and packing
    answer = (answer_override or _extract_text_from_responses(resp_inf) or "")
    sources = (reranked or []) + (web_context or [])
    # Sentinel detection: if the model indicates no supported sources, suppress Sources section and returned sources.
    _ans = (answer or "").rstrip()
    # Handle sentinel - check for NO_SUPPORTED_SOURCES in answer text to remove irrelevant sources section
    if _ans.endswith("NO_SUPPORTED_SOURCES"):
        # Remove sentinel from final answer text
        if "\n" in _ans:
            _ans = _ans.rsplit("\n", 1)[0].rstrip()
        answer = _ans
        sources = []  # JSON: no sources returned
        sources_section = ""  # No sources block appended
   
    # Heuristic: if the model indicates lack of supporting context, even without sentinel
    lower_ans = _ans.lower()
    if (
        "the provided context does not contain" in lower_ans
        or "the context provided does not contain" in lower_ans
        or "provided context does not" in lower_ans
        or "context does not" in lower_ans
    ):
        answer = _ans
        sources = []
        sources_section = ""
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
        "tools_used": sorted({t for t in used_tools if t}) if used_tools else [],
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

    deps = {
        "db": db,
        "cache": _SUMMARY_CACHE,
        "settings": settings,
        "get_client": get_client,
        "list_tools": list_tools,
        "get_executor": get_executor,
        "get_web_context": (lambda q, existing: []),  # stateless: no auto web
        "style": "flat",
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
        return {
            "answer": out.get("answer", ""),
            "response": out.get("answer", ""),  # legacy compatibility for frontend expecting 'response'
            "metrics": out.get("metrics", {"vectors_retrieved": 0}),
            "turn_metrics": out.get("turn_metrics", {}),
            "conversation_totals": out.get("conversation_totals", {}),
            "tools_used": out.get("tools_used", []),
            "rewrite_display": out.get("rewrite_display", {}),
        }
    except Exception as e:
        logger.exception("[PIPELINE] handle_chat orchestrator failed: %s", e)
        return {
            "answer": "Sorry, something went wrong.",
            "response": "Sorry, something went wrong.",
            "metrics": {"vectors_retrieved": 0},
        }
