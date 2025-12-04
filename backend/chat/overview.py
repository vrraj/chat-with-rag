# backend/chat/overview.py
# Lightweight "corpus overview" pipeline that can be called from chat_manager.
# It samples a small set of docs/snippets from the vector DB and asks the LLM
# to summarize the scope of the corpus with citations.
#
# This module is self-contained and does NOT mutate global state.

from __future__ import annotations

import json
import logging
import random
import re
from collections import Counter
from typing import Any, Dict, List, Tuple
from urllib.parse import urlparse

logger = logging.getLogger("backend.chat.overview")

# ---------------------------
# Intent matching
# ---------------------------

_INTENT_PATTERNS = [
    r"\bwhat\s+(?:data|docs?|content|knowledge)\s+(?:do\s+(?:you|we)\s+)?have\??",
    r"\bwhat\s+can\s+(?:you|we)\s+(?:answer|talk\s+about|cover)\b",
    r"\bshow\s+me\s+(?:examples|samples)\b",
    r"\bcorpus\s+(?:overview|summary)\b",
    r"\bwhat'?s\s+in\s+(?:here|this)\b",
    r"\bwhat\s+topics\s+(?:are|do\s+you)\s+have\b",
]
_INTENT_RE = re.compile("|".join(_INTENT_PATTERNS), re.IGNORECASE)

def match_overview_intent(message: str) -> Tuple[bool, str]:
    """
    Return (is_overview, matched_pattern) for queries like 'what data do you have?'.
    """
    msg = (message or "").strip()
    m = _INTENT_RE.search(msg)
    return (m is not None, m.group(0) if m else "")

# ---------------------------
# Sampling helpers
# ---------------------------

def _norm_host(src: str | None) -> str:
    if not src:
        return "unknown"
    try:
        u = urlparse(src.strip())
        return (u.netloc or src).lower()
    except Exception:
        return str(src).lower()

def _extract_text(payload: Dict[str, Any]) -> str:
    # Try common fields in order; fall back to stringifying
    for k in ("text", "chunk", "content", "page_content", "snippet", "body"):
        if k in payload and payload[k]:
            v = payload[k]
            if isinstance(v, list):
                v = " ".join(str(x) for x in v)
            return str(v)
    # As a last resort, try nested metadata->text
    meta = payload.get("metadata") or {}
    if isinstance(meta, dict):
        for k in ("text", "content", "summary"):
            if k in meta and meta[k]:
                return str(meta[k])
    return ""

def _extract_source(payload: Dict[str, Any]) -> str:
    # Prefer canonical URL if present; else any 'source' string.
    for k in ("url", "source", "uri", "href"):
        v = payload.get(k)
        if v:
            return str(v)
    meta = payload.get("metadata") or {}
    if isinstance(meta, dict):
        for k in ("url", "source", "uri", "href"):
            if k in meta and meta[k]:
                return str(meta[k])
    return ""

def _dedup_key(payload: Dict[str, Any]) -> str:
    # Use doc_id/url if available; else hash the first 120 chars of text.
    for k in ("doc_id", "document_id", "id"):
        if k in payload and payload[k]:
            return f"id:{payload[k]}"
    src = _extract_source(payload)
    if src:
        return f"url:{src.strip().lower()}"
    text = _extract_text(payload)
    return f"txt:{(text[:120] if text else '').strip().lower()}"

def sample_corpus(
    db: Any,
    *,
    sample_size: int = 30,
    per_host_cap: int = 5,
    seed: int | None = None,
) -> List[Dict[str, Any]]:
    """
    Return a deduped, lightly stratified sample of payloads/snippets from Qdrant via the
    provided db wrapper. Tries multiple strategies:
      1) db.scroll(...) if available
      2) db.search_similar(...) using neutral probe queries
    Degrades gracefully; returns as many as it can collect (possibly 0).
    """
    rnd = random.Random(seed)
    want = max(1, int(sample_size))
    per_host_cap = max(1, int(per_host_cap))

    kept: List[Dict[str, Any]] = []
    seen: set[str] = set()
    per_host = Counter()

    # Strategy 1: scroll if available
    if hasattr(db, "scroll"):
        logger.debug("[OVERVIEW] using db.scroll for sampling (target=%d, host_cap=%d, seed=%s)", want, per_host_cap, seed)
        offset = None
        guard = 0
        try:
            while len(kept) < want and guard < 200:
                guard += 1
                # Try to fetch a small page
                out = db.scroll(limit=64, with_payload=True, with_vectors=False, offset=offset)  # type: ignore[attr-defined]
                points = out.get("points") or out.get("result") or []
                # randomize order within page for variety
                rnd.shuffle(points)
                for p in points:
                    payload = p.get("payload", {}) if isinstance(p, dict) else {}
                    if not isinstance(payload, dict) or not payload:
                        continue
                    key = _dedup_key(payload)
                    if key in seen:
                        continue
                    src = _extract_source(payload)
                    host = _norm_host(src)
                    if per_host[host] >= per_host_cap:
                        continue
                    text = _extract_text(payload)
                    if not text:
                        continue
                    kept.append({
                        "text": text,
                        "source": src,
                        "host": host,
                        "payload": payload,
                    })
                    seen.add(key)
                    per_host[host] += 1
                    if len(kept) >= want:
                        break
                offset = out.get("next_page_offset") or out.get("offset") or None
                if not offset:
                    break
        except Exception as e:
            logger.debug("[OVERVIEW] scroll sampling failed, will try probe-search: %s", e)

    # Strategy 2: probe queries via search_similar
    if len(kept) < want and hasattr(db, "search_similar"):
        probes = ["the", "introduction", "overview", "guide", "data", "about", "summary", "faq"]
        rnd.shuffle(probes)
        logger.debug("[OVERVIEW] using search_similar probes (remaining=%d)", want - len(kept))
        for q in probes:
            if len(kept) >= want:
                break
            try:
                res = db.search_similar(query=q, top_k=min(64, want * 2), score_threshold=0.0)
            except TypeError:
                # older signature without named args
                res = db.search_similar(q, top_k=min(64, want * 2))
            except Exception as e:
                logger.debug("[OVERVIEW] probe '%s' failed: %s", q, e)
                continue
            # randomize within results
            rnd.shuffle(res)
            for r in res:
                payload = r.get("payload", {}) if isinstance(r, dict) else {}
                if not isinstance(payload, dict) or not payload:
                    continue
                key = _dedup_key(payload)
                if key in seen:
                    continue
                src = _extract_source(payload)
                host = _norm_host(src)
                if per_host[host] >= per_host_cap:
                    continue
                text = _extract_text(payload)
                if not text:
                    continue
                kept.append({
                    "text": text,
                    "source": src,
                    "host": host,
                    "payload": payload,
                })
                seen.add(key)
                per_host[host] += 1
                if len(kept) >= want:
                    break

    logger.debug("[OVERVIEW] sampled=%d unique_hosts=%d (target=%d, cap=%d)", len(kept), len(per_host), want, per_host_cap)
    return kept[:want]

# ---------------------------
# Prompt + run
# ---------------------------

def _format_context_lines(items: List[Dict[str, Any]]) -> Tuple[str, List[Dict[str, Any]]]:
    lines = []
    sources: List[Dict[str, Any]] = []
    for i, it in enumerate(items, 1):
        src = it.get("source") or ""
        host = it.get("host") or _norm_host(src)
        text = (it.get("text") or "").strip()
        if len(text) > 1200:
            text = text[:1200].rstrip() + "…"
        lines.append(f"[{i}] {text}")
        sources.append({"tag": i, "source": src, "host": host})
    return "\n".join(lines), sources

_OVERVIEW_JSON_INSTR = """You are a librarian. Based ONLY on the snippets in Context, produce a concise catalog of what this corpus contains.
Return STRICT JSON with fields:
{
  "topics": [ {"name": "string","evidence":[<int>, ...]} ],
  "doc_types": [ {"type":"wiki|pdf|page|code|other","evidence":[<int>, ...]} ],
  "sources": [ {"host":"string","count_estimate": "few|dozens|many"} ],
  "date_hints": "string",
  "gaps_or_limits": "string",
  "good_followups": ["string", "string", "string"]
}
Do not speculate. Use ONLY evidence in Context. Cite using the numeric indices from Context in the evidence arrays.
"""

def build_overview_prompt(context_lines: str) -> str:
    return (
        _OVERVIEW_JSON_INSTR.strip()
        + "\n\nContext:\n"
        + context_lines.strip()
        + "\n\nTask: Produce the JSON object now. No prose outside JSON."
    )

def _extract_output_text(resp: Any) -> str:
    """
    Best-effort extraction of plain text from OpenAI Responses API response object.
    Compatible with newer response.output_text or older content arrays.
    """
    # Newer SDKs
    txt = getattr(resp, "output_text", None)
    if isinstance(txt, str) and txt.strip():
        return txt

    # Older shape: response.output[0].content[0].text
    try:
        output = getattr(resp, "output", None) or []
        if output and isinstance(output, list):
            content = output[0].get("content", [])
            if content and isinstance(content, list):
                part = content[0]
                if isinstance(part, dict):
                    t = part.get("text") or part.get("value")
                    if isinstance(t, str):
                        return t
    except Exception:
        pass

    # Fallback: stringify
    try:
        return json.dumps(resp, default=str)
    except Exception:
        return str(resp)

def run_overview(*, deps: Dict[str, Any], req: Dict[str, Any]) -> Dict[str, Any]:
    """
    Orchestrates an overview turn. Returns dict shaped like run_pipeline's output:
    {
      "answer": str,
      "sources": List[Dict],    # [{tag, source, host}]
      "metrics": Dict[str, Any],
      "turn_metrics": Dict[str, Any],
      "conversation_totals": Dict[str, Any],
      "tools_used": List[Any],
    }
    """
    settings = deps.get("settings")
    db = deps.get("db")
    get_client = deps.get("get_client")
    log_origin = str(deps.get("log_origin", "overview"))
    req_id = deps.get("request_id", "")

    params = (req or {}).get("params") or {}
    sample_size = int(params.get("overview_sample_size", 30) or 30)
    per_host_cap = int(params.get("overview_per_host_cap", 5) or 5)
    seed = params.get("overview_seed")
    try:
        seed = int(seed) if seed is not None else None
    except Exception:
        seed = None

    # 1) Sample
    items = sample_corpus(db, sample_size=sample_size, per_host_cap=per_host_cap, seed=seed)
    ctx_str, sources = _format_context_lines(items)
    logger.debug("[OVERVIEW] (%s#%s) sample=%d per_host_cap=%d hosts=%d", log_origin, req_id, len(items), per_host_cap, len({s['host'] for s in sources}))

    if not items:
        return {
            "answer": "I couldn't find any indexed documents to summarize yet.",
            "sources": [],
            "metrics": {"vectors_retrieved": 0},
            "turn_metrics": {},
            "conversation_totals": {},
            "tools_used": [],
        }

    # 2) Prompt
    prompt = build_overview_prompt(ctx_str)

    # 3) Call model
    model = getattr(settings, "inference_model", "gpt-4o")
    temperature = float(getattr(settings, "inference_temperature", 0.3))
    max_out = int(getattr(settings, "max_inference_output_tokens", 300))

    client = get_client()
    logger.info("[OVERVIEW] (%s#%s) Attempting Responses API with model=%s", log_origin, req_id, model)
    resp = client.responses.create(
        model=model,
        input=prompt,
        temperature=temperature,
        max_output_tokens=max_out,
    )
    txt = _extract_output_text(resp).strip()

    # 4) Parse JSON and render
    overview_json = {}
    try:
        overview_json = json.loads(txt)
    except Exception:
        # If the model returned prose, wrap it
        overview_json = {"notes": txt}

    # Friendly render with guardrails
    lines = []
    if "topics" in overview_json and isinstance(overview_json["topics"], list):
            lines.append("**What the corpus covers (sample-based):**")
            for t in overview_json["topics"][:8]:
                name = t.get("name", "").strip()
                ev = t.get("evidence") or []
                if name:
                    cite = f" [{','.join(str(x) for x in ev)}]" if ev else ""
                    lines.append(f"- {name}{cite}")
    if "doc_types" in overview_json and isinstance(overview_json["doc_types"], list):
            lines.append("\n**Document types observed:**")
            kinds = []
            for d in overview_json["doc_types"][:8]:
                t = d.get("type")
                if t:
                    kinds.append(str(t))
            if kinds:
                lines.append("- " + ", ".join(sorted(set(kinds))))
    if "gaps_or_limits" in overview_json and overview_json["gaps_or_limits"]:
            lines.append("\n**Obvious gaps or limits:**")
            lines.append("- " + str(overview_json["gaps_or_limits"]).strip())
    if "good_followups" in overview_json and isinstance(overview_json["good_followups"], list):
            lines.append("\n**Good follow‑ups:**")
            for q in overview_json["good_followups"][:5]:
                lines.append(f"- {q}")

    human = "\n".join(lines).strip() or overview_json.get("notes") or "Here's an overview based on a small sample of the index."

    # Append Sources: block similar to your normal answers
    src_lines = []
    for s in sources:
        tag = s.get("tag")
        src = s.get("source") or ""
        if not src:
            continue
        src_lines.append(f"[{tag}] {src}")
    if src_lines:
        human += "\n\nSources:\n" + "\n".join(src_lines)

    return {
        "answer": human,
        "sources": sources,
        "metrics": {"vectors_retrieved": len(items)},
        "turn_metrics": {},
        "conversation_totals": {},
        "tools_used": [],
    }
