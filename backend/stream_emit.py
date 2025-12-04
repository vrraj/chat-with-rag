# backend/stream_emit.py
"""
stream_emit.py — small helper for emitting SSE *stage* events to the streaming layer.

What this is:
    A very thin, dependency‑light wrapper around `backend.stream_registry` that
    lets any backend module enqueue “stage” updates for a given `query_id`.
    It hides the details of which queue to use and whether we’re on the same
    asyncio loop as the consumer, and adds consistent logging and timestamps.

Why:
    - Decouples pipeline/orchestrator logic (e.g., chat_manager.py) from the
      queue/loop mechanics needed by SSE.
    - Makes it safe to call from background tasks or different event loops.
    - Central place to add diagnostics without touching business logic.

Public functions:
    - emit_stage(query_id, stage, **extra) -> bool
        Enqueue a stage event. Returns True if the event was queued.
        `extra` can include arbitrary keys; common ones: `final`, `finalContent`.
        A `ts` field (seconds since epoch) is added if not provided.
    - close_stream(query_id) -> None
        Signal that no more stage events are expected for `query_id`.

Notes:
    - This module intentionally imports `backend.stream_registry` lazily and
      tolerates import errors to make isolated unit testing easier.
    - Payloads are dictionaries; JSON serialization is performed by the SSE
      endpoint (`stream_stages.py`) just before sending over the wire.
"""

from typing import Any, Dict, Optional
import json
import time
import logging

logger = logging.getLogger(__name__)

try:
    # The registry owns queues and handles thread/loop‑safe put/get.
    import backend.stream_registry as _sr  # type: ignore
except Exception as _e:  # pragma: no cover
    _sr = None
    logger.warning("stream_emit: stream_registry import failed: %s", _e)


# --- thin pass‑throughs so call‑sites don’t import the registry directly -----
def ensure_queue_for(query_id: str) -> Any:
    """Create (or fetch) the queue for this query_id in the registry."""
    if _sr and hasattr(_sr, "ensure_queue_for"):
        return _sr.ensure_queue_for(query_id)
    return None


def put_stage_update(query_id: str, payload: Dict[str, Any]) -> bool:
    """Non‑threadsafe put; only use when you are on the consumer loop."""
    if _sr and hasattr(_sr, "put_stage_update"):
        return bool(_sr.put_stage_update(query_id, payload))
    return False


def put_stage_update_threadsafe(query_id: str, payload: Dict[str, Any]) -> bool:
    """Thread/loop‑agnostic put; safe to call from workers/background threads."""
    if _sr and hasattr(_sr, "put_stage_update_threadsafe"):
        return bool(_sr.put_stage_update_threadsafe(query_id, payload))
    return False


def in_same_loop_as_consumer(query_id: str) -> bool:
    """Best‑effort hint whether current context shares the consumer loop."""
    if _sr and hasattr(_sr, "in_same_loop_as_consumer"):
        return bool(_sr.in_same_loop_as_consumer(query_id))
    return False


def close_stream(query_id: str) -> None:
    """Signal that producers are done for this query_id."""
    if _sr and hasattr(_sr, "close_stream"):
        try:
            _sr.close_stream(query_id)
        except Exception:  # pragma: no cover
            logger.exception("close_stream(%s) failed", query_id)


# ------------------------------- main API -----------------------------------
def _build_payload(query_id: str, stage: str, extra: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Compose the event payload and normalize a few common fields."""
    payload: Dict[str, Any] = {"type": "stage", "stage": stage, "id": query_id}
    # Add timestamp if caller didn't set one
    payload["ts"] = extra.get("ts") if isinstance(extra, dict) and "ts" in extra else time.time()

    if extra:
        merged = dict(extra)
        # If someone passed a JSON string via `payload`, merge it (best effort)
        if "payload" in merged and isinstance(merged["payload"], str):
            try:
                inner = json.loads(merged["payload"])
                if isinstance(inner, dict):
                    merged.update(inner)
            except Exception:
                # If it's not valid JSON, just ignore; upstream logging will show details
                pass
            finally:
                merged.pop("payload", None)
        # Avoid overwriting core keys with wrong types, but allow explicit overrides
        for k, v in merged.items():
            if k in ("type", "stage", "id") and not isinstance(v, str):
                continue
            payload[k] = v
    return payload


def emit_stage(query_id: str, stage: str, **extra: Any) -> bool:
    """
    Emit a single stage event for the given `query_id`.

    Examples:
        >>> emit_stage("abc123", "Query Rewrite")
        >>> emit_stage("abc123", "Final Answer", final=True, finalContent="...")

    Returns:
        True if the event was enqueued into the stream queue; False otherwise.
    """
    try:
        q = ensure_queue_for(query_id)
        try:
            logger.info("  [SSE-DIAG] registry_mod_id=%s queue_obj_id=%s for %s",
                        id(_sr) if _sr else None, id(q) if q is not None else None, query_id)
        except Exception:
            pass

        payload = _build_payload(query_id, stage, extra)

        ok = False
        # Preferred: threadsafe path
        try:
            ok = put_stage_update_threadsafe(query_id, payload)
        except Exception:  # pragma: no cover
            logger.exception("threadsafe enqueue raised for %s", query_id)

        # Fallback: only if same loop as consumer
        if not ok:
            try:
                if in_same_loop_as_consumer(query_id):
                    put_stage_update(query_id, payload)
                    ok = True
                    logger.info("  Fallback direct enqueue (same loop) succeeded for %s", query_id)
                else:
                    logger.warning("  Skipping direct enqueue for %s (not same loop); threadsafe path failed",
                                   query_id)
            except Exception:  # pragma: no cover
                logger.exception("  Fallback direct enqueue failed for %s", query_id)

        try:
            logger.info("  Enqueued=%s payload=%s (type=%s)",
                        ok, json.dumps(payload, ensure_ascii=False), type(payload).__name__)
        except Exception:
            pass

        return ok
    except Exception:  # pragma: no cover
        logger.exception("emit_stage(%s, %s) failed", query_id, stage)
        return False


# Optional tiny convenience for the final event.
def emit_final(query_id: str, final_text: str, **extra: Any) -> bool:
    """
    Convenience wrapper to send the Final Answer event with content.
    Usage:
        emit_final(qid, text, citations=[...], tokens=123)
    """
    return emit_stage(query_id, "Final Answer", final=True, finalContent=final_text, **extra)