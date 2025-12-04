"""In-process SSE stream registry for per-query stage events."""

from __future__ import annotations

import asyncio
from asyncio import AbstractEventLoop
from typing import Dict, Optional
import json
from collections import deque
from typing import Deque, Any
import sys as _sys
import logging

logger = logging.getLogger("backend.stream_registry")
# Ensure a single module instance even if imported via different paths
_sys.modules.setdefault("stream_registry", _sys.modules[__name__])

# Store per-query queues. Payloads are strings; we also send None as a sentinel to close the stream.
_queues: Dict[str, asyncio.Queue[Optional[str]]] = {}

# Track the event loop that is consuming each query's SSE stream.
_consumer_loops: Dict[str, AbstractEventLoop] = {}

# Backlog per query for events emitted before the consumer loop is registered.
_backlogs: Dict[str, Deque[str]] = {}


def register_query(query_id: str) -> asyncio.Queue[Optional[str]]:
    """Create (if needed) and return a per-query stage queue.

    Idempotent: if the queue already exists, it is returned unchanged.
    """
    q = _queues.get(query_id)
    if q is None:
        q = asyncio.Queue(maxsize=256)
        _queues[query_id] = q
        if query_id not in _backlogs:
            _backlogs[query_id] = deque(maxlen=512)
    return q


def register_consumer_loop(query_id: str, loop: AbstractEventLoop) -> None:
    """Associate a consumer event loop with this query's queue."""
    _consumer_loops[query_id] = loop
    # Ensure a queue exists so producers can immediately enqueue
    register_query(query_id)

    # Drain any backlog that accumulated before the consumer loop was known.
    backlog = _backlogs.get(query_id)
    q = _queues.get(query_id)
    if backlog and q:
        while backlog:
            payload = backlog.popleft()
            if loop and loop.is_running():
                loop.call_soon_threadsafe(q.put_nowait, payload)
            else:
                try:
                    q.put_nowait(payload)
                except asyncio.QueueFull:
                    try:
                        q.get_nowait()
                    except Exception:
                        pass
                    try:
                        q.put_nowait(payload)
                    except Exception:
                        pass
    # Log backlog drain result
    try:
        drained_count = 0 if backlog is None else 0  # placeholder, we can't know now; keep light log
        logger.debug("register_consumer_loop: consumer set for %s; backlog size now=%d", query_id, 0 if backlog is None else len(backlog))
    except Exception:
        pass


def unregister_consumer_loop(query_id: str) -> None:
    """Remove the consumer loop association for this query."""
    _consumer_loops.pop(query_id, None)


def get_stage_queue_for_query(query_id: str, _user=None) -> Optional[asyncio.Queue[Optional[str]]]:
    """Return the existing stage queue for a query, or None if not registered."""
    return _queues.get(query_id)


def get_queue(query_id: str) -> Optional[asyncio.Queue[Optional[str]]]:
    """Alias for get_stage_queue_for_query; kept for brevity in callers."""
    return _queues.get(query_id)


def ensure_queue_for(query_id: str) -> asyncio.Queue[Optional[str]]:
    """Return existing queue or create one if missing."""
    return register_query(query_id)


def in_same_loop_as_consumer(query_id: str) -> bool:
    """Return True if currently running on the same loop as the consumer for this query."""
    try:
        current = asyncio.get_running_loop()
    except RuntimeError:
        # Not in any running loop (likely in a worker thread)
        return False
    loop = _consumer_loops.get(query_id)
    return loop is not None and loop is current


def put_stage_update(query_id: str, message: str | dict) -> bool:
    """Synchronous: enqueue a stage update from the current context.

    If the consumer loop is known:
      - If we're on that loop, put_nowait directly.
      - Otherwise, schedule via call_soon_threadsafe.
    If the consumer loop is not known yet, append to backlog so events preserve order
    until the SSE endpoint attaches and drains.
    """
    # Ensure queue/backlog exist
    q = _queues.get(query_id) or register_query(query_id)

    # Normalize message to a JSON string
    if isinstance(message, dict):
        payload = json.dumps(message, ensure_ascii=False)
    else:
        payload = str(message)

    loop = _consumer_loops.get(query_id)

    if loop and loop.is_running():
        try:
            if in_same_loop_as_consumer(query_id):
                q.put_nowait(payload)
            else:
                loop.call_soon_threadsafe(q.put_nowait, payload)
            return True
        except asyncio.QueueFull:
            # Drop oldest then retry to avoid blocking
            try:
                q.get_nowait()
            except Exception:
                pass
            if in_same_loop_as_consumer(query_id):
                q.put_nowait(payload)
            else:
                loop.call_soon_threadsafe(q.put_nowait, payload)
            return True

    # No consumer loop yet: backlog to preserve ordering until client attaches
    bl = _backlogs.get(query_id)
    if bl is None:
        bl = deque(maxlen=512)
        _backlogs[query_id] = bl
    try:
        logger.debug("put_stage_update: backlogging for %s (consumer unknown); backlog_len=%d", query_id, 0 if bl is None else len(bl))
    except Exception:
        pass
    bl.append(payload)
    return True


def put_stage_update_threadsafe(query_id: str, message: str | dict) -> bool:
    """Non-async: enqueue from any thread using the *consumer's* loop (if known).
    Returns True if scheduled/enqueued.
    """
    queue = _queues.get(query_id)
    if queue is None:
        queue = register_query(query_id)

    # Normalize message to string for the SSE layer.
    if isinstance(message, dict):
        payload = json.dumps(message, ensure_ascii=False)
    else:
        payload = str(message)

    loop = _consumer_loops.get(query_id)

    if loop and loop.is_running():
        # Ensure the queue operation runs in the consumer's event loop
        loop.call_soon_threadsafe(queue.put_nowait, payload)
        return True

    # No consumer loop yet: backlog the payload so it flushes in order once the SSE client registers.
    bl = _backlogs.get(query_id)
    if bl is None:
        bl = deque(maxlen=512)
        _backlogs[query_id] = bl
    try:
        logger.debug("put_stage_update_threadsafe: backlogging for %s (consumer unknown); backlog_len=%d", query_id, 0 if bl is None else len(bl))
    except Exception:
        pass
    bl.append(payload)
    return True


def unregister_query(query_id: str) -> None:
    """Cleanup a query queue when the client disconnects. Enqueue None as a sentinel."""
    queue = _queues.pop(query_id, None)
    loop = _consumer_loops.pop(query_id, None)
    _backlogs.pop(query_id, None)
    if queue is not None:
        if loop and loop.is_running():
            loop.call_soon_threadsafe(queue.put_nowait, None)
        else:
            try:
                queue.put_nowait(None)
            except asyncio.QueueFull:
                # If full, drop one and then put sentinel
                try:
                    queue.get_nowait()
                except Exception:
                    pass
                try:
                    queue.put_nowait(None)
                except Exception:
                    pass
    try:
        logger.debug("unregister_query: closed and cleaned %s", query_id)
    except Exception:
        pass


__all__ = [
    "register_query",
    "register_consumer_loop",
    "unregister_consumer_loop",
    "get_stage_queue_for_query",
    "get_queue",
    "ensure_queue_for",
    "put_stage_update",
    "put_stage_update_threadsafe",
    "unregister_query",
    "in_same_loop_as_consumer",
]