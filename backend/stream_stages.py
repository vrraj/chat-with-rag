from __future__ import annotations

import asyncio
import json
import logging
from typing import AsyncIterator

from fastapi import APIRouter, Request, HTTPException, status, Query
from sse_starlette.sse import EventSourceResponse

from backend.stream_registry import (
    ensure_queue_for,
    register_consumer_loop,
    unregister_consumer_loop,
)

log = logging.getLogger(__name__)

# Exported router for main.py (app.include_router(router, prefix="/chat"))
router = APIRouter()

# --- Helper to ensure we don't double-encode JSON strings ---
def _normalize_sse_data(evt):
    """Return a string suitable for SSE 'data:' field.
    Accepts dicts/lists (json-dumped once) or strings (passed through).
    """
    if isinstance(evt, (dict, list)):
        return json.dumps(evt, ensure_ascii=False)
    # For any non-dict payload (including str), cast to str without extra dumps
    return str(evt)


async def _stage_event_generator(request: Request, query_id: str) -> AsyncIterator[dict]:
    """
    Yields SSE events for this query_id.
    Each yield is a dict like {"data": "<json>"} that sse_starlette formats to "data: ...".
    """
    queue = ensure_queue_for(query_id)
    loop = asyncio.get_running_loop()
    register_consumer_loop(query_id, loop)

    # Initial ACK
    initial = {"type": "debug", "stage": "_connected", "id": query_id, "note": "sse-connected"}
    yield {"data": _normalize_sse_data(initial)}
    log.info("[SSE] initial ACK for %s: %s", query_id, initial)
    log.info("[SSE] using loop_id=%s queue_obj_id=%s for %s", id(loop), id(queue), query_id)

    # Pre-drain any items enqueued before the client connected
    try:
        while True:
            item = queue.get_nowait()
            log.info("[SSE] pre-drain -> %s (type=%s)", item, type(item).__name__)
            try:
                yield {"data": _normalize_sse_data(item)}
            finally:
                try:
                    queue.task_done()
                except Exception:
                    pass
    except asyncio.QueueEmpty:
        pass

    try:
        while True:
            # Stop if client disconnected
            if await request.is_disconnected():
                log.info("[SSE] client disconnected; ending for %s", query_id)
                break

            try:
                payload = await asyncio.wait_for(queue.get(), timeout=12.0)
            except asyncio.TimeoutError:
                # keepalive payload (so simple consumers that only print data lines see activity)
                keep = {"type": "keepalive", "id": query_id}
                yield {"data": _normalize_sse_data(keep)}
                continue

            if payload is None:  # close sentinel
                log.info("[SSE] received close sentinel; ending for %s", query_id)
                break

            log.info("[SSE] dequeued -> %s (type=%s)", payload, type(payload).__name__)
            try:
                yield {"data": _normalize_sse_data(payload)}
            finally:
                try:
                    queue.task_done()
                except Exception:
                    pass
    finally:
        unregister_consumer_loop(query_id)
        log.info("[SSE] unregistered consumer loop for %s", query_id)


@router.get(
    "/stream/stages",
    name="stream_stages",
    summary="SSE stream of pipeline stages for a given query_id",
)
async def stream_stages(request: Request, query_id: str = Query(..., min_length=1)) -> EventSourceResponse:
    qid = (query_id or "").strip()
    if not qid:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="query_id is required")
    log.info("[SSE] connected for %s", qid)
    # Note: no unsupported kwargs (e.g., no `retry=`).
    return EventSourceResponse(_stage_event_generator(request, qid))