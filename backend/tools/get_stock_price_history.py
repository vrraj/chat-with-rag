from __future__ import annotations

from typing import Any, Dict, List
import logging

import requests

from .get_timeseries_sparklines_svg import generate_timeseries_sparklines


TOOL_NAME = "get_stock_price_history"

logger = logging.getLogger(__name__)


def tool_definition() -> Dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": TOOL_NAME,
            "description": (
                "Primary tool for stock price history chart requests. Use this whenever the user asks "
                "for a stock symbol's history/trend/chart (for example: AAPL 5 day, MSFT 3 months, NVDA 6 months, one year). "
                "Fetches data from /api/price-history/stocks and returns both normalized points and rendered SVG."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "symbol": {
                        "type": "string",
                        "description": "Stock ticker symbol to chart, e.g. AAPL, MSFT, NVDA.",
                    },
                    "period": {
                        "type": "string",
                        "enum": ["5D", "3M", "6M", "1Y", "5 day", "3 months", "6 months", "one year"],
                        "default": "1Y",
                        "description": "Requested time range. Accepts canonical values (5D, 3M, 6M, 1Y) and natural phrases (5 day, 3 months, 6 months, one year).",
                    },
                    "title": {
                        "type": "string",
                        "description": "Optional chart title override.",
                    },
                },
                "required": ["symbol"],
                "additionalProperties": False,
            },
        },
    }


def _normalize_period(period: str | None) -> str:
    raw = (period or "1Y").strip().lower()
    mapping = {
        "5d": "5D",
        "5 day": "5D",
        "5 days": "5D",
        "3m": "3M",
        "3 month": "3M",
        "3 months": "3M",
        "6m": "6M",
        "6 month": "6M",
        "6 months": "6M",
        "1y": "1Y",
        "1 year": "1Y",
        "one year": "1Y",
    }
    return mapping.get(raw, raw.upper() if raw.upper() in {"5D", "3M", "6M", "1Y"} else "1Y")


def _row_to_point(row: Dict[str, Any]) -> Dict[str, Any] | None:
    date_value = row.get("d") or row.get("date") or row.get("t")
    close_value = row.get("c")
    if close_value is None:
        close_value = row.get("close")
    if close_value is None:
        close_value = row.get("v")
    if date_value is None or close_value is None:
        return None
    try:
        return {"d": str(date_value)[:10], "c": round(float(close_value), 2)}
    except Exception:
        return None


def _extract_points(payload: Any, symbol: str) -> List[Dict[str, Any]]:
    if isinstance(payload, list):
        rows = payload
    elif isinstance(payload, dict):
        key = symbol.upper()
        candidate = payload.get(key)
        if isinstance(candidate, list):
            rows = candidate
        else:
            # Handle items[0].history structure from price-history API
            items = payload.get("items")
            if isinstance(items, list) and items:
                first_item = items[0]
                if isinstance(first_item, dict):
                    history = first_item.get("history")
                    if isinstance(history, list):
                        rows = history
                    else:
                        rows = (
                            payload.get("data")
                            or payload.get("prices")
                            or payload.get("history")
                            or payload.get("results")
                            or []
                        )
                        if isinstance(rows, dict):
                            nested = rows.get(key)
                            if isinstance(nested, list):
                                rows = nested
                            else:
                                rows = rows.get("data") or rows.get("prices") or rows.get("history") or []
                else:
                    rows = (
                        payload.get("data")
                        or payload.get("prices")
                        or payload.get("history")
                        or payload.get("results")
                        or []
                    )
                    if isinstance(rows, dict):
                        nested = rows.get(key)
                        if isinstance(nested, list):
                            rows = nested
                        else:
                            rows = rows.get("data") or rows.get("prices") or rows.get("history") or []
            else:
                rows = (
                    payload.get("data")
                    or payload.get("prices")
                    or payload.get("history")
                    or payload.get("results")
                    or []
                )
                if isinstance(rows, dict):
                    nested = rows.get(key)
                    if isinstance(nested, list):
                        rows = nested
                    else:
                        rows = rows.get("data") or rows.get("prices") or rows.get("history") or []
    else:
        rows = []

    points: List[Dict[str, Any]] = []
    for item in rows:
        if not isinstance(item, dict):
            continue
        point = _row_to_point(item)
        if point:
            points.append(point)
    return points


def _resolve_endpoint_url(tool_runtime: Dict[str, Any] | None) -> str:
    runtime = tool_runtime if isinstance(tool_runtime, dict) else {}
    endpoint = runtime.get("endpoint") if isinstance(runtime.get("endpoint"), dict) else {}
    endpoint_type = str(endpoint.get("type") or "").strip()
    endpoint_url = str(endpoint.get("url") or "").strip()
    if not endpoint_type or not endpoint_url:
        raise ValueError("Missing runtime.endpoint.type or runtime.endpoint.url for get_stock_price_history")
    if endpoint_type.lower() != "rest":
        raise ValueError(f"Unsupported endpoint type '{endpoint_type}' for get_stock_price_history")
    return endpoint_url


def _fetch_price_history(symbol: str, period: str, endpoint_url: str) -> List[Dict[str, Any]]:
    logger.info(
        "[TOOL_API] get_stock_price_history request method=GET url=%s params=%s timeout=%s",
        endpoint_url,
        {"symbols": symbol.upper(), "period": period},
        12,
    )
    response = requests.get(
        endpoint_url,
        params={"symbols": symbol.upper(), "period": period},
        timeout=12,
    )
    logger.info(
        "[TOOL_API] get_stock_price_history response status=%s url=%s",
        response.status_code,
        response.url,
    )
    response.raise_for_status()
    payload = response.json()
    points = _extract_points(payload, symbol)
    if not points:
        raise ValueError("Price history response did not include parseable points")
    return points


def run(args: Dict[str, Any] | None, chat_context: List[Dict[str, str]] | None = None, **kwargs: Any) -> Dict[str, Any]:
    _ = chat_context
    args = args or {}

    symbol = (args.get("symbol") or "AAPL").strip().upper()
    period = _normalize_period(args.get("period"))
    title = (args.get("title") or "").strip() or f"{symbol} {period}"
    endpoint_url = _resolve_endpoint_url(kwargs.get("tool_runtime"))

    try:
        points = _fetch_price_history(symbol, period, endpoint_url)
    except Exception as ex:
        try:
            logger.debug(
                "[get_stock_price_history] fetch failed symbol=%s period=%s err=%s",
                symbol,
                period,
                ex,
                exc_info=True,
            )
        except Exception:
            pass
        return {
            "symbol": symbol,
            "period": period,
            "svg": "",
            "summary": f"Could not fetch price history for {symbol} over {period}: {ex}",
            "data_points": 0,
            "data": [],
        }

    rendered = generate_timeseries_sparklines(
        data=points,
        period=period,
        title=title,
        width=760,
        height=320,
        margin={"top": 16, "right": 20, "bottom": 44, "left": 58},
        up_color="#16a34a",
        down_color="#dc2626",
        grid_color="rgba(148,163,184,0.35)",
        axis_color="#94a3b8",
        label_color="#64748b",
    )

    return {
        "symbol": symbol,
        "period": period,
        "data": points,
        "svg": rendered.get("svg", ""),
        "summary": rendered.get("summary", ""),
        "data_points": rendered.get("data_points", len(points)),
    }


__all__ = ["TOOL_NAME", "tool_definition", "run"]
