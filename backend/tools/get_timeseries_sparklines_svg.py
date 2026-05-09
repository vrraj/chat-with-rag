from __future__ import annotations

from typing import Any, Dict, List

from timeseries_svg import BarChartRenderer, SparklineRenderer, TimeSeriesChartRenderer


TOOL_NAME = "get_timeseries_sparklines_svg"


_BASE_SERIES: List[Dict[str, float | str]] = [
    {"d": "2024-01-01", "v": 100},
    {"d": "2024-01-02", "v": 102.5},
    {"d": "2024-01-03", "v": 101.2},
    {"d": "2024-01-04", "v": 105},
    {"d": "2024-01-05", "v": 103.8},
    {"d": "2024-01-08", "v": 107},
    {"d": "2024-01-09", "v": 103},
    {"d": "2024-01-10", "v": 100},
    {"d": "2024-01-11", "v": 90},
    {"d": "2024-01-12", "v": 95},
    {"d": "2024-01-15", "v": 99},
    {"d": "2024-01-16", "v": 105},
    {"d": "2024-01-17", "v": 106},
    {"d": "2024-01-18", "v": 100},
    {"d": "2024-01-19", "v": 110},
]


def _shift_dataset(series: List[Dict[str, float | str]], multiplier: float, offset: float) -> List[Dict[str, float | str]]:
    out: List[Dict[str, float | str]] = []
    for row in series:
        out.append({"d": row["d"], "v": round(float(row["v"]) * multiplier + offset, 2)})
    return out


def _monthly_temperatures() -> List[Dict[str, float | str]]:
    return [
        {"d": "2024-01-01", "v": 42.0},
        {"d": "2024-02-01", "v": 45.0},
        {"d": "2024-03-01", "v": 52.0},
        {"d": "2024-04-01", "v": 61.0},
        {"d": "2024-05-01", "v": 69.0},
        {"d": "2024-06-01", "v": 77.0},
        {"d": "2024-07-01", "v": 82.0},
        {"d": "2024-08-01", "v": 80.0},
        {"d": "2024-09-01", "v": 73.0},
        {"d": "2024-10-01", "v": 63.0},
        {"d": "2024-11-01", "v": 52.0},
        {"d": "2024-12-01", "v": 44.0},
    ]


_STATIC_DATASETS: Dict[str, List[Dict[str, float | str]]] = {
    "MSFT": _BASE_SERIES,
    "NVDA": _shift_dataset(_BASE_SERIES, multiplier=1.22, offset=8),
    "AAPL": _shift_dataset(_BASE_SERIES, multiplier=0.92, offset=6),
    "API_LATENCY": _shift_dataset(_BASE_SERIES, multiplier=0.78, offset=-42),
    "CPU_USAGE": _shift_dataset(_BASE_SERIES, multiplier=0.55, offset=-8),
    "MONTHLY_TEMPERATURES": _monthly_temperatures(),
}


def tool_definition() -> Dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": TOOL_NAME,
            "description": (
                "Return server-rendered SVG for sparkline, trend line chart, or bar chart from static "
                "time-series data. Supports symbols like MSFT/NVDA/AAPL and non-financial datasets like "
                "api_latency/cpu_usage/monthly_temperatures."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "symbol": {
                        "type": "string",
                        "description": "Dataset key, e.g. MSFT, NVDA, AAPL, api_latency, cpu_usage, monthly_temperatures.",
                    },
                    "period": {
                        "type": "string",
                        "enum": ["5D", "1W", "2W", "1M", "3M", "6M", "1Y"],
                        "default": "1M",
                        "description": "Time window for slicing supported by chart renderers.",
                    },
                    "chart_type": {
                        "type": "string",
                        "enum": ["sparkline", "trend", "bar"],
                        "default": "sparkline",
                        "description": "Type of SVG to generate.",
                    },
                    "title": {
                        "type": "string",
                        "description": "Optional title for trend/bar charts.",
                    },
                },
                "required": ["symbol"],
                "additionalProperties": False,
            },
        },
    }


def get_timeseries_data(symbol: str) -> List[Dict[str, float | str]]:
    key = (symbol or "").strip().upper()
    return _STATIC_DATASETS.get(key) or _STATIC_DATASETS["MSFT"]


def _build_summary(symbol: str, period: str, values: List[float]) -> str:
    if len(values) < 2:
        return f"{symbol} has insufficient data for period {period}."
    delta = values[-1] - values[0]
    trend = "upward" if delta >= 0 else "downward"
    return f"{symbol} trended {trend} over {period} ({values[0]:.2f} → {values[-1]:.2f})."


def run(args: Dict[str, Any] | None, chat_context: List[Dict[str, str]] | None = None, **_: Any) -> Dict[str, Any]:
    _ = chat_context
    args = args or {}

    symbol = (args.get("symbol") or "MSFT").strip()
    period = (args.get("period") or "1M").strip()
    chart_type = (args.get("chart_type") or "sparkline").strip().lower()
    title = (args.get("title") or "").strip() or None

    data = get_timeseries_data(symbol)
    values = [float(row.get("v", 0.0)) for row in data]

    if chart_type == "trend":
        renderer = TimeSeriesChartRenderer(width=760, height=320)
        svg = renderer.render(data, period=period, title=title or f"{symbol.upper()} {period} Trend")
    elif chart_type == "bar":
        renderer = BarChartRenderer(width=760, height=320)
        svg = renderer.render(data, period=period, title=title or f"{symbol.upper()} {period} Bars")
    else:
        renderer = SparklineRenderer(width=180, height=56)
        svg = renderer.render(data)

    return {
        "symbol": symbol.upper(),
        "period": period,
        "chart_type": chart_type,
        "svg": svg,
        "summary": _build_summary(symbol.upper(), period, values),
        "data_points": len(data),
    }


__all__ = ["TOOL_NAME", "tool_definition", "run", "get_timeseries_data"]
