from __future__ import annotations

from typing import Any, Dict, List

from timeseries_svg import TimeSeriesChartRenderer

def _normalize_period(period: str | None) -> str:
    val = (period or "1Y").strip().upper()
    return val if val in {"5D", "3M", "6M", "1Y"} else "1Y"


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


def _normalize_points(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    points: List[Dict[str, Any]] = []
    for item in rows:
        if not isinstance(item, dict):
            continue
        point = _row_to_point(item)
        if point:
            points.append(point)
    return points


def _build_summary(period: str, values: List[float]) -> str:
    if len(values) < 2:
        return f"Insufficient data for period {period}."
    delta = values[-1] - values[0]
    trend = "upward" if delta >= 0 else "downward"
    return f"Series trended {trend} over {period} ({values[0]:.2f} → {values[-1]:.2f})."


def generate_timeseries_sparklines(
    data: List[Dict[str, Any]],
    *,
    period: str = "1Y",
    title: str | None = None,
    width: int = 760,
    height: int = 320,
    margin: Dict[str, int] | None = None,
    up_color: str = "#16a34a",
    down_color: str = "#dc2626",
    grid_color: str = "rgba(148,163,184,0.35)",
    axis_color: str = "#94a3b8",
    label_color: str = "#64748b",
) -> Dict[str, Any]:
    points = _normalize_points(data or [])
    if not points:
        return {
            "period": _normalize_period(period),
            "svg": "",
            "summary": "No parseable time-series points were provided.",
            "data_points": 0,
        }

    normalized_period = _normalize_period(period)
    values = [float(row.get("c", 0.0)) for row in points]
    renderer = TimeSeriesChartRenderer(
        width=int(width),
        height=int(height),
        margin=margin or {"top": 16, "right": 20, "bottom": 44, "left": 58},
        up_color=up_color,
        down_color=down_color,
        grid_color=grid_color,
        axis_color=axis_color,
        label_color=label_color,
    )
    svg = renderer.render(points, period=normalized_period, title=title)

    return {
        "period": normalized_period,
        "svg": svg,
        "summary": _build_summary(normalized_period, values),
        "data_points": len(points),
    }


__all__ = ["generate_timeseries_sparklines"]
