"""
Weather tool for the Responses API using Open‑Meteo.

Design:
- Exposes a function-style tool definition compatible with OpenAI Responses API.
- `run` accepts args and chat context; it will use `location` if provided,
  otherwise tries a simple heuristic to infer it from recent chat messages.
- Uses Open‑Meteo Geocoding API to resolve the location and Open‑Meteo
  Forecast API to obtain current, max, and min temperatures. Average is
  computed as the mid-point of daily max/min. Output includes date (Day/Mon),
  high, low, average, and current temperatures in both Celsius and Kelvin.

Notes:
- Open‑Meteo is a free, no‑auth API. Network errors are handled gracefully.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
import re
import requests
import logging


TOOL_NAME = "get_weather"

logger = logging.getLogger(__name__)


def tool_definition() -> Dict[str, Any]:
    """Return a Responses API tool definition for weather lookup."""
    return {
        "type": "function",
        "function": {
            "name": TOOL_NAME,
            "description": (
                "Get weather for a location (city/state/country). If `location` is omitted,"
                " the tool may infer it from the recent chat context. Returns date (Day/Mon),"
                " high, low, average, and current temperatures in Celsius and Kelvin."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "Location name, e.g. 'San Francisco' or 'Bavaria, Germany'.",
                    },
                    "unit": {
                        "type": "string",
                        "enum": ["C", "K", "F"],
                        "description": "Preferred display unit (C, K, or F). Output includes all three.",
                        "default": "C",
                    },
                },
                "required": [],
                "additionalProperties": False,
            },
        },
    }


# --- Implementation ---

@dataclass
class Weather:
    location: str
    high_c: float
    low_c: float
    avg_c: float
    current_c: float

    @property
    def high_k(self) -> float:
        return self.high_c + 273.15

    @property
    def low_k(self) -> float:
        return self.low_c + 273.15

    @property
    def avg_k(self) -> float:
        return self.avg_c + 273.15

    @property
    def current_k(self) -> float:
        return self.current_c + 273.15


def _infer_location_from_chat(chat_context: List[Dict[str, str]]) -> Optional[str]:
    """Very lightweight inference: look for patterns like 'in <Location>' or 'at <Location>'."""
    if not chat_context:
        return None
    text = " \n".join(m.get("content", "") or "" for m in chat_context[-5:])
    # Try explicit patterns first
    m = re.search(r"\b(?:in|at) ([A-Z][\w\-]+(?:[\s,]+[A-Z][\w\-]+){0,3})\b", text)
    if m:
        return m.group(1).strip()
    # Fallback: last capitalized token sequence
    caps = re.findall(r"\b([A-Z][\w\-]+(?:\s+[A-Z][\w\-]+){0,3})\b", text)
    return caps[-1].strip() if caps else None


def _geocode_location(name: str, *, timeout: float = 7.0) -> Optional[Tuple[float, float, str]]:
    """Resolve a free-form place name to (lat, lon, display_name) via Open‑Meteo Geocoding API."""
    try:
        resp = requests.get(
            "https://geocoding-api.open-meteo.com/v1/search",
            params={"name": name, "count": 1, "language": "en", "format": "json"},
            timeout=timeout,
        )
        resp.raise_for_status()
        data = resp.json() or {}
        results = data.get("results") or []
        if not results:
            return None
        r = results[0]
        lat, lon = float(r.get("latitude")), float(r.get("longitude"))
        # Build a readable display name
        parts = [p for p in [r.get("name"), r.get("admin1"), r.get("country")] if p]
        display = ", ".join(parts) if parts else name
        return lat, lon, display
    except Exception as ex:
        try:
            logger.debug("[get_weather] geocode failed name=%r err=%s", name, ex, exc_info=True)
        except Exception:
            pass
        return None


def _fetch_weather(location: str, *, timeout: float = 10.0) -> Optional[Weather]:
    """Fetch weather for a location using Open‑Meteo.

    Steps:
    - Geocode the location to lat/lon.
    - Fetch daily max/min and current temperature.
    - Compute average as (max+min)/2.
    """
    geo = _geocode_location(location, timeout=timeout)
    if not geo:
        return None
    lat, lon, display = geo
    try:
        resp = requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": lat,
                "longitude": lon,
                "daily": ["temperature_2m_max", "temperature_2m_min"],
                "current_weather": True,
                "timezone": "auto",
            },
            headers={"User-Agent": "chat-with-rag/1.0 (+https://github.com/)"},
            timeout=timeout,
        )
        resp.raise_for_status()
        data = resp.json() or {}
        daily = data.get("daily") or {}
        tmax = (daily.get("temperature_2m_max") or [None])[0]
        tmin = (daily.get("temperature_2m_min") or [None])[0]
        curr = (data.get("current_weather") or {}).get("temperature")
        if tmax is None or tmin is None:
            return None
        avg = (float(tmax) + float(tmin)) / 2.0
        current = float(curr) if curr is not None else avg
        return Weather(
            location=display,
            high_c=round(float(tmax), 1),
            low_c=round(float(tmin), 1),
            avg_c=round(float(avg), 1),
            current_c=round(float(current), 1),
        )
    except Exception as ex:
        try:
            logger.debug("[get_weather] forecast failed location=%r lat=%s lon=%s err=%s", location, lat, lon, ex, exc_info=True)
        except Exception:
            pass
        return None


def _format_date() -> str:
    now = datetime.now()
    # Format as "Day/Mon" e.g., "11/Sep"
    return f"{now.day:02d}/{now.strftime('%b')}"


def _format_output(w: Weather, preferred_unit: str = "C") -> str:
    day = _format_date()
    pu = (preferred_unit or "C").upper()
    if pu not in {"C", "K", "F"}:
        pu = "C"

    def c_to_f(c: float) -> float:
        return c * 9.0 / 5.0 + 32.0

    def fmt_triplet(c_val: float) -> str:
        values = {
            "C": f"{c_val:.1f}°C",
            "K": f"{c_val + 273.15:.1f}K",
            "F": f"{c_to_f(c_val):.1f}°F",
        }
        order = [pu] + [u for u in ("C", "K", "F") if u != pu]
        return " / ".join(values[u] for u in order)

    return (
        f"{w.location} — {day}\n"
        f"High: {fmt_triplet(w.high_c)}\n"
        f"Low: {fmt_triplet(w.low_c)}\n"
        f"Average: {fmt_triplet(w.avg_c)}\n"
        f"Current: {fmt_triplet(w.current_c)}"
    )


def run(args: Dict[str, Any] | None, chat_context: List[Dict[str, str]] | None = None, **_: Any) -> str:
    """Execute the weather tool.

    Args:
        args: Dict with optional keys: {"location": str, "unit": "C"|"K"}
        chat_context: Recent chat messages for optional location inference
    Returns:
        A formatted multi-line string with the weather summary.
    """
    args = args or {}
    chat_context = chat_context or []
    location = (args.get("location") or "").strip()
    unit = (args.get("unit") or "C").strip().upper()

    if not location:
        inferred = _infer_location_from_chat(chat_context)
        if inferred:
            location = inferred
        else:
            # Sensible default to avoid failures
            location = "Unknown Location"

    weather = _fetch_weather(location)
    if not weather:
        # One simple retry for common phrasing artifacts (e.g., "Region of X")
        simplified = re.sub(r"\bof\b", ",", location, flags=re.IGNORECASE).strip(" ,")
        if simplified and simplified.lower() != location.lower():
            weather = _fetch_weather(simplified)
            if weather:
                return _format_output(weather, preferred_unit=unit)
        return (
            f"Could not fetch weather for '{location}'. "
            "This may be due to an ambiguous place name or network/SSL restrictions. "
            "Try a more specific location like 'Mount Whitney, CA' or 'Lone Pine, CA'."
        )
    return _format_output(weather, preferred_unit=unit)


__all__ = ["TOOL_NAME", "tool_definition", "run"]
