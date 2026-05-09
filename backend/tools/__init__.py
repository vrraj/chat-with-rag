"""
Lightweight tool registry for Responses API-style tools.

Each tool module should export two callables:
- tool_definition() -> dict  # Returns a Responses API tool definition
- run(args: dict, chat_context: list[dict], **kwargs) -> str | dict  # Executes the tool

This package exposes helpers to list tool specs and retrieve executors
without wiring anything into chat_manager yet.
"""

from typing import Any, Callable, Dict, List

# Import tool modules here to register them
from . import get_weather as _get_weather
from . import web_search as _web_search
from . import get_nearby_airports as _get_nearby_airports
from . import get_timeseries_sparklines_svg as _get_timeseries_sparklines_svg


def _flatten_tool_spec(spec: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize any tool spec into flattened Responses API format."""
    if not isinstance(spec, dict):
        raise TypeError("tool spec must be a dict")
    if "function" in spec and isinstance(spec["function"], dict):
        fn = spec["function"]
        return {
            "type": spec.get("type", "function"),
            "name": fn.get("name"),
            "description": fn.get("description"),
            "parameters": fn.get("parameters") or {"type": "object", "properties": {}},
        }
    # assume already flattened
    return {
        "type": spec.get("type", "function"),
        "name": spec.get("name"),
        "description": spec.get("description"),
        "parameters": spec.get("parameters") or {"type": "object", "properties": {}},
    }


def list_tools() -> List[Dict[str, Any]]:
    """Return tool definitions ready for `tools=[...]` in the Responses API."""
    return [
        _flatten_tool_spec(_get_weather.tool_definition()),
        _flatten_tool_spec(_web_search.tool_definition()),
        _flatten_tool_spec(_get_nearby_airports.tool_definition()),
        _flatten_tool_spec(_get_timeseries_sparklines_svg.tool_definition()),
    ]


def get_executor(name: str) -> Callable[..., Any] | None:
    """Return the executor function for a tool by name, or None if not found."""
    mapping: Dict[str, Callable[..., Any]] = {
        _get_weather.TOOL_NAME: _get_weather.run,
        _web_search.TOOL_NAME: _web_search.run,
        _get_nearby_airports.TOOL_NAME: _get_nearby_airports.run,
        _get_timeseries_sparklines_svg.TOOL_NAME: _get_timeseries_sparklines_svg.run,
    }
    return mapping.get(name)


__all__ = [
    "list_tools",
    "get_executor",
]
