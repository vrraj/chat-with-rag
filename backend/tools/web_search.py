"""
Web search tool for the Responses API.

Design:
- Tool definition exposes a function with `query` and `num_results` parameters.
- `run` will use `query` if provided; otherwise it falls back to the last
  user message in `chat_context`. It returns a concise, readable text block
  of results. It can also consider `existing_context` to avoid duplicates.

This wraps the existing `backend.chat.web_search.WebSearchClient` for parity
with current search behavior elsewhere in the codebase.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from backend.chat.web_search import WebSearchClient


TOOL_NAME = "web_search"


def tool_definition() -> Dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": TOOL_NAME,
            "description": (
                "Perform a web search to gather additional context. If `query` is omitted,"
                " the tool will use the latest user message from the chat context."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query; if omitted the last user message is used.",
                    },
                    "num_results": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 10,
                        "default": 3,
                        "description": "Number of results to return (1-10).",
                    },
                },
                "required": [],
                "additionalProperties": False,
            },
        },
    }


def _last_user_message(chat_context: List[Dict[str, str]]) -> Optional[str]:
    for m in reversed(chat_context or []):
        if (m.get("role") or "").lower() == "user":
            return m.get("content") or None
    # If roles are not labeled, fall back to the last content
    return (chat_context or [{}])[-1].get("content") if chat_context else None


def _format_results(results: List[Dict[str, Any]]) -> str:
    if not results:
        return "No web results were found."
    lines = ["Web results:"]
    for i, item in enumerate(results, start=1):
        title = item.get("title") or "(untitled)"
        snippet = item.get("snippet") or ""
        url = item.get("url") or ""
        lines.append(f"{i}. {title}\n   {snippet}\n   {url}")
    return "\n".join(lines)


def run(
    args: Dict[str, Any] | None,
    chat_context: List[Dict[str, str]] | None = None,
    *,
    existing_context: List[Dict[str, Any]] | None = None,
) -> str:
    """Execute the web search tool.

    Args:
        args: Dict with optional keys: {"query": str, "num_results": int}
        chat_context: Recent chat messages for fallback query
        existing_context: Current context items to avoid duplicates
    Returns:
        A formatted string summarizing the search results.
    """
    args = args or {}
    chat_context = chat_context or []
    existing_context = existing_context or []

    query = (args.get("query") or "").strip()
    if not query:
        lm = _last_user_message(chat_context)
        query = lm.strip() if lm else ""

    num_results = args.get("num_results") if isinstance(args.get("num_results"), int) else 3
    num_results = max(1, min(10, int(num_results)))

    client = WebSearchClient()
    # Use existing behavior to filter duplicates and cap results
    results = client.get_additional_context(query, existing_context) if query else []
    results = (results or [])[:num_results]
    return _format_results(results)


__all__ = ["TOOL_NAME", "tool_definition", "run"]

