"""Example: Call the FastAPI /chat route with params.

Run this after starting the app (e.g. `make start`).

This script sends a single stateless chat request with a params payload
similar to what the frontend chat UI (`frontend/chat.html`) uses.
"""

import uuid
from typing import Any, Dict

import requests


BASE_URL = "http://localhost:8000"


def build_payload(message: str, show_processing_steps: bool = True) -> Dict[str, Any]:
    """Build a ChatRequest-style payload for POST /chat."""
    query_id = uuid.uuid4().hex[:8]
    conversation_id = "example-conversation-1"

    return {
        "message": message,
        "use_web_search": False,
        "history": [],
        "params": {
            # Retrieval
            "top_k": 8,
            "score_threshold": 0.35,
            # Summarization / history window
            "chat_history_window_turns": 2,
            "raw_tail_turns": 2,
            "summarizer_max_input_tokens": 400,
            "summarizer_max_output_tokens": 200,
            # Inference
            "temperature": 0.4,
            "top_p": 0.9,
            "max_output_tokens": 300,
            # Query rewrite
            "enable_query_rewrite": True,
            "rewrite_confidence_threshold": 0.67,
            "rewrite_tail_turns": 1,
            # Tools
            "use_tools": False,
            # Per-turn control of intermediate SSE stages
            "show_processing_steps": show_processing_steps,
            # UX / observability
            "query_id": query_id,
            "conversation_id": conversation_id,
        },
    }


def call_chat(message: str, show_processing_steps: bool = True) -> Dict[str, Any]:
    """Call POST /chat and return the parsed JSON response."""
    payload = build_payload(message, show_processing_steps=show_processing_steps)

    resp = requests.post(f"{BASE_URL}/chat", json=payload, timeout=60)
    resp.raise_for_status()
    return resp.json()


def main() -> None:
    message = "Give me a short overview of how this RAG chat pipeline works."

    print("Sending request to /chat ...")
    data = call_chat(message, show_processing_steps=True)

    print("\n--- /chat response ---")
    print("Answer:", data.get("answer") or data.get("response"))
    print("Metrics:", data.get("metrics"))
    print("Turn metrics:", data.get("turn_metrics"))
    print("Conversation totals:", data.get("conversation_totals"))
    print("Tools used:", data.get("tools_used"))
    print("Rewrite display:", data.get("rewrite_display"))


if __name__ == "__main__":
    main()
