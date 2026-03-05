"""Test script for llm_client OpenAI route.

Usage:
    python scripts/test_openai_handler.py "Your prompt here"

This exercises the exact call shape:

    llm_client.generate(
        model_key="openai:gpt-4o-mini",
        input=prompt_or_messages,
        temperature=0.2,
        max_output_tokens=800,
        tools=tools,
        stream=False,
    )

It assumes:
    - OPENAI_API_KEY is available in the environment (optionally via .env)
    - backend.llm.llm_client is importable
"""

import os
import sys
from typing import Any, List, Dict

# Make the project root importable so we can resolve `llm` and `backend` when
# this file is executed as a script from any working directory.
# __file__ -> examples/llm_tests/test_openai_handler.py
# Going up three levels lands at the repo root: chat-with-rag
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

try:
    # Prefer python-dotenv so we can load from .env in the project root.
    from dotenv import load_dotenv
except Exception:  # pragma: no cover
    load_dotenv = None  # type: ignore[assignment]

# Try both import layouts for the shared client.
llm_client = None
_llm_import_error: Any = None
try:  # pragma: no cover
    from backend.llm.llm_client import generate, embed, get_pricing_for_model as _client
    llm_client = _client
except Exception as e:  # pragma: no cover
    _llm_import_error = e
    try:
        from backend.llm.llm_client import generate, embed as _client  # type: ignore[assignment]
        llm_client = _client
        _llm_import_error = None
    except Exception as e2:
        _llm_import_error = e2
        llm_client = None  # type: ignore[assignment]


def get_prompt_from_argv() -> str:
    if len(sys.argv) > 1:
        return " ".join(sys.argv[1:])
    return input("Enter a prompt for OpenAI via llm_client: ")


def main() -> None:
    # Load env vars from .env if available.
    if load_dotenv is not None:
        load_dotenv()

    if llm_client is None:
        print("[ERROR] Could not import llm_client from backend.llm.llm_client.")
        if _llm_import_error is not None:
            print(f"[DEBUG] Last llm_client import error: {_llm_import_error}")
        sys.exit(1)

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("[WARNING] OPENAI_API_KEY is not set in the environment.")
        print("llm_client.generate() may still succeed if your app loads config elsewhere,")
        print("but for this standalone test it is recommended to export OPENAI_API_KEY or use .env.")

    prompt = get_prompt_from_argv().strip()
    if not prompt:
        print("[ERROR] Empty prompt; nothing to send.")
        sys.exit(1)

    # For this test we send a simple user prompt string; your handler will pass it as `input`.
    prompt_or_messages: Any = prompt

    # No tools by default; you can edit this list to test tools wiring.
    tools: List[Dict[str, Any]] = []

    print("=== Test 1: llm_client.generate (OpenAI route, stream=False) ===")
    print("Model: openai:gpt-4o-mini")
    print("Temperature: 0.2")
    print("Max output tokens: 800")
    print(f"Prompt: {prompt}")
    print("------------------------------------------------------------")

    try:
        resp = generate(
            model_key="openai:gpt-4o-mini",
            input=prompt,
            temperature=0.2,
            max_output_tokens=800,
            tools=tools,
            stream=False,
        )
    except Exception as e:
        print(f"[ERROR] llm_client.generate failed: {e}")
        sys.exit(1)

    # Best-effort extraction of text and usage from a Responses-style object.
    text = getattr(resp, "text", "") or ""
    print("\n=== OpenAI Response via llm_client.generate ===")
    print(text or "<no text returned>")

    usage = getattr(resp, "usage", None)
    if usage is not None:
        print("\n=== Usage (best-effort) ===")
        # Support both dict-like and attribute-like usage objects.
        for field in ("prompt_tokens", "input_tokens", "completion_tokens", "output_tokens", "total_tokens"):
            if isinstance(usage, dict):
                val = usage.get(field)
            else:
                val = getattr(usage, field, None)
            if val is not None:
                print(f"{field}: {val}")


if __name__ == "__main__":
    main()
