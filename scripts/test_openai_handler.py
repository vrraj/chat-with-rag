"""Test script for LLMHandler OpenAI route.

Usage:
    python scripts/test_openai_handler.py "Your prompt here"

This exercises the exact call shape:

    llm_handler.responses.create(
        model="gpt-4o-mini",
        input=prompt_or_messages,
        temperature=0.2,
        max_output_tokens=800,
        tools=tools,
        stream=False,
    )

It assumes:
    - OPENAI_API_KEY is available in the environment (optionally via .env)
    - backend.llm.llm_handler or llm.llm_handler is importable
"""

import os
import sys
from typing import Any, List, Dict

# Make the project root importable so we can resolve `llm` and `backend` when
# this file is executed as a script from any working directory.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

try:
    # Prefer python-dotenv so we can load from .env in the project root.
    from dotenv import load_dotenv
except Exception:  # pragma: no cover
    load_dotenv = None  # type: ignore[assignment]

# Try both import layouts for the shared handler.
llm_handler = None
try:  # pragma: no cover
    from llm.llm_handler import llm_handler as _handler
    llm_handler = _handler
except Exception:  # pragma: no cover
    try:
        from backend.llm.llm_handler import llm_handler as _handler  # type: ignore[assignment]
        llm_handler = _handler
    except Exception:
        llm_handler = None  # type: ignore[assignment]


def get_prompt_from_argv() -> str:
    if len(sys.argv) > 1:
        return " ".join(sys.argv[1:])
    return input("Enter a prompt for OpenAI via llm_handler: ")


def main() -> None:
    # Load env vars from .env if available.
    if load_dotenv is not None:
        load_dotenv()

    if llm_handler is None:
        print("[ERROR] Could not import llm_handler from llm.llm_handler or backend.llm.llm_handler.")
        sys.exit(1)

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("[WARNING] OPENAI_API_KEY is not set in the environment.")
        print("llm_handler._get_openai() may still succeed if your app loads config elsewhere,")
        print("but for this standalone test it is recommended to export OPENAI_API_KEY or use .env.")

    prompt = get_prompt_from_argv().strip()
    if not prompt:
        print("[ERROR] Empty prompt; nothing to send.")
        sys.exit(1)

    # For this test we send a simple user prompt string; your handler will pass it as `input`.
    prompt_or_messages: Any = prompt

    # No tools by default; you can edit this list to test tools wiring.
    tools: List[Dict[str, Any]] = []

    print("=== Test 1: llm_handler.responses.create (OpenAI route, stream=False) ===")
    print("Model: gpt-4o-mini")
    print("Temperature: 0.2")
    print("Max output tokens: 800")
    print(f"Prompt: {prompt}")
    print("------------------------------------------------------------")

    try:
        resp = llm_handler.responses.create(
            model="gpt-4o-mini",
            input=prompt_or_messages,
            temperature=0.2,
            max_output_tokens=800,
            tools=tools,
            stream=False,
        )
    except Exception as e:
        print(f"[ERROR] llm_handler.responses.create failed: {e}")
        sys.exit(1)

    # Best-effort extraction of text and usage from a Responses-style object.
    text = getattr(resp, "output_text", "") or ""
    print("\n=== OpenAI Response via llm_handler.responses.create ===")
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
