import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict

# Ensure project root (one level up from scripts/) is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.llm.llm_handler import llm_handler

logger = logging.getLogger(__name__)


def run_test(
    provider: str,
    model: str,
    prompt: str,
    stream: bool,
    debug_thoughts: bool,
    max_output_tokens: int | None,
    reasoning_effort: str | None,
) -> None:
    """Call LLMHandler.create and print both raw response and LLMResult-like view.

    This script does NOT change llm_handler behavior; it only exercises it.
    """

    provider = (provider or "openai").strip().lower()
    kwargs: Dict[str, Any] = {
        "model": model,
        "input": prompt,
        "stream": stream,
    }

    if max_output_tokens is not None:
        kwargs["max_output_tokens"] = max_output_tokens
    if reasoning_effort is not None:
        kwargs["reasoning_effort"] = reasoning_effort

    # Adapter-level flag used by Gemini path; safe to pass always.
    if debug_thoughts:
        kwargs["debug_thoughts"] = True

    print("=== LLMHandler.create call ===")
    print(f"provider={provider} model={model} stream={stream} debug_thoughts={debug_thoughts}")

    if provider == "openai":
        # For openai we call llm_handler.create directly; chat_manager uses responses facade,
        # but this path is simpler for manual testing.
        resp = llm_handler.create(provider="openai", **kwargs)

        if stream:
            print("--- Streaming events ---")
            for ev in resp:  # type: ignore[assignment]
                # AdapterEvent has .type and optional .delta
                etype = getattr(ev, "type", None)
                delta = getattr(ev, "delta", None)
                print(json.dumps({"type": etype, "delta": delta}))
        else:
            print("--- Raw OpenAI Responses-like object ---")
            # Best-effort summary of the object
            summary = {
                "type": str(type(resp)),
                "model": getattr(resp, "model", None),
                "id": getattr(resp, "id", None),
                "output_text": getattr(resp, "output_text", None),
                "usage": getattr(resp, "usage", None),
                "finish_reason": getattr(resp, "finish_reason", None),
            }
            print(json.dumps(summary, default=str, indent=2))

            # Build LLMResult for inspection without changing handler contract.
            from backend.llm.llm_handler import LLMHandler  # type: ignore

            # Use the singleton instance's helper via its class to avoid changing imports.
            # NOTE: _build_llm_result_from_openai is an internal helper; this script is for
            # local debugging only.
            result = llm_handler._build_llm_result_from_openai(resp, provider="openai")  # type: ignore[attr-defined]
            print("--- LLMResult view ---")
            print(json.dumps(result, default=str, indent=2))

    elif provider == "gemini":
        # Provider-aware path: use llm_handler.create which routes into _gemini_call
        resp = llm_handler.create(provider="gemini", **kwargs)

        if stream:
            print("--- Streaming events ---")
            for ev in resp:  # type: ignore[assignment]
                etype = getattr(ev, "type", None)
                delta = getattr(ev, "delta", None)
                print(json.dumps({"type": etype, "delta": delta}))
        else:
            print("--- OpenAI-compatible adapter response summary ---")
            summary = {
                "type": str(type(resp)),
                "model": getattr(resp, "model", None),
                "id": getattr(resp, "id", None),
                "output_text": getattr(resp, "output_text", None),
                "usage": getattr(resp, "usage", None),
                "finish_reason": getattr(resp, "finish_reason", None),
            }
            print(json.dumps(summary, default=str, indent=2))

            # For Gemini via adapter, we can still reuse the same helper since the
            # adapter returns Responses-compatible objects.
            result = llm_handler._build_llm_result_from_openai(resp, provider="gemini")  # type: ignore[attr-defined]
            print("--- LLMResult view ---")
            print(json.dumps(result, default=str, indent=2))

    else:
        raise SystemExit(f"Unsupported provider: {provider}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Manual test harness for LLMHandler.create")
    parser.add_argument("--provider", default="openai", help="Provider: openai or gemini")
    parser.add_argument("--model", required=True, help="Model identifier (registry key or native name)")
    parser.add_argument("--prompt", default="Hello from test_llm_handler_responses", help="Prompt text")
    parser.add_argument("--stream", action="store_true", help="Use streaming responses")
    parser.add_argument("--debug-thoughts", action="store_true", help="Request debug_thoughts (Gemini reasoning models)")
    parser.add_argument("--max-output-tokens", type=int, default=None, help="Max output tokens")
    parser.add_argument("--reasoning-effort", type=str, default=None, help="Reasoning effort value (e.g., low, medium, high)")
    parser.add_argument("--log-level", default="INFO", help="Logging level")

    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))

    run_test(
        provider=args.provider,
        model=args.model,
        prompt=args.prompt,
        stream=args.stream,
        debug_thoughts=args.debug_thoughts,
        max_output_tokens=args.max_output_tokens,
        reasoning_effort=args.reasoning_effort,
    )


if __name__ == "__main__":
    main()
