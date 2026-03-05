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

from backend.llm.llm_client import generate

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
    """Call llm_client.generate and print both raw response and LLMResult-like view.

    This script does NOT change llm_client behavior; it only exercises it.
    """

    provider = (provider or "openai").strip().lower()
    kwargs: Dict[str, Any] = {}

    if max_output_tokens is not None:
        kwargs["max_output_tokens"] = max_output_tokens
    if reasoning_effort is not None:
        kwargs["reasoning_effort"] = reasoning_effort

    # Adapter-level flag used by Gemini path; safe to pass always.
    if debug_thoughts:
        kwargs["debug_thoughts"] = True

    print("=== Using llm_client.generate call ===")
    print(f"model_key={model} stream={stream} debug_thoughts={debug_thoughts}")

    # Convert provider-specific model to model_key format
    model_key = f"{provider}:{model}" if ":" not in model else model
    
    # Use llm_client.generate (normalized response)
    resp = generate(model_key=model_key, input=prompt, stream=stream, **kwargs)

    if stream:
        print("--- Streaming events ---")
        for ev in resp:  # type: ignore[assignment]
            # AdapterEvent has .type and optional .delta
            etype = getattr(ev, "type", None)
            delta = getattr(ev, "delta", None)
            print(json.dumps({"type": etype, "delta": delta}))
    else:
        print("--- Normalized LLMResult ---")
        # generate() returns normalized response
        summary = {
            "type": str(type(resp)),
            "text": resp.get("text"),
            "reasoning": resp.get("reasoning"),
            "usage": resp.get("usage"),
            "finish_reason": resp.get("finish_reason"),
        }
        print(json.dumps(summary, default=str, indent=2))

    # Note: generate() already returns normalized response, no need for additional processing


def main() -> None:
    parser = argparse.ArgumentParser(description="Manual test harness for llm_client.generate")
    parser.add_argument("--provider", default="openai", help="Provider: openai or gemini")
    parser.add_argument("--model", required=True, help="Model identifier (registry key or native name)")
    parser.add_argument("--prompt", default="Hello from test_llm_client_responses", help="Prompt text")
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
