#!/usr/bin/env python3
"""Single-call OpenAI Responses+tools test.

This script sends ONE OpenAI Responses API-style request via the llm-adapter package, including:

- your natural-language question as the input
- the full list of registered tools from `backend.tools.list_tools()`

It lets you verify whether the model plans tool usage in a single
inference step for questions such as:
  "Where is Mount Kilimanjaro and what is the current weather there?"

Usage (from repo root):

  python -m scripts.test_openai_tools_single_call \
      --question "Where is Mount Kilimanjaro and what is the current weather there?" \
      --model gpt-4.1

The script will print:
- the question
- which model was used
- which tools were advertised to the model
- the final text answer
- any tool calls the model decided to emit
- token usage and finish reason
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Any, Dict, List

# Ensure project root is on PYTHONPATH
THIS_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from backend.llm.llm_client import generate
from backend.tools import list_tools  # type: ignore


def _build_responses_input(question: str) -> List[Dict[str, Any]]:
    """Build a minimal Responses API `input` payload for a user question.

    We use the new Responses content format: a list of messages, each with
    `role` and `content`, where `content` is a list of blocks. Here we just
    send a single `input_text` block.
    """

    return [
        {
            "role": "user",
            "content": [
                {
                    "type": "input_text",
                    "text": question,
                }
            ],
        }
    ]


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Single OpenAI Responses call with tools to test complex questions "
            "(e.g., location + current weather)."
        )
    )
    parser.add_argument(
        "--question",
        type=str,
        required=True,
        help=(
            "Natural-language question to send as the single user message, "
            "e.g. 'Where is Mount Kilimanjaro and what is the current weather there?'"
        ),
    )
    parser.add_argument(
        "--model",
        type=str,
        default=os.getenv("OPENAI_RESPONSES_MODEL", "gpt-4.1"),
        help=(
            "OpenAI model identifier to use with the Responses API "
            "(default: env OPENAI_RESPONSES_MODEL or 'gpt-4.1')."
        ),
    )
    parser.add_argument(
        "--max-output-tokens",
        type=int,
        default=800,
        help="Maximum output tokens to request from the model (default: 800).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.2,
        help="Sampling temperature (default: 0.2).",
    )

    args = parser.parse_args()

    question: str = args.question.strip()
    if not question:
        print("Question cannot be empty.")
        return 1

    model: str = args.model.strip()

    # Prepare input and tools list
    input_payload = _build_responses_input(question)
    tools = list_tools()

    print("================ TEST: Single OpenAI Responses Call with Tools ================")
    print(f"Model:    {model}")
    print(f"Question: {question}")
    print("\nTools advertised to the model:")
    for t in tools:
        name = t.get("name") or t.get("function", {}).get("name")
        desc = t.get("description") or t.get("function", {}).get("description")
        print(f"  - {name}: {desc}")

    print("\nCalling OpenAI via llm_client.generate(...) in ONE request...\n")

    # Use llm_client.generate for normalized response
    resp = generate(
        model_key=model,
        input=input_payload,
        tools=tools,
        max_output_tokens=int(args.max_output_tokens),
        temperature=float(args.temperature),
    )

    # generate() already returns normalized response
    text = str(resp.get("text") or "")
    finish_reason = resp.get("finish_reason")
    usage = resp.get("usage") or {}
    tool_calls = resp.get("tool_calls") or []

    print("================ MODEL ANSWER ================\n")
    print(text)
    print("\n================ METADATA ================")
    print(f"Finish reason: {finish_reason}")
    print("Usage (canonical tokens):")
    for k in [
        "input_tokens",
        "cached_tokens",
        "output_tokens",
        "reasoning_tokens",
        "completion_tokens",
        "total_tokens",
    ]:
        v = usage.get(k)
        print(f"  {k}: {v}")

    print("\nTool calls inferred by the model (single-step planning):")
    if not tool_calls:
        print("  (none)")
    else:
        for i, tc in enumerate(tool_calls, start=1):
            print(f"  [{i}] name={tc.get('name')} id={tc.get('id')}")
            print(f"      args={tc.get('args')}")

    print("\nDone.")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
