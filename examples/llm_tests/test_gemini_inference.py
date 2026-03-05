"""Simple script to sanity-check Gemini inference.

Usage:
    python examples/llm_tests/test_gemini_inference.py "Your prompt here"

Requirements:
    - Environment variable GEMINI_API_KEY must be set (e.g. via your .env)
    - Environment variable GEMINI_OPENAI_BASE_URL must point to your Gemini OpenAI adapter base URL
    - Package `openai` must be installed, e.g.:
        pip install openai
"""

import os
import sys
from typing import Optional

# Make the project root importable so we can resolve `llm` and `backend` when
# this file is executed as a script from any working directory.
# __file__ -> examples/llm_tests/test_gemini_inference.py
# Going up three levels lands at the repo root: chat-with-rag
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

try:
    # Prefer python-dotenv if available so the script can read from .env directly.
    from dotenv import load_dotenv
except Exception:  # pragma: no cover
    load_dotenv = None  # type: ignore[assignment]

try:
    from openai import OpenAI
except Exception:  # pragma: no cover
    OpenAI = None  # type: ignore[assignment]

# Optional: import the shared LLM client so we can test the Gemini path via the same abstraction
llm_client_available = True
_llm_import_error: Optional[BaseException] = None
try:  # pragma: no cover
    from backend.llm.llm_client import generate
except Exception as e:  # pragma: no cover
    _llm_import_error = e
    llm_client_available = False


def get_prompt_from_argv() -> str:
    if len(sys.argv) > 1:
        return " ".join(sys.argv[1:])
    return input("Enter a prompt for Gemini: ")

def main() -> None:
    # Load variables from .env if python-dotenv is installed.
    if load_dotenv is not None:
        load_dotenv()

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("[ERROR] GEMINI_API_KEY is not set in the environment.")
        print("Ensure your .env exports GEMINI_API_KEY before running this script.")
        sys.exit(1)

    base_url = os.getenv("GEMINI_OPENAI_BASE_URL")
    if not base_url:
        print("[ERROR] GEMINI_OPENAI_BASE_URL is not set in the environment.")
        print("It should point to your Gemini OpenAI-compatible adapter base URL.")
        sys.exit(1)

    if OpenAI is None:
        print("[ERROR] openai package is not installed.")
        print("Install it with: pip install openai")
        sys.exit(1)

    # Allow override via env; otherwise use a reasonable default model name for your adapter.
    model_name = os.getenv("GEMINI_MODEL_NAME", "models/gemini-2.5-flash")

    prompt = get_prompt_from_argv().strip()
    if not prompt:
        print("[ERROR] Empty prompt; nothing to send.")
        sys.exit(1)

    # Debug: print key (masked), base URL, and model name so we can verify configuration.
    masked_key = api_key[:4] + "..." + api_key[-4:] if len(api_key) > 8 else "<short>"
    print(f"Using GEMINI_API_KEY: {masked_key}")
    print(f"Using adapter base URL: {base_url}")
    print(f"Using model: {model_name}")
    print("Sending prompt to Gemini via OpenAI chat.completions adapter...\n")

    try:
        client = OpenAI(api_key=api_key, base_url=base_url)
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
        )
    except Exception as e:
        print(f"[ERROR] Gemini (adapter) call failed: {e}")
        sys.exit(1)

    # Extract text from the first choice, OpenAI Chat Completions-style.
    text: str = ""
    try:
        if response and getattr(response, "choices", None):
            choice0 = response.choices[0]
            if hasattr(choice0, "message") and getattr(choice0.message, "content", None):
                text = choice0.message.content
    except Exception:
        pass

    print("=== Gemini Response (via chat.completions adapter) ===")
    print(text or "<no text returned>")

    usage = getattr(response, "usage", None)
    if usage is not None:
        print("\n=== Usage (best-effort) ===")
        for field in ("prompt_tokens", "completion_tokens", "total_tokens"):
            val = getattr(usage, field, None) if not isinstance(usage, dict) else usage.get(field)
            if val is not None:
                print(f"{field}: {val}")

    # --- Second test: call Gemini via shared llm_client, if available ---
    if llm_client_available:
        print("\n==============================")
        print("Now calling Gemini via llm_client.generate (model_key)")
        print("==============================\n")

        try:
            # Use llm_client.generate with normalized response
            client_resp = generate(
                model_key="gemini:openai-2.5-flash-lite",
                input=prompt,
                stream=False,
            )
        except Exception as e:
            print(f"[ERROR] llm_client.generate failed: {e}")
        else:
            # Extract text from normalized response
            text2: str = client_resp.get("text", "") or ""

            print("=== Gemini Response via llm_client (normalized) ===")
            print(text2 or "<no text returned>")
    else:
        print("\n[INFO] llm_client is not importable; skipping client-based Gemini test.")
        if _llm_import_error is not None:
            print(f"[DEBUG] Last llm_client import error: {_llm_import_error}")


if __name__ == "__main__":
    main()
