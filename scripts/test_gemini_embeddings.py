"""Simple manual test for Gemini embeddings via LLMHandler.

Usage (from project root, with venv active):

    GEMINI_API_KEY=... \
    GEMINI_OPENAI_BASE_URL=... \
    python scripts/test_gemini_embeddings.py

This uses the OpenAI-compatible Gemini adapter path, not the native
`gemini_sdk` client.
"""

from __future__ import annotations

from typing import Any
import sys
from pathlib import Path


# Ensure project root (which contains the `backend` package) is on sys.path
_THIS_FILE = Path(__file__).resolve()
_PROJECT_ROOT = _THIS_FILE.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from backend.llm.llm_handler import llm_handler


def main() -> None:
    # Adjust these as needed for your environment / registry
    model = "gemini-embedding-001"  # adapter model or registry key
    text = "Hello from Gemini embeddings"
    dimensions = 768  # must match what your adapter/backend expects

    print(f"Requesting Gemini embedding: model={model!r}, dimensions={dimensions}, text={text!r}")

    resp: Any = llm_handler.create_embedding(
        provider="gemini",
        model=model,
        input=text,
        dimensions=dimensions,
    )

    # The exact response shape depends on your adapter; this is a best-effort
    # inspection that should work with OpenAI-style embeddings.create.
    print("Raw embedding response type:", type(resp))
    try:
        data = getattr(resp, "data", None)
        if isinstance(data, list) and data:
            embedding = getattr(data[0], "embedding", None)
            if embedding is not None:
                print("First embedding length:", len(embedding))
                print("First 8 embedding values:", embedding[:8])
            else:
                print("Response.data[0] has no 'embedding' attribute; raw item:", data[0])
        else:
            print("Response has no 'data' list; raw response:", resp)
    except Exception as e:  # pragma: no cover - debug helper
        print("Error while inspecting embedding response:", e)
        print("Raw response:", resp)


if __name__ == "__main__":
    main()
