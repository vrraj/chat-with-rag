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
import os
import sys
from pathlib import Path


# Ensure project root (which contains the `backend` package) is on sys.path
_THIS_FILE = Path(__file__).resolve()
_PROJECT_ROOT = _THIS_FILE.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from backend.llm.llm_handler import llm_handler


def test_adapter_embedding() -> None:
    """Test Gemini embeddings via the OpenAI-compatible adapter (llm_handler)."""

    # Adjust these as needed for your environment / registry
    model = "text-embedding-004"  # adapter model or registry key
    text = "Hello from Gemini embeddings"
    dimensions = 768  # must match what your adapter/backend expects

    print(f"Requesting Gemini embedding (adapter): model={model!r}, dimensions={dimensions}, text={text!r}")

    resp: Any = llm_handler.create_embedding(
        provider="gemini",
        model=model,
        input=text,
        dimensions=dimensions,
    )

    # The exact response shape depends on your adapter; this is a best-effort
    # inspection that should work with OpenAI-style embeddings.create.
    print("Raw embedding response type:", type(resp))

    # Dump high-level structure
    try:
        from pprint import pprint

        print("\n=== Top-level attributes ===")
        top_level = {
            "has_data": hasattr(resp, "data"),
            "has_usage": hasattr(resp, "usage"),
            "dir": [n for n in dir(resp) if not n.startswith("__")],
        }
        pprint(top_level)

        print("\n=== usage field (if any) ===")
        usage = getattr(resp, "usage", None)
        pprint(usage)

        print("\n=== data[0] summary ===")
        data = getattr(resp, "data", None)
        if isinstance(data, list) and data:
            item0 = data[0]
            summary = {
                "type": type(item0),
                "has_embedding": hasattr(item0, "embedding"),
                "keys_or_dir": getattr(item0, "keys", None)() if hasattr(item0, "keys") else [n for n in dir(item0) if not n.startswith("__")],
            }
            pprint(summary)

            embedding = getattr(item0, "embedding", None)
            if embedding is not None:
                print("First embedding length:", len(embedding))
                print("First 8 embedding values:", embedding[:8])
        else:
            print("Response has no 'data' list; raw response:")
            pprint(resp)
    except Exception as e:  # pragma: no cover - debug helper
        print("Error while inspecting embedding response:", e)
        print("Raw response:", resp)


def test_native_count_tokens() -> None:
    """Test native Gemini SDK count_tokens for an embedding model."""

    try:
        from google import genai  # type: ignore
    except Exception as e:  # pragma: no cover - optional dependency
        print("google-genai not available; cannot run native count_tokens test:", e)
        return

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("GEMINI_API_KEY is not set; skipping native count_tokens test.")
        return

    client = genai.Client(api_key=api_key)

    # You can adjust the model to match your embedding model if desired.
    model = "text-embedding-004"
    contents = "Hello from Gemini"

    print(f"\nRequesting native count_tokens: model={model!r}, contents={contents!r}")
    try:
        resp = client.models.count_tokens(
            model=model,
            contents=contents,
        )
        # Shape depends on SDK version; most expose total_tokens.
        total_tokens = getattr(resp, "total_tokens", None)
        print("Native count_tokens response:", resp)
        print("Token count (total_tokens):", total_tokens)
    except Exception as e:
        print("Error calling native count_tokens:", e)


def test_native_embedding() -> None:
    """Test native Gemini SDK embedding and inspect response/usage."""

    try:
        from google import genai  # type: ignore
    except Exception as e:  # pragma: no cover - optional dependency
        print("google-genai not available; cannot run native embedding test:", e)
        return

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("GEMINI_API_KEY is not set; skipping native embedding test.")
        return

    client = genai.Client(api_key=api_key)

    # NOTE: The exact embedding model and method may differ depending on SDK version.
    # You may need to adjust `model` or use `client.models.embed_content`/`embed_text`.
    model = "text-embedding-004"  # native embedding model id; may need adjustment per SDK/docs
    contents = "Hello from Gemini native embedding"

    print(f"\nRequesting native embedding: model={model!r}, contents={contents!r}")
    try:
        # Best-effort call; adjust to your installed google-genai version if needed.
        resp = client.models.embed_content(
            model=model,
            contents=contents,
        )
    except Exception as e:
        print("Error calling native embed_content:", e)
        return

    try:
        from pprint import pprint

        print("Native embedding response:", resp)

        # Inspect common fields; actual structure depends on SDK version.
        print("\n=== Native embedding attributes ===")
        top_level = {
            "dir": [n for n in dir(resp) if not n.startswith("__")],
        }
        pprint(top_level)

        # Try to locate embeddings/values
        emb = getattr(resp, "embeddings", None)
        if isinstance(emb, list) and emb:
            values = getattr(emb[0], "values", None)
            if values is not None:
                print("First native embedding length:", len(values))
                print("First 8 native embedding values:", values[:8])
        # Try to locate any usage-like metadata if exposed
        usage_meta = getattr(resp, "usage_metadata", None)
        if usage_meta is not None:
            print("\nusage_metadata:")
            pprint(usage_meta)
    except Exception as e:  # pragma: no cover - debug helper
        print("Error while inspecting native embedding response:", e)


def main() -> None:
    test_adapter_embedding()
    test_native_count_tokens()
    test_native_embedding()


if __name__ == "__main__":
    main()
