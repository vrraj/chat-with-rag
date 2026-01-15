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
    model = "gemini-embedding-001"  # adapter model or registry key
    text = "Hello from Gemini embeddings"
    dimensions = 768  # must match what your adapter/backend expects

    print(f"Requesting Gemini embedding (adapter, no normalization): model={model!r}, dimensions={dimensions}, text={text!r}")

    resp_no: Any = llm_handler.create_embedding(
        provider="gemini",
        model=model,
        input=text,
        dimensions=dimensions,
    )

    print("Raw embedding response type (no_norm):", type(resp_no))

    print(f"\nRequesting Gemini embedding (adapter, with normalization): model={model!r}, dimensions={dimensions}, text={text!r}")
    resp_yes: Any = llm_handler.create_embedding(
        provider="gemini",
        model=model,
        input=text,
        dimensions=dimensions,
        normalize_embedding=True,
    )

    print("Raw embedding response type (norm):", type(resp_yes))

    # Inspect first embedding and L2 norms before/after normalization using
    # a tiny pure-Python norm helper.
    try:
        import math

        def _first_vec(r: Any) -> list[float] | None:
            data = getattr(r, "data", None)
            if isinstance(data, list) and data:
                item0 = data[0]
                return getattr(item0, "embedding", None)
            return None

        def _l2_norm(vec: list[float]) -> float:
            s = 0.0
            for x in vec:
                try:
                    fx = float(x)
                except Exception:
                    fx = 0.0
                s += fx * fx
            return math.sqrt(s)

        v_no = _first_vec(resp_no)
        v_yes = _first_vec(resp_yes)

        if v_no is not None:
            print("First embedding length (adapter no_norm):", len(v_no))
            print("First 8 embedding values (adapter no_norm):", v_no[:8])
            print("L2 norm (adapter no_norm):", _l2_norm(v_no))
        else:
            print("No embedding found in adapter no_norm response")

        if v_yes is not None:
            print("First embedding length (adapter norm):", len(v_yes))
            print("First 8 embedding values (adapter norm):", v_yes[:8])
            print("L2 norm (adapter norm):", _l2_norm(v_yes))
        else:
            print("No embedding found in adapter norm response")

    except Exception as e:  # pragma: no cover - debug helper
        print("Error while inspecting adapter embedding responses:", e)


def test_native_via_llm_handler() -> None:
    """Test native Gemini SDK embeddings via llm_handler (gemini_native provider).

    This exercises the self-contained native embedding route in LLMHandler while
    preserving the OpenAI-style embeddings response shape (data[].embedding, usage).
    """

    # Use the registry key for the native embedding profile so routing can
    # consult endpoint and capabilities (task_type, output_dimensionality).
    model = "gemini:native-embed"
    text = "Hello from Gemini native embeddings via llm_handler"
    # These match the defaults in model_registry for gemini:native-embed but
    # are provided explicitly here to exercise call-time overrides.
    task_type = "RETRIEVAL_DOCUMENT"
    output_dimensionality = 1536

    print(f"\nRequesting Gemini native embedding via llm_handler (no normalization): model={model!r}, text={text!r}")

    # First call: no normalization
    resp_no_norm: Any = llm_handler.create_embedding(
        provider="gemini",  # endpoint=="gemini_sdk" for this key routes to native SDK
        model=model,
        input=text,
        task_type=task_type,
        output_dimensionality=output_dimensionality,
    )

    print("Raw native-embedding response type (no_norm):", type(resp_no_norm))

    # Second call: with normalize_embedding=True
    print(f"\nRequesting Gemini native embedding via llm_handler (with normalization): model={model!r}, text={text!r}")
    resp_norm: Any = llm_handler.create_embedding(
        provider="gemini",
        model=model,
        input=text,
        task_type=task_type,
        output_dimensionality=output_dimensionality,
        normalize_embedding=True,
    )

    print("Raw native-embedding response type (norm):", type(resp_norm))

    # Inspect shapes and (optionally) L2 norms before/after normalization.
    try:
        from pprint import pprint
        import numpy as np  # type: ignore

        def _get_first_embedding_vec(r: Any) -> list[float] | None:
            data = getattr(r, "data", None)
            if isinstance(data, list) and data:
                item0 = data[0]
                return getattr(item0, "embedding", None)
            return None

        v_no = _get_first_embedding_vec(resp_no_norm)
        v_yes = _get_first_embedding_vec(resp_norm)

        print("\n=== Native via llm_handler: top-level attributes (no_norm) ===")
        top_level_no = {
            "has_data": hasattr(resp_no_norm, "data"),
            "has_usage": hasattr(resp_no_norm, "usage"),
            "dir": [n for n in dir(resp_no_norm) if not n.startswith("__")],
        }
        pprint(top_level_no)

        print("\n=== Native via llm_handler: top-level attributes (norm) ===")
        top_level_yes = {
            "has_data": hasattr(resp_norm, "data"),
            "has_usage": hasattr(resp_norm, "usage"),
            "dir": [n for n in dir(resp_norm) if not n.startswith("__")],
        }
        pprint(top_level_yes)

        if v_no is not None:
            arr_no = np.asarray(v_no, dtype="float32")
            print("First embedding length (no_norm):", len(v_no))
            print("First 8 embedding values (no_norm):", v_no[:8])
            print("L2 norm (no_norm):", float(np.linalg.norm(arr_no)))
        else:
            print("No embedding found in no_norm response")

        if v_yes is not None:
            arr_yes = np.asarray(v_yes, dtype="float32")
            print("First embedding length (norm):", len(v_yes))
            print("First 8 embedding values (norm):", v_yes[:8])
            print("L2 norm (norm):", float(np.linalg.norm(arr_yes)))
        else:
            print("No embedding found in norm response")

    except Exception as e:  # pragma: no cover - debug helper
        print("Error while inspecting native llm_handler embedding responses:", e)


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
    model = "gemini-embedding-001"
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
    model = "gemini-embedding-001"  # native embedding model id; may need adjustment per SDK/docs
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
    # test_adapter_embedding(). # enable to test OpenAI compatible adapter
    test_native_via_llm_handler()
    # test_native_count_tokens() # enable to test native count_tokens
    #test_native_embedding() # enable to test native embedding without llm_handler


if __name__ == "__main__":
    main()
