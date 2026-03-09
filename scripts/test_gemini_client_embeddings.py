"""Test script for llm_client Gemini embeddings route.

Usage:
    python scripts/test_gemini_client_embeddings.py "Your text here"

Requirements:
    - GEMINI_API_KEY must be set (e.g., via .env)
    - The adapter must support Gemini embeddings

This exercises the call shape:

    llm_client.embed(
        model_key="gemini:openai-2.5-flash-lite",
        model="gemini-embedding-001",
        input=text,
        # dimensions defaults to 1536 if not provided
    )
"""

import os
import sys
import logging
from typing import Any

# Make the project root importable so we can resolve `llm` and `backend` when
# this file is executed as a script from any working directory.
# __file__ -> examples/llm_tests/test_gemini_embeddings.py
# Going up three levels lands at the repo root: chat-with-rag
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from backend.llm.llm_client import embed

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_text_from_argv() -> str:
    if len(sys.argv) > 1:
        return " ".join(sys.argv[1:])
    return input("Enter text to embed with Gemini: ")


def main() -> None:
    """Simple smoke test for llm_client.embed using Gemini.

    Requirements:
      - GEMINI_API_KEY must be set in the environment.
      - The specified embedding model must be available to your key.
    """

    api_key = os.getenv("GEMINI_API_KEY")
    base_url = os.getenv("GEMINI_OPENAI_BASE_URL")
    if not api_key:
        logger.error("GEMINI_API_KEY is not set; cannot run Gemini embedding test")
        return
    if not base_url:
        logger.error("GEMINI_OPENAI_BASE_URL is not set; cannot run Gemini embedding test")
        return

    # Model and dimensions: align with backend/core/config.py defaults.
    model = os.getenv("GEMINI_EMBEDDING_MODEL", "gemini-embedding-001")
    # Optional override for dimensions; if unset, llm_client defaults to 1536.
    dim_env = os.getenv("GEMINI_EMBEDDING_DIMENSIONS", "")
    dimensions: Any
    if dim_env.strip():
        try:
            dimensions = int(dim_env)
        except ValueError:
            logger.warning("Invalid GEMINI_EMBEDDING_DIMENSIONS=%r; falling back to client default", dim_env)
            dimensions = None
    else:
        dimensions = None

    text = get_text_from_argv().strip()
    if not text:
        logger.error("Empty text; nothing to embed")
        return

    logger.info("Testing llm_client.embed with provider='gemini', model=%s", model)

    kwargs: dict[str, Any] = {
        "provider": "gemini",
        "model": model,
        "input": text,
    }
    if isinstance(dimensions, int) and dimensions > 0:
        kwargs["dimensions"] = dimensions

    try:
        resp = embed(
            model_key="gemini:native-embed",
            texts=text,
            dimensions=dimensions
        )
    except Exception as e:
        logger.exception("Gemini embedding call failed: %s", e)
        return

    try:
        data = getattr(resp, "data", None) or []
        if not data:
            logger.error("No data field on embedding response: %r", resp)
            return
        embedding = getattr(data[0], "embedding", None)
        if embedding is None:
            logger.error("No embedding field on first data item: %r", data[0])
            return

        logger.info("Gemini embedding vector length: %d", len(embedding))
        sample_size = min(8, len(embedding))
        logger.info("Gemini embedding sample (first %d values): %s", sample_size, embedding[:sample_size])
    except Exception as e:
        logger.exception("Failed to inspect Gemini embedding response: %s", e)
        return

    usage = getattr(resp, "usage", None)
    if usage is not None:
        try:
            prompt_tokens = getattr(usage, "prompt_tokens", None)
            total_tokens = getattr(usage, "total_tokens", None)
            if isinstance(usage, dict):
                prompt_tokens = usage.get("prompt_tokens", prompt_tokens)
                total_tokens = usage.get("total_tokens", total_tokens)
            logger.info("Gemini embedding usage: prompt_tokens=%s total_tokens=%s", prompt_tokens, total_tokens)
        except Exception as e:
            logger.warning("Failed to log Gemini embedding usage: %s", e)

    logger.info("llm_client.embed Gemini smoke test completed successfully")


if __name__ == "__main__":
    main()
