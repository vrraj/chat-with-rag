import os
import sys
import logging

# Make the project root importable so we can resolve `llm` and `backend` when
# this file is executed as a script from any working directory.
# __file__ -> examples/llm_tests/test_llm_client_embeddings.py
# Going up three levels lands at the repo root: chat-with-rag
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from backend.llm.llm_client import embed

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """Simple smoke test for llm_client.embed using OpenAI.

    Requirements:
      - OPENAI_API_KEY must be set in the environment.
      - The specified embedding model must be available to your key.
    """

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        logger.error("OPENAI_API_KEY is not set; cannot run embedding test")
        return

    # Default to a common OpenAI embedding model; override via env if desired.
    model = os.getenv("TEST_EMBEDDING_MODEL", "text-embedding-3-small")
    text = os.getenv("TEST_EMBEDDING_TEXT", "Hello from llm_client.embed test")

    logger.info("Testing llm_client.embed with model=%s", model)

    try:
        resp = embed(
            model_key="openai:embed_small",
            texts=text
        )
    except Exception as e:
        logger.exception("Embedding call failed: %s", e)
        return

    # Expect OpenAI-style response: resp.data[0].embedding and optional resp.usage
    try:
        data = getattr(resp, "data", None) or []
        if not data:
            logger.error("No data field on embedding response: %r", resp)
            return
        embedding = getattr(data[0], "embedding", None)
        if embedding is None:
            logger.error("No embedding field on first data item: %r", data[0])
            return

        logger.info("Embedding vector length: %d", len(embedding))
        try:
            # Log a small prefix of the vector for inspection (without flooding logs).
            sample_size = min(8, len(embedding))
            sample = embedding[:sample_size]
            logger.info("Embedding sample (first %d values): %s", sample_size, sample)
        except Exception as e:
            logger.warning("Failed to log embedding sample: %s", e)

        usage = getattr(resp, "usage", None)
        if usage is not None:
            # usage may be an object or dict; handle both.
            prompt_tokens = getattr(usage, "prompt_tokens", None)
            total_tokens = getattr(usage, "total_tokens", None)
            if isinstance(usage, dict):
                prompt_tokens = usage.get("prompt_tokens", prompt_tokens)
                total_tokens = usage.get("total_tokens", total_tokens)
            logger.info(
                "Usage: prompt_tokens=%s total_tokens=%s",
                prompt_tokens,
                total_tokens,
            )
    except Exception as e:
        logger.exception("Failed to inspect embedding response: %s", e)
        return

    logger.info("llm_client.embed OpenAI smoke test completed successfully")


if __name__ == "__main__":
    main()
