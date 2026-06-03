from typing import List

from backend.llm.llm_client import embed
from backend.retrieval.schemas import EmbeddingSpec, EmbeddingResult


class HostedEmbeddingProvider:
    def embed(self, texts: List[str], spec: EmbeddingSpec) -> EmbeddingResult:
        result = embed(
            model_key=spec.model,
            texts=texts,
        )
        vectors = []
        usage = {}
        if isinstance(result, dict):
            vectors = result.get("vectors") or result.get("embeddings") or result.get("data") or []
            usage = result.get("usage", {})
        else:
            vectors = getattr(result, "vectors", None) or getattr(result, "embeddings", None) or getattr(result, "data", None) or []
            usage = getattr(result, "usage", {}) or {}

        return EmbeddingResult(
            vectors=vectors,
            model=spec.model,
            dimensions=spec.dimensions,
            runtime=spec.runtime,
            usage=usage,
            metadata={
                "provider": spec.provider,
            },
        )
