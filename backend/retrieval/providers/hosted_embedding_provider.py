from typing import List

from backend.llm.llm_client import embed
from backend.retrieval.schemas import EmbeddingSpec, EmbeddingResult


class HostedEmbeddingProvider:
    def embed(self, texts: List[str], spec: EmbeddingSpec) -> EmbeddingResult:
        result = embed(
            model_key=spec.model,
            input=texts,
        )

        vectors = result.get("vectors") or result.get("embeddings") or []

        return EmbeddingResult(
            vectors=vectors,
            model=spec.model,
            dimensions=spec.dimensions,
            runtime=spec.runtime,
            usage=result.get("usage", {}) if isinstance(result, dict) else {},
            metadata={
                "provider": spec.provider,
            },
        )
