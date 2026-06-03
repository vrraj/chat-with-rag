from typing import List

from backend.retrieval.schemas import EmbeddingSpec, EmbeddingResult
from backend.retrieval.providers.hosted_embedding_provider import HostedEmbeddingProvider
from backend.retrieval.providers.fastembed_embedding_provider import FastEmbedEmbeddingProvider


class EmbeddingRouter:
    def __init__(self):
        self.providers = {
            "hosted": HostedEmbeddingProvider(),
            "fastembed": FastEmbedEmbeddingProvider(),
        }

    def embed(self, texts: List[str], spec: EmbeddingSpec) -> EmbeddingResult:
        provider = self.providers.get(spec.runtime)
        if provider is None:
            raise ValueError(f"Unsupported embedding runtime: {spec.runtime}")
        return provider.embed(texts, spec)
