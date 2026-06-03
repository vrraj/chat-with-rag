from typing import List

from backend.retrieval.schemas import RerankSpec, RerankResult
from backend.retrieval.providers.llm_rerank_provider import LLMRerankProvider


class RerankRouter:
    def __init__(self):
        self.providers = {
            "llm": LLMRerankProvider(),
            # later:
            # "fastembed": FastEmbedRerankProvider(),
        }

    def rerank(self, query: str, documents: List[str], spec: RerankSpec) -> RerankResult:
        provider = self.providers.get(spec.runtime)
        if provider is None:
            raise ValueError(f"Unsupported rerank runtime: {spec.runtime}")
        return provider.rerank(query, documents, spec)
