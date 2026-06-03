from abc import ABC, abstractmethod
from typing import List
from backend.retrieval.schemas import EmbeddingSpec, EmbeddingResult, RerankSpec, RerankResult


class EmbeddingProvider(ABC):
    @abstractmethod
    def embed(self, texts: List[str], spec: EmbeddingSpec) -> EmbeddingResult:
        pass


class RerankProvider(ABC):
    @abstractmethod
    def rerank(self, query: str, documents: List[str], spec: RerankSpec) -> RerankResult:
        pass
