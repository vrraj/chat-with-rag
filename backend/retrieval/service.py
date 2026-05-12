from typing import Any, Dict, List

from backend.retrieval.embedding_router import EmbeddingRouter
from backend.retrieval.rerank_router import RerankRouter
from backend.retrieval.schemas import EmbeddingSpec, RerankSpec, RetrievalResult


class RetrievalService:
    def __init__(self, qdrant_db):
        self.qdrant_db = qdrant_db
        self.embedding_router = EmbeddingRouter()
        self.rerank_router = RerankRouter()

    def retrieve(
        self,
        *,
        query: str,
        namespace: str,
        top_k: int,
        embedding_spec: EmbeddingSpec,
        rerank_spec: RerankSpec | None = None,
        rerank_enabled: bool = True,
    ) -> RetrievalResult:
        embedding_result = self.embedding_router.embed([query], embedding_spec)
        query_vector = embedding_result.vectors[0]

        candidates = self.qdrant_db.search_similar_by_embedding(
            query_embedding=query_vector,
            limit=top_k,
        )

        reranked = candidates

        if rerank_enabled and rerank_spec and candidates:
            docs = []
            for c in candidates:
                payload = c.get("payload") or {}
                docs.append(
                    payload.get("text")
                    or payload.get("snippet")
                    or payload.get("content")
                    or ""
                )

            rerank_result = self.rerank_router.rerank(
                query=query,
                documents=docs,
                spec=rerank_spec,
            )

            reranked = [candidates[item.index] for item in rerank_result.items]

        context_text = self._format_context(reranked)

        return RetrievalResult(
            query=query,
            rewritten_query=None,
            candidates=candidates,
            reranked=reranked,
            context_text=context_text,
            sources=self._extract_sources(reranked),
            metadata={
                "embedding": embedding_result.metadata,
                "embedding_usage": embedding_result.usage,
            },
        )

    def _format_context(self, items: List[Dict[str, Any]]) -> str:
        lines = []
        for i, item in enumerate(items or []):
            payload = item.get("payload") or {}
            text = payload.get("text") or payload.get("snippet") or payload.get("content") or ""
            section = payload.get("section") or "N/A"
            subsection = payload.get("subsection") or "N/A"
            lines.append(f"[{i+1}] {text} (Section: {section} > {subsection})")
        return "\n".join(lines)

    def _extract_sources(self, items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        sources = []
        for i, item in enumerate(items or []):
            payload = item.get("payload") or {}
            sources.append({
                "index": i + 1,
                "url": payload.get("url") or "unknown",
                "section": payload.get("section") or "N/A",
                "subsection": payload.get("subsection") or "N/A",
            })
        return sources
