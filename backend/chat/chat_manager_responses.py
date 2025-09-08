# backend/chat/chat_manager_responses.py
from typing import List, Dict, Any
from fastapi import HTTPException
from openai import AsyncOpenAI

from backend.core.config import settings
from backend.embeddings.embeddings_manager import EmbeddingsManager
from backend.db.qdrant_db import QdrantDB
from backend.chat.web_search import WebSearchClient

class ChatManagerResponses:
    """
    Chat orchestrator using the OpenAI Responses API.
    - Retrieval (Qdrant) + optional web context
    - Simple prompt packing
    - Ready to add tools later (agent-style)
    """
    def __init__(self):
        self.embeddings_manager = EmbeddingsManager()
        self.qdrant_db = QdrantDB(
            host=settings.qdrant_host,
            port=settings.qdrant_port,
            collection_name=settings.collection_name,
        )
        self.web_search = WebSearchClient()
        self.history: List[Dict[str, str]] = []
        self.client = AsyncOpenAI()

    async def chat(self, message: str, use_web_search: bool = False) -> Dict[str, Any]:
        try:
            # 1) Retrieve from Qdrant
            search_ctx = await self._search_similar(message, limit=5)

            # 2) Optional web context
            web_ctx: List[Dict[str, Any]] = []
            if use_web_search:
                web_ctx = await self._get_web_context(message, search_ctx)

            combined = (search_ctx or []) + (web_ctx or [])

            # 3) Build compact context block for grounding
            def line(item: Dict[str, Any]) -> str:
                title = item.get("title") or item.get("metadata", {}).get("title") or ""
                snippet = item.get("snippet") or item.get("text") or ""
                return f"- {title}: {snippet}"
            context_block = "\n".join(line(it) for it in combined[:5])

            # 4) Responses API call (tools can be added later via tools=[...])
            kwargs = {
                "model": getattr(settings, "inference_model", getattr(settings, "CHAT_MODEL", "gpt-4o-mini")),
                "input": [
                    {"role": "system", "content": (
                        "You are a helpful assistant. Use the provided context to answer concisely and cite sources."
                    )},
                    {"role": "system", "content": f"Context:\n{context_block}"},
                    {"role": "user", "content": message},
                ],
                "temperature": getattr(settings, "inference_temperature", 0.7),
                "max_output_tokens": getattr(settings, "max_inference_output_tokens", 1000),
            }
            if getattr(settings, "inference_top_p", None) is not None:
                kwargs["top_p"] = settings.inference_top_p
            if getattr(settings, "inference_presence_penalty", None) is not None:
                kwargs["presence_penalty"] = settings.inference_presence_penalty
            if getattr(settings, "inference_frequency_penalty", None) is not None:
                kwargs["frequency_penalty"] = settings.inference_frequency_penalty

            resp = await self.client.responses.create(**kwargs)

            # 5) Extract text across common SDK shapes
            answer = (
                getattr(resp, "output_text", None)
                or (resp.output[0].content[0].text if getattr(resp, "output", None) else None)
                or (getattr(resp, "choices", [])[0].message.content if getattr(resp, "choices", None) else None)
            )
            if not answer:
                answer = "Sorry, I couldn't produce a response."

            # 6) Save thin history (optional)
            self.history.append({"role": "user", "content": message})
            self.history.append({"role": "assistant", "content": answer})

            return {"response": answer, "sources": combined}

        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Chat (Responses API) error: {e}")

    async def _search_similar(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        result = self.qdrant_db.search_similar(query, limit=limit)
        if hasattr(result, "__await__"):
            return await result
        return result

    async def _get_web_context(self, query: str, existing: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        result = self.web_search.get_additional_context(query, existing)
        if hasattr(result, "__await__"):
            return await result
        return result
