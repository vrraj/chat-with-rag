from typing import Any, Dict, Optional

from pydantic import BaseModel, Field


class RetrievalEvalRequest(BaseModel):
    query: str
    active_domain: Optional[str] = ""
    search_mode: str = "dense"
    score_threshold: Optional[float] = 0.35
    top_k: int = 8
    query_filter: Optional[Dict[str, Any]] = None
    exact: bool = False
    with_payload: bool = True

    use_colbert: bool = False
    colbert_score_threshold: float = 0.0
    max_items_for_cross_encoder: int = 8
    reranked_top_n: int = 5


class RetrievalEvalResponse(BaseModel):
    domain: Dict[str, Any]
    retrieval: Dict[str, Any]
    colbert: Optional[Dict[str, Any]] = None
    reranked: Dict[str, Any]
    payload_echo: Dict[str, Any] = Field(default_factory=dict)
