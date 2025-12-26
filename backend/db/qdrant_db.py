from typing import Dict, Any, Optional, List
from qdrant_client.http.models import Filter, FieldCondition, MatchValue, Batch, PointIdsList, models
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from backend.core.schemas import PayloadUpdateRequest
import logging
import openai
from backend.core.config import settings
from backend.embeddings.specs import resolve_embedding_spec
from backend.llm.llm_handler import llm_handler
from openai import OpenAI
import math

logger = logging.getLogger(__name__)

class QdrantDB:
    def __init__(self, host: str, port: int, collection_name: str):
        """
        Initialize QdrantDB connection
        
        Args:
            host: Qdrant host address
            port: Qdrant port number
            collection_name: Name of the collection to use
        """
        self.client = QdrantClient(host=host, port=port)
        self.collection_name = collection_name
        self._openai_client = None  # lazy init
        self.last_embedding_usage: Dict[str, int] = {"input_tokens": 0, "total_tokens": 0}
        
        # Ensure target exists. Use get_collection so aliases resolve correctly.
        try:
            self.client.get_collection(collection_name)
            logger.debug("Using existing collection or alias %s", collection_name)
        except Exception:
            logger.warning("Collection or alias %s not found. Creating collection %s…", collection_name, collection_name)
            try:
                self.client.create_collection(
                    collection_name=collection_name,
                    vectors_config=VectorParams(size=int(settings.vector_size), distance=Distance.COSINE),
                )
            except Exception as e:
                logger.exception("Failed to create collection %s", collection_name)
                raise

    def get_openai_client(self):
        """Lazily initialize and return the OpenAI client."""
        if self._openai_client is None:
            logger.debug("Initializing OpenAI client for embeddings")
            try:
                self._openai_client = OpenAI(api_key=settings.openai_api_key)
            except Exception as e:
                logger.exception("Failed to create OpenAI client: %s", e)
                raise
        return self._openai_client

    def _build_filter(self, query_filter: Optional[Dict]) -> Optional[models.Filter]:
        """Translate a simple dict (e.g., {"url": "...", "source": "..."})
        into a Qdrant Filter. Special-case: `url` maps to `url_lower` and is
        lower-cased before matching. Other keys are used as-is for exact matches.
        Returns None if no filter is provided.
        """
        if not query_filter:
            return None
        must: List[models.FieldCondition] = []
        for k, v in query_filter.items():
            if v is None:
                continue
            if k == "url":
                must.append(
                    models.FieldCondition(
                        key="url_lower",
                        match=models.MatchValue(value=str(v).lower()),
                    )
                )
            else:
                must.append(
                    models.FieldCondition(
                        key=k,
                        match=models.MatchValue(value=v),
                    )
                )
        return models.Filter(must=must) if must else None

    def update_payload_by_url(self, request: PayloadUpdateRequest) -> int:
        """
        Update a specific payload field for all chunks matching the given URL.
        
        Args:
            request: PayloadUpdateRequest containing URL, meta_key, and meta_value
            
        Returns:
            Number of points updated
        """
        try:
            url = request.url
            url_lower = url.lower()

            # Create filter that matches only url_lower using must clause
            filter_by_url = Filter(
                must=[
                    FieldCondition(
                        key="url_lower",
                        match=MatchValue(value=url_lower)
                    )
                ]
            )

            # Count matches via a lightweight scroll (no payload/vectors)
            offset = None
            updated_count = 0
            logger.debug("Counting points to update via scroll...")
            while True:
                points, offset = self.client.scroll(
                    collection_name=self.collection_name,
                    scroll_filter=filter_by_url,
                    with_payload=False,
                    with_vectors=False,
                    limit=1000,
                    offset=offset,
                )
                updated_count += len(points)
                if not points or offset is None:
                    break

            # Bulk update payload using a filter selector
            self.client.set_payload(
                collection_name=self.collection_name,
                payload={request.meta_key: request.meta_value},
                points=models.Filter(must=[
                    models.FieldCondition(
                        key="url_lower",
                        match=models.MatchValue(value=url_lower),
                    )
                ]),
            )

            return updated_count
        except Exception as e:
            logger.exception("Failed to update payload for URL %s: %s", url, e)
            raise

    def generate_embeddings(self, text: str) -> List[float]:
        """Generate embeddings for Qdrant queries.

        Backward-compatible behavior:
        - When `settings.embedding_model` is an OpenAI model id string
          (legacy), we continue to use the OpenAI client created via
          `get_openai_client()`.
        - Newer configs may set `embedding_model` to a provider key
          ("openai" or "gemini"); in that case we route via llm_handler.
        """
        try:
            spec = resolve_embedding_spec(settings)
            provider = spec.get("provider", "openai")
            model = spec.get("model")

            # Legacy / default path: OpenAI via local OpenAI client.
            use_legacy_openai = (
                provider == "openai"
                and isinstance(getattr(settings, "embedding_model", None), str)
                and getattr(settings, "embedding_model", None) not in ("openai", "gemini")
            )

            if use_legacy_openai:
                response = self.get_openai_client().embeddings.create(
                    input=text,
                    model=model,
                )
            else:
                if llm_handler is None:
                    raise ValueError("llm_handler is not available for provider-aware embeddings")
                kwargs: Dict[str, Any] = {
                    "provider": provider,
                    "model": model,
                    "input": text,
                }
                dims = spec.get("dimensions")
                if provider == "gemini" and isinstance(dims, int) and dims > 0:
                    kwargs["dimensions"] = dims
                response = llm_handler.embeddings.create(**kwargs)
            # Record token usage if the SDK returns it
            try:
                usage = getattr(response, "usage", None)
                if usage is None and isinstance(response, dict):
                    usage = response.get("usage")
                input_toks = 0
                total_toks = 0
                if usage is not None:
                    # usage may be an object or dict; prefer prompt/input tokens if present
                    if isinstance(usage, dict):
                        input_toks = int(usage.get("prompt_tokens") or usage.get("input_tokens") or usage.get("total_tokens") or 0)
                        total_toks = int(usage.get("total_tokens") or input_toks)
                    else:
                        pt = getattr(usage, "prompt_tokens", None)
                        it = getattr(usage, "input_tokens", None)
                        tt = getattr(usage, "total_tokens", None)
                        input_toks = int((pt or it or tt or 0))
                        total_toks = int((tt or input_toks))
                self.last_embedding_usage = {"input_tokens": input_toks, "total_tokens": total_toks}
            except Exception:
                # If anything goes wrong reading usage, fall back to zeros
                self.last_embedding_usage = {"input_tokens": 0, "total_tokens": 0}

            return response.data[0].embedding
        except Exception as e:
            logger.exception("Error generating embeddings: %s", e)
            self.last_embedding_usage = {"input_tokens": 0, "total_tokens": 0}
            raise
            
    def search_similar(
        self,
        query: str,
        limit: int = settings.top_k,
        score_threshold: float = settings.score_threshold,
        query_filter: Optional[Dict] = None,
        with_vectors: bool = False,
        with_payload: bool = True,
        exact: Optional[bool] = True,
    ) -> List[Dict[str, Any]]:
        """
        Search for similar vectors in the collection
        
        Args:
            query: Query string to search for
            limit: Maximum number of results to return
            score_threshold: Minimum similarity score (0-1)
            query_filter: Optional plain dict filter (e.g., {"url": "..."}); will be converted to Qdrant Filter.
            with_vectors: Whether to include vectors in the response
            with_payload: Whether to include payload in the response
            exact: If True (default), use exact search; if False, allow approximate (HNSW) search.
            
        Returns:
            List of search results with scores and payloads
        """
        # Removed commented-out print lines for cleanliness.
        
        try:
            # Generate embedding for the query
            logger.debug(
                "Generating embeddings for query (truncated): %s",
                (query[:settings.debug_log_truncate_chars] + "…")
                if (getattr(settings, "debug_verbose", False) and len(query) > getattr(settings, "debug_log_truncate_chars", 500))
                else query
            )
            query_embedding = self.generate_embeddings(query)

            # Convert simple dict to Qdrant Filter if provided
            qdrant_filter = self._build_filter(query_filter)

            # Prepare search parameters
            search_params = models.SearchParams(exact=exact) if exact is not None else None

            # Perform the search
            search_results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                query_filter=qdrant_filter,
                limit=limit,
                score_threshold=score_threshold,
                with_vectors=with_vectors,
                with_payload=with_payload,
                search_params=search_params
            )

            logger.debug("Qdrant search returned %d results", len(search_results))

            # Format the results
            results = []
            if getattr(settings, "debug_verbose", False):
                for i, hit in enumerate(search_results[:10]):
                    try:
                        logger.debug("Qdrant result %d score=%.4f id=%s", i + 1, hit.score, getattr(hit, 'id', 'n/a'))
                        if getattr(hit, 'payload', None) and 'text' in hit.payload:
                            preview = str(hit.payload.get('text') or '')
                            maxc = int(getattr(settings, "debug_log_truncate_chars", 500))
                            if len(preview) > maxc:
                                preview = preview[:maxc] + '…'
                            logger.debug("Qdrant retrieved preview: %s", preview[:100])
                    except Exception:
                        # don't let logging issues affect query flow
                        pass

            for hit in search_results:
                result = {
                    'id': hit.id,
                    'score': hit.score,
                    'payload': hit.payload
                }
                if with_vectors:
                    result['vector'] = hit.vector
                results.append(result)

            return results

        except Exception as e:
            logger.exception("Error in search_similar: %s", e)
            raise

    def search_similar_by_embedding(
        self,
        query_embedding: List[float],
        limit: Optional[int] = settings.top_k,
        query_filter: Optional[Dict] = None,
        score_threshold: Optional[float] = settings.score_threshold,
        with_payload: Optional[bool] = None,
        exact: Optional[bool] = None,
    ) -> List[Dict]:
        """
        Search for similar content using a precomputed embedding - internal method wher eyou already have the embeddings and just need to requery.
        Note: 'exact' is applied via SearchParams; Qdrant client's 'search' does not accept 'exact' as a top-level parameter.
        """
        try:
            # Basic validation
            if len(query_embedding) != int(settings.vector_size):
                raise ValueError(
                    f"Embedding dim {len(query_embedding)} != expected {settings.vector_size}"
                )
            if any(
                (v is None) or (isinstance(v, float) and (math.isnan(v) or math.isinf(v)))
                for v in query_embedding
            ):
                raise ValueError("Embedding contains NaN/Inf/None")

            # Set defaults for optional parameters
            score_threshold = score_threshold if score_threshold is not None else 0.35
            exact = exact if exact is not None else True
            with_payload = with_payload if with_payload is not None else True
            limit = limit if limit is not None else 8

            qdrant_filter = self._build_filter(query_filter)
            q_search_params = models.SearchParams(exact=exact)

            # Execute search with the prepared parameters
            results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                limit=limit,
                query_filter=qdrant_filter,
                score_threshold=score_threshold,
                with_payload=with_payload,
                search_params=q_search_params
            )
            logger.debug("Found %d results for similarity search", len(results))

            return [
                {"score": result.score, "payload": result.payload}
                for result in results
            ]
        except Exception as e:
            logger.exception("Error searching Qdrant with embedding: %s", e)
            raise

    def get_chunks_by_url(self, url: str, limit: int = 100) -> List[Dict]:
        """
        Retrieve chunks from Qdrant that match the given URL in the payload.
        
        Args:
            url: URL to match
            limit: Maximum number of chunks to return
            
        Returns:
            List of chunks with their payloads
        """
        try:
            url_lower = url.lower()
            filter_by_url = Filter(
                must=[
                    FieldCondition(
                        key="url_lower",
                        match=MatchValue(value=url_lower)
                    )
                ]
            )

            points, _ = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=filter_by_url,
                limit=limit,
                with_payload=True,
                with_vectors=False
            )
            
            return [
                {
                    "id": point.id,
                    "payload": point.payload
                }
                for point in points
            ]
        except Exception as e:
            logger.exception("Failed to get chunks for URL %s: %s", url, e)
            return []

    def delete_by_url(self, url: str) -> int:
        """
        Delete all points matching the given URL.
        
        Args:
            url: URL to match
            
        Returns:
            Number of points deleted
        """
        try:
            logger.debug("Deleting points for URL: %s", url)
            url_lower = url.lower()
            filter_by_url = Filter(
                must=[
                    FieldCondition(
                        key="url_lower",
                        match=MatchValue(value=url_lower)
                    )
                ]
            )
            logger.debug("Filter by URL: %s", filter_by_url)

            # Count matches with a lightweight scroll
            offset = None
            total = 0
            while True:
                points, offset = self.client.scroll(
                    collection_name=self.collection_name,
                    scroll_filter=filter_by_url,
                    with_payload=False,
                    with_vectors=False,
                    limit=1000,
                    offset=offset,
                )
                total += len(points)
                if not points or offset is None:
                    break

            if total == 0:
                return 0

            # Bulk delete using a filter selector
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=models.Filter(must=[
                    models.FieldCondition(
                        key="url_lower",
                        match=models.MatchValue(value=url_lower),
                    )
                ]),
            )

            return total
        except Exception as e:
            logger.exception("Failed to delete points for URL %s: %s", url, e)
            raise

    def count_points_by_url(self, url: str) -> int:
        """
        Count points in the collection that match the given URL by `url_lower` payload.

        Args:
            url: URL to match (case-insensitive; compared against `url_lower`).

        Returns:
            Total number of points/vectors found for the URL.
        """
        try:
            url_lower = url.lower()
            filter_by_url = Filter(
                must=[
                    FieldCondition(
                        key="url_lower",
                        match=MatchValue(value=url_lower)
                    )
                ]
            )
            offset = None
            total = 0
            while True:
                points, offset = self.client.scroll(
                    collection_name=self.collection_name,
                    scroll_filter=filter_by_url,
                    with_payload=False,
                    with_vectors=False,
                    limit=1000,
                    offset=offset,
                )
                total += len(points)
                if not points or offset is None:
                    break
            return total
        except Exception as e:
            logger.exception("Failed to count points for URL %s: %s", url, e)
            raise

    def get_chunks_by_base_url(self, base_url: str, limit: int = 100) -> List[Dict]:
        """Retrieve chunks from Qdrant that match the given base URL via `base_url_lower`.

        Args:
            base_url: Base URL to match (will be lowercased).
            limit: Maximum number of chunks to return.

        Returns:
            List of chunks with their payloads.
        """
        try:
            base_url_lower = (base_url or "").lower()
            filter_by_base = Filter(
                must=[
                    FieldCondition(
                        key="base_url_lower",
                        match=MatchValue(value=base_url_lower),
                    )
                ]
            )

            points, _ = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=filter_by_base,
                limit=limit,
                with_payload=True,
                with_vectors=False,
            )

            return [
                {"id": point.id, "payload": point.payload}
                for point in points
            ]
        except Exception as e:
            logger.exception("Failed to get chunks for base URL %s: %s", base_url, e)
            return []

    def delete_by_base_url(self, base_url: str) -> int:
        """Delete all points whose `base_url_lower` matches the given base URL.

        Args:
            base_url: Base URL to match (case-insensitive; matched against `base_url_lower`).

        Returns:
            Number of points deleted.
        """
        try:
            logger.debug("Deleting points for base URL: %s", base_url)
            base_url_lower = (base_url or "").lower()
            filter_by_base = Filter(
                must=[
                    FieldCondition(
                        key="base_url_lower",
                        match=MatchValue(value=base_url_lower),
                    )
                ]
            )
            logger.debug("Filter by base URL: %s", filter_by_base)

            # Count matches with a lightweight scroll
            offset = None
            total = 0
            while True:
                points, offset = self.client.scroll(
                    collection_name=self.collection_name,
                    scroll_filter=filter_by_base,
                    with_payload=False,
                    with_vectors=False,
                    limit=1000,
                    offset=offset,
                )
                total += len(points)
                if not points or offset is None:
                    break

            if total == 0:
                return 0

            # Bulk delete using a filter selector
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="base_url_lower",
                            match=models.MatchValue(value=base_url_lower),
                        )
                    ]
                ),
            )

            return total
        except Exception as e:
            logger.exception(
                "Failed to delete points for base URL %s: %s", base_url, e
            )
            raise

    def count_points_by_base_url(self, base_url: str) -> int:
        """Count points in the collection that match the given base URL by `base_url_lower`.

        Args:
            base_url: Base URL to match (case-insensitive; compared against `base_url_lower`).

        Returns:
            Total number of points/vectors found for the base URL.
        """
        try:
            base_url_lower = (base_url or "").lower()
            filter_by_base = Filter(
                must=[
                    FieldCondition(
                        key="base_url_lower",
                        match=MatchValue(value=base_url_lower),
                    )
                ]
            )
            offset = None
            total = 0
            while True:
                points, offset = self.client.scroll(
                    collection_name=self.collection_name,
                    scroll_filter=filter_by_base,
                    with_payload=False,
                    with_vectors=False,
                    limit=1000,
                    offset=offset,
                )
                total += len(points)
                if not points or offset is None:
                    break
            return total
        except Exception as e:
            logger.exception(
                "Failed to count points for base URL %s: %s", base_url, e
            )
            raise
