from typing import Dict, Any, Optional, List
from qdrant_client.http.models import Filter, FieldCondition, MatchValue, Batch, PointIdsList, models
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from backend.core.schemas import PayloadUpdateRequest
import logging
from backend.core.config import settings
from backend.llm.llm_client import embed, get_model_info
import math

logger = logging.getLogger(__name__)

class QdrantDB:
    def __init__(self, host: str, port: int, collection_name: str, embedding_model_key: Optional[str] = None):
        """
        Initialize QdrantDB connection
        
        Args:
            host: Qdrant host address
            port: Qdrant port number
            collection_name: Name of the collection to use
        """
        self.client = QdrantClient(host=host, port=port)
        self.collection_name = collection_name
        self.embedding_model_key = str(embedding_model_key or settings.embedding_model_key)
        self.last_embedding_usage: Dict[str, int] = {"input_tokens": 0, "total_tokens": 0}
        
        # Ensure target exists. Use get_collection so aliases resolve correctly.
        try:
            self.client.get_collection(collection_name)
            logger.debug("Using existing collection or alias %s", collection_name)
        except Exception:
            logger.warning("Collection or alias %s not found. Creating collection %s…", collection_name, collection_name)
            try:
                vector_size = self._get_expected_dense_vector_size()
                
                # Check vector type from domain config
                vector_type = settings.vector_type
                
                if vector_type == "hybrid":
                    # Create collection with named dense + sparse vectors
                    self.client.create_collection(
                        collection_name=collection_name,
                        vectors_config={
                            "dense": VectorParams(size=vector_size, distance=Distance.COSINE, on_disk=True)
                        },
                        sparse_vectors_config={
                            "sparse": models.SparseVectorParams(
                                index=models.SparseIndexParams(on_disk=True)
                            )
                        }
                    )
                    logger.info("Created hybrid collection %s with named vectors: dense (%d) + sparse", collection_name, vector_size)
                elif vector_type == "dense":
                    # Create collection with named dense vector only
                    self.client.create_collection(
                        collection_name=collection_name,
                        vectors_config={
                            "dense": VectorParams(size=vector_size, distance=Distance.COSINE, on_disk=True)
                        }
                    )
                    logger.info("Created collection %s with named dense vector (%d)", collection_name, vector_size)
                else:
                    # Create collection with unnamed vector (existing behavior)
                    self.client.create_collection(
                        collection_name=collection_name,
                        vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
                    )
                    logger.info("Created collection %s with unnamed vector (%d)", collection_name, vector_size)
            except Exception as e:
                logger.exception("Failed to create collection %s", collection_name)
                raise

    
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

    def _get_collection_vector_capabilities(self) -> Dict[str, bool]:
        """Return whether current collection has dense and/or sparse vectors configured."""
        has_dense = False
        has_sparse = False
        try:
            collection_info = self.client.get_collection(self.collection_name)
            params = getattr(getattr(collection_info, "config", None), "params", None)
            vectors_cfg = getattr(params, "vectors", None)
            sparse_cfg = getattr(params, "sparse_vectors", None)

            if isinstance(vectors_cfg, dict):
                has_dense = bool(vectors_cfg)
            else:
                has_dense = vectors_cfg is not None

            if isinstance(sparse_cfg, dict):
                has_sparse = bool(sparse_cfg)
            else:
                has_sparse = sparse_cfg is not None
        except Exception:
            pass
        return {"has_dense": has_dense, "has_sparse": has_sparse}

    def _get_expected_dense_vector_size(self) -> int:
        """Resolve expected dense vector size.

        Priority:
        1) Model registry (llm_adapter) capabilities for self.embedding_model_key
        2) local_models_registry-derived dense config dimensions
        3) Collection vector config
        4) settings.vector_size
        """
        # 1) llm_adapter/model registry capabilities
        try:
            info = get_model_info(model_key=self.embedding_model_key)
            dims = (getattr(info, "capabilities", {}) or {}).get("dimensions") if info is not None else None
            if isinstance(dims, int) and dims > 0:
                return int(dims)
        except Exception:
            pass

        # 2) local_models_registry (via retrieval config loader)
        try:
            from backend.retrieval.config_loader import get_model_config
            dense_cfg = get_model_config("dense") or {}
            dims = dense_cfg.get("dimensions")
            if dims is not None:
                dims_i = int(dims)
                if dims_i > 0:
                    return dims_i
        except Exception:
            pass

        # 3) collection config
        try:
            collection_info = self.client.get_collection(self.collection_name)
            vectors_cfg = collection_info.config.params.vectors
            if isinstance(vectors_cfg, dict):
                dense_cfg = vectors_cfg.get("dense")
                size = getattr(dense_cfg, "size", None)
            else:
                size = getattr(vectors_cfg, "size", None)
            if isinstance(size, int) and size > 0:
                return int(size)
        except Exception:
            pass
        return int(settings.vector_size)

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
        """Generate embeddings for USER QUERIES (search/retrieval).

        Uses gemini_embed_type_query config for optimal query performance.
        Uses embedding_model_key to resolve model via model registry.
        """
        try:
            model_key = self.embedding_model_key

            if ":" not in str(model_key):
                import os
                from backend.retrieval.embedding_router import EmbeddingRouter
                from backend.retrieval.schemas import EmbeddingSpec
                from backend.retrieval.config_loader import get_model_config

                dense_config = {}
                try:
                    dense_config = get_model_config("dense")
                except Exception:
                    dense_config = {}

                cache_dir = os.getenv("FASTEMBED_CACHE_PATH", dense_config.get("cache_dir", ""))
                cache_dir = os.path.expandvars(os.path.expanduser(str(cache_dir))) if cache_dir else None

                try:
                    dims = int(dense_config.get("dimensions")) if dense_config.get("dimensions") is not None else None
                except Exception:
                    dims = None

                spec = EmbeddingSpec(
                    task="embedding",
                    runtime="fastembed",
                    provider="local",
                    model=str(model_key),
                    dimensions=dims,
                    normalize=True,
                    batch_size=32,
                    device=dense_config.get("device"),
                    extra={"cache_dir": cache_dir} if cache_dir else {},
                    vector_type="dense",
                )

                embedding_result = EmbeddingRouter().embed([text], spec)
                vectors = embedding_result.vectors or []
                if not vectors:
                    raise ValueError("No embedding vectors returned from FastEmbed provider")
                self.last_embedding_usage = {"input_tokens": 0, "total_tokens": 0}
                # Ensure we return a list, not a tuple (Qdrant query_points requires list)
                embedding = vectors[0]
                if isinstance(embedding, tuple):
                    embedding = list(embedding)
                return embedding

            provider = str(model_key).split(":", 1)[0].lower()

            kwargs: Dict[str, Any] = {
                "provider": provider,
                "model": model_key,
                "input": text,
            }
            dims = None
            try:
                model_info = get_model_info(model_key=model_key)
                dims = (getattr(model_info, "capabilities", {}) or {}).get("dimensions") if model_info is not None else None
            except Exception:
                dims = None
            if provider == "gemini" and isinstance(dims, int) and dims > 0:
                kwargs["dimensions"] = dims
                # Apply config-driven task type for user queries
                try:
                    from backend.core.config import settings as _settings  # type: ignore
                    task_type = getattr(_settings, "gemini_embed_type_query", "RETRIEVAL_QUERY")
                    kwargs["task_type"] = task_type
                    logger.debug(f"[GEMINI QUERY] Using task_type={task_type} for user query")
                except Exception:
                    pass
                # Apply the same config-driven normalization flag used for
                # indexing so that Gemini query embeddings are treated
                # consistently with content embeddings.
                try:
                    from backend.core.config import settings as _settings  # type: ignore

                    kwargs["normalize_embedding"] = bool(getattr(_settings, "gemini_embedding_normalize", False))
                except Exception:
                    pass
            # Remove provider from kwargs since it's inferred from model_key
            kwargs_for_embed = {k: v for k, v in kwargs.items() if k != "provider"}
            
            response = embed(model_key=model_key, texts=text, **kwargs_for_embed)
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

            # Handle different response structures
            embedding_data = response.data[0]
            if hasattr(embedding_data, 'embedding'):
                embedding = embedding_data.embedding
                if isinstance(embedding, tuple):
                    embedding = list(embedding)
                return embedding
            elif isinstance(embedding_data, list):
                return embedding_data
            elif isinstance(embedding_data, tuple):
                return list(embedding_data)
            else:
                raise ValueError(f"Unexpected embedding response structure: {type(embedding_data)}")
        except Exception as e:
            logger.exception("Error generating embeddings: %s", e)
            self.last_embedding_usage = {"input_tokens": 0, "total_tokens": 0}
            raise

    def generate_sparse_embeddings(self, text: str) -> Dict[str, List[float]]:
        """Generate sparse query embeddings for hybrid retrieval."""
        try:
            import os
            from backend.retrieval.embedding_router import EmbeddingRouter
            from backend.retrieval.schemas import EmbeddingSpec
            from backend.retrieval.config_loader import get_model_config

            sparse_config = get_model_config("sparse")
            sparse_model = sparse_config.get("name")
            if not sparse_model:
                raise ValueError("Sparse model name missing from retrieval config")

            cache_dir = os.getenv("FASTEMBED_CACHE_PATH", sparse_config.get("cache_dir", ""))
            cache_dir = os.path.expandvars(os.path.expanduser(str(cache_dir))) if cache_dir else None

            spec = EmbeddingSpec(
                task="embedding",
                runtime="fastembed",
                provider="local",
                model=str(sparse_model),
                dimensions=None,
                normalize=False,
                batch_size=32,
                device=sparse_config.get("device"),
                extra={"cache_dir": cache_dir} if cache_dir else {},
                vector_type="sparse",
            )

            embedding_result = EmbeddingRouter().embed([text], spec)
            vectors = embedding_result.vectors or []
            if not vectors:
                return {"indices": [], "values": []}

            sparse_vector = vectors[0]
            if not isinstance(sparse_vector, dict):
                return {"indices": [], "values": []}

            return {
                "indices": sparse_vector.get("indices") or [],
                "values": sparse_vector.get("values") or [],
            }
        except Exception as e:
            logger.exception("Error generating sparse embeddings: %s", e)
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
        """Dense-only search using direct query_points.

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
            if isinstance(query_embedding, tuple):
                query_embedding = list(query_embedding)

            # Convert simple dict to Qdrant Filter if provided
            qdrant_filter = self._build_filter(query_filter)

            # Prepare search parameters
            search_params = models.SearchParams(exact=exact) if exact is not None else None

            # Check if collection uses named vectors
            try:
                collection_info = self.client.get_collection(self.collection_name)
                vectors_config = collection_info.config.params.vectors
                has_named_vectors = isinstance(vectors_config, dict) and "dense" in vectors_config
            except Exception:
                has_named_vectors = False

            # Perform the search (Qdrant v1.18+ uses query_points instead of search)
            if has_named_vectors:
                response = self.client.query_points(
                    collection_name=self.collection_name,
                    query=query_embedding,
                    query_filter=qdrant_filter,
                    limit=limit,
                    score_threshold=score_threshold,
                    with_vectors=with_vectors,
                    with_payload=with_payload,
                    search_params=search_params,
                    using="dense"  # Use named dense vector
                )
            else:
                response = self.client.query_points(
                    collection_name=self.collection_name,
                    query=query_embedding,
                    query_filter=qdrant_filter,
                    limit=limit,
                    score_threshold=score_threshold,
                    with_vectors=with_vectors,
                    with_payload=with_payload,
                    search_params=search_params
                )
            search_results = response.points

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

    def search_similar_hybrid(
        self,
        query: str,
        limit: int = settings.top_k,
        score_threshold: Optional[float] = settings.score_threshold,
        query_filter: Optional[Dict] = None,
        with_payload: bool = True,
        exact: Optional[bool] = True,
    ) -> List[Dict[str, Any]]:
        """Hybrid search using prefetch + RRF fusion on dense + sparse vectors.

        Falls back to dense when sparse vectors missing or fusion yields no hits.
        """
        try:
            caps = self._get_collection_vector_capabilities()
            if not (caps.get("has_dense") and caps.get("has_sparse")):
                return self.search_similar(
                    query=query,
                    limit=limit,
                    score_threshold=score_threshold if score_threshold is not None else settings.score_threshold,
                    query_filter=query_filter,
                    with_payload=with_payload,
                    exact=exact,
                )

            dense_query_embedding = self.generate_embeddings(query)

            sparse_query_embedding = self.generate_sparse_embeddings(query)
            sparse_indices = sparse_query_embedding.get("indices") or []
            sparse_values = sparse_query_embedding.get("values") or []

            # If sparse query cannot be formed, gracefully fall back to dense search.
            if not sparse_indices or not sparse_values:
                return self.search_similar(
                    query=query,
                    limit=limit,
                    score_threshold=score_threshold if score_threshold is not None else settings.score_threshold,
                    query_filter=query_filter,
                    with_payload=with_payload,
                    exact=exact,
                )

            qdrant_filter = self._build_filter(query_filter)
            search_params = models.SearchParams(exact=exact) if exact is not None else None

            prefetch_dense = models.Prefetch(
                query=dense_query_embedding,
                using="dense",
                limit=limit,
                filter=qdrant_filter,
                params=search_params,
            )
            prefetch_sparse = models.Prefetch(
                query=models.SparseVector(indices=sparse_indices, values=sparse_values),
                using="sparse",
                limit=limit,
                filter=qdrant_filter,
                params=search_params,
            )

            # Do not apply dense-style score_threshold to RRF fusion scores.
            # Hybrid confidence is controlled by prefetch limits + final limit;
            # optional thresholding should happen after reranking, if needed.
            response = self.client.query_points(
                collection_name=self.collection_name,
                prefetch=[prefetch_dense, prefetch_sparse],
                query=models.FusionQuery(fusion=models.Fusion.RRF),
                limit=limit,
                with_payload=with_payload,
            )
            search_results = response.points

            # If hybrid fusion yields no hits (often due to threshold/fusion dynamics),
            # return dense results instead of empty output.
            if not search_results:
                return self.search_similar(
                    query=query,
                    limit=limit,
                    score_threshold=score_threshold if score_threshold is not None else settings.score_threshold,
                    query_filter=query_filter,
                    with_payload=with_payload,
                    exact=exact,
                )

            return [
                {
                    "id": hit.id,
                    "score": hit.score,
                    "payload": hit.payload,
                }
                for hit in search_results
            ]
        except Exception as e:
            logger.exception("Error in search_similar_hybrid: %s", e)
            raise

    def search_similar_sparse(
        self,
        query: str,
        limit: int = settings.top_k,
        score_threshold: Optional[float] = settings.score_threshold,
        query_filter: Optional[Dict] = None,
        with_payload: bool = True,
        exact: Optional[bool] = True,
    ) -> List[Dict[str, Any]]:
        """Sparse-only search using direct query_points.

        Falls back to dense when sparse vectors missing.
        """
        try:
            caps = self._get_collection_vector_capabilities()
            if not caps.get("has_sparse"):
                return self.search_similar(
                    query=query,
                    limit=limit,
                    score_threshold=score_threshold if score_threshold is not None else settings.score_threshold,
                    query_filter=query_filter,
                    with_payload=with_payload,
                    exact=exact,
                )

            sparse_query_embedding = self.generate_sparse_embeddings(query)
            sparse_indices = sparse_query_embedding.get("indices") or []
            sparse_values = sparse_query_embedding.get("values") or []

            qdrant_filter = self._build_filter(query_filter)
            search_params = models.SearchParams(exact=exact) if exact is not None else None

            response = self.client.query_points(
                collection_name=self.collection_name,
                query=models.SparseVector(indices=sparse_indices, values=sparse_values),
                using="sparse",
                query_filter=qdrant_filter,
                limit=limit,
                score_threshold=score_threshold,
                with_payload=with_payload,
                search_params=search_params,
            )
            search_results = response.points

            return [
                {
                    "id": hit.id,
                    "score": hit.score,
                    "payload": hit.payload,
                }
                for hit in search_results
            ]
        except Exception as e:
            logger.exception("Error in search_similar_sparse: %s", e)
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
        Search for similar content using a precomputed embedding - internal method where you already have the embeddings and just need to requery.
        Note: 'exact' is applied via SearchParams; Qdrant client's 'search' does not accept 'exact' as a top-level parameter.
        """
        try:
            # Basic validation
            if isinstance(query_embedding, tuple):
                query_embedding = list(query_embedding)

            expected_dim = self._get_expected_dense_vector_size()
            if len(query_embedding) != expected_dim:
                raise ValueError(
                    f"Embedding dim {len(query_embedding)} != expected {expected_dim}"
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

            # Check if collection uses named vectors
            try:
                collection_info = self.client.get_collection(self.collection_name)
                vectors_config = collection_info.config.params.vectors
                has_named_vectors = isinstance(vectors_config, dict) and "dense" in vectors_config
            except Exception:
                has_named_vectors = False

            # Execute search with the prepared parameters
            if has_named_vectors:
                results = self.client.query_points(
                    collection_name=self.collection_name,
                    query=query_embedding,
                    query_filter=qdrant_filter,
                    limit=limit,
                    score_threshold=score_threshold,
                    with_payload=with_payload,
                    search_params=q_search_params,
                    using="dense"  # Use named dense vector
                )
                points = results.points
            else:
                results = self.client.search(
                    collection_name=self.collection_name,
                    query_vector=query_embedding,
                    limit=limit,
                    query_filter=qdrant_filter,
                    score_threshold=score_threshold,
                    with_payload=with_payload,
                    search_params=q_search_params
                )
                points = results
            logger.debug("Found %d results for similarity search", len(points))

            return [
                {"score": result.score, "payload": result.payload}
                for result in points
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
