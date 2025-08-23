from typing import Dict, Any, Optional, List
from qdrant_client.http.models import Filter, FieldCondition, MatchValue, Batch, PointIdsList, models
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from backend.embeddings.schemas import PayloadUpdateRequest
import logging
import openai
from backend.core.config import settings
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
        self.openai_client = OpenAI(api_key=settings.openai_api_key)
        
        # Ensure collection exists (robust: list and create if missing)
        try:
            existing = self.client.get_collections().collections
            if not any(c.name == collection_name for c in existing):
                logger.warning(f"Collection '{collection_name}' not found. Creating...")
                self.client.recreate_collection(
                    collection_name=collection_name,
                    vectors_config=VectorParams(size=int(settings.vector_size), distance=Distance.COSINE),
                )
            else:
                logger.debug(f"Using existing collection '{collection_name}'")
        except Exception as e:
            logger.error(f"Failed to verify/create collection '{collection_name}': {e}")
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
            logger.error(f"Failed to update payload for URL {url}: {e}")
            raise

    def generate_embeddings(self, text: str) -> List[float]:
        """
        Generate embeddings using OpenAI's API
        
        Args:
            text: Text to generate embeddings for
            
        Returns:
            List of floats representing the embedding vector
        """
        try:
            response = self.openai_client.embeddings.create(
                input=text,
                model=settings.embedding_model
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"Error generating embeddings: {str(e)}")
            raise
            
    def search_similar(
        self,
        query: str,
        limit: int = 5,
        score_threshold: float = 0.35,
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
        try:
            # Generate embeddings for the query
            query_vector = self.generate_embeddings(query)
            logger.debug(f"In search Generated embedding vector of length: {len(query_vector)}")

            # Build/accept a Qdrant filter
            if isinstance(query_filter, models.Filter):
                q_filter = query_filter
            else:
                q_filter = self._build_filter(query_filter)

            # Build search params (apply 'exact' if provided)
            search_params = models.SearchParams(exact=exact) if exact is not None else None

            # Perform the search
            search_results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                limit=limit,
                score_threshold=score_threshold,
                query_filter=q_filter,
                with_vectors=with_vectors,
                with_payload=with_payload,
                search_params=search_params,
            )
            
            # Format the results
            results = []
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
            logger.error(f"Error in search_similar: {str(e)}")
            raise

    def search_similar_by_embedding(
        self,
        query_embedding: List[float],
        limit: Optional[int] = 8,
        query_filter: Optional[Dict] = None,
        score_threshold: Optional[float] = None,
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
            logger.debug(f"Found {len(results)} results for similarity search")

            return [
                {"score": result.score, "payload": result.payload}
                for result in results
            ]
        except Exception as e:
            logger.error(f"Error searching Qdrant with embedding: {e}")
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
            logger.error(f"Failed to get chunks for URL {url}: {e}")
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
            logger.debug(f"Deleting points for URL: {url}")
            url_lower = url.lower()
            filter_by_url = Filter(
                must=[
                    FieldCondition(
                        key="url_lower",
                        match=MatchValue(value=url_lower)
                    )
                ]
            )
            logger.debug(f"Filter by URL: {filter_by_url}")

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
            logger.error(f"Failed to delete points for URL {url}: {e}")
            raise
