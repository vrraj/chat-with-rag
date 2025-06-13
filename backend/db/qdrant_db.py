from typing import Dict, Any, Optional, List
from qdrant_client.http.models import Filter, FieldCondition, MatchValue, Batch, PointIdsList, models
from qdrant_client import QdrantClient
from backend.embeddings.schemas import PayloadUpdateRequest
import logging
import openai
from backend.core.config import settings

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
        self.openai_client = openai.OpenAI(api_key=settings.openai_api_key)
        
        # Ensure collection exists
        if not self.client.get_collection(collection_name=collection_name):
            logger.warning(f"Collection {collection_name} does not exist. Creating...")
            self.client.create_collection(
                collection_name=collection_name,
                vector_size=settings.vector_size,  # Default for OpenAI embeddings
                distance="Cosine"
            )

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

            offset = None
            updated_count = 0
            seen_ids = set()
            logger.debug("Starting payload update scroll loop...")
            while True:
                points, offset = self.client.scroll(
                    collection_name=self.collection_name,
                    scroll_filter=filter_by_url,
                    with_payload=True,
                    offset=offset
                )
                logger.debug(f"Scroll returned {len(points)} points. Next offset: {offset}")

                for point in points:
                    if point.id in seen_ids:
                        logger.warning(f"Repeated point ID: {point.id}")
                    else:
                        seen_ids.add(point.id)
                    logger.debug(f"[DEBUG] Read point ID {point.id} with payload: {point.payload}")
                    self.client.set_payload(
                        collection_name=self.collection_name,
                        payload={request.meta_key: request.meta_value},
                        points=[point.id]
                    )
                    logger.debug(f"[DEBUG] Updated point ID {point.id} with {request.meta_key}: {request.meta_value}")
                    updated_count += 1

                if not points or offset is None:
                    break

            return updated_count
        except Exception as e:
            logger.error(f"Failed to update payload for URL {url}: {e}")
            raise

    def search_similar(self, query: str, limit: int = 5, query_filter: Optional[Dict] = None) -> List[Dict]:
        """
        Search for similar content using a query string
        
        Args:
            query: Search query string
            limit: Number of results to return
            query_filter: Optional Qdrant filter to narrow the search (e.g., by URL)
        
        Returns:
            List of search results with scores and content
        """
        try:
            # Generate embedding for the query
            logger.debug(f"Generating embedding for query: {query}")
            query_embedding = self.openai_client.embeddings.create(
                input=query,
                model=settings.embedding_model
            ).data[0].embedding
            logger.debug(f"Tokens used - prompt: {len(query)}, total: {len(query)}")
            logger.debug(f"Received embedding vector of length: {len(query_embedding)}")

            # Create filter if provided
            qdrant_filter = None
            if query_filter:
                url = query_filter["url"]
                url_lower = url.lower()
                qdrant_filter = models.Filter(
                    must=[
                        models.FieldCondition(
                            key="url_lower",
                            match=models.MatchValue(value=url_lower)
                        )
                    ]
                )
            
            # Perform the search
            results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                limit=limit,
                query_filter=qdrant_filter
            )
            logger.debug(f"Found {len(results)} results for similarity search")
            
            return [{
                "score": result.score,
                "payload": result.payload
            } for result in results]
        except Exception as e:
            logger.error(f"Error searching Qdrant: {e}")
            raise
        """
        Search for similar content using a query embedding
        
        Args:
            query_embedding: Embedding vector for the search query
            limit: Number of results to return
            query_filter: Optional Qdrant filter to narrow the search (e.g., by URL)
        
        Returns:
            List of search results with scores and content
        """
        try:
            qdrant_filter = None
            if query_filter:
                url = query_filter["url"]
                url_lower = url.lower()
                qdrant_filter = models.Filter(
                    must=[
                        models.FieldCondition(
                            key="url_lower",
                            match=models.MatchValue(value=url_lower)
                        )
                    ]
                )
            
            results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                limit=limit,
                qdrant_filter=qdrant_filter
            )
            logger.debug(f"Found {len(results)} results for similarity search")
            
            return [{
                "score": result.score,
                "payload": result.payload
            } for result in results]
        except Exception as e:
            logger.error(f"Error searching Qdrant: {e}")
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
            points, _ = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=filter_by_url,
                with_payload=False
            )
            logger.debug(f"Points: {points}")
            point_ids = [point.id for point in points]
            logger.debug(f"Point IDs: {point_ids}")
            if not points:
                return 0

            self.client.delete(
                collection_name=self.collection_name,
                points_selector=PointIdsList(points=point_ids)
            )
            
            return len(point_ids)
        except Exception as e:
            logger.error(f"Failed to delete points for URL {url}: {e}")
            raise
