from typing import Dict, Any, Optional, List
from qdrant_client.http.models import Filter, FieldCondition, MatchValue, Batch
from qdrant_client import QdrantClient
from backend.embeddings.schemas import PayloadUpdateRequest
import logging

logger = logging.getLogger(__name__)

class QdrantDB:
    def __init__(self, client: QdrantClient, collection_name: str):
        self.client = client
        self.collection_name = collection_name

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

            # Create filter that matches either url or url_lower
            filter_by_url = Filter(
                should=[
                    FieldCondition(
                        key="url",
                        match=MatchValue(value=url)
                    ),
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
                should=[
                    FieldCondition(
                        key="url",
                        match=MatchValue(value=url)
                    ),
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
            url_lower = url.lower()
            filter_by_url = Filter(
                should=[
                    FieldCondition(
                        key="url",
                        match=MatchValue(value=url)
                    ),
                    FieldCondition(
                        key="url_lower",
                        match=MatchValue(value=url_lower)
                    )
                ]
            )

            points, _ = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=filter_by_url,
                with_payload=False
            )
            
            if not points:
                return 0

            point_ids = [point.id for point in points]
            self.client.delete(
                collection_name=self.collection_name,
                points=point_ids
            )
            
            return len(point_ids)
        except Exception as e:
            logger.error(f"Failed to delete points for URL {url}: {e}")
            raise
