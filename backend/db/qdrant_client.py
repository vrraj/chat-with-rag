import uuid
from datetime import datetime
from qdrant_client import QdrantClient
from qdrant_client.http import models
from typing import List, Dict, Optional
from backend.core.config import settings
from qdrant_client.http.models import Batch
import hashlib
import logging

logger = logging.getLogger(__name__)

class QdrantStorage:
    def __init__(self, collection_name: Optional[str] = None):
        self.client = QdrantClient(
            host=settings.qdrant_host,
            port=settings.qdrant_port
        )
        self.collection_name = str(collection_name or settings.collection_name)

    def create_collection(self):
        """
        Ensure collection exists. Does nothing if already created.
        """
        try:
            self.client.get_collection(self.collection_name)
            #logger.info("Qdrant: collection %s already exists; skipping creation", self.collection_name)
        except Exception:
            #logger.info("Qdrant: creating collection %s", self.collection_name)
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=settings.vector_size,
                    distance=models.Distance.COSINE
                ),
                on_disk_payload=True
            )
            self.client.update_collection(
                collection_name=self.collection_name,
                optimizer_config=models.OptimizersConfigDiff(
                    indexing_threshold=0,
                    memmap_threshold=10000
                )
            )
            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name="document_type",
                field_schema=models.PayloadSchemaType.KEYWORD
            )
            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name="url",
                field_schema=models.PayloadSchemaType.KEYWORD
            )
            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name="domain",
                field_schema=models.PayloadSchemaType.KEYWORD
            )
            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name="url",
                field_schema=models.PayloadSchemaType.KEYWORD
            )

    def delete_by_url(self, url: str):
        """
        Delete all points with the given URL
        Args:
            url: URL to delete points for
        """
        try:
            #logger.info("Qdrant: deleting points for URL=%s", url)
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=models.FilterSelector(
                    filter=models.Filter(
                        must=[
                            models.FieldCondition(
                                key="payload.url",
                                match=models.MatchValue(value=url)
                            )
                        ]
                    )
                )
            )
            #logger.info("Qdrant: deleted points for URL=%s", url)
        except Exception as e:
            logger.exception("Qdrant: failed to delete points for URL=%s: %s", url, e)
            raise

    def add_embeddings(self, embeddings: List[Dict], batch_size: int = 100, vector_type: Optional[str] = None):
        """
        Add embeddings to the collection
        Args:
            embeddings: List of dictionaries containing id, vector, and payload
            batch_size: Size of batches to process
            vector_type: None (unnamed), "dense", or "hybrid" (dense + sparse)
        """
        import numpy as np
        from qdrant_client import models

        def make_point_id(url: str, chunk_index: int) -> str:
            base_uuid = uuid.uuid5(uuid.NAMESPACE_URL, url)
            return str(uuid.uuid5(base_uuid, str(chunk_index)))

        # Deletion of points by URL is now handled upstream in main.py to avoid redundant deletes.
        # The following block is intentionally commented out.
        # all_urls = {item["payload"].get("url", "") for item in embeddings if item["payload"].get("url", "")}
        # print(f"[DEBUG] URLs to delete before upsert: {all_urls}")
        # for url in all_urls:
        #     self.delete_by_url(url)

        for i in range(0, len(embeddings), batch_size):
            batch = embeddings[i:i + batch_size]

            for index, item in enumerate(batch):
                url = item["payload"].get("url", "")
                item["id"] = make_point_id(url, i + index)

            # Validate vectors based on vector_type
            if vector_type == "hybrid":
                for item in batch:
                    assert "dense" in item and "sparse" in item, f"Missing dense/sparse vectors for ID: {item['id']}"
                    assert isinstance(item["dense"], list), f"Invalid dense vector format for ID: {item['id']}"
                    assert isinstance(item["sparse"], dict), f"Invalid sparse vector format for ID: {item['id']}"
                    assert "indices" in item["sparse"] and "values" in item["sparse"], f"Missing indices/values in sparse vector for ID: {item['id']}"
            elif vector_type == "dense":
                for item in batch:
                    assert "dense" in item, f"Missing dense vector for ID: {item['id']}"
                    assert isinstance(item["dense"], list), f"Invalid dense vector format for ID: {item['id']}"
            else:
                for item in batch:
                    if item["vector"] is None:
                        logger.error("Qdrant: vector is None for id=%s", item['id'])
                    elif not isinstance(item["vector"], list):
                        logger.error("Qdrant: vector is not list for id=%s", item['id'])
                    elif not all(isinstance(x, float) for x in item["vector"]):
                        logger.error("Qdrant: vector has non-floats for id=%s", item['id'])
                    assert item["vector"] is not None, f"Missing vector for ID: {item['id']}"
                    assert isinstance(item["vector"], list), f"Invalid vector format for ID: {item['id']}"
                    assert all(isinstance(x, float) for x in item["vector"]), f"Vector contains non-floats for ID: {item['id']}"

            # Convert dense vectors to float32
            for item in batch:
                if vector_type == "hybrid":
                    item["dense"] = list(np.array(item["dense"], dtype=np.float32))
                elif vector_type == "dense":
                    item["dense"] = list(np.array(item["dense"], dtype=np.float32))
                else:
                    item["vector"] = list(np.array(item["vector"], dtype=np.float32))

            for point in batch:
                payload = point["payload"]
                from pprint import pformat
                try:
                    preview = pformat(payload)
                except Exception:
                    preview = str(payload)
                #logger.debug("Qdrant: payload for %s %s", point['id'], preview[:getattr(settings, 'debug_log_truncate_chars', 500)] if getattr(settings, 'debug_verbose', False) else "<hidden>")
                # Use the full original payload as-is to retain all custom metadata fields (e.g., section_index, title, etc.)
                enriched_payload = payload
                if "url" in enriched_payload:
                    enriched_payload["url_lower"] = enriched_payload["url"].lower()
                # Insert debug print block for URL presence
                if not enriched_payload.get("url"):
                    logger.warning("Qdrant: missing URL in payload for id=%s", point['id'])
                else:
                    logger.debug("Qdrant: url for id=%s -> %s", point['id'], enriched_payload['url'])
                # Automatically add/update timestamp when indexing to track freshness
                enriched_payload["updated_at"] = datetime.utcnow().isoformat()
                point["payload"] = enriched_payload
                #logger.info("Qdrant: upserting point id=%s url=%s", point['id'], enriched_payload.get('url'))

            try:
                # Build points based on vector_type
                if vector_type == "hybrid":
                    points = [
                        models.PointStruct(
                            id=point["id"],
                            vector={
                                "dense": point["dense"],
                                "sparse": models.SparseVector(
                                    indices=point["sparse"]["indices"],
                                    values=point["sparse"]["values"]
                                )
                            },
                            payload=point["payload"]
                        )
                        for point in batch
                    ]
                elif vector_type == "dense":
                    points = [
                        models.PointStruct(
                            id=point["id"],
                            vector={"dense": point["dense"]},
                            payload=point["payload"]
                        )
                        for point in batch
                    ]
                else:
                    points = [
                        models.PointStruct(
                            id=point["id"],
                            vector=point["vector"],
                            payload=point["payload"]
                        )
                        for point in batch
                    ]
                
                # Upsert the points directly - Qdrant will update existing points with same ID
                self.client.upsert(
                    collection_name=self.collection_name,
                    points=points
                )
                #logger.info("Qdrant: inserted batch size=%d", len(batch))
                #logger.debug("Qdrant: batch upsert complete")
            except Exception as e:
                logger.exception("Qdrant: failed to insert batch: %s", e)
                return False
        return True

    def search(self, query_vector: List[float], limit: int = 5, query_filter: Optional[Dict] = None):
        """
        Search for similar embeddings
        Args:
            query_vector: Vector to search for
            limit: Number of results to return
            query_filter: Optional filter conditions
        Returns:
            List of search results with scores and payloads
        """
        # Check if collection uses named vectors
        try:
            collection_info = self.client.get_collection(self.collection_name)
            vectors_config = collection_info.config.params.vectors
            has_named_vectors = isinstance(vectors_config, dict) and "dense" in vectors_config
        except Exception:
            has_named_vectors = False

        # Perform search (Qdrant v1.18+ uses query_points instead of search)
        if has_named_vectors:
            response = self.client.query_points(
                collection_name=self.collection_name,
                query=query_vector,
                limit=limit,
                query_filter=query_filter,
                using="dense"  # Use named dense vector
            )
        else:
            response = self.client.query_points(
                collection_name=self.collection_name,
                query=query_vector,
                limit=limit,
                query_filter=query_filter
            )
        return response.points

    def delete_by_id(self, ids: List[str]):
        """
        Delete embeddings by their IDs
        Args:
            ids: List of IDs to delete
        """
        self.client.delete(
            collection_name=self.collection_name,
            points_selector=models.PointIdsList(points=ids)
        )

