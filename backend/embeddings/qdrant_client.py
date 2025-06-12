import uuid
from datetime import datetime
from qdrant_client import QdrantClient
from qdrant_client.http import models
from typing import List, Dict, Optional
from backend.core.config import settings
from qdrant_client.http.models import Batch
import hashlib

class QdrantStorage:
    def __init__(self):
        self.client = QdrantClient(
            host=settings.qdrant_host,
            port=settings.qdrant_port
        )
        self.collection_name = settings.collection_name

    def create_collection(self):
        """
        Ensure collection exists. Does nothing if already created.
        """
        try:
            self.client.get_collection(self.collection_name)
            print(f"[INFO] Collection '{self.collection_name}' already exists. Skipping creation.")
        except Exception:
            print(f"[INFO] Creating collection '{self.collection_name}'...")
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
            print(f"[DEBUG] Deleting points with URL: {url}")
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
            print(f"[DEBUG] Successfully deleted points with URL: {url}")
        except Exception as e:
            print(f"[ERROR] Failed to delete points with URL {url}: {e}")
            raise

    def add_embeddings(self, embeddings: List[Dict], batch_size: int = 100):
        """
        Add embeddings to the collection
        Args:
            embeddings: List of dictionaries containing id, vector, and payload
            batch_size: Size of batches to process
        """
        import numpy as np

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

            for item in batch:
                if item["vector"] is None:
                    print(f"[ERROR] Vector is None for ID: {item['id']}")
                elif not isinstance(item["vector"], list):
                    print(f"[ERROR] Vector is not a list for ID: {item['id']}")
                elif not all(isinstance(x, float) for x in item["vector"]):
                    print(f"[ERROR] Vector has non-float elements for ID: {item['id']}")
                assert item["vector"] is not None, f"Missing vector for ID: {item['id']}"
                assert isinstance(item["vector"], list), f"Invalid vector format for ID: {item['id']}"
                assert all(isinstance(x, float) for x in item["vector"]), f"Vector contains non-floats for ID: {item['id']}"

            for item in batch:
                item["vector"] = list(np.array(item["vector"], dtype=np.float32))

            for point in batch:
                payload = point["payload"]
                print(f"[DEBUG] Payload for point {point['id']}: {payload}")
                # Use the full original payload as-is to retain all custom metadata fields (e.g., section_index, title, etc.)
                enriched_payload = payload
                if "url" in enriched_payload:
                    enriched_payload["url_lower"] = enriched_payload["url"].lower()
                # Insert debug print block for URL presence
                if not enriched_payload.get("url"):
                    print(f"[WARN] Missing URL in payload for ID: {point['id']}")
                else:
                    print(f"[DEBUG] URL for ID {point['id']}: {enriched_payload['url']}")
                # Automatically add/update timestamp when indexing to track freshness
                enriched_payload["updated_at"] = datetime.utcnow().isoformat()
                point["payload"] = enriched_payload
                print(f"[DEBUG] Inserting point ID: {point['id']} for URL: {enriched_payload.get('url')}")

            #print("[DEBUG] Prepared payloads with keys: text, title, description, url, date")
            #print("[DEBUG] Sample payload:", batch[0]["payload"])

            try:
                # Upsert the points directly - Qdrant will update existing points with same ID
                self.client.upsert(
                    collection_name=self.collection_name,
                    points=[
                        models.PointStruct(
                            id=point["id"],
                            vector=point["vector"],
                            payload=point["payload"]
                        )
                        for point in batch
                    ]
                )
                print(f"[DEBUG] Inserted batch of {len(batch)} vectors.")
                print("[DEBUG] Batch upsert complete.")
            except Exception as e:
                print(f"[ERROR] Failed to insert batch: {e}")
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
        return self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            limit=limit,
            query_filter=query_filter
        )

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