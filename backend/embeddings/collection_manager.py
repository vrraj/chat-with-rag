from qdrant_client import QdrantClient
from qdrant_client.http import models
from typing import Optional
from backend.core.config import settings

class CollectionManager:
    def __init__(self, client: QdrantClient):
        self.client = client
        self.collection_name = settings.collection_name
        self._collection_initialized = False

    def ensure_collection(self) -> bool:
        """Ensure the collection exists and is properly configured"""
        try:
            if not self._collection_initialized:
                try:
                    self.client.get_collection(self.collection_name)
                    print(f"[INFO] Collection '{self.collection_name}' already exists")
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
                self._collection_initialized = True
            return True
        except Exception as e:
            print(f"Error ensuring collection: {e}")
            return False
