import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from qdrant_client import QdrantClient
from qdrant_client.http import models
from backend.core.config import settings

def create_collection():
    client = QdrantClient(host=settings.qdrant_host, port=settings.qdrant_port)
    collection_name = settings.collection_name
    vector_size = settings.vector_size  # Make sure this is defined in config.py
    print(f"[INFO] Collection attributes from config.py: '{collection_name}' with vector size {vector_size}")
    try:
        # Check if collection exists
        client.get_collection(collection_name=collection_name)
        user_input = input(f"[INFO] Collection '{collection_name}' already exists. Overwrite? (y/N): ").strip().lower()
        if user_input != 'y':
            print("[INFO] Aborting. Collection not modified.")
            return
        print(f"[DEBUG] Recreating collection '{collection_name}' with vector size {vector_size}")
        client.recreate_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(
                size=vector_size,
                distance=models.Distance.COSINE
            ),
            on_disk_payload=True
        )
        print(f"[SUCCESS] Collection '{collection_name}' recreated.")

    except Exception as e:
        print(f"[DEBUG] Collection '{collection_name}' does not exist or error occurred: {e}")
        print(f"[DEBUG] Creating collection '{collection_name}' with vector size {vector_size}")
        client.recreate_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(
                size=vector_size,
                distance=models.Distance.COSINE
            ),
            on_disk_payload=True
        )
        print(f"[SUCCESS] Collection '{collection_name}' created.")

if __name__ == "__main__":
    create_collection()