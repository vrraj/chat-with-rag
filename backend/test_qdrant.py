from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

def test_qdrant_upsert_and_search():
    print("Connecting to Qdrant...")
    client = QdrantClient(host="localhost", port=6333)

    print("Deleting existing collection (if any)...")
    client.delete_collection(collection_name="test_vectors")

    print("Recreating test collection...")
    client.recreate_collection(
        collection_name="test_vectors",
        vectors_config=VectorParams(size=4, distance=Distance.COSINE)
    )

    print("Verifying collection config...")
    info = client.get_collection("test_vectors")
    print("Available collections:", client.get_collections())
    print("✅ Actual vector config:", info.config.params.vectors)
    assert info.config.params.vectors.size == 4

    print("Upserting a test point...")
    upsert_result = client.upsert(
        collection_name="test_vectors",
        points=[
            PointStruct(
                id=1,
                vector=[0.1, 0.2, 0.3, 0.4],
                payload={"label": "test"}
            )
        ],
        wait=True
    )
    print("Upsert result:", upsert_result)

    print("Searching for the test point...")
    results = client.search(
        collection_name="test_vectors",
        query_vector=[0.1, 0.2, 0.3, 0.4],
        limit=1,
        with_vectors=True
    )
    print("Search results:", results)

    print("Running assertions...")
    assert len(results) == 1
    assert results[0].id == 1
    assert results[0].payload["label"] == "test"
    assert results[0].vector is not None

    print("Test completed successfully.")
