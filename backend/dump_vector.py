from qdrant_client import QdrantClient
from qdrant_client.models import ScrollRequest

client = QdrantClient(host="localhost", port=6333)

# Scroll through the collection
response = client.scroll(
    collection_name="test_vectors",
    scroll_filter=None,
    with_payload=True,
    with_vectors=True,
    limit=100
)

for point in response[0]:
    print(f"ID: {point.id}")
    print(f"Vector: {point.vector}")
    print(f"Payload: {point.payload}")
    print("—" * 30)