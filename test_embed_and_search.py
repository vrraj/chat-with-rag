import os
from dotenv import load_dotenv
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

# 1. Load API key and initialize clients
load_dotenv()
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
qdrant = QdrantClient(host="localhost", port=6333)

# 2. Example texts to store
documents = [
    "ChatGPT is useful for writing and coding.",
    "Qdrant is a high-performance vector database.",
    "The Inca Trail leads to Machu Picchu in Peru.",
    "FastAPI helps you build web APIs quickly.",
    "OpenAI develops AI models like GPT-4."
]

# 3. Generate embeddings
response = openai_client.embeddings.create(
    input=documents,
    model="text-embedding-3-small"
)

# embeddings = [item.embedding for item in response.data]
embeddings = []
for i, item in enumerate(response.data):
    vector = item.embedding
    embeddings.append(vector)
    print(f"\n🔢 Embedding for: \"{documents[i]}\"")
    print(f"Length: {len(vector)}")
    print(vector[:10], "...")  # Print first 10 dimensions for readability

# 4. Create/reset collection in Qdrant
collection = "multi_text_demo"
qdrant.recreate_collection(
    collection_name=collection,
    vectors_config=VectorParams(size=len(embeddings[0]), distance=Distance.COSINE)
)

# 5. Upload embeddings with payloads
points = [
    PointStruct(id=i, vector=embeddings[i], payload={"text": documents[i]})
    for i in range(len(documents))
]
qdrant.upsert(collection_name=collection, points=points)

# 6. Search for the most similar to this new query
query = "What tool helps build Python APIs fast?"
query_vector = openai_client.embeddings.create(
    input=[query],
    model="text-embedding-3-small"
).data[0].embedding

results = qdrant.search(
    collection_name=collection,
    query_vector=query_vector,
    limit=3,
    with_payload=True
)

# 7. Print matches
print("🔍 Top matches:")
for hit in results:
    print(f"Score: {hit.score:.4f}, Text: {hit.payload['text']}")
