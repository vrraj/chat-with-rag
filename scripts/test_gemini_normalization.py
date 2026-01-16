import math
import os
from google import genai
import numpy as np
from qdrant_client import QdrantClient

# Initialize client with API key
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    print("Error: GEMINI_API_KEY environment variable not set")
    print("Please set it with: export GEMINI_API_KEY=your_key_here")
    exit(1)

client = genai.Client(api_key=api_key)

texts = [
    "RAG systems combine retrieval with generation",
    "Gemini embeddings support multiple output dimensions",
    "Normalization matters for cosine similarity"
]

# Request embeddings
response = client.models.embed_content(
    model="models/gemini-embedding-001",
    contents=texts,
    config=genai.types.EmbedContentConfig(
        output_dimensionality=1536,
        task_type="RETRIEVAL_DOCUMENT"
    ),
)

def l2_normalize(vec):
    mag = math.sqrt(sum(x * x for x in vec))
    return [x / mag for x in vec] if mag > 0 else vec

embeddings = [
    l2_normalize(e.values)
    for e in response.embeddings
]

# Print results for manual normalization
print(f"Generated {len(embeddings)} embeddings (manual normalization)")
print(f"Embedding dimension: {len(embeddings[0])}")
print(f"First embedding (first 5 values): {embeddings[0][:5]}")
print(f"L2 norm of first embedding: {math.sqrt(sum(x * x for x in embeddings[0])):.6f}")

# Test numpy normalization
def l2_normalize_np(vec):
    v = np.asarray(vec, dtype=np.float32)
    norm = np.linalg.norm(v)
    return (v / norm).tolist() if norm > 0 else vec

embeddings_np = [
    l2_normalize_np(e.values)
    for e in response.embeddings
]

print(f"\nGenerated {len(embeddings_np)} embeddings (numpy normalization)")
print(f"Embedding dimension: {len(embeddings_np[0])}")
print(f"First embedding (first 5 values): {embeddings_np[0][:5]}")
print(f"L2 norm of first embedding: {math.sqrt(sum(x * x for x in embeddings_np[0])):.6f}")

# Compare results
diff = max(abs(a - b) for a, b in zip(embeddings[0], embeddings_np[0]))
print(f"\nMax difference between manual and numpy normalization: {diff:.10f}")

# Test Qdrant automatic normalization
def test_qdrant_normalization():
    print("\n" + "="*60)
    print("TESTING QDRANT AUTOMATIC NORMALIZATION")
    print("="*60)
    
    # Connect to Qdrant
    qdrant_client = QdrantClient(host="localhost", port=6333)
    collection_name = "document_index_gemini"
    
    try:
        # Get the first raw (non-normalized) Gemini embedding
        raw_embedding = response.embeddings[0].values
        
        # Calculate L2 norm of raw embedding
        raw_norm = math.sqrt(sum(x * x for x in raw_embedding))
        
        print(f"Raw embedding L2 norm: {raw_norm:.6f}")
        print(f"Raw embedding (first 8 values): {[round(x, 6) for x in raw_embedding[:8]]}")
        
        # Insert raw embedding into Qdrant
        test_point_id = 999999  # Use a high ID to avoid conflicts
        
        qdrant_client.upsert(
            collection_name=collection_name,
            points=[{
                "id": test_point_id,
                "vector": raw_embedding,  # Insert WITHOUT normalization
                "payload": {
                    "text": texts[0],
                    "test": "normalization_test",
                    "raw_norm": raw_norm
                }
            }]
        )
        print(f"Inserted raw embedding into Qdrant (point ID: {test_point_id})")
        
        # Retrieve the embedding from Qdrant
        retrieved = qdrant_client.retrieve(
            collection_name=collection_name,
            ids=[test_point_id],
            with_vectors=True
        )
        
        if retrieved:
            retrieved_vector = retrieved[0].vector
            retrieved_norm = math.sqrt(sum(x * x for x in retrieved_vector))
            
            print(f"Retrieved embedding L2 norm: {retrieved_norm:.6f}")
            print(f"Retrieved embedding (first 8 values): {[round(x, 6) for x in retrieved_vector[:8]]}")
            
            # Compare norms
            norm_difference = abs(raw_norm - retrieved_norm)
            print(f"\nNorm difference: {norm_difference:.10f}")
            
            if norm_difference < 1e-6:
                print("✅ Qdrant did NOT normalize the vector (norms are identical)")
            else:
                if abs(retrieved_norm - 1.0) < 1e-6:
                    print("✅ Qdrant AUTOMATICALLY normalized the vector to unit norm")
                else:
                    print(f"⚠️  Qdrant changed the norm but not to exactly 1.0")
            
            # Clean up - delete the test point
            qdrant_client.delete(
                collection_name=collection_name,
                points_selector=[test_point_id]
            )
            print(f"Cleaned up test point {test_point_id}")
            
        else:
            print("❌ Failed to retrieve the test point from Qdrant")
            
    except Exception as e:
        print(f"❌ Error during Qdrant test: {e}")
        print("Make sure Qdrant is running and the collection exists")

# Run the Qdrant test
test_qdrant_normalization()
