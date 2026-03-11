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
try:
    response = client.models.embed_content(
        model="models/gemini-embedding-001",
        contents=texts,
        config=genai.types.EmbedContentConfig(
            output_dimensionality=1536,
            task_type="RETRIEVAL_DOCUMENT"
        ),
    )
    
    # Manual L2 normalization
    def l2_normalize(vec):
        mag = math.sqrt(sum(x * x for x in vec))
        return [x / mag for x in vec] if mag > 0 else vec

    embeddings_manual = [
        l2_normalize(e.values)
        for e in response.embeddings
    ]
    
    # NumPy L2 normalization
    def l2_normalize_np(vec):
        v = np.asarray(vec, dtype=np.float32)
        norm = np.linalg.norm(v)
        return (v / norm).tolist() if norm > 0 else vec

    embeddings_numpy = [
        l2_normalize_np(e.values)
        for e in response.embeddings
    ]
    
    print(f"Generated {len(embeddings_manual)} embeddings (manual normalization)")
    print(f"Embedding dimension: {len(embeddings_manual[0])}")
    print(f"First embedding (first 5 values): {embeddings_manual[0][:5]}")
    print(f"L2 norm of first embedding: {math.sqrt(sum(x * x for x in embeddings_manual[0])):.6f}")
    
    print(f"\nGenerated {len(embeddings_numpy)} embeddings (numpy normalization)")
    print(f"Embedding dimension: {len(embeddings_numpy[0])}")
    print(f"First embedding (first 5 values): {embeddings_numpy[0][:5]}")
    print(f"L2 norm of first embedding: {math.sqrt(sum(x * x for x in embeddings_numpy[0])):.6f}")
    
    # Compare results
    diff = max(abs(a - b) for a, b in zip(embeddings_manual[0], embeddings_numpy[0]))
    print(f"\nMax difference between manual and numpy normalization: {diff:.10f}")
    
except Exception as e:
    print(f"Error generating embeddings: {e}")
    exit(1)
