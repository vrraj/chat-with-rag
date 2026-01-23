import math
import os
from google import genai

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
    
    def l2_normalize(vec):
        mag = math.sqrt(sum(x * x for x in vec))
        return [x / mag for x in vec] if mag > 0 else vec

    embeddings = [
        l2_normalize(e.values)
        for e in response.embeddings
    ]
    
    print(f"Generated {len(embeddings)} embeddings")
    print(f"Embedding dimension: {len(embeddings[0])}")
    print(f"First embedding (first 5 values): {embeddings[0][:5]}")
    print(f"L2 norm of first embedding: {math.sqrt(sum(x * x for x in embeddings[0])):.6f}")
    
except Exception as e:
    print(f"Error generating embeddings: {e}")
    exit(1)
