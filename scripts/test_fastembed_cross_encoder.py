import os
from fastembed.rerank.cross_encoder import TextCrossEncoder

# 1. Point to your permanent models folder
cache_path = os.path.expanduser("~/models/fastembed_cache")

# 2. Initialize the Cross-Encoder
# Note: This model is ~1.1GB, so it will trigger the hotspot download
model = TextCrossEncoder(model_name="BAAI/bge-reranker-base", cache_dir=cache_path)

def test_cross_encoder_rerank(query, documents):
    print(f"🎯 Cross-Encoder Rerank for: '{query}'")
    
    # 1. Rerank returns a list of floats (scores)
    # The order matches the input list of documents
    scores = list(model.rerank(query, documents))
    
    # 2. Pair them up: [(score, doc), (score, doc)...]
    results = list(zip(scores, documents))
    
    # 3. Sort by the score (index 0) in descending order
    reranked = sorted(results, key=lambda x: x[0], reverse=True)
    
    # 4. Print the results
    for idx, (score, doc) in enumerate(reranked):
        print(f"{idx+1}. [{score:.4f}] {doc}")

# --- TEST DATA (The hard version/symbol test) ---
docs = [
    "Release notes for LLM-Adapter v2.4.1: fixed prompt registry bugs.",
    "Release notes for LLM-Adapter v2.4.2: added support for Late Interaction.",
    "Architecture update: Use the -> symbol for mapping prompts to the registry.",
    "Architecture update: Use the => symbol for mapping prompts to the registry."
]

if __name__ == "__main__":
    test_cross_encoder_rerank("v2.4.2 changes", docs)
    print("-" * 30)
    test_cross_encoder_rerank("mapping with -> symbol", docs)
