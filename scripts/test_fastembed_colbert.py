import numpy as np
import os
from fastembed import LateInteractionTextEmbedding

# Use environment variable if available, otherwise fallback to default
model_path = os.path.expanduser(os.getenv("FASTEMBED_CACHE_PATH", "~/models/fastembed_cache"))

model = LateInteractionTextEmbedding(
    model_name="colbert-ir/colbertv2.0",
    cache_dir=model_path
)

def compute_maxsim(query_embeddings, doc_embeddings):
    # Standard MaxSim interaction logic
    sim_matrix = np.dot(query_embeddings, doc_embeddings.T)
    max_sim_per_query_token = np.max(sim_matrix, axis=1)
    return np.sum(max_sim_per_query_token)

def test_version_and_symbol_sensitivity(query, documents):
    print(f"🔍 Query: '{query}'")
    
    # Generate token-level matrices
    query_emb = list(model.query_embed(query))[0]
    doc_embs = list(model.embed(documents))
    
    results = []
    for i, doc_matrix in enumerate(doc_embs):
        score = compute_maxsim(query_emb, doc_matrix)
        results.append({"doc": documents[i], "score": float(score)})
    
    # Rank results
    for idx, item in enumerate(sorted(results, key=lambda x: x['score'], reverse=True)):
        print(f"{idx+1}. [{item['score']:.4f}] {item['doc']}")

# --- MULTI-VERSION & SYMBOL DATASET ---
docs = [
    "Release notes for LLM-Adapter v2.4.1: fixed prompt registry bugs.",
    "Release notes for LLM-Adapter v2.4.2: added support for Late Interaction.",
    "Architecture update: Use the -> symbol for mapping prompts to the registry.",
    "Architecture update: Use the => symbol for mapping prompts to the registry.",
    "Generic documentation for the prompt registry system."
]

if __name__ == "__main__":
    # Test 1: Version Precision
    test_version_and_symbol_sensitivity("v2.4.2 changes", docs)
    print("-" * 30)
    # Test 2: Symbol Precision
    test_version_and_symbol_sensitivity("mapping with -> symbol", docs)
    print("-" * 30)
    # Test 3: Exact Version Precision
    test_version_and_symbol_sensitivity("exact changes specifically for version v2.4.2 only.", docs)
