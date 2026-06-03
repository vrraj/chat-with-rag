from fastembed import TextEmbedding, LateInteractionTextEmbedding

# 1. High-quality Dense Embedding
dense_model = TextEmbedding(model_name="BAAI/bge-large-en-v1.5")

# 2. The ColBERT Reranker (This is the 'Secret Sauce' for version precision)
colbert_model = LateInteractionTextEmbedding(model_name="colbert-ir/colbertv2.0")

print("✅ Systems Ready: Using BGE-Large + ColBERT v2.0")
