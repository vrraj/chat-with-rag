from fastembed import TextEmbedding, SparseTextEmbedding, LateInteractionTextEmbedding

def check_bge_m3_support():
    target = "BAAI/bge-m3"
    
    # BGE-M3 is unique because it appears in all three categories
    dense_models = [m['model'] for m in TextEmbedding.list_supported_models()]
    sparse_models = [m['model'] for m in SparseTextEmbedding.list_supported_models()]
    late_models = [m['model'] for m in LateInteractionTextEmbedding.list_supported_models()]

    print(f"Checking support for: {target}\n" + "-"*30)
    
    results = {
        "Dense Embedding": target in dense_models,
        "Sparse (Lexical)": target in sparse_models,
        "Late Interaction (ColBERT)": target in late_models
    }

    for feature, supported in results.items():
        status = "✅ SUPPORTED" if supported else "❌ NOT FOUND"
        print(f"{feature:30} : {status}")

    if all(results.values()):
        print("\nConclusion: BGE-M3 is fully integrated in this version.")
    elif any(results.values()):
        print("\nConclusion: Partial support found. Update fastembed for full features.")
    else:
        print("\nConclusion: BGE-M3 not found. Please run: pip install --upgrade fastembed")

if __name__ == "__main__":
    check_bge_m3_support()
