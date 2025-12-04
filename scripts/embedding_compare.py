import argparse
import math
from typing import List

import openai

from backend.core.config import settings


def embed_text(text: str) -> List[float]:
    """Generate an embedding for the given text using the configured model."""
    client = openai.OpenAI(api_key=settings.openai_api_key)
    response = client.embeddings.create(input=text, model=settings.embedding_model)
    return response.data[0].embedding


def dot(a: List[float], b: List[float]) -> float:
    if len(a) != len(b):
        raise ValueError("Embedding vectors must have the same length")
    return sum(x * y for x, y in zip(a, b))


def l2_norm(a: List[float]) -> float:
    return math.sqrt(sum(x * x for x in a))


def cosine_similarity(a: List[float], b: List[float]) -> float:
    denom = l2_norm(a) * l2_norm(b)
    if denom == 0:
        raise ValueError("Cannot compute cosine similarity with a zero vector")
    return dot(a, b) / denom


def normalized_dot_product(a: List[float], b: List[float]) -> float:
    """
    Dot product after L2-normalizing both vectors.
    Note: This is equivalent to cosine similarity.
    """
    na = l2_norm(a)
    nb = l2_norm(b)
    if na == 0 or nb == 0:
        raise ValueError("Cannot compute normalized dot product with a zero vector")
    a_norm = [x / na for x in a]
    b_norm = [y / nb for y in b]
    return dot(a_norm, b_norm)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute cosine similarity and normalized dot product between embeddings of two texts using the configured embedding model."
    )
    # Make inputs optional; prompt if missing
    parser.add_argument("text1", nargs="?", help="First input text")
    parser.add_argument("text2", nargs="?", help="Second input text")
    args = parser.parse_args()

    text1 = args.text1 or input("Enter input string 1: ")
    text2 = args.text2 or input("Enter input string to compare: ")

    print("Embedding input 1...", flush=True)
    emb1 = embed_text(text1)
    print("Embedding input 2...", flush=True)
    emb2 = embed_text(text2)

    cos = cosine_similarity(emb1, emb2)
    ndp = normalized_dot_product(emb1, emb2)

    print("Results:")
    print(f"- Cosine similarity: {cos:.6f}")
    print(f"- Dot product (normalized): {ndp:.6f}")


if __name__ == "__main__":
    main()
