"""Embeddings package public API.

Expose only embeddings-native components from this package.
"""

from .embeddings_manager import EmbeddingsManager

__all__ = [
    "EmbeddingsManager",
]
