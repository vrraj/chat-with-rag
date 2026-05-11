from typing import List

from backend.retrieval.schemas import EmbeddingSpec, EmbeddingResult


class FastEmbedEmbeddingProvider:
    def __init__(self):
        self._models = {}

    def _get_model(self, spec: EmbeddingSpec):
        from fastembed import TextEmbedding

        key = f"{spec.model}:{spec.device or 'default'}"
        if key not in self._models:
            kwargs = {}
            if spec.extra:
                kwargs.update(spec.extra)

            self._models[key] = TextEmbedding(
                model_name=spec.model,
                **kwargs,
            )

        return self._models[key]

    def embed(self, texts: List[str], spec: EmbeddingSpec) -> EmbeddingResult:
        model = self._get_model(spec)
        vectors = [list(v) for v in model.embed(texts, batch_size=spec.batch_size)]

        return EmbeddingResult(
            vectors=vectors,
            model=spec.model,
            dimensions=spec.dimensions,
            runtime=spec.runtime,
            usage={
                "input_text_count": len(texts),
                "local": True,
            },
            metadata={
                "provider": spec.provider,
                "batch_size": spec.batch_size,
            },
        )
