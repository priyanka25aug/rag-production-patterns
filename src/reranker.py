"""
Cross-encoder reranker: re-scores retrieved chunks by relevance to the query.
Mock mode uses query-chunk token overlap; live mode uses a cross-encoder model.
"""
from typing import List
from .retriever import RetrievedChunk


class MockReranker:
    """Reranks by bi-gram overlap with query."""

    def rerank(self, query: str, chunks: List[RetrievedChunk]) -> List[RetrievedChunk]:
        query_bigrams = self._bigrams(query.lower())

        def score(chunk):
            chunk_bigrams = self._bigrams(chunk.text.lower())
            overlap = len(query_bigrams & chunk_bigrams)
            return overlap / max(len(query_bigrams), 1)

        reranked = sorted(chunks, key=score, reverse=True)
        for i, chunk in enumerate(reranked):
            chunk.score = 1.0 - i * 0.1
        return reranked

    def _bigrams(self, text: str) -> set:
        words = text.split()
        return {(words[i], words[i + 1]) for i in range(len(words) - 1)}


class CrossEncoderReranker:
    """Production cross-encoder reranker using sentence-transformers."""

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model_name = model_name
        self._model = None

    def _load(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder
            self._model = CrossEncoder(self.model_name)

    def rerank(self, query: str, chunks: List[RetrievedChunk]) -> List[RetrievedChunk]:
        self._load()
        pairs = [(query, chunk.text) for chunk in chunks]
        scores = self._model.predict(pairs)
        for chunk, score in zip(chunks, scores):
            chunk.score = float(score)
        return sorted(chunks, key=lambda c: c.score, reverse=True)
