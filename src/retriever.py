"""
Vector retriever using FAISS with mock fallback (no sentence-transformers needed in mock mode).
"""
from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class RetrievedChunk:
    text: str
    chunk_id: str
    score: float
    doc_id: str


class MockVectorRetriever:
    """Keyword-overlap mock retriever — no FAISS or embedding model needed."""

    def __init__(self, top_k: int = 5):
        self.top_k = top_k
        self._chunks = []

    def index(self, chunks: list):
        self._chunks = chunks

    def retrieve(self, query: str) -> List[RetrievedChunk]:
        query_words = set(query.lower().split())
        scored = []
        for chunk in self._chunks:
            chunk_words = set(chunk.text.lower().split())
            overlap = len(query_words & chunk_words)
            score = overlap / max(len(query_words), 1)
            scored.append((chunk, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        return [
            RetrievedChunk(
                text=c.text,
                chunk_id=c.chunk_id,
                score=s,
                doc_id=c.doc_id,
            )
            for c, s in scored[: self.top_k]
            if s > 0
        ]


class FAISSRetriever:
    """Production FAISS retriever — requires sentence-transformers and faiss-cpu."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", top_k: int = 5):
        self.top_k = top_k
        self._index = None
        self._chunks = []
        self._model_name = model_name

    def index(self, chunks: list):
        import numpy as np
        import faiss
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(self._model_name)
        self._chunks = chunks
        embeddings = model.encode([c.text for c in chunks], show_progress_bar=False)
        embeddings = np.array(embeddings, dtype="float32")
        faiss.normalize_L2(embeddings)
        self._index = faiss.IndexFlatIP(embeddings.shape[1])
        self._index.add(embeddings)
        self._model = model

    def retrieve(self, query: str) -> List[RetrievedChunk]:
        import numpy as np
        import faiss

        q_emb = self._model.encode([query])
        q_emb = np.array(q_emb, dtype="float32")
        faiss.normalize_L2(q_emb)
        scores, indices = self._index.search(q_emb, self.top_k)
        return [
            RetrievedChunk(
                text=self._chunks[i].text,
                chunk_id=self._chunks[i].chunk_id,
                score=float(scores[0][j]),
                doc_id=self._chunks[i].doc_id,
            )
            for j, i in enumerate(indices[0])
            if i >= 0
        ]
