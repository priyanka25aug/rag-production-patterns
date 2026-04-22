"""
Full RAG pipeline: rewrite → retrieve → rerank → threshold → generate.
"""
from dataclasses import dataclass
from typing import List
from .chunker import HierarchicalChunker
from .query_rewriter import QueryRewriter
from .retriever import MockVectorRetriever, RetrievedChunk
from .reranker import MockReranker
from .uncertainty import UncertaintyThreshold


@dataclass
class RAGResponse:
    answer: str
    sources: List[RetrievedChunk]
    queries_used: List[str]
    top_score: float
    routed_to_human: bool


class RAGPipeline:
    def __init__(
        self,
        rewriter: QueryRewriter = None,
        retriever: MockVectorRetriever = None,
        reranker: MockReranker = None,
        uncertainty: UncertaintyThreshold = None,
        mock_mode: bool = True,
    ):
        self.rewriter = rewriter or QueryRewriter(mock_mode=mock_mode)
        self.retriever = retriever or MockVectorRetriever(top_k=5)
        self.reranker = reranker or MockReranker()
        self.uncertainty = uncertainty or UncertaintyThreshold(min_score=0.2)
        self.mock_mode = mock_mode

    def index_documents(self, documents: List[str], doc_ids: List[str] = None):
        chunker = HierarchicalChunker()
        all_chunks = []
        for i, doc in enumerate(documents):
            doc_id = (doc_ids or [])[i] if doc_ids and i < len(doc_ids) else f"doc{i}"
            all_chunks.extend(chunker.chunk(doc, doc_id=doc_id))
        self.retriever.index(all_chunks)
        return len(all_chunks)

    def query(self, question: str) -> RAGResponse:
        queries = self.rewriter.rewrite(question)
        all_chunks = []
        seen = set()
        for q in queries:
            for chunk in self.retriever.retrieve(q):
                if chunk.chunk_id not in seen:
                    all_chunks.append(chunk)
                    seen.add(chunk.chunk_id)

        reranked = self.reranker.rerank(question, all_chunks) if all_chunks else []
        decision = self.uncertainty.evaluate(reranked)

        if not decision.should_answer:
            return RAGResponse(
                answer="I don't have enough relevant information to answer confidently. Please consult a specialist.",
                sources=[],
                queries_used=queries,
                top_score=decision.top_score,
                routed_to_human=True,
            )

        top_chunks = reranked[:3]
        context = "\n\n".join(c.text for c in top_chunks)

        if self.mock_mode:
            answer = f"[Mock answer based on {len(top_chunks)} retrieved chunks]\nContext preview: {context[:200]}..."
        else:
            answer = self._llm_generate(question, context)

        return RAGResponse(
            answer=answer,
            sources=top_chunks,
            queries_used=queries,
            top_score=decision.top_score,
            routed_to_human=False,
        )

    def _llm_generate(self, question: str, context: str) -> str:
        raise NotImplementedError("Set mock_mode=True for testing without API keys")
