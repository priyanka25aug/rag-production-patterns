import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from src.pipeline import RAGPipeline
from src.query_rewriter import QueryRewriter
from src.uncertainty import UncertaintyThreshold
from src.retriever import MockVectorRetriever, RetrievedChunk


DOCS = [
    "EPS earnings per share quarterly revenue guidance financial results beat expectations.",
    "Counterparty credit risk default exposure regulatory compliance SEC filing.",
    "Trade confirmation settlement CUSIP ISIN T+2 OTC derivatives counterparty.",
]


@pytest.fixture
def pipeline():
    p = RAGPipeline(mock_mode=True)
    p.index_documents(DOCS, doc_ids=["doc0", "doc1", "doc2"])
    return p


def test_pipeline_indexes_documents():
    p = RAGPipeline(mock_mode=True)
    n = p.index_documents(DOCS)
    assert n > 0


def test_pipeline_returns_answer_for_relevant_query(pipeline):
    response = pipeline.query("What is EPS earnings quarterly revenue?")
    assert response.answer
    assert not response.routed_to_human or response.top_score < 0.2


def test_pipeline_escalates_for_irrelevant_query(pipeline):
    # A query with zero overlap with indexed docs should route to human
    response = pipeline.query("Quantum entanglement physics experiment")
    assert response.routed_to_human is True


def test_query_rewriter_returns_multiple_queries():
    rewriter = QueryRewriter(mock_mode=True, n_queries=3)
    queries = rewriter.rewrite("What is the EPS for Q3?")
    assert len(queries) == 3
    assert all(isinstance(q, str) for q in queries)


def test_uncertainty_threshold_blocks_low_score():
    threshold = UncertaintyThreshold(min_score=0.5)
    low_chunks = [RetrievedChunk(text="text", chunk_id="c1", score=0.1, doc_id="d1")]
    decision = threshold.evaluate(low_chunks)
    assert decision.should_answer is False


def test_uncertainty_threshold_passes_high_score():
    threshold = UncertaintyThreshold(min_score=0.1)
    high_chunks = [RetrievedChunk(text="text", chunk_id="c1", score=0.8, doc_id="d1")]
    decision = threshold.evaluate(high_chunks)
    assert decision.should_answer is True


def test_pipeline_sources_returned(pipeline):
    response = pipeline.query("earnings revenue EPS quarterly")
    if not response.routed_to_human:
        assert len(response.sources) > 0
