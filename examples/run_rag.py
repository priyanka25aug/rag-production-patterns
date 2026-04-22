#!/usr/bin/env python3
"""Working RAG demo — no LLM or embedding model needed (mock mode)."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.pipeline import RAGPipeline

SAMPLE_DOCS = [
    """
## Q3 Earnings Call Transcript

Revenue for Q3 2024 came in at $4.2B, beating analyst expectations by 5%.
EPS of $1.87 exceeded the consensus estimate of $1.72. Management raised
full-year revenue guidance to $16.5B from $15.8B, citing strong performance
in the institutional trading division and continued growth in the asset
management segment.
""",
    """
## Risk Factors — SEC Annual Filing

The company faces significant counterparty credit risk in its derivatives
portfolio. Market volatility, including elevated VIX readings, has increased
mark-to-market losses on certain positions. The company is subject to
regulatory oversight by the SEC and FINRA. Non-compliance with capital
adequacy requirements could result in enforcement actions.
""",
    """
## Trade Operations — Settlement Procedures

All equity trades settle on a T+2 basis under standard market conventions.
Trade confirmations are issued within one hour of execution. Counterparty
verification is required for all OTC derivatives. CUSIP and ISIN identifiers
must be validated before settlement instructions are released.
""",
]

pipeline = RAGPipeline(mock_mode=True)
n_chunks = pipeline.index_documents(SAMPLE_DOCS, doc_ids=["earnings-q3", "risk-filing", "trade-ops"])
print(f"Indexed {n_chunks} chunks from {len(SAMPLE_DOCS)} documents\n")

QUESTIONS = [
    "What was the EPS guidance for Q3 earnings?",
    "What are the counterparty credit risk factors?",
    "How does trade settlement work for OTC derivatives?",
    "What is the capital adequacy formula?",  # Low confidence — should route to human
]

print("=" * 70)
for q in QUESTIONS:
    response = pipeline.query(q)
    print(f"\nQ: {q}")
    print(f"Queries used: {response.queries_used}")
    print(f"Top score: {response.top_score:.2f} | Human escalation: {response.routed_to_human}")
    print(f"Answer: {response.answer[:200]}")
    if response.sources:
        print(f"Sources: {[s.chunk_id for s in response.sources]}")
