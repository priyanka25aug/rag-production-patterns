# RAG Production Patterns

Production RAG: hierarchical chunking, query rewriting, re-ranking, uncertainty thresholding.

## Pipeline

```
User Query
    │
    ▼
Query Rewriter ──→ 3 targeted search queries
    │
    ▼
Vector Retriever (FAISS / mock) ──→ top-k chunks per query
    │
    ▼
Cross-Encoder Reranker ──→ reordered by relevance
    │
    ▼
Uncertainty Threshold ──→ confidence < threshold → route to human
    │
    ▼
LLM Generation (or mock answer)
```

## Quick Start

```bash
pip install -r requirements.txt
python examples/run_rag.py   # no API keys or GPU needed (mock mode)
pytest tests/
```

## Key Design Decisions

### Hierarchical Chunking
Splits by document structure (sections → paragraphs) rather than fixed token windows.
Preserves semantic coherence and parent-child chunk relationships for citation.

### Query Rewriting
Rewrites the user query into 3 targeted sub-queries before retrieval.
Improves recall by covering different phrasings of the same intent.

### Cross-Encoder Reranking
After vector retrieval (fast), re-scores top chunks with a cross-encoder (accurate).
Bi-directional attention captures query-chunk relevance that embeddings miss.

### Uncertainty Thresholding
If the top retrieved chunk score is below threshold, the pipeline routes to human review
instead of generating a low-confidence answer. Prevents hallucination on out-of-domain queries.

## Mock vs Live Mode

| Component | Mock mode | Live mode |
|-----------|-----------|-----------|
| Embeddings | keyword overlap | sentence-transformers |
| Vector store | in-memory list | FAISS |
| Reranker | bigram overlap | cross-encoder/ms-marco |
| Generator | template | OpenAI/Anthropic/Vertex |

Set `mock_mode=False` and provide API keys in `.env` for production use.
