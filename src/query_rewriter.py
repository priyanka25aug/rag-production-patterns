"""
Query rewriter: expands a user query into multiple targeted search queries
to improve retrieval coverage (HyDE / multi-query pattern).
"""
from typing import List


# Domain-aware query expansion templates
EXPANSION_TEMPLATES = {
    "financial": [
        "{query}",
        "financial analysis {query}",
        "{query} risk factors regulatory",
    ],
    "default": [
        "{query}",
        "explain {query} in detail",
        "{query} examples context",
    ],
}

FINANCIAL_KEYWORDS = {"earnings", "revenue", "risk", "filing", "trade", "sec", "compliance", "credit"}


class QueryRewriter:
    def __init__(self, mock_mode: bool = True, n_queries: int = 3):
        self.mock_mode = mock_mode
        self.n_queries = n_queries

    def rewrite(self, query: str) -> List[str]:
        if self.mock_mode:
            return self._template_rewrite(query)
        return self._llm_rewrite(query)

    def _template_rewrite(self, query: str) -> List[str]:
        domain = "financial" if any(kw in query.lower() for kw in FINANCIAL_KEYWORDS) else "default"
        templates = EXPANSION_TEMPLATES[domain][: self.n_queries]
        return [t.format(query=query) for t in templates]

    def _llm_rewrite(self, query: str) -> List[str]:
        raise NotImplementedError("Set mock_mode=True for testing without API keys")
