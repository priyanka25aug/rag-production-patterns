"""
Uncertainty thresholding: if retrieval confidence is too low, route to human
instead of hallucinating a low-quality answer.
"""
from dataclasses import dataclass
from typing import List, Optional
from .retriever import RetrievedChunk


@dataclass
class UncertaintyDecision:
    should_answer: bool
    top_score: float
    threshold: float
    reason: str
    chunks: List[RetrievedChunk]


class UncertaintyThreshold:
    def __init__(self, min_score: float = 0.3, min_chunks: int = 1):
        self.min_score = min_score
        self.min_chunks = min_chunks

    def evaluate(self, chunks: List[RetrievedChunk]) -> UncertaintyDecision:
        if not chunks:
            return UncertaintyDecision(
                should_answer=False,
                top_score=0.0,
                threshold=self.min_score,
                reason="No chunks retrieved",
                chunks=[],
            )

        top_score = chunks[0].score
        sufficient = top_score >= self.min_score and len(chunks) >= self.min_chunks

        return UncertaintyDecision(
            should_answer=sufficient,
            top_score=top_score,
            threshold=self.min_score,
            reason=(
                f"Top score {top_score:.2f} ≥ threshold {self.min_score:.2f}"
                if sufficient
                else f"Top score {top_score:.2f} < threshold {self.min_score:.2f} — routing to human"
            ),
            chunks=chunks,
        )
