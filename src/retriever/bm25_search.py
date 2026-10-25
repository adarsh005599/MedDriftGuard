"""
BM25 sparse keyword retriever for MedDriftGuard.

Uses rank_bm25 to perform lexical keyword matching over knowledge base chunks.
"""

from __future__ import annotations

import re
from typing import List, Tuple
from rank_bm25 import BM25Okapi
from src.common.schemas import RetrievedChunk


def tokenize_text(text: str) -> List[str]:
    """Tokenize text into lowercase alphanumeric tokens."""
    return re.findall(r"\w+", text.lower())


class BM25Retriever:
    """Sparse keyword search index using BM25Okapi."""

    def __init__(self, chunks: List[RetrievedChunk], k1: float = 1.5, b: float = 0.75):
        self.chunks = chunks
        self.k1 = k1
        self.b = b
        self.corpus_tokens = [tokenize_text(c.text) for c in chunks]
        self.bm25 = BM25Okapi(self.corpus_tokens, k1=k1, b=b)

    def search(self, query: str, top_k: int = 10) -> List[Tuple[RetrievedChunk, float]]:
        """
        Search for matching chunks using BM25 keyword matching.
        Returns a list of (RetrievedChunk, score) tuples sorted in descending order.
        """
        query_tokens = tokenize_text(query)
        if not query_tokens or not self.chunks:
            return []

        scores = self.bm25.get_scores(query_tokens)
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

        results = []
        for rank, idx in enumerate(top_indices, start=1):
            chunk = self.chunks[idx].model_copy()
            score = float(scores[idx])
            chunk.score = score
            chunk.rank = rank
            results.append((chunk, score))

        return results
