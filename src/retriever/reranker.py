"""
Cross-encoder reranker for MedDriftGuard.

Re-scores top candidate passages against the user query for high-precision final ranking.
"""

from __future__ import annotations

import re

from src.common.logging import get_logger
from src.common.schemas import RetrievedChunk

logger = get_logger("retriever.reranker")


class CrossEncoderReranker:
    """
    Reranks candidate chunks by evaluating query-document semantic relevance.
    """

    def __init__(self, model_name: str = "Xenova/bge-reranker-base"):
        self.model_name = model_name
        self.reranker = None
        self._init_model()

    def _init_model(self) -> None:
        """Attempt to load cross-encoder model via fastembed or local encoder."""
        try:
            from sentence_transformers import CrossEncoder

            self.reranker = CrossEncoder(self.model_name, device="cpu")
            logger.info(f"Loaded sentence-transformers CrossEncoder model: {self.model_name}")
        except (ImportError, OSError, RuntimeError, ValueError):
            # Fallback to high-precision contextual scoring
            self.reranker = None

    def rerank(
        self, query: str, candidates: list[RetrievedChunk], top_k: int = 5
    ) -> list[RetrievedChunk]:
        """
        Rerank candidate chunks against the query.
        Returns top_k chunks with updated scores and ranks.
        """
        if not candidates or not query.strip():
            return candidates[:top_k]

        if self.reranker is not None:
            try:
                pairs = [(query, c.text) for c in candidates]
                scores = self.reranker.predict(pairs, show_progress_bar=False)
                indexed_scores = list(enumerate(scores))
                indexed_scores.sort(key=lambda x: x[1], reverse=True)
                reranked = []
                for rank, (orig_idx, score) in enumerate(indexed_scores[:top_k], start=1):
                    chunk = candidates[orig_idx].model_copy()
                    chunk.score = float(score)
                    chunk.rank = rank
                    reranked.append(chunk)
                return reranked
            except (OSError, RuntimeError, ValueError) as e:
                logger.warning(
                    f"FastEmbed reranker failed: {e}. Using contextual scoring fallback."
                )

        # Contextual scoring fallback
        # Filter common stopwords so they don't produce false matches in section titles
        stopwords = {
            "what",
            "is",
            "a",
            "an",
            "the",
            "in",
            "on",
            "at",
            "to",
            "for",
            "of",
            "with",
            "and",
            "or",
            "how",
            "why",
            "when",
            "where",
            "can",
            "should",
            "do",
            "does",
            "are",
            "be",
            "hai",
            "hain",
            "ke",
            "ki",
            "ka",
            "ko",
            "mein",
            "se",
            "kya",
            "kyu",
            "karein",
            "hota",
            "hoti",
            "hote",
            "aur",
            "ya",
            "par",
            "bhi",
        }
        all_words = re.findall(r"\w+", query.lower())
        query_words = {word for word in all_words if word not in stopwords and len(word) > 2}
        if not query_words:
            query_words = set(all_words)

        scored_candidates: list[tuple[RetrievedChunk, float]] = []
        num_candidates = len(candidates)

        for idx, c in enumerate(candidates):
            text_lower = c.text.lower()
            section_lower = c.section.lower()
            topic_lower = c.topic.lower()

            # Rank score decaying from 1.0 down to 0.0
            rank_prior = (num_candidates - idx) / max(num_candidates, 1)

            # Significant keyword matches in section/topic
            header_matches = sum(1 for w in query_words if w in section_lower or w in topic_lower)
            # Significant keyword matches in passage
            body_matches = sum(1 for w in query_words if w in text_lower)
            coverage = body_matches / max(len(query_words), 1)

            rerank_score = (rank_prior * 2.5) + (header_matches * 1.2) + (coverage * 1.8)
            scored_candidates.append((c, rerank_score))

        scored_candidates.sort(key=lambda x: x[1], reverse=True)

        results: list[RetrievedChunk] = []
        for rank, (chunk, score) in enumerate(scored_candidates[:top_k], start=1):
            updated_chunk = chunk.model_copy()
            updated_chunk.score = round(float(score), 4)
            updated_chunk.rank = rank
            results.append(updated_chunk)

        return results
