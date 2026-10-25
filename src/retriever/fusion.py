"""
Reciprocal Rank Fusion (RRF) for combining sparse and dense retrieval rankings.
"""

from __future__ import annotations

from typing import Dict, List, Tuple
from src.common.schemas import RetrievedChunk


def reciprocal_rank_fusion(
    rankings: List[List[Tuple[RetrievedChunk, float]]],
    k: int = 60,
    top_k: int = 10,
) -> List[Tuple[RetrievedChunk, float]]:
    """
    Combines multiple ranked lists of chunks using Reciprocal Rank Fusion.
    
    Formula:
        RRF_score(d) = sum_{m in rankings} (1.0 / (k + rank_m(d)))
    
    Args:
        rankings: List of ranked lists, where each list contains (RetrievedChunk, original_score) tuples.
        k: Smoothing constant (default: 60).
        top_k: Number of combined results to return.

    Returns:
        List of (RetrievedChunk, rrf_score) tuples sorted in descending order.
    """
    chunk_map: Dict[str, RetrievedChunk] = {}
    fused_scores: Dict[str, float] = {}

    for ranking in rankings:
        for rank, (chunk, _) in enumerate(ranking, start=1):
            cid = chunk.chunk_id
            if cid not in chunk_map:
                chunk_map[cid] = chunk.model_copy()
                fused_scores[cid] = 0.0

            fused_scores[cid] += 1.0 / (k + rank)

    # Sort chunks by fused score descending
    sorted_chunk_ids = sorted(fused_scores.keys(), key=lambda cid: fused_scores[cid], reverse=True)[:top_k]

    fused_results: List[Tuple[RetrievedChunk, float]] = []
    for rank, cid in enumerate(sorted_chunk_ids, start=1):
        chunk = chunk_map[cid]
        score = float(fused_scores[cid])
        chunk.score = score
        chunk.rank = rank
        fused_results.append((chunk, score))

    return fused_results
