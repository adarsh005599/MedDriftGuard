"""
Master Hybrid Retriever for MedDriftGuard.

Integrates BM25 sparse search, dense semantic embeddings, Reciprocal Rank Fusion (RRF),
and cross-encoder reranking under a unified, drift-resistant interface.

IMPORTANT ARCHITECTURAL RULE:
The retriever accepts ONLY a clean standalone query string for the current turn.
It strictly prohibits receiving raw multi-turn chat history.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

import yaml

from src.common.logging import get_logger
from src.common.schemas import RetrievedChunk
from src.retriever.bm25_search import BM25Retriever
from src.retriever.dense_search import DenseRetriever
from src.retriever.fusion import reciprocal_rank_fusion
from src.retriever.reranker import CrossEncoderReranker

logger = get_logger("retriever.hybrid")

RetrieverMode = Literal["dense_only", "bm25_only", "hybrid", "hybrid_rerank"]


class HybridRetriever:
    """
    Hybrid retriever combining dense vector search and sparse BM25 search
    with RRF and cross-encoder reranking.
    """

    def __init__(
        self,
        chunks: list[RetrievedChunk] | None = None,
        chunks_path: Path = Path("kb/chunks.jsonl"),
        mode: RetrieverMode | None = None,
        rrf_k: int | None = None,
        dense_model_name: str | None = None,
        reranker_model_name: str | None = None,
        config_path: Path = Path("configs/retriever.yaml"),
    ):
        config = self._load_config(config_path)
        retriever_config = config.get("retriever", {})
        dense_config = config.get("dense", {})
        sparse_config = config.get("sparse", {})
        reranker_config = config.get("reranker", {})

        self.mode = mode or retriever_config.get("type", "hybrid")
        self.rrf_k = rrf_k if rrf_k is not None else retriever_config.get("rrf_k", 60)
        self.chunks = chunks if chunks is not None else self._load_chunks(chunks_path)

        # Initialize sub-components
        self.bm25 = BM25Retriever(
            self.chunks,
            k1=sparse_config.get("k1", 1.5),
            b=sparse_config.get("b", 0.75),
        )
        self.dense = DenseRetriever(
            self.chunks,
            model_name=dense_model_name
            or dense_config.get(
                "model_name", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
            ),
        )
        self.reranker = CrossEncoderReranker(
            model_name=reranker_model_name
            or reranker_config.get("model_name", "BAAI/bge-reranker-large")
        )
        logger.info(
            f"Initialized HybridRetriever with {len(self.chunks)} chunks in '{self.mode}' mode."
        )

    @staticmethod
    def _load_config(config_path: Path) -> dict:
        if not config_path.exists():
            logger.warning(f"Retriever config {config_path} not found; using built-in defaults.")
            return {}
        with config_path.open("r", encoding="utf-8") as config_file:
            config = yaml.safe_load(config_file) or {}
        return config

    @staticmethod
    def _load_chunks(chunks_path: Path) -> list[RetrievedChunk]:
        """Loads and parses chunks from JSONL file."""
        if not chunks_path.exists():
            logger.warning(f"Chunks file {chunks_path} not found. Returning empty list.")
            return []

        chunks: list[RetrievedChunk] = []
        with chunks_path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    chunks.append(RetrievedChunk(**data))
        return chunks

    def retrieve(
        self,
        standalone_query: str,
        k: int = 5,
        mode_override: RetrieverMode | None = None,
    ) -> list[RetrievedChunk]:
        """
        Retrieve the top-k most relevant knowledge base chunks for a standalone query.

        Args:
            standalone_query: Clean English standalone search query representing the current turn's intent.
            k: Number of chunks to retrieve (default: 5).
            mode_override: Optional override for retrieval mode ("dense_only", "bm25_only", "hybrid", "hybrid_rerank").

        Raises:
            TypeError / ValueError: If conversation history or non-string object is passed.
        """
        # Architectural guardrail: strict prevention of chat history leakage into retriever
        if not isinstance(standalone_query, str):
            raise TypeError(
                f"Retriever only accepts a single standalone query string. "
                f"Received object of type: {type(standalone_query).__name__} (chat history is forbidden)."
            )

        cleaned_query = standalone_query.strip()
        if not cleaned_query or k <= 0 or not self.chunks:
            return []

        active_mode = mode_override or self.mode
        candidate_pool_size = max(k * 4, 20)

        if active_mode == "bm25_only":
            bm25_results = self.bm25.search(cleaned_query, top_k=k)
            return [chunk for chunk, _ in bm25_results]

        if active_mode == "dense_only":
            dense_results = self.dense.search(cleaned_query, top_k=k)
            return [chunk for chunk, _ in dense_results]

        # Hybrid modes: fetch from both sparse and dense
        bm25_candidates = self.bm25.search(cleaned_query, top_k=candidate_pool_size)
        dense_candidates = self.dense.search(cleaned_query, top_k=candidate_pool_size)

        fused_candidates = reciprocal_rank_fusion(
            [bm25_candidates, dense_candidates],
            k=self.rrf_k,
            top_k=candidate_pool_size if active_mode == "hybrid_rerank" else k,
        )

        if active_mode == "hybrid":
            return [chunk for chunk, _ in fused_candidates[:k]]

        # "hybrid_rerank": apply cross-encoder reranker
        candidates_to_rerank = [chunk for chunk, _ in fused_candidates]
        final_reranked = self.reranker.rerank(cleaned_query, candidates_to_rerank, top_k=k)
        return final_reranked
