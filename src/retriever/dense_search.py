"""
Dense semantic retriever for MedDriftGuard.

Uses FastEmbed (ONNX-accelerated BGE multilingual/embedding models) to perform
vector similarity search with cosine distance over knowledge base chunks.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional, Tuple
import numpy as np

from src.common.schemas import RetrievedChunk
from src.common.logging import get_logger

logger = get_logger("retriever.dense")


class DenseRetriever:
    """Dense vector retriever using local FastEmbed ONNX models."""

    def __init__(
        self,
        chunks: List[RetrievedChunk],
        model_name: str = "BAAI/bge-small-en-v1.5",
        cache_dir: Optional[str] = None,
    ):
        self.chunks = chunks
        self.model_name = model_name
        self.embeddings: Optional[np.ndarray] = None
        self._init_encoder(model_name, cache_dir)
        if self.chunks:
            self.index_chunks()

    def _init_encoder(self, model_name: str, cache_dir: Optional[str]) -> None:
        """Initialize the ONNX embedding model."""
        try:
            from fastembed import TextEmbedding

            os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
            self.encoder = TextEmbedding(model_name=model_name, cache_dir=cache_dir)
            logger.info(f"Initialized DenseRetriever with FastEmbed model: {model_name}")
        except Exception as e:
            logger.warning(f"Could not load FastEmbed model '{model_name}': {e}. Using fallback vectorizer.")
            self.encoder = None

    def _embed_texts(self, texts: List[str]) -> np.ndarray:
        """Embed a list of text strings into normalized float32 vectors."""
        if self.encoder is not None:
            vectors = list(self.encoder.embed(texts))
            arr = np.array(vectors, dtype=np.float32)
            # L2 normalize
            norms = np.linalg.norm(arr, axis=1, keepdims=True)
            norms[norms == 0] = 1e-10
            return arr / norms
        else:
            # Fallback deterministic pseudo-semantic bag-of-character-trigrams vectorizer
            dim = 256
            matrix = np.zeros((len(texts), dim), dtype=np.float32)
            for i, text in enumerate(texts):
                for word in text.lower().split():
                    for j in range(max(1, len(word) - 2)):
                        trigram = word[j : j + 3]
                        idx = hash(trigram) % dim
                        matrix[i, idx] += 1.0
                norm = np.linalg.norm(matrix[i])
                if norm > 0:
                    matrix[i] /= norm
            return matrix

    def index_chunks(self) -> None:
        """Compute and store embeddings for all knowledge base chunks."""
        texts = [c.text for c in self.chunks]
        logger.info(f"Computing dense embeddings for {len(texts)} chunks...")
        self.embeddings = self._embed_texts(texts)

    def search(self, query: str, top_k: int = 10) -> List[Tuple[RetrievedChunk, float]]:
        """
        Search for matching chunks using cosine similarity.
        Returns a list of (RetrievedChunk, similarity_score) sorted in descending order.
        """
        if not query or not query.strip() or self.embeddings is None or not self.chunks:
            return []

        query_vec = self._embed_texts([query])[0]
        # Dot product of normalized vectors equals cosine similarity
        similarities = np.dot(self.embeddings, query_vec)
        top_indices = np.argsort(similarities)[::-1][:top_k]

        results = []
        for rank, idx in enumerate(top_indices, start=1):
            chunk = self.chunks[idx].model_copy()
            score = float(similarities[idx])
            chunk.score = score
            chunk.rank = rank
            results.append((chunk, score))

        return results
