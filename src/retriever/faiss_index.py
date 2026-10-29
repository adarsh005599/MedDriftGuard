"""Optional FAISS index persistence for provenance-rich retrieved chunks.

The metadata sidecar is JSON, not pickle.  Loading validates the index dimension
and chunk mapping before any search result is returned.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Protocol

import numpy as np

from src.common.schemas import RetrievedChunk


class EmbeddingProvider(Protocol):
    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


class FaissIndexError(RuntimeError):
    """Raised for unavailable or invalid FAISS index operations."""


class FaissChunkIndex:
    """Build, persist, load, and search a FAISS inner-product index."""

    def __init__(self, embeddings: EmbeddingProvider):
        self.embeddings = embeddings
        self.index = None
        self.chunks: list[RetrievedChunk] = []

    @staticmethod
    def _faiss():
        try:
            import faiss  # type: ignore

            return faiss
        except ImportError as error:
            raise FaissIndexError("FAISS support requires the optional 'faiss' dependency") from error

    @staticmethod
    def _normalise(vectors: list[list[float]] | np.ndarray) -> np.ndarray:
        array = np.asarray(vectors, dtype="float32")
        if array.ndim != 2 or array.shape[0] == 0 or array.shape[1] == 0:
            raise FaissIndexError("Embedding provider returned no usable vectors")
        norms = np.linalg.norm(array, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return array / norms

    def build(self, chunks: list[RetrievedChunk]) -> None:
        self.chunks = list(chunks)
        if not self.chunks:
            self.index = None
            return
        vectors = self._normalise(self.embeddings.embed_documents([chunk.text for chunk in self.chunks]))
        faiss = self._faiss()
        self.index = faiss.IndexFlatIP(vectors.shape[1])
        self.index.add(vectors)

    def save(self, index_path: Path, metadata_path: Path | None = None) -> None:
        if self.index is None or not self.chunks:
            raise FaissIndexError("Cannot save an empty FAISS index")
        faiss = self._faiss()
        index_path.parent.mkdir(parents=True, exist_ok=True)
        metadata_path = metadata_path or index_path.with_suffix(index_path.suffix + ".json")
        faiss.write_index(self.index, str(index_path))
        metadata_path.write_text(
            json.dumps([chunk.model_dump(mode="json") for chunk in self.chunks], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def load(self, index_path: Path, metadata_path: Path | None = None) -> None:
        if not index_path.exists():
            raise FaissIndexError(f"FAISS index does not exist: {index_path}")
        metadata_path = metadata_path or index_path.with_suffix(index_path.suffix + ".json")
        if not metadata_path.exists():
            raise FaissIndexError(f"FAISS metadata sidecar does not exist: {metadata_path}")
        faiss = self._faiss()
        try:
            index = faiss.read_index(str(index_path))
            records = json.loads(metadata_path.read_text(encoding="utf-8"))
            chunks = [RetrievedChunk(**record) for record in records]
        except Exception as error:
            raise FaissIndexError("Could not load the FAISS index and JSON metadata sidecar") from error
        if index.ntotal != len(chunks):
            raise FaissIndexError("FAISS vector count does not match metadata count")
        self.index = index
        self.chunks = chunks

    def search(self, query: str, top_k: int = 5) -> list[tuple[RetrievedChunk, float]]:
        if not query.strip() or top_k <= 0 or self.index is None or not self.chunks:
            return []
        vector = self._normalise([self.embeddings.embed_query(query)])
        scores, indices = self.index.search(vector, min(top_k, len(self.chunks)))
        return [
            (self.chunks[int(index)], float(score))
            for score, index in zip(scores[0], indices[0])
            if int(index) >= 0
        ]
