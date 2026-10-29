"""Tests for the optional legacy PDF/FAISS integration boundary."""

import pytest

from src.ingest.pdf_processor import PDFIngestError, PDFSource, _split_text
from src.retriever.faiss_index import FaissChunkIndex, FaissIndexError


def test_unapproved_pdf_source_cannot_become_trusted_evidence():
    source = PDFSource(
        source_id="upload",
        title="Uploaded document",
        url="https://example.org/document.pdf",
        license="Unknown",
        topic="health",
        date_accessed="2026-10-29",
    )
    with pytest.raises(PDFIngestError, match="not approved"):
        source.validate()


def test_pdf_chunk_split_is_deterministic_and_overlapping():
    chunks = _split_text("one two three four five six seven", chunk_size=4, overlap=1)
    assert chunks == ["one two three four", "four five six seven"]


def test_empty_faiss_index_search_is_safe():
    class Embeddings:
        def embed_documents(self, texts):
            return [[1.0, 0.0] for _ in texts]

        def embed_query(self, text):
            return [1.0, 0.0]

    index = FaissChunkIndex(Embeddings())
    assert index.search("fever") == []


def test_empty_faiss_index_cannot_be_persisted():
    class Embeddings:
        def embed_documents(self, texts):
            return [[1.0, 0.0] for _ in texts]

        def embed_query(self, text):
            return [1.0, 0.0]

    index = FaissChunkIndex(Embeddings())
    with pytest.raises(FaissIndexError):
        index.save("missing.faiss")
