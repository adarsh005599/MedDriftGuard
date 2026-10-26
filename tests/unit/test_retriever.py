"""
Unit tests for MedDriftGuard Hybrid Retriever (Phase 2).
Validates BM25, Dense semantic search, Reciprocal Rank Fusion,
Reranker scoring, and the architectural refusal of multi-turn chat history.
"""

from pathlib import Path

import pytest

from src.common.schemas import RetrievedChunk
from src.retriever.bm25_search import BM25Retriever
from src.retriever.dense_search import DenseRetriever
from src.retriever.fusion import reciprocal_rank_fusion
from src.retriever.hybrid_retriever import HybridRetriever
from src.retriever.reranker import CrossEncoderReranker


@pytest.fixture
def sample_chunks():
    """Create sample chunks for isolated unit testing."""
    return [
        RetrievedChunk(
            chunk_id="test_fever_01",
            source="MedlinePlus",
            url="https://example.com/fever",
            date_accessed="2026-10-25",
            section="Overview",
            language="en",
            topic="fever",
            license="Public Domain",
            text="Fever is an elevation in body temperature often caused by viral infections.",
        ),
        RetrievedChunk(
            chunk_id="test_cough_01",
            source="MedlinePlus",
            url="https://example.com/cough",
            date_accessed="2026-10-25",
            section="Overview",
            language="en",
            topic="cough",
            license="Public Domain",
            text="A cough helps clear mucus and foreign substances from airways and lungs.",
        ),
        RetrievedChunk(
            chunk_id="test_dengue_01",
            source="WHO",
            url="https://example.com/dengue",
            date_accessed="2026-10-25",
            section="Overview",
            language="en",
            topic="dengue",
            license="CC BY-NC-SA 3.0 IGO",
            text="Dengue is transmitted by Aedes mosquitoes and causes high fever and severe joint pain.",
        ),
    ]


def test_bm25_exact_keyword_matching(sample_chunks):
    retriever = BM25Retriever(sample_chunks)
    results = retriever.search("elevation in body temperature", top_k=2)

    assert len(results) > 0
    top_chunk, score = results[0]
    assert top_chunk.chunk_id == "test_fever_01"
    assert score > 0
    assert top_chunk.rank == 1


def test_empty_retrieval_indexes_and_queries_are_safe(tmp_path):
    bm25 = BM25Retriever([])
    assert bm25.search("fever") == []
    assert bm25.search("fever", top_k=0) == []

    retriever = HybridRetriever(chunks=[], config_path=Path("configs/retriever.yaml"))
    assert retriever.retrieve("fever") == []
    assert retriever.retrieve("   ") == []
    assert retriever.retrieve("fever", k=0) == []

    missing_index = HybridRetriever(chunks_path=tmp_path / "missing.jsonl")
    assert missing_index.retrieve("fever") == []


def test_hybrid_retriever_uses_configured_models(sample_chunks):
    retriever = HybridRetriever(chunks=sample_chunks)

    assert (
        retriever.dense.model_name == "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )
    assert retriever.reranker.model_name == "BAAI/bge-reranker-large"
    assert retriever.mode == "hybrid"


def test_chunk_loader_reports_malformed_jsonl(tmp_path):
    malformed = tmp_path / "malformed.jsonl"
    malformed.write_text("{not valid json}\n", encoding="utf-8")

    with pytest.raises(ValueError):
        HybridRetriever(chunks_path=malformed)

    invalid_record = tmp_path / "invalid_record.jsonl"
    invalid_record.write_text('{"chunk_id": "missing required fields"}\n', encoding="utf-8")
    with pytest.raises(ValueError):
        HybridRetriever(chunks_path=invalid_record)


def test_empty_jsonl_index_is_safe(tmp_path):
    empty_index = tmp_path / "empty.jsonl"
    empty_index.write_text("", encoding="utf-8")

    retriever = HybridRetriever(chunks_path=empty_index)

    assert retriever.retrieve("standalone query") == []


def test_dense_fallback_embeddings_are_reproducible(sample_chunks):
    first = DenseRetriever(sample_chunks)
    second = DenseRetriever(sample_chunks)

    if first.encoder is None and second.encoder is None:
        assert (first.embeddings == second.embeddings).all()


def test_dense_retriever_cosine_similarity(sample_chunks):
    dense = DenseRetriever(sample_chunks, model_name="BAAI/bge-small-en-v1.5")
    results = dense.search("mosquito-borne tropical fever and joint pain", top_k=1)

    assert len(results) == 1
    top_chunk, score = results[0]
    assert top_chunk.chunk_id == "test_dengue_01"
    assert -1.0 <= score <= 1.0


def test_reciprocal_rank_fusion(sample_chunks):
    c1, c2, c3 = sample_chunks
    list1 = [(c1, 10.0), (c2, 5.0)]
    list2 = [(c2, 8.0), (c3, 4.0)]

    fused = reciprocal_rank_fusion([list1, list2], k=60, top_k=3)
    assert len(fused) == 3

    # c2 was rank 2 in list1 and rank 1 in list2 => (1/62) + (1/61)
    # c1 was rank 1 in list1 => (1/61)
    # c2 should have highest fused score
    top_chunk, top_score = fused[0]
    assert top_chunk.chunk_id == "test_cough_01"
    assert top_score > fused[1][1]


def test_cross_encoder_reranker(sample_chunks):
    reranker = CrossEncoderReranker()
    reranked = reranker.rerank(
        query="cough and mucus",
        candidates=sample_chunks,
        top_k=2,
    )
    assert len(reranked) == 2
    assert reranked[0].chunk_id == "test_cough_01"
    assert reranked[0].rank == 1


def test_hybrid_retriever_strictly_prohibits_chat_history(sample_chunks):
    retriever = HybridRetriever(chunks=sample_chunks)

    # Valid string query should succeed
    results = retriever.retrieve("fever treatment", k=2)
    assert isinstance(results, list)

    # Passing raw chat history list must raise TypeError
    with pytest.raises(TypeError, match="chat history is forbidden"):
        retriever.retrieve([{"role": "user", "content": "fever"}])  # type: ignore

    # Passing raw dictionary must raise TypeError
    with pytest.raises(TypeError, match="chat history is forbidden"):
        retriever.retrieve({"history": ["turn 1", "turn 2"]})  # type: ignore


def test_hybrid_retriever_all_modes_execute(sample_chunks):
    retriever = HybridRetriever(chunks=sample_chunks)
    modes = ["bm25_only", "dense_only", "hybrid", "hybrid_rerank"]

    for m in modes:
        res = retriever.retrieve("mosquito bite fever", k=2, mode_override=m)  # type: ignore
        assert len(res) > 0
        assert all(isinstance(c, RetrievedChunk) for c in res)
        assert res[0].rank == 1
