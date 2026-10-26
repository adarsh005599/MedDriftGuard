from eval.retrieval_eval import evaluate_mode, generate_markdown_report
from src.common.schemas import RetrievedChunk


def test_retrieval_evaluator_computes_recall_and_mrr():
    chunks = [
        RetrievedChunk(
            chunk_id=f"chunk_{index}",
            source="Test",
            url="https://example.com/topic",
            date_accessed="2026-10-26",
            section="Overview",
            language="en",
            topic="topic",
            license="Test license",
            text=f"This is a sufficiently long test passage for chunk {index}.",
        )
        for index in range(1, 12)
    ]

    class FakeRetriever:
        def __init__(self):
            self.calls = []

        def retrieve(self, query, k, mode_override):
            self.calls.append((query, k, mode_override))
            return chunks

    retriever = FakeRetriever()
    questions = [
        {"query": "standalone query", "gold_chunk_id": "chunk_3", "script": "en"},
        {"query": "another query", "gold_chunk_id": "not_present", "script": "roman_hindi"},
    ]

    result = evaluate_mode(retriever, questions, "bm25_only")

    assert result["total_queries"] == 2
    assert result["recall_at_5"] == 0.5
    assert result["recall_at_10"] == 0.5
    assert result["mrr"] == 0.1667
    assert retriever.calls == [
        ("standalone query", 10, "bm25_only"),
        ("another query", 10, "bm25_only"),
    ]


def test_retrieval_report_discloses_backends_and_observed_results(tmp_path):
    report_path = tmp_path / "retrieval_results.md"
    results = [
        {
            "mode": "bm25_only",
            "recall_at_5": 0.7,
            "recall_at_10": 0.8,
            "mrr": 0.5,
            "by_script": {},
        },
        {
            "mode": "dense_only",
            "recall_at_5": 0.9,
            "recall_at_10": 0.95,
            "mrr": 0.7,
            "by_script": {},
            "dense_backend": "FastEmbed:test-model",
        },
        {
            "mode": "hybrid",
            "recall_at_5": 0.85,
            "recall_at_10": 0.95,
            "mrr": 0.65,
            "by_script": {},
            "dense_backend": "FastEmbed:test-model",
        },
        {
            "mode": "hybrid_rerank",
            "recall_at_5": 0.85,
            "recall_at_10": 0.95,
            "mrr": 0.68,
            "by_script": {},
            "dense_backend": "FastEmbed:test-model",
            "reranker_backend": "deterministic-contextual-fallback",
        },
    ]

    generate_markdown_report(results, report_path)

    report = report_path.read_text(encoding="utf-8")
    assert "FastEmbed:test-model" in report
    assert "deterministic-contextual-fallback" in report
    assert "did not exceed" in report
    assert "changed MRR" in report
    assert "clinical correctness" in report
