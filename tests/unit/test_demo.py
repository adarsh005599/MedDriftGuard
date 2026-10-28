import json

from demo import EvidenceExcerptGenerator, build_offline_demo_components, process_question
from src.common.schemas import RetrievedChunk


def sample_chunk():
    return RetrievedChunk(
        chunk_id="demo_cough_01",
        source="MedlinePlus",
        url="https://example.com/cough",
        date_accessed="2026-10-27",
        section="Home care",
        language="en",
        topic="cough",
        license="Public Domain",
        text="[Cough | Home Care]\nWarm fluids may soothe an irritated throat.",
    )


class RecordingRetriever:
    def __init__(self, chunks):
        self.chunks = chunks
        self.query = None

    def retrieve(self, standalone_query, k=5):
        self.query = standalone_query
        return self.chunks


def test_offline_demo_retrieves_and_verifies_safe_answer():
    chunk = sample_chunk()
    retriever = RecordingRetriever([chunk])
    generator, verifier = build_offline_demo_components()

    result = process_question(
        "What home care is mentioned for a cough?", retriever, generator, verifier
    )

    assert retriever.query == result.standalone_query
    assert result.chunks == [chunk]
    assert "[demo_cough_01]" in result.verified_answer.final_text
    assert not result.verified_answer.abstained
    assert result.verified_answer.citations == ["demo_cough_01"]


def test_demo_missing_evidence_shows_safe_abstention():
    retriever = RecordingRetriever([])
    generator, verifier = build_offline_demo_components()

    result = process_question(
        "What home care is mentioned for a cough?",
        retriever,
        generator,
        verifier,
        simulate_missing_evidence=True,
    )

    assert result.evidence_missing_simulated
    assert result.chunks == []
    assert result.verified_answer.abstained
    assert result.verified_answer.citations == []


def test_demo_excerpt_avoids_source_wording_that_looks_like_a_diagnosis():
    payload = {
        "evidence": [
            {
                "chunk_id": "fever_01",
                "text": "[Fever | Symptoms]\nYou have a fever when your temperature rises. Dehydration may occur during fevers.",
            }
        ]
    }

    result = json.loads(EvidenceExcerptGenerator().generate("system", json.dumps(payload)))

    assert result["citations"] == ["fever_01"]
    assert "You have a fever" not in result["short_answer"]
    assert "Dehydration may occur during fevers" in result["short_answer"]
