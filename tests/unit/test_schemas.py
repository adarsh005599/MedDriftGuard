"""
Unit tests for MedDriftGuard Pydantic data schemas.
Validates model constraints, serialization, edge cases, and safety rules.
"""

from datetime import datetime, timezone
import pytest
from pydantic import ValidationError

from src.common.schemas import (
    Claim,
    LedgerFact,
    LedgerUpdate,
    Message,
    PipelineTrace,
    RetrievedChunk,
    TriageResult,
    VerifiedAnswer,
)
from src.common.i18n import detect_script
from src.common.seeds import set_seed


def test_set_seed_executes_without_error():
    set_seed(42)


def test_detect_script():
    assert detect_script("I have had a mild headache for three days.") == "en"
    assert detect_script("मुझे सिरदर्द और बुखार है।") == "devanagari"
    assert detect_script("mujhe 3 din se bukhar aur khansi hai") == "roman_hindi"
    assert detect_script("") == "unknown"


# --- Message Tests ---

def test_message_valid():
    msg = Message(
        role="user",
        content="Hello doctor, I have a sore throat.",
        turn_id=1,
        language="en",
        script="en",
    )
    assert msg.role == "user"
    assert msg.turn_id == 1
    assert isinstance(msg.timestamp, datetime)


def test_message_rejects_empty_content():
    with pytest.raises(ValidationError):
        Message(role="user", content="")


def test_message_rejects_invalid_role():
    with pytest.raises(ValidationError):
        Message(role="invalid_role", content="hello")  # type: ignore


# --- LedgerFact Tests ---

def test_ledger_fact_valid():
    fact = LedgerFact(
        fact_id="fact_001",
        fact_type="symptom",
        value="dry cough",
        turn_id=1,
        status="active",
        source_text="sirf khansi hai",
    )
    assert fact.fact_id == "fact_001"
    assert fact.status == "active"
    assert fact.fact_type == "symptom"
    assert fact.value == "dry cough"


def test_ledger_fact_retraction():
    fact = LedgerFact(
        fact_id="fact_002",
        fact_type="symptom",
        value="fever",
        turn_id=2,
        status="retracted",
        supersedes="fact_001",
        source_text="nahi, bukhar nahi hai",
    )
    assert fact.status == "retracted"
    assert fact.supersedes == "fact_001"


def test_ledger_fact_rejects_empty_value():
    with pytest.raises(ValidationError):
        LedgerFact(
            fact_id="f1",
            fact_type="symptom",
            value="   ",
            turn_id=0,
        )


def test_ledger_fact_rejects_invalid_fact_type():
    with pytest.raises(ValidationError):
        LedgerFact(
            fact_id="f1",
            fact_type="doctor_diagnosis",  # type: ignore
            value="malaria",
            turn_id=0,
        )


# --- LedgerUpdate Tests ---

def test_ledger_update_valid():
    update = LedgerUpdate(
        op="retract",
        fact_type="symptom",
        value="fever",
        supersedes="fact_001",
        source_text="bukhar nahi hai",
    )
    assert update.op == "retract"
    assert update.value == "fever"


# --- RetrievedChunk Tests ---

def test_retrieved_chunk_valid():
    chunk = RetrievedChunk(
        chunk_id="mp_fever_0001",
        source="MedlinePlus",
        url="https://medlineplus.gov/fever.html",
        date_accessed="2026-10-25",
        section="Overview and Symptoms",
        language="en",
        topic="fever",
        license="Public Domain (U.S. Government)",
        text="A fever is a body temperature that is higher than normal. It is a symptom of many illnesses.",
        score=0.92,
        rank=1,
    )
    assert chunk.chunk_id == "mp_fever_0001"
    assert chunk.license == "Public Domain (U.S. Government)"
    assert chunk.score == 0.92


def test_retrieved_chunk_rejects_empty_license():
    with pytest.raises(ValidationError):
        RetrievedChunk(
            chunk_id="mp_fever_0002",
            source="MedlinePlus",
            url="https://medlineplus.gov/fever.html",
            date_accessed="2026-10-25",
            section="Overview",
            language="en",
            topic="fever",
            license="  ",  # Must have valid license
            text="Fever is a high temperature.",
        )


def test_retrieved_chunk_rejects_dosage_instructions():
    with pytest.raises(ValidationError, match="prohibited pharmaceutical dosing"):
        RetrievedChunk(
            chunk_id="mp_bad_0001",
            source="Unknown",
            url="https://example.com",
            date_accessed="2026-10-25",
            section="Dosage",
            language="en",
            topic="headache",
            license="Open License",
            text="Take 500mg tablets every 4 hours for relief.",
        )


# --- Claim Tests ---

def test_claim_valid():
    claim = Claim(
        claim_id="claim_001",
        text="Drinking fluids helps prevent dehydration during fever.",
        supported=True,
        confidence=0.95,
        cited_chunk_ids=["mp_fever_0001"],
        justification="Text explicitly mentions staying hydrated.",
        label="supported",
    )
    assert claim.claim_id == "claim_001"
    assert claim.supported is True
    assert claim.confidence == 0.95
    assert claim.label == "supported"


def test_claim_confidence_range():
    with pytest.raises(ValidationError):
        Claim(
            claim_id="c2",
            text="Some claim",
            confidence=1.5,  # Out of [0, 1] bounds
        )


# --- TriageResult Tests ---

def test_triage_result_emergency():
    res = TriageResult(
        is_emergency=True,
        category="emergency",
        confidence=0.99,
        matched_rules=["chest_pain_severe"],
        emergency_message="Call 112 or 108 immediately.",
    )
    assert res.is_emergency is True
    assert res.category == "emergency"
    assert len(res.matched_rules) == 1


# --- VerifiedAnswer Tests ---

def test_verified_answer():
    ans = VerifiedAnswer(
        final_text="Fever is a temporary increase in body temperature [mp_fever_0001].",
        claims=[
            Claim(
                claim_id="c1",
                text="Fever is a temporary increase in body temperature.",
                supported=True,
                label="supported",
            )
        ],
        unsupported_count=0,
        abstained=False,
        citations=["mp_fever_0001"],
    )
    assert not ans.abstained
    assert len(ans.citations) == 1


# --- PipelineTrace Tests ---

def test_pipeline_trace_comprehensive():
    fact = LedgerFact(
        fact_id="f1",
        fact_type="symptom",
        value="headache",
        turn_id=0,
        status="active",
        source_text="sar dard hai",
    )
    chunk = RetrievedChunk(
        chunk_id="chunk_01",
        source="WHO",
        url="https://who.int/headache",
        date_accessed="2026-10-25",
        section="Primary Headaches",
        language="en",
        topic="headache",
        license="CC BY-NC-SA 3.0 IGO",
        text="Headache disorders are among the most common disorders of the nervous system.",
    )
    trace = PipelineTrace(
        trace_id="tr_001",
        conv_id="conv_123",
        turn_id=0,
        user_message="mujhe sar dard hai",
        detected_language="hi",
        detected_script="roman_hindi",
        triage_decision=TriageResult(is_emergency=False, category="routine"),
        retrieval_query="headache symptoms and general care",
        ledger_state_after=[fact],
        retrieved_chunks=[chunk],
        final_answer="Sar dard ek aam samsya hai [chunk_01].",
        citations=["chunk_01"],
        abstained=False,
        stage_latencies={"retriever": 0.045, "generator": 0.210},
        total_latency=0.312,
    )
    assert trace.trace_id == "tr_001"
    assert len(trace.retrieved_chunks) == 1
    assert trace.ledger_state_after[0].value == "headache"
    # Ensure serializable to JSON
    json_repr = trace.model_dump_json()
    assert "conv_123" in json_repr
