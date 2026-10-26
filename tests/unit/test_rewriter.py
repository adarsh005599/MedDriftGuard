"""
Unit tests and acceptance tests for MedDriftGuard Rewriter (Phase 3).
Validates standalone query reformulation, topic shift detection, pronoun resolution,
and executes all 30 hand-written test conversations from tests/fixtures/conversations_p3.json.
"""

import json
from pathlib import Path

import requests

from src.ledger.patient_ledger import PatientLedger
from src.rewriter.rule_extractor import SYMPTOM_LEXICON, RuleBasedExtractor
from src.rewriter.standalone_rewriter import StandaloneRewriter


def _canonical_expected_symptoms(values):
    canonical = {SYMPTOM_LEXICON.get(value.lower(), value.lower()) for value in values}
    for broad, specific in (
        ("fever", {"high fever"}),
        ("cough", {"dry cough", "productive cough"}),
        ("headache", {"tension headache", "severe headache"}),
    ):
        if canonical.intersection(specific):
            canonical.discard(broad)
    return canonical


class FakeLLMClient:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def generate(self, system_prompt, user_prompt):
        self.calls.append((system_prompt, json.loads(user_prompt)))
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


def test_rule_extractor_negation_and_retraction():
    extractor = RuleBasedExtractor()
    updates = extractor.extract_updates("nahi bukhar nahi hai, sirf khansi hai")

    ops = [u.op for u in updates]
    vals = [u.value for u in updates]

    assert "retract" in ops
    assert "add" in ops
    assert "fever" in vals
    assert "cough" in vals


def test_topic_shift_detection():
    extractor = RuleBasedExtractor()
    assert extractor.detect_topic_shift("chalo ye chhodo, mujhe pet mein dard hai") is True
    assert extractor.detect_topic_shift("forget that, I have another question") is True
    assert extractor.detect_topic_shift("यह छोड़ो, दूसरा सवाल एसिडिटी का है") is True
    assert extractor.detect_topic_shift("mujhe 2 din se bukhar hai") is False


def test_pronoun_followup_resolution():
    ledger = PatientLedger()
    rewriter = StandaloneRewriter()

    # Turn 1: User reports dry cough
    res1 = rewriter.rewrite(
        user_message="I have dry cough for 3 days",
        ledger=ledger,
        turn_id=1,
    )
    ledger.apply_updates(res1.ledger_updates, turn_id=1)
    assert "dry cough" in ledger.get_active_symptoms()

    # Turn 2: User asks pronoun follow-up
    res2 = rewriter.rewrite(
        user_message="What is its home care and treatment?",
        ledger=ledger,
        turn_id=2,
    )
    assert "dry cough" in res2.standalone_query_en.lower()
    assert "home care" in res2.standalone_query_en.lower()


def test_prompted_rewriter_validates_and_uses_mock_response():
    output = {
        "standalone_query_en": "home care for cough",
        "detected_script": "en",
        "language": "en",
        "topic_shift": False,
        "ledger_updates": [
            {
                "op": "add",
                "fact_type": "symptom",
                "value": "cough",
                "source_text": "cough",
            },
            {
                "op": "add",
                "fact_type": "age_group",
                "value": "adult",
                "source_text": "adult",
            },
        ],
    }
    client = FakeLLMClient([json.dumps(output)])
    rewriter = StandaloneRewriter(llm_client=client)

    result = rewriter.rewrite("I am an adult with cough", PatientLedger(), turn_id=4)

    assert result.standalone_query_en == "home care for cough"
    assert result.detected_script == "en"
    assert result.language == "en"
    assert [update.fact_type for update in result.ledger_updates] == ["symptom", "age_group"]
    assert client.calls[0][1]["last_turns"] == []


def test_prompted_rewriter_retries_malformed_json_then_succeeds():
    output = {
        "standalone_query_en": "cough warning signs",
        "detected_script": "en",
        "language": "en",
        "topic_shift": False,
        "ledger_updates": [],
    }
    client = FakeLLMClient(["not json", json.dumps(output)])
    result = StandaloneRewriter(llm_client=client, max_retries=1).rewrite(
        "What are cough warning signs?", PatientLedger()
    )

    assert result.standalone_query_en == "cough warning signs"
    assert len(client.calls) == 2


def test_prompted_rewriter_accepts_documented_facts_with_verbatim_sources():
    message = "I am an adult, allergic to pollen, take cetirizine, and have asthma."
    updates = [
        ("age_group", "adult"),
        ("allergy_mentioned", "pollen"),
        ("medication_mentioned", "cetirizine"),
        ("condition_mentioned", "asthma"),
    ]
    output = {
        "standalone_query_en": "general asthma information",
        "detected_script": "en",
        "language": "en",
        "topic_shift": False,
        "ledger_updates": [
            {
                "op": "add",
                "fact_type": fact_type,
                "value": value,
                "source_text": value,
            }
            for fact_type, value in updates
        ],
    }
    result = StandaloneRewriter(llm_client=FakeLLMClient([json.dumps(output)])).rewrite(
        message, PatientLedger()
    )

    assert [(update.fact_type, update.value) for update in result.ledger_updates] == updates


def test_prompted_rewriter_rejects_unquoted_user_facts_and_falls_back():
    output = {
        "standalone_query_en": "fever and cough care",
        "detected_script": "en",
        "language": "en",
        "topic_shift": False,
        "ledger_updates": [
            {
                "op": "add",
                "fact_type": "symptom",
                "value": "fever",
                "source_text": "fever",
            }
        ],
    }
    client = FakeLLMClient([json.dumps(output), json.dumps(output)])
    result = StandaloneRewriter(llm_client=client, max_retries=1).rewrite(
        "I only have a cough", PatientLedger()
    )

    assert [(update.fact_type, update.value) for update in result.ledger_updates] == [
        ("symptom", "cough")
    ]
    assert len(client.calls) == 2
    assert "fever" not in result.standalone_query_en.lower()


def test_prompted_rewriter_falls_back_when_local_service_is_unavailable():
    client = FakeLLMClient([requests.ConnectionError("Ollama is not running")])
    result = StandaloneRewriter(llm_client=client, max_retries=0).rewrite(
        "I have a cough", PatientLedger()
    )

    assert "cough" in result.standalone_query_en.lower()
    assert [(update.fact_type, update.value) for update in result.ledger_updates] == [
        ("symptom", "cough")
    ]


def test_empty_input_and_three_script_detection():
    rewriter = StandaloneRewriter()
    empty = rewriter.rewrite("  ", PatientLedger())
    assert empty.standalone_query_en == ""
    assert empty.ledger_updates == []

    for message, expected_script, expected_language in (
        ("I have a cough", "en", "en"),
        ("मुझे खांसी है", "devanagari", "hi"),
        ("mujhe khansi hai", "roman_hindi", "hi"),
    ):
        result = rewriter.rewrite(message, PatientLedger())
        assert result.detected_script == expected_script
        assert result.language == expected_language


def test_retriever_receives_only_standalone_query():
    class RecordingRetriever:
        def __init__(self):
            self.query = None

        def retrieve(self, standalone_query, k=5):
            self.query = standalone_query
            return []

    ledger = PatientLedger()
    rewriter = StandaloneRewriter()
    first = rewriter.rewrite("I have a cough", ledger, turn_id=1)
    ledger.apply_updates(first.ledger_updates, turn_id=1)
    previous_turn = {"role": "user", "content": "I have a cough"}
    result = rewriter.rewrite(
        "What is its home care?",
        ledger,
        last_turns=[previous_turn],
        turn_id=2,
    )
    retriever = RecordingRetriever()

    retriever.retrieve(result.standalone_query_en, k=5)

    assert retriever.query == result.standalone_query_en
    assert retriever.query == "home care and hydration management for cough"
    assert previous_turn["content"] not in retriever.query


def test_ambiguous_followup_without_context_does_not_invent_topic():
    result = StandaloneRewriter().rewrite("What about that?", PatientLedger())

    assert result.ledger_updates == []
    assert result.standalone_query_en == "What about that?"


def test_rewriter_logs_do_not_include_patient_query(caplog):
    StandaloneRewriter().rewrite("I have a cough", PatientLedger(), turn_id=7)

    assert "I have a cough" not in caplog.text
    assert "cough symptoms causes and home care" not in caplog.text
    assert "Turn 7" in caplog.text


def test_topic_shift_does_not_reuse_old_ledger_symptom():
    ledger = PatientLedger()
    rewriter = StandaloneRewriter()
    first = rewriter.rewrite("I have a cough", ledger, turn_id=1)
    ledger.apply_updates(first.ledger_updates, turn_id=1)

    result = rewriter.rewrite(
        "Forget that, what are dengue warning signs?",
        ledger,
        last_turns=[{"role": "user", "content": "I have a cough"}],
        turn_id=2,
    )

    assert result.topic_shift is True
    assert "dengue" in result.standalone_query_en.lower()
    assert "cough" not in result.standalone_query_en.lower()
    assert "cough" in ledger.get_active_symptoms()
    assert result.ledger_updates == []


def test_30_conversations_acceptance_suite():
    """
    Acceptance test for Phase 3:
    Executes all 30 hand-written multi-turn test conversations (English, Roman Hinglish,
    and Devanagari Hindi) to verify correction handling, topic shifts, and standalone queries.
    """
    fixtures_path = Path("tests/fixtures/conversations_p3.json")
    assert fixtures_path.exists(), "conversations_p3.json fixture must exist"

    data = json.loads(fixtures_path.read_text(encoding="utf-8"))
    conversations = data["conversations"]
    assert len(conversations) == 30, f"Expected 30 test conversations, found {len(conversations)}"

    rewriter = StandaloneRewriter()

    for conv in conversations:
        conv_id = conv["conv_id"]
        ledger = PatientLedger()
        history = []

        for turn in conv["turns"]:
            turn_id = turn["turn_id"]
            user_msg = turn["user_message"]

            res = rewriter.rewrite(
                user_message=user_msg,
                ledger=ledger,
                last_turns=history[-2:] if history else [],
                turn_id=turn_id,
            )

            # Apply updates to ledger
            ledger.apply_updates(res.ledger_updates, turn_id=turn_id)

            expected_active = _canonical_expected_symptoms(turn["expected_active_symptoms"])
            actual_active = {value.lower() for value in ledger.get_active_symptoms()}
            assert actual_active == expected_active, (
                f"[{conv_id} Turn {turn_id}] Expected active facts {sorted(expected_active)}, "
                f"found {sorted(actual_active)}"
            )

            # 1. Verify topic shift detection if expected
            if turn["expected_topic_shift"]:
                assert (
                    res.topic_shift is True
                ), f"[{conv_id} Turn {turn_id}] Expected topic shift to be True for: {user_msg}"

            # 2. Verify retractions
            for expected_retract in turn["expected_retractions"]:
                active_symptoms = [s.lower() for s in ledger.get_active_symptoms()]
                assert (
                    expected_retract.lower() not in active_symptoms
                ), f"[{conv_id} Turn {turn_id}] Retracted symptom '{expected_retract}' still active in ledger! Active: {active_symptoms}"

            # 3. Verify query constraints
            query_lower = res.standalone_query_en.lower()
            for must_contain in turn["query_must_contain"]:
                assert (
                    must_contain.lower() in query_lower
                ), f"[{conv_id} Turn {turn_id}] Query '{res.standalone_query_en}' missing '{must_contain}'"

            for must_not in turn["query_must_not_contain"]:
                assert (
                    must_not.lower() not in query_lower
                ), f"[{conv_id} Turn {turn_id}] Query '{res.standalone_query_en}' contains forbidden '{must_not}'"

            # Record history
            history.append({"role": "user", "content": user_msg})
