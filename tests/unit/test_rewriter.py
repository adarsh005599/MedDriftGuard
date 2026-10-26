"""
Unit tests and acceptance tests for MedDriftGuard Rewriter (Phase 3).
Validates standalone query reformulation, topic shift detection, pronoun resolution,
and executes all 30 hand-written test conversations from tests/fixtures/conversations_p3.json.
"""

import json
from pathlib import Path
import pytest

from src.ledger.patient_ledger import PatientLedger
from src.rewriter.standalone_rewriter import StandaloneRewriter
from src.rewriter.rule_extractor import RuleBasedExtractor


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

            # 1. Verify topic shift detection if expected
            if turn["expected_topic_shift"]:
                assert res.topic_shift is True, (
                    f"[{conv_id} Turn {turn_id}] Expected topic shift to be True for: {user_msg}"
                )

            # 2. Verify retractions
            for expected_retract in turn["expected_retractions"]:
                active_symptoms = [s.lower() for s in ledger.get_active_symptoms()]
                assert expected_retract.lower() not in active_symptoms, (
                    f"[{conv_id} Turn {turn_id}] Retracted symptom '{expected_retract}' still active in ledger! Active: {active_symptoms}"
                )

            # 3. Verify query constraints
            query_lower = res.standalone_query_en.lower()
            for must_contain in turn["query_must_contain"]:
                assert must_contain.lower() in query_lower, (
                    f"[{conv_id} Turn {turn_id}] Query '{res.standalone_query_en}' missing '{must_contain}'"
                )

            for must_not in turn["query_must_not_contain"]:
                assert must_not.lower() not in query_lower, (
                    f"[{conv_id} Turn {turn_id}] Query '{res.standalone_query_en}' contains forbidden '{must_not}'"
                )

            # Record history
            history.append({"role": "user", "content": user_msg})
