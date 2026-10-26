"""
Unit tests for MedDriftGuard Patient-State Ledger (Phase 3).
Validates versioned fact addition, retractions, corrections, superseding pointers,
and privacy-safe summary formatting.
"""

import pytest
from src.common.schemas import LedgerFact, LedgerUpdate
from src.ledger.patient_ledger import PatientLedger


def test_ledger_add_fact():
    ledger = PatientLedger()
    update = LedgerUpdate(
        op="add",
        fact_type="symptom",
        value="high fever",
        source_text="I have high fever",
    )
    fact = ledger.apply_update(update, turn_id=1)

    assert fact.fact_id == "fact_001"
    assert fact.status == "active"
    assert fact.value == "high fever"
    assert len(ledger.get_active_facts()) == 1


def test_ledger_retract_fact():
    ledger = PatientLedger()
    # Turn 1: Add fever
    ledger.apply_update(
        LedgerUpdate(op="add", fact_type="symptom", value="fever"), turn_id=1
    )
    assert len(ledger.get_active_facts()) == 1

    # Turn 2: Retract fever ("nahi bukhar nahi hai")
    retract_update = LedgerUpdate(
        op="retract",
        fact_type="symptom",
        value="fever",
        source_text="nahi bukhar nahi hai",
    )
    retract_fact = ledger.apply_update(retract_update, turn_id=2)

    assert retract_fact.status == "retracted"
    assert retract_fact.supersedes == "fact_001"

    # Active facts should now be empty
    assert len(ledger.get_active_facts()) == 0
    # Retracted facts should contain the retracted fact
    retracted = ledger.get_retracted_facts()
    assert len(retracted) == 2  # original is retracted, retraction entry is recorded


def test_ledger_correct_fact_chain():
    ledger = PatientLedger()
    # Turn 1: User says cough
    f1 = ledger.apply_update(
        LedgerUpdate(op="add", fact_type="symptom", value="cough"), turn_id=1
    )
    assert f1.status == "active"

    # Turn 2: User corrects to dry cough
    f2 = ledger.apply_update(
        LedgerUpdate(op="correct", fact_type="symptom", value="dry cough", supersedes=f1.fact_id),
        turn_id=2,
    )
    assert f2.status == "active"
    assert f2.supersedes == f1.fact_id

    # f1 should now be marked corrected
    assert f1.status == "corrected"
    active = ledger.get_active_facts()
    assert len(active) == 1
    assert active[0].value == "dry cough"


def test_ledger_summary_string():
    ledger = PatientLedger()
    ledger.apply_update(
        LedgerUpdate(op="add", fact_type="symptom", value="dry cough"), turn_id=1
    )
    ledger.apply_update(
        LedgerUpdate(op="add", fact_type="duration", value="3 days"), turn_id=1
    )

    summary = ledger.to_summary_string()
    assert "Symptom: dry cough" in summary
    assert "Duration: 3 days" in summary


def test_ledger_never_stores_unmentioned_info():
    ledger = PatientLedger()
    ledger.apply_update(
        LedgerUpdate(op="add", fact_type="symptom", value="headache"), turn_id=1
    )
    active = ledger.get_active_facts()

    # Verify no diagnostic extrapolation occurred
    assert len(active) == 1
    assert active[0].value == "headache"
    assert active[0].fact_type == "symptom"
    assert active[0].value != "brain tumor"
