"""
Data schemas for MedDriftGuard.

All core pipeline structures, facts, chunks, claims, and analysis traces are
strictly typed and validated using Pydantic v2 models.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Any, Literal, Optional
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

# Allowed fact types according to project specification
FactType = Literal[
    "age_group",
    "symptom",
    "duration",
    "medication_mentioned",
    "allergy_mentioned",
    "condition_mentioned",
    "user_correction",
]

# Status of facts in the patient-state ledger
FactStatus = Literal["active", "corrected", "retracted"]

# Triage classification categories
TriageCategory = Literal["emergency", "urgent", "routine", "non_medical"]

# Language scripts
ScriptType = Literal["en", "devanagari", "roman_hindi", "unknown"]

# Claim verification verdict labels
ClaimVerdictLabel = Literal["supported", "contradicted", "not_enough_info"]


class Message(BaseModel):
    """Represents a conversation turn message."""

    model_config = ConfigDict(extra="forbid")

    role: Literal["user", "assistant", "system"]
    content: str = Field(..., min_length=1, description="Message text content")
    turn_id: Optional[int] = Field(default=None, ge=0, description="Sequential turn index")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    language: Optional[str] = Field(default=None, description="ISO language code (e.g. 'en', 'hi')")
    script: Optional[ScriptType] = Field(default=None, description="Writing script")


class LedgerFact(BaseModel):
    """
    A single versioned fact stated explicitly by the patient.
    The ledger only tracks user-stated facts, never inferred medical diagnoses.
    """

    model_config = ConfigDict(extra="forbid")

    fact_id: str = Field(..., min_length=1, description="Unique fact identifier (e.g. 'f_001')")
    fact_type: FactType = Field(..., description="Category of user-stated fact")
    value: str = Field(..., min_length=1, description="User-reported value")
    turn_id: int = Field(..., ge=0, description="Turn index where fact was stated/updated")
    status: FactStatus = Field(default="active", description="Lifecycle state of this fact")
    supersedes: Optional[str] = Field(
        default=None, description="ID of earlier fact replaced or retracted by this fact"
    )
    source_text: Optional[str] = Field(
        default=None, description="Original user utterance text containing this fact"
    )
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @field_validator("value")
    @classmethod
    def validate_value_not_empty(cls, v: str) -> str:
        cleaned = v.strip()
        if not cleaned:
            raise ValueError("Fact value cannot be blank or whitespace only")
        return cleaned


class LedgerUpdate(BaseModel):
    """Instruction for updating the patient-state ledger from a single user turn."""

    model_config = ConfigDict(extra="forbid")

    op: Literal["add", "retract", "correct"]
    fact_type: FactType
    value: str = Field(..., min_length=1)
    supersedes: Optional[str] = None
    source_text: Optional[str] = None


class RetrievedChunk(BaseModel):
    """
    A single passage retrieved from the curated medical knowledge base.
    Must always have verified license metadata and non-empty text.
    """

    model_config = ConfigDict(extra="forbid")

    chunk_id: str = Field(..., min_length=1, description="Deterministic chunk identifier")
    source: str = Field(..., min_length=1, description="Knowledge base source name (e.g. MedlinePlus)")
    url: str = Field(..., min_length=1, description="Source webpage URL")
    date_accessed: str = Field(..., min_length=4, description="Date content was scraped/indexed (YYYY-MM-DD)")
    section: str = Field(..., min_length=1, description="Document section header")
    language: str = Field(default="en", description="Language code of chunk text")
    topic: str = Field(..., min_length=1, description="Medical topic (e.g. 'fever', 'asthma')")
    license: str = Field(..., min_length=1, description="License or reuse terms (e.g. 'Public Domain')")
    text: str = Field(..., min_length=10, description="Cleaned chunk text content")
    score: Optional[float] = Field(default=None, description="Retrieval or reranking score")
    rank: Optional[int] = Field(default=None, ge=1, description="Rank position in retrieval output")

    @field_validator("license")
    @classmethod
    def validate_license(cls, v: str) -> str:
        cleaned = v.strip()
        if not cleaned:
            raise ValueError("Knowledge base chunk must have an explicit, non-empty license")
        return cleaned

    @field_validator("text")
    @classmethod
    def validate_safety_no_dosage_tables(cls, v: str) -> str:
        # Warning check: MedDriftGuard chunks should not contain explicit dosage tables
        # Example pattern: "dosage: 500mg every 4 hours" or "500 mg tablets"
        dosage_regex = re.compile(
            r"\b\d+\s*(?:mg|ml|mcg|iu|tablets?|capsules?)(?:\s+(?:tablets?|capsules?|drops?|pills?))?\s+(?:every|daily|per\s+day|bd|tds|od|hours?)\b",
            re.IGNORECASE,
        )
        if dosage_regex.search(v):
            raise ValueError("Chunk contains prohibited pharmaceutical dosing prescription text")
        return v


class Claim(BaseModel):
    """An atomic factual claim extracted from a generated response."""

    model_config = ConfigDict(extra="forbid")

    claim_id: str = Field(..., min_length=1, description="Unique claim identifier (e.g. 'c_01')")
    text: str = Field(..., min_length=1, description="Atomic, self-contained statement")
    supported: Optional[bool] = Field(
        default=None, description="True if evidence entails this claim"
    )
    confidence: Optional[float] = Field(
        default=None, ge=0.0, le=1.0, description="NLI entailment model confidence"
    )
    cited_chunk_ids: list[str] = Field(
        default_factory=list, description="IDs of knowledge base chunks supporting this claim"
    )
    justification: Optional[str] = Field(
        default=None, description="Explanation or supporting sentence from evidence"
    )
    label: Optional[ClaimVerdictLabel] = Field(
        default=None, description="Categorical verifier outcome"
    )


class TriageResult(BaseModel):
    """Outcome of the red-flag safety triage gate."""

    model_config = ConfigDict(extra="forbid")

    is_emergency: bool = Field(..., description="True if immediate medical attention is required")
    category: TriageCategory = Field(..., description="Assigned triage category")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="Detection confidence")
    matched_rules: list[str] = Field(
        default_factory=list, description="Identifiers of matched red-flag lexicon rules"
    )
    emergency_message: Optional[str] = Field(
        default=None, description="Fixed safe response for emergency or self-harm triggers"
    )


class VerifiedAnswer(BaseModel):
    """Final answer produced by the generator and validated by the claim verifier."""

    model_config = ConfigDict(extra="forbid")

    final_text: str = Field(..., description="Safe final response text delivered to the patient")
    claims: list[Claim] = Field(default_factory=list, description="List of verified atomic claims")
    unsupported_count: int = Field(default=0, ge=0, description="Number of unsupported claims rejected")
    abstained: bool = Field(default=False, description="True if system safely abstained due to insufficient evidence")
    citations: list[str] = Field(default_factory=list, description="Unique chunk IDs cited in the answer")


class PipelineTrace(BaseModel):
    """
    Comprehensive execution trace for a single conversation turn.
    Records stage inputs and outputs to enable scientific evaluation of drift.
    """

    model_config = ConfigDict(extra="allow")

    trace_id: str = Field(..., min_length=1, description="Unique trace identifier")
    conv_id: str = Field(..., min_length=1, description="Conversation session ID")
    turn_id: int = Field(..., ge=0, description="Zero-indexed turn number")
    user_message: str = Field(..., min_length=1, description="Raw input text from user")
    detected_language: Optional[str] = Field(default=None, description="Detected ISO language")
    detected_script: Optional[ScriptType] = Field(default=None, description="Detected script")

    # Stage 1: Triage
    triage_decision: Optional[TriageResult] = Field(default=None, description="Triage gate result")

    # Stage 2: Rewriter
    rewriter_input: Optional[dict[str, Any]] = Field(default=None, description="Inputs passed to rewriter")
    rewriter_output: Optional[dict[str, Any]] = Field(default=None, description="Outputs from rewriter")
    retrieval_query: Optional[str] = Field(default=None, description="Clean standalone English query")

    # Stage 3: Ledger
    ledger_state_before: list[LedgerFact] = Field(default_factory=list)
    ledger_updates: list[LedgerUpdate] = Field(default_factory=list)
    ledger_state_after: list[LedgerFact] = Field(default_factory=list)

    # Stage 4: Retriever (evidence store)
    retrieved_chunks: list[RetrievedChunk] = Field(default_factory=list)

    # Stage 5: Generator
    generator_input: Optional[dict[str, Any]] = Field(default=None)
    raw_generated_answer: Optional[str] = Field(default=None)

    # Stage 6: Verifier
    atomic_claims: list[Claim] = Field(default_factory=list)
    verifier_verdicts: list[dict[str, Any]] = Field(default_factory=list)
    regeneration_count: int = Field(default=0, ge=0)

    # Final Output
    final_answer: str = Field(default="")
    citations: list[str] = Field(default_factory=list)
    abstained: bool = Field(default=False)

    # Performance & Profiling
    stage_latencies: dict[str, float] = Field(default_factory=dict, description="Latency per stage in seconds")
    total_latency: Optional[float] = Field(default=None, description="Total turn latency in seconds")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
