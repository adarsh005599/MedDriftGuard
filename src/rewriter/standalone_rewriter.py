"""
Standalone Query Rewriter for MedDriftGuard.

Converts conversational user messages into clean, standalone English retrieval queries.
Resolves pronouns, detects topic shifts, and extracts structured ledger updates.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import re
from typing import Any, Dict, List, Optional

from src.common.schemas import LedgerUpdate, ScriptType
from src.common.i18n import detect_script
from src.ledger.patient_ledger import PatientLedger
from src.rewriter.rule_extractor import RuleBasedExtractor, SYMPTOM_LEXICON
from src.common.logging import get_logger

logger = get_logger("rewriter.standalone")

# Ordered list of more specific symptoms that should be preferred when
# multiple symptoms co-exist in the ledger.
_SPECIFICITY_ORDER = [
    "migraine",
    "tension headache",
    "severe headache",
    "dry cough",
    "productive cough",
    "high fever",
    "dengue awareness",
    "acidity and heartburn",
    "acidity and gerd",
    "acidity and acid reflux",
    "hypertension",
    "diarrhoea",
    "dehydration",
    "headache",
    "cough",
    "fever",
]


def _pick_primary_symptom(symptoms: List[str]) -> str:
    """
    Choose the most topically specific symptom from a list.
    Prefers more specific variants (e.g. 'migraine' over 'headache').
    Falls back to the last symptom in the list when no preference applies.
    """
    if not symptoms:
        return ""
    lower_map = {s.lower(): s for s in symptoms}
    for candidate in _SPECIFICITY_ORDER:
        if candidate in lower_map:
            return lower_map[candidate]
    return symptoms[-1]


@dataclass
class RewriterResult:
    """Output from the conversational rewriter."""
    standalone_query_en: str
    detected_script: ScriptType
    language: str
    topic_shift: bool
    ledger_updates: List[LedgerUpdate] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "standalone_query_en": self.standalone_query_en,
            "detected_script": self.detected_script,
            "language": self.language,
            "topic_shift": self.topic_shift,
            "ledger_updates": [u.model_dump() for u in self.ledger_updates],
        }


class StandaloneRewriter:
    """
    Reformulates user queries into standalone search intent and extracts ledger updates.
    """

    def __init__(self, mode: str = "prompted_baseline"):
        self.mode = mode
        self.extractor = RuleBasedExtractor()

    def rewrite(
        self,
        user_message: str,
        ledger: PatientLedger,
        last_turns: Optional[List[Dict[str, str]]] = None,
        turn_id: int = 1,
    ) -> RewriterResult:
        """
        Produce a clean standalone English query and extract ledger updates.
        """
        user_text = user_message.strip()
        script = detect_script(user_text)
        language = "en" if script == "en" else "hi"

        # Step 1: Detect topic shift
        topic_shift = self.extractor.detect_topic_shift(user_text)

        # Step 2: Extract ledger updates (symptoms, retractions, durations)
        ledger_updates = self.extractor.extract_updates(user_text)

        # Step 3: Check for pronoun follow-up
        is_followup = self.extractor.is_pronoun_followup(user_text)

        # Identify primary active symptoms from new updates or existing ledger
        new_add_symptoms = [u.value for u in ledger_updates if u.op == "add" and u.fact_type == "symptom"]
        retracted_in_turn = {u.value.lower() for u in ledger_updates if u.op == "retract" and u.fact_type == "symptom"}
        if new_add_symptoms:
            # On a topic-shift or plain new report, use the newly added symptoms
            primary_symptom = _pick_primary_symptom(new_add_symptoms)
        else:
            # For pronoun follow-ups or queries without new symptoms, use ledger state,
            # excluding symptoms that were just retracted in this very turn.
            candidate_symptoms = [
                s for s in ledger.get_active_symptoms()
                if s.lower() not in retracted_in_turn
            ]
            primary_symptom = _pick_primary_symptom(candidate_symptoms)

        # Step 4: Formulate the standalone English search query
        standalone_query = self._formulate_standalone_query(
            user_text=user_text,
            primary_symptom=primary_symptom,
            topic_shift=topic_shift,
            is_followup=is_followup,
            script=script,
            ledger_updates=ledger_updates,
        )

        logger.info(
            f"Rewriter (Turn {turn_id}): Script={script}, Shift={topic_shift}, "
            f"Updates={len(ledger_updates)}, Standalone='{standalone_query}'"
        )

        return RewriterResult(
            standalone_query_en=standalone_query,
            detected_script=script,
            language=language,
            topic_shift=topic_shift,
            ledger_updates=ledger_updates,
        )

    def _formulate_standalone_query(
        self,
        user_text: str,
        primary_symptom: str,
        topic_shift: bool,
        is_followup: bool,
        script: ScriptType,
        ledger_updates: List[LedgerUpdate],
    ) -> str:
        """Constructs the English search query based on intent resolution."""
        text_lower = user_text.lower()

        # If it's a pronoun follow-up (e.g. "what is its home care?", "iski dawa kya hai?")
        if is_followup and primary_symptom:
            # Causes / warning signs check — must come BEFORE care to prevent कारण being swallowed
            if any(w in text_lower for w in [
                "cause", "karan", "kyu", "why", "warning", "लक्षण", "कारण", "खतरे",
            ]):
                return f"causes risk factors and warning signs of {primary_symptom}"
            # Doctor / when to seek care
            if any(w in text_lower for w in [
                "doctor", "hospital", "emergency", "kab dikhaye", "दिखाना", "डॉक्टर", "dikhana",
            ]):
                return f"when to see a doctor for {primary_symptom}"
            # Home care / treatment (broad catch — includes "करें", "aaram", "ors", "bachaav" etc.)
            if any(w in text_lower for w in [
                "care", "upay", "ilaj", "remedy", "home", "treatment", "dawa", "aaram",
                "bachav", "bachaav", "घरेलू", "उपाय", "इलाज", "नियम", "राहत", "करें",
                "karein", "kare", "karna", "ors", "ओआरएस",
            ]):
                return f"home care and hydration management for {primary_symptom}"
            return f"general clinical guidance and overview for {primary_symptom}"

        # If user explicitly corrected / negated a symptom (e.g. "nahi bukhar nahi hai, sirf khansi hai")
        active_adds = [u.value for u in ledger_updates if u.op == "add" and u.fact_type == "symptom"]
        if active_adds:
            symptom_str = " and ".join(active_adds)
            if any(w in text_lower for w in ["care", "upay", "ilaj", "home"]):
                return f"home care and relief for {symptom_str}"
            if any(w in text_lower for w in [
                "doctor", "kab dikhaye", "serious", "दिखाना", "डॉक्टर", "dikhana",
            ]):
                return f"when to see a doctor for {symptom_str}"
            return f"{symptom_str} symptoms causes and home care"

        # If user asked about existing symptom directly (non-pronoun, non-new-symptom)
        if primary_symptom:
            if any(w in text_lower for w in [
                "emergency", "doctor", "kab dikhaye", "severe", "दिखाना", "डॉक्टर", "dikhana",
            ]):
                return f"when to see a doctor for {primary_symptom}"
            if any(w in text_lower for w in ["cause", "karan", "kyu", "why", "warning", "कारण"]):
                return f"causes risk factors and warning signs of {primary_symptom}"
            return f"{primary_symptom} symptoms causes and management"

        # General English search fallback (translating common concepts if no named symptom)
        clean_query = user_text
        # Remove topic shift fillers
        clean_query = re.sub(
            r"\b(?:chalo\s+ye\s+chhodo|forget\s+that|leave\s+that|ye\s+chodo|dusra\s+sawal)\b",
            "",
            clean_query,
            flags=re.IGNORECASE,
        ).strip()

        if clean_query:
            return clean_query
        return "general health information overview"
