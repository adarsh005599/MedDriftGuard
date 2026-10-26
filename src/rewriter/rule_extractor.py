"""
Deterministic rule-based extractor for MedDriftGuard Rewriter.

Provides an offline, zero-latency extraction baseline for patient statements,
corrections, retractions, durations, and topic shifts across English,
Devanagari Hindi, and Roman Hinglish.
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Set, Tuple
from src.common.schemas import LedgerUpdate

# Comprehensive multilingual lexicon for symptoms
SYMPTOM_LEXICON: Dict[str, str] = {
    # Fever
    "fever": "fever",
    "high fever": "high fever",
    "temperature": "fever",
    "bukhar": "fever",
    "bukhaar": "fever",
    "tez bukhar": "high fever",
    "बुखार": "fever",
    "तेज बुखार": "high fever",
    
    # Cough
    "cough": "cough",
    "dry cough": "dry cough",
    "productive cough": "productive cough",
    "khansi": "cough",
    "khaansi": "cough",
    "sukhi khansi": "dry cough",
    "balgam wali khansi": "productive cough",
    "खांसी": "cough",
    "सूखी खांसी": "dry cough",
    
    # Headache / Migraine
    "headache": "headache",
    "migraine": "migraine",
    "tension headache": "tension headache",
    "sar dard": "headache",
    "sir dard": "headache",
    "tez sar dard": "severe headache",
    "sir me dard": "headache",
    "सिरदर्द": "headache",
    "माइग्रेन": "migraine",

    # Acidity & GERD
    "acidity": "acidity and acid reflux",
    "heartburn": "acidity and heartburn",
    "acid reflux": "acidity and acid reflux",
    "gerd": "acidity and gerd",
    "gas": "acidity and acid reflux",
    "seene mein jalan": "acidity and heartburn",
    "chhati mein jalan": "acidity and heartburn",
    "khatti dakar": "acidity and acid reflux",
    "सीने में जलन": "acidity and heartburn",
    "एसिडिटी": "acidity and acid reflux",

    # Diarrhoea & Dehydration
    "diarrhoea": "diarrhoea",
    "diarrhea": "diarrhoea",
    "loose motions": "diarrhoea",
    "dehydration": "dehydration",
    "dast": "diarrhoea",
    "patle dast": "diarrhoea",
    "pani ki kami": "dehydration",
    "दस्त": "diarrhoea",
    "डिहाइड्रेशन": "dehydration",

    # Hypertension
    "high blood pressure": "hypertension",
    "high bp": "hypertension",
    "hypertension": "hypertension",
    "high bp reading": "hypertension",
    "हाई बीपी": "hypertension",
    "उच्च रक्तचाप": "hypertension",

    # Dengue
    "dengue": "dengue awareness",
    "breakbone fever": "dengue awareness",
    "dengue bukhar": "dengue awareness",
    "डेंगू": "dengue awareness",
}

# Negation and retraction trigger patterns
NEGATION_PATTERNS = [
    re.compile(r"\b(?:no|not|no\s+longer|don't\s+have|without)\s+([a-zA-Z\s]+)", re.IGNORECASE),
    re.compile(r"([a-zA-Z\s]+)\s+(?:nahi|nhi|na)\s+hai\b", re.IGNORECASE),
    re.compile(r"\b(?:nahi|nhi)\s+([a-zA-Z\s]+)\s+(?:nahi|nhi)\b", re.IGNORECASE),
    re.compile(r"\b(?:nahi|nhi)\s+([a-zA-Z\s]+)", re.IGNORECASE),
    re.compile(r"([\u0900-\u097F\s]+)\s+नहीं\s+है", re.IGNORECASE),
    re.compile(r"नहीं\s+([\u0900-\u097F\s]+)", re.IGNORECASE),
]

# Duration patterns
DURATION_PATTERNS = [
    re.compile(r"\b(\d+\s*(?:days?|weeks?|months?|hours?))\b", re.IGNORECASE),
    re.compile(r"\b(\d+\s*din\s*(?:se)?)\b", re.IGNORECASE),
    re.compile(r"\b(kal\s+se|yesterday)\b", re.IGNORECASE),
    re.compile(r"\b(ek\s+hafte\s*se|one\s+week)\b", re.IGNORECASE),
    re.compile(r"(\d+\s*दिन\s*(?:से)?)", re.IGNORECASE),
    re.compile(r"(कल\s+से)", re.IGNORECASE),
]

# Topic shift indicators
TOPIC_SHIFT_PATTERNS = [
    re.compile(r"\b(?:forget\s+that|leave\s+that|different\s+question|by\s+the\s+way|another\s+thing|on\s+another\s+note)\b", re.IGNORECASE),
    re.compile(r"\b(?:chalo\s+ye\s+chhodo|ye\s+chodo|ye\s+chhodo|dusra\s+sawal|alag\s+baat|ek\s+aur\s+baat)\b", re.IGNORECASE),
    re.compile(r"(यह\s+छोड़ो|दूसरा\s+सवाल|एक\s+और\s+बात)", re.IGNORECASE),
]

# Pronoun follow-up markers
PRONOUN_FOLLOWUP_PATTERNS = [
    re.compile(r"\b(?:its|it|this|that)\s+(?:treatment|home\s+care|cure|causes?|remedy|diet|advice|warning\s+signs)\b", re.IGNORECASE),
    re.compile(r"\b(?:iska|iski|iske|uska|uski|uske)\s+(?:ilaj|upay|home\s+care|dawa|lakshan|karan|bachaav|bachav)\b", re.IGNORECASE),
    re.compile(r"\b(?:isme|usme|issme|ussme)\s+(?:kya\s+(?:karein|kare|karna\s+chahiye)|aaram|doctor|bachav)\b", re.IGNORECASE),
    re.compile(r"(इसका\s+इलाज|इसके\s+घरेलू\s+उपाय|इसके\s+उपाय|इसमें\s+क्या\s+करें|इसके\s+लक्षण|इसके\s+कारण|इसके\s+नियम|इसके\s+बचाव)", re.IGNORECASE),
]


class RuleBasedExtractor:
    """Extracts structured ledger updates and reformulates queries using pattern rules."""

    @classmethod
    def detect_topic_shift(cls, text: str) -> bool:
        """Check if user explicitly indicated changing the conversation topic."""
        for pattern in TOPIC_SHIFT_PATTERNS:
            if pattern.search(text):
                return True
        return False

    @classmethod
    def is_pronoun_followup(cls, text: str) -> bool:
        """Check if user asked a follow-up referring back via pronoun."""
        for pattern in PRONOUN_FOLLOWUP_PATTERNS:
            if pattern.search(text):
                return True
        return False

    @classmethod
    def extract_updates(cls, text: str) -> List[LedgerUpdate]:
        """
        Extract fact additions, retractions, and corrections from patient text.
        """
        updates: List[LedgerUpdate] = []
        text_lower = text.lower()

        # Step 1: Detect retractions (negations)
        retracted_symptoms: Set[str] = set()

        for neg_pat in NEGATION_PATTERNS:
            for match in neg_pat.finditer(text):
                negated_phrase = match.group(1).lower().strip()
                for raw_sym, canonical in SYMPTOM_LEXICON.items():
                    if raw_sym in negated_phrase:
                        retracted_symptoms.add(canonical)
                        updates.append(
                            LedgerUpdate(
                                op="retract",
                                fact_type="symptom",
                                value=canonical,
                                source_text=match.group(0),
                            )
                        )

        # Step 2: Detect affirmed symptoms (not negated)
        for raw_sym, canonical in SYMPTOM_LEXICON.items():
            if raw_sym in text_lower:
                # If this symptom was explicitly negated in this utterance, skip adding it
                if canonical in retracted_symptoms:
                    continue

                # Avoid duplicate additions in the same turn
                if not any(u.op == "add" and u.value == canonical for u in updates):
                    updates.append(
                        LedgerUpdate(
                            op="add",
                            fact_type="symptom",
                            value=canonical,
                            source_text=text,
                        )
                    )

        # Step 3: Detect durations
        for dur_pat in DURATION_PATTERNS:
            match = dur_pat.search(text)
            if match:
                dur_val = match.group(1).strip()
                updates.append(
                    LedgerUpdate(
                        op="add",
                        fact_type="duration",
                        value=dur_val,
                        source_text=match.group(0),
                    )
                )
                break

        return updates
