"""
Deterministic rule-based extractor for MedDriftGuard Rewriter.

Provides an offline, zero-latency extraction baseline for patient statements,
corrections, retractions, durations, and topic shifts across English,
Devanagari Hindi, and Roman Hinglish.
"""

from __future__ import annotations

import re

from src.common.schemas import LedgerUpdate

# Comprehensive multilingual lexicon for symptoms
SYMPTOM_LEXICON: dict[str, str] = {
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
    "severe headache": "severe headache",
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
    re.compile(
        r"\b(?:forget\s+that|leave\s+that|different\s+question|by\s+the\s+way|another\s+thing|on\s+another\s+note)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:chalo\s+ye\s+chhodo|ye\s+chodo|ye\s+chhodo|dusra\s+sawal|alag\s+baat|ek\s+aur\s+baat)\b",
        re.IGNORECASE,
    ),
    re.compile(r"(यह\s+छोड़ो|दूसरा\s+सवाल|एक\s+और\s+बात)", re.IGNORECASE),
]

# Pronoun follow-up markers
PRONOUN_FOLLOWUP_PATTERNS = [
    re.compile(
        r"\b(?:its|it|this|that)\s+(?:treatment|home\s+care|cure|causes?|remedy|diet|advice|warning\s+signs)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:iska|iski|iske|uska|uski|uske)\s+(?:ilaj|upay|home\s+care|dawa|lakshan|karan|bachaav|bachav)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:isme|usme|issme|ussme)\s+(?:kya\s+(?:karein|kare|karna\s+chahiye)|aaram|doctor|bachav)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"(इसका\s+इलाज|इसके\s+घरेलू\s+उपाय|इसके\s+उपाय|इसमें\s+क्या\s+करें|इसके\s+लक्षण|इसके\s+कारण|इसके\s+नियम|इसके\s+बचाव)",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?:इसका|इसके|इसमें|उसका|उसके|उसमें).{0,24}(?:लक्षण|कारण|खतरे|चेतावनी|देखभाल|उपाय)",
        re.IGNORECASE,
    ),
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
    def extract_updates(cls, text: str) -> list[LedgerUpdate]:
        """
        Extract fact additions, retractions, and corrections from patient text.
        """
        updates: list[LedgerUpdate] = []
        text_lower = text.lower()
        mentions: list[tuple[int, int, str, str]] = []
        for raw_sym, canonical in sorted(
            SYMPTOM_LEXICON.items(), key=lambda item: len(item[0]), reverse=True
        ):
            pattern = re.compile(rf"(?<!\w){re.escape(raw_sym)}(?!\w)", re.IGNORECASE)
            for match in pattern.finditer(text):
                if any(
                    match.start() < end and match.end() > start for start, end, _, _ in mentions
                ):
                    continue
                mentions.append((match.start(), match.end(), raw_sym, canonical))

        mentions.sort(key=lambda item: item[0])
        affirmed_symptoms: set[str] = set()
        retracted_symptoms: set[str] = set()
        is_question = bool(
            re.search(
                r"\?|\b(?:what|when|where|why|how|can|should|does|is)\b|क्या|कब|कैसे",
                text,
                re.IGNORECASE,
            )
        )
        negation_before = re.compile(
            r"(?:\bno\s+longer\s+(?:have\s+)?|\bdo\s+not\s+have\s+|\bdon't\s+have\s+|"
            r"\bno\s+|\bnot\s+|\bwithout\s+|\b(?:nahi|nhi|na)\s+|नहीं\s*)$",
            re.IGNORECASE,
        )
        negation_after = re.compile(
            r"^\s*(?:anymore|no\s+longer|is\s+gone|has\s+gone|nahi(?:\s+hai)?|nhi(?:\s+hai)?|"
            r"नहीं(?:\s+है)?)",
            re.IGNORECASE,
        )

        for start, end, raw_sym, canonical in mentions:
            prefix = text_lower[max(0, start - 40) : start]
            suffix = text_lower[end : end + 24]
            is_negated = bool(negation_before.search(prefix) or negation_after.search(suffix))
            if is_question and not is_negated:
                continue
            operation = "retract" if is_negated else "add"
            target = retracted_symptoms if is_negated else affirmed_symptoms
            if canonical in target:
                continue
            target.add(canonical)
            updates.append(
                LedgerUpdate(
                    op=operation,
                    fact_type="symptom",
                    value=canonical,
                    source_text=text[start:end],
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
