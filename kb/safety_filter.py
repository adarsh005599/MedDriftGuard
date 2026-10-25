"""
Safety filter for the MedDriftGuard Knowledge Base.

Enforces the core safety rule:
NO PHARMACEUTICAL DRUG DOSAGES, TITRATION TABLES, OR SPECIFIC MEDICATION AMOUNTS
SHALL BE INDEXED IN THE KNOWLEDGE BASE OR PASSED TO GENERATORS.
"""

import re
from typing import Tuple

# Patterns matching pharmaceutical drug dosages, units, and frequencies
DOSAGE_PATTERNS = [
    # Explicit dosage with units: e.g. 500mg, 10 ml, 250 mcg, 1000 IU
    re.compile(
        r"\b\d+(?:\.\d+)?\s*(?:mg|milligrams?|ml|milliliters?|mcg|micrograms?|iu|tablets?|capsules?|drops?|pills?|syrup)(?:\s+(?:tablets?|capsules?|drops?|pills?|syrup|suspension|solution|liquid))?\s+(?:every|daily|per\s+day|bd|tds|od|times\s+a\s+day|hourly|hours?|q\d+h)\b",
        re.IGNORECASE,
    ),
    # Dosing instructions: e.g. "dose: 2 tablets", "take 500 mg"
    re.compile(
        r"\b(?:dose|dosage|take|administer|prescribed)\s*:\s*\d+(?:\.\d+)?\s*(?:mg|ml|mcg|tablets?|capsules?|drops?)\b",
        re.IGNORECASE,
    ),
    # Dosage ranges: e.g. "250-500 mg every 6 hours"
    re.compile(
        r"\b\d+\s*(?:-|to)\s*\d+\s*(?:mg|ml|mcg|tablets?)\s+(?:every|daily|per\s+day|hours?)\b",
        re.IGNORECASE,
    ),
    # Adult/Child specific dosing: e.g. "adults: 1-2 tablets daily"
    re.compile(
        r"\b(?:adults?|children|infants?|pediatric)\s*:\s*\d+(?:\.\d+)?\s*(?:mg|ml|tablets?)\b",
        re.IGNORECASE,
    ),
]

# Sensitive dosing table markers
DOSAGE_TABLE_HEADER_PATTERNS = [
    re.compile(r"\b(?:dosage\s+table|dosing\s+chart|recommended\s+dosage|dosage\s+and\s+administration)\b", re.IGNORECASE)
]


def contains_drug_dosage(text: str) -> bool:
    """Check if the text contains any explicit pharmaceutical drug dosage or frequency."""
    for pattern in DOSAGE_PATTERNS:
        if pattern.search(text):
            return True
    return False


def strip_drug_dosages(text: str) -> Tuple[str, int]:
    """
    Remove explicit pharmaceutical drug dosage sentences or phrases from text,
    replacing them with safe general clinician advisory text.
    Returns (cleaned_text, number_of_redactions).
    """
    redaction_count = 0
    cleaned_lines = []

    for line in text.splitlines():
        # If the line looks like a dosage table row or header, skip it entirely
        if any(h.search(line) for h in DOSAGE_TABLE_HEADER_PATTERNS):
            redaction_count += 1
            continue

        modified_line = line
        for pattern in DOSAGE_PATTERNS:
            matches = list(pattern.finditer(modified_line))
            if matches:
                redaction_count += len(matches)
                modified_line = pattern.sub("[Consult a doctor or pharmacist for medication guidance]", modified_line)

        cleaned_lines.append(modified_line)

    return "\n".join(cleaned_lines), redaction_count
