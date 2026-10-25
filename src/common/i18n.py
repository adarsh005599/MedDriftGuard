"""
Script and language utilities for English, Hindi (Devanagari), and Hinglish (Roman Hindi).
"""

import re
from typing import Literal

ScriptType = Literal["en", "devanagari", "roman_hindi", "unknown"]

DEVANAGARI_RANGE = re.compile(r"[\u0900-\u097F]")


def detect_script(text: str) -> ScriptType:
    """
    Detect whether the input text is written in Devanagari script,
    English, or Roman-script Hindi (Hinglish).
    """
    if not text or not text.strip():
        return "unknown"

    # Check for Devanagari characters
    if DEVANAGARI_RANGE.search(text):
        return "devanagari"

    # Common Hinglish/Roman-Hindi stop words and markers
    hinglish_markers = {
        "hai", "hain", "kya", "kyu", "kyun", "nahi", "nhi", "mujhe", "mera", "meri",
        "dard", "bukhar", "khansi", "sar", "sir", "pet", "gala", "dawa", "dawai",
        "se", "ko", "ka", "ki", "ke", "ho", "raha", "rahi", "rahe", "tha", "thi",
        "the", "aur", "ya", "par", "bhi", "bahut", "sirf", "lekin", "kab", "kahan"
    }

    words = re.findall(r"\b[a-zA-Z]+\b", text.lower())
    if not words:
        return "en"

    matches = sum(1 for w in words if w in hinglish_markers)
    ratio = matches / len(words)

    # If at least 15% of words or 2+ distinct markers are Roman-Hindi keywords
    if ratio >= 0.15 or matches >= 2:
        return "roman_hindi"

    return "en"
