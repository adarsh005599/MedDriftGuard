"""Demo-mode chat orchestration shared by the API and presentation UI."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from demo import build_offline_demo_components, process_question
from src.common.i18n import detect_script
from src.ledger.patient_ledger import PatientLedger
from src.retriever.hybrid_retriever import HybridRetriever

DISCLAIMER = "This is general health information, not a diagnosis."
GREETING_TEXT = (
    "Hello — I can help with general health-information questions in English, Hindi, "
    "or Hinglish. Please tell me the symptom or health topic you would like to understand."
)
CLARIFICATION_TEXT = (
    "Please enter a meaningful health-information question or symptom so I can look for "
    "relevant evidence."
)
IRRELEVANT_EVIDENCE_TEXT = (
    "I do not have relevant evidence in the available sources to answer that safely. "
    "Please ask a general health-information question about a supported topic, or consult a clinician."
)
_GREETING_PATTERN = re.compile(
    r"^(?:hi|hello|hey|namaste|namaskar|good\s+(?:morning|afternoon|evening))(?:\s+there)?[!,. ]*$",
    re.IGNORECASE,
)
_REPEATED_TEXT_PATTERN = re.compile(r"^(.)\1{2,}$", re.DOTALL)
_WORD_PATTERN = re.compile(r"[^\W\d_]{3,}", re.UNICODE)
_QUERY_STOPWORDS = {
    "about",
    "and",
    "are",
    "can",
    "does",
    "for",
    "from",
    "have",
    "home",
    "how",
    "information",
    "is",
    "it",
    "management",
    "mentioned",
    "the",
    "this",
    "what",
    "when",
    "with",
}


def input_kind(message: str) -> str:
    """Classify only greetings and clearly non-meaningful input before retrieval."""
    compact = " ".join(message.split())
    if not compact:
        return "clarify"
    if _GREETING_PATTERN.fullmatch(compact):
        return "greeting"
    letters_only = "".join(character for character in compact if character.isalpha())
    if len(letters_only) < 2 or _REPEATED_TEXT_PATTERN.fullmatch(letters_only.casefold()):
        return "clarify"
    return "query"


def evidence_is_relevant(query: str, chunks: list[object]) -> bool:
    """Require at least one meaningful rewritten-query term in retrieved evidence.

    This lightweight guard is deliberately conservative: it does not diagnose or
    classify health topics, and it only rejects an answer when none of the useful
    query terms occur in any retrieved chunk's actual metadata or text.
    """
    terms = {
        term.casefold()
        for term in _WORD_PATTERN.findall(query)
        if term.casefold() not in _QUERY_STOPWORDS
    }
    if not terms or not chunks:
        return False
    evidence_text = " ".join(
        " ".join(
            str(getattr(chunk, field_name, "")) for field_name in ("topic", "section", "text")
        ).casefold()
        for chunk in chunks
    )
    return any(term in evidence_text for term in terms)


@dataclass
class DemoChatService:
    """Keep per-conversation ledgers while using the existing offline demo path.

    This deliberately uses the deterministic generator and verifier from
    ``demo.py``.  It is a presentation-safe adapter, not a live clinical model
    service.
    """

    retriever: object
    generator: object
    verifier: object
    ledgers: dict[str, PatientLedger] = field(default_factory=dict)
    turn_counts: dict[str, int] = field(default_factory=dict)

    @classmethod
    def create_default(cls) -> DemoChatService:
        generator, verifier = build_offline_demo_components()
        return cls(retriever=HybridRetriever(), generator=generator, verifier=verifier)

    def reply(self, message: str, conversation_id: str) -> dict:
        disposition = input_kind(message)
        script = detect_script(message)
        language = "hi" if script in {"devanagari", "roman_hindi"} else "en"
        if disposition == "greeting":
            return self._safe_response(conversation_id, GREETING_TEXT, language, script)
        if disposition == "clarify":
            return self._safe_response(conversation_id, CLARIFICATION_TEXT, language, script)

        ledger = self.ledgers.setdefault(conversation_id, PatientLedger())
        turn_id = self.turn_counts.get(conversation_id, 0) + 1
        result = process_question(
            message,
            self.retriever,
            self.generator,
            self.verifier,
            ledger=ledger,
            turn_id=turn_id,
        )
        self.turn_counts[conversation_id] = turn_id
        if not result.verified_answer.abstained and not evidence_is_relevant(
            result.standalone_query, result.chunks
        ):
            return self._safe_response(
                conversation_id,
                IRRELEVANT_EVIDENCE_TEXT,
                result.detected_language,
                result.detected_script,
            )
        cited_ids = set(result.verified_answer.citations)
        sources = [
            {
                "chunk_id": chunk.chunk_id,
                "source": chunk.source,
                "url": chunk.url,
                "section": chunk.section,
                "topic": chunk.topic,
                "license": chunk.license,
                "excerpt": chunk.text,
            }
            for chunk in result.chunks
            if chunk.chunk_id in cited_ids
        ]
        return {
            "conversation_id": conversation_id,
            "answer": result.verified_answer.final_text,
            "citations": result.verified_answer.citations,
            "sources": sources,
            "abstained": result.verified_answer.abstained,
            "detected_language": result.detected_language,
            "detected_script": result.detected_script,
            "demo_mode": True,
            "mode_note": "DEMO/MOCK MODE: generation and verification are deterministic mocks; retrieval uses the local knowledge base.",
        }

    @staticmethod
    def _safe_response(conversation_id: str, answer: str, language: str, script: str) -> dict:
        """Return a no-evidence response without fabricated citations or sources."""
        return {
            "conversation_id": conversation_id,
            "answer": f"{answer}\n\n{DISCLAIMER}",
            "citations": [],
            "sources": [],
            "abstained": True,
            "detected_language": language,
            "detected_script": script,
            "demo_mode": True,
            "mode_note": "DEMO/MOCK MODE: generation and verification are deterministic mocks; retrieval uses the local knowledge base.",
        }
