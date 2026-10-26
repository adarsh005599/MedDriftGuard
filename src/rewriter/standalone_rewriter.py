"""
Standalone Query Rewriter for MedDriftGuard.

Converts conversational user messages into clean, standalone English retrieval queries.
Resolves pronouns, detects topic shifts, and extracts structured ledger updates.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import requests
import yaml

from src.common.i18n import detect_script
from src.common.logging import get_logger
from src.common.schemas import LedgerUpdate, ScriptType
from src.ledger.patient_ledger import PatientLedger
from src.rewriter.prompt_templates import REWRITER_SYSTEM_PROMPT
from src.rewriter.rule_extractor import SYMPTOM_LEXICON, RuleBasedExtractor

logger = get_logger("rewriter.standalone")


class OllamaRewriterClient:
    """Small local-only Ollama client for prompted rewriter responses."""

    def __init__(
        self,
        model_name: str = "qwen2.5:3b",
        base_url: str = "http://127.0.0.1:11434",
        timeout: float = 30.0,
    ):
        self.model_name = model_name
        self.endpoint = f"{base_url.rstrip('/')}/api/generate"
        self.timeout = timeout

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        response = requests.post(
            self.endpoint,
            json={
                "model": self.model_name,
                "system": system_prompt,
                "prompt": user_prompt,
                "stream": False,
                "format": "json",
            },
            timeout=self.timeout,
        )
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, dict):
            raise TypeError("Ollama response payload must be a JSON object")
        generated = payload.get("response")
        if not isinstance(generated, str):
            raise TypeError("Ollama response did not contain generated text")
        return generated


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


def _pick_primary_symptom(symptoms: list[str]) -> str:
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
    ledger_updates: list[LedgerUpdate] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
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

    def __init__(
        self,
        mode: str = "prompted_baseline",
        llm_client: OllamaRewriterClient | None = None,
        max_retries: int = 1,
    ):
        self.mode = mode
        self.extractor = RuleBasedExtractor()
        self.llm_client = llm_client
        self.max_retries = max(0, max_retries)

    @classmethod
    def from_config(
        cls,
        config_path: Path = Path("configs/pipeline.yaml"),
        base_url: str = "http://127.0.0.1:11434",
        timeout: float = 30.0,
        max_retries: int = 1,
    ) -> StandaloneRewriter:
        """Create a rewriter using the local model name in pipeline.yaml."""
        with config_path.open("r", encoding="utf-8") as config_file:
            config = yaml.safe_load(config_file) or {}
        model_name = config.get("stages", {}).get("rewriter", {}).get("model_name", "qwen2.5:3b")
        return cls(
            mode="prompted_baseline",
            llm_client=OllamaRewriterClient(
                model_name=model_name,
                base_url=base_url,
                timeout=timeout,
            ),
            max_retries=max_retries,
        )

    def rewrite(
        self,
        user_message: str,
        ledger: PatientLedger,
        last_turns: list[dict[str, str]] | None = None,
        turn_id: int = 1,
    ) -> RewriterResult:
        """
        Produce a clean standalone English query and extract ledger updates.
        """
        user_text = user_message.strip()
        script = detect_script(user_text)
        language = "en" if script == "en" else "hi"

        if not user_text:
            return RewriterResult("", script, language, False, [])

        if self.llm_client is not None:
            prompted_result = self._rewrite_with_llm(
                user_text=user_text,
                ledger=ledger,
                last_turns=(last_turns or [])[-2:],
                turn_id=turn_id,
                script=script,
                language=language,
            )
            if prompted_result is not None:
                return prompted_result

        # Step 1: Detect topic shift
        topic_shift = self.extractor.detect_topic_shift(user_text)

        # Step 2: Extract ledger updates (symptoms, retractions, durations)
        ledger_updates = self.extractor.extract_updates(user_text)

        # Step 3: Check for pronoun follow-up
        is_followup = self.extractor.is_pronoun_followup(user_text)

        # Identify primary active symptoms from new updates or existing ledger
        new_add_symptoms = [
            u.value for u in ledger_updates if u.op == "add" and u.fact_type == "symptom"
        ]
        retracted_in_turn = {
            u.value.lower()
            for u in ledger_updates
            if u.op == "retract" and u.fact_type == "symptom"
        }
        if new_add_symptoms:
            # On a topic-shift or plain new report, use the newly added symptoms
            primary_symptom = _pick_primary_symptom(new_add_symptoms)
        elif topic_shift:
            primary_symptom = self._topic_from_recent_turns(
                [{"role": "user", "content": user_text}]
            )
        else:
            # For pronoun follow-ups or queries without new symptoms, use ledger state,
            # excluding symptoms that were just retracted in this very turn.
            candidate_symptoms = (
                []
                if topic_shift
                else [
                    symptom
                    for symptom in ledger.get_active_symptoms()
                    if not any(
                        symptom.lower() == retracted
                        or symptom.lower().endswith(f" {retracted}")
                        or retracted.endswith(f" {symptom.lower()}")
                        for retracted in retracted_in_turn
                    )
                ]
            )
            primary_symptom = _pick_primary_symptom(candidate_symptoms)
            if not primary_symptom:
                primary_symptom = self._topic_from_recent_turns(
                    [{"role": "user", "content": user_text}]
                )
            if not primary_symptom and is_followup:
                primary_symptom = self._topic_from_recent_turns(last_turns or [])

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
            f"Updates={len(ledger_updates)}"
        )

        return RewriterResult(
            standalone_query_en=standalone_query,
            detected_script=script,
            language=language,
            topic_shift=topic_shift,
            ledger_updates=ledger_updates,
        )

    def _rewrite_with_llm(
        self,
        user_text: str,
        ledger: PatientLedger,
        last_turns: list[dict[str, str]],
        turn_id: int,
        script: ScriptType,
        language: str,
    ) -> RewriterResult | None:
        prompt_payload = json.dumps(
            {
                "ledger": ledger.to_summary_string(),
                "last_turns": last_turns,
                "user_msg": user_text,
            },
            ensure_ascii=False,
        )
        for attempt in range(self.max_retries + 1):
            try:
                raw_response = self.llm_client.generate(REWRITER_SYSTEM_PROMPT, prompt_payload)
                data = json.loads(raw_response)
                result = self._validate_llm_result(data, user_text, last_turns, script, language)
                return RewriterResult(
                    standalone_query_en=result["standalone_query_en"],
                    detected_script=script,
                    language=language,
                    topic_shift=result["topic_shift"],
                    ledger_updates=result["ledger_updates"],
                )
            except (
                ValueError,
                TypeError,
                KeyError,
                json.JSONDecodeError,
                requests.RequestException,
            ) as error:
                logger.warning(
                    f"Prompted rewriter response rejected on attempt {attempt + 1}; "
                    f"using retry or rule fallback ({type(error).__name__})."
                )
        return None

    def _validate_llm_result(
        self,
        data: Any,
        user_text: str,
        last_turns: list[dict[str, str]],
        script: ScriptType,
        language: str,
    ) -> dict[str, Any]:
        required_fields = {
            "standalone_query_en",
            "detected_script",
            "language",
            "topic_shift",
            "ledger_updates",
        }
        if not isinstance(data, dict) or set(data) != required_fields:
            raise ValueError("Rewriter JSON must contain exactly the documented fields")
        query = data["standalone_query_en"]
        if not isinstance(query, str) or not query.strip():
            raise ValueError("standalone_query_en must be a non-empty string")
        if data["detected_script"] not in {"en", "devanagari", "roman_hindi"}:
            raise ValueError("detected_script is not supported")
        if data["language"] not in {"en", "hi"}:
            raise ValueError("language must be en or hi")
        if not isinstance(data["topic_shift"], bool):
            raise TypeError("topic_shift must be a boolean")
        if not isinstance(data["ledger_updates"], list):
            raise TypeError("ledger_updates must be a list")

        query_lower = query.casefold()
        if any(
            turn.get("content", "").strip() and turn["content"].strip().casefold() in query_lower
            for turn in last_turns
        ):
            raise ValueError("Standalone query contains a raw previous turn")

        updates = []
        for raw_update in data["ledger_updates"]:
            update_data = dict(raw_update)
            source_text = update_data.get("source_text")
            if not isinstance(source_text, str) or not source_text.strip():
                raise ValueError("Every prompted ledger update needs a verbatim source_text")
            if source_text.casefold() not in user_text.casefold():
                raise ValueError(
                    "Ledger update source_text must come from the current user message"
                )
            update = LedgerUpdate(**update_data)
            if update.fact_type == "symptom":
                supported = self.extractor.extract_updates(source_text)
                if not any(
                    candidate.fact_type == update.fact_type
                    and candidate.op == update.op
                    and candidate.value == update.value
                    for candidate in supported
                ):
                    raise ValueError(
                        "Prompted symptom update is not supported by explicit user wording"
                    )
            elif update.value.casefold() not in source_text.casefold():
                raise ValueError("Non-symptom ledger value must appear in its source_text")
            updates.append(update)

        return {
            "standalone_query_en": query.strip(),
            "detected_script": script,
            "language": language,
            "topic_shift": data["topic_shift"],
            "ledger_updates": updates,
        }

    @staticmethod
    def _topic_from_recent_turns(last_turns: list[dict[str, str]]) -> str:
        """Resolve a pronoun from recent user wording without storing it as a patient fact."""
        for turn in reversed(last_turns[-2:]):
            if turn.get("role") != "user":
                continue
            message = turn.get("content", "").lower()
            for alias, canonical in sorted(
                SYMPTOM_LEXICON.items(), key=lambda item: len(item[0]), reverse=True
            ):
                if re.search(rf"(?<!\w){re.escape(alias)}(?!\w)", message):
                    return canonical
        return ""

    def _formulate_standalone_query(
        self,
        user_text: str,
        primary_symptom: str,
        topic_shift: bool,
        is_followup: bool,
        script: ScriptType,
        ledger_updates: list[LedgerUpdate],
    ) -> str:
        """Constructs the English search query based on intent resolution."""
        text_lower = user_text.lower()

        # If it's a pronoun follow-up (e.g. "what is its home care?", "iski dawa kya hai?")
        if is_followup and primary_symptom:
            # Causes / warning signs check — must come BEFORE care to prevent कारण being swallowed
            if any(
                w in text_lower
                for w in [
                    "cause",
                    "karan",
                    "kyu",
                    "why",
                    "warning",
                    "लक्षण",
                    "कारण",
                    "खतरे",
                ]
            ):
                return f"causes risk factors and warning signs of {primary_symptom}"
            # Doctor / when to seek care
            if any(
                w in text_lower
                for w in [
                    "doctor",
                    "hospital",
                    "emergency",
                    "kab dikhaye",
                    "दिखाना",
                    "डॉक्टर",
                    "dikhana",
                ]
            ):
                return f"when to see a doctor for {primary_symptom}"
            # Home care / treatment (broad catch — includes "करें", "aaram", "ors", "bachaav" etc.)
            if any(
                w in text_lower
                for w in [
                    "care",
                    "upay",
                    "ilaj",
                    "remedy",
                    "home",
                    "treatment",
                    "dawa",
                    "aaram",
                    "bachav",
                    "bachaav",
                    "घरेलू",
                    "उपाय",
                    "इलाज",
                    "नियम",
                    "राहत",
                    "करें",
                    "karein",
                    "kare",
                    "karna",
                    "ors",
                    "ओआरएस",
                ]
            ):
                return f"home care and hydration management for {primary_symptom}"
            return f"general clinical guidance and overview for {primary_symptom}"

        # If user explicitly corrected / negated a symptom (e.g. "nahi bukhar nahi hai, sirf khansi hai")
        active_adds = [
            u.value for u in ledger_updates if u.op == "add" and u.fact_type == "symptom"
        ]
        if active_adds:
            symptom_str = " and ".join(active_adds)
            if any(w in text_lower for w in ["care", "upay", "ilaj", "home"]):
                return f"home care and relief for {symptom_str}"
            if any(
                w in text_lower
                for w in [
                    "doctor",
                    "kab dikhaye",
                    "serious",
                    "दिखाना",
                    "डॉक्टर",
                    "dikhana",
                ]
            ):
                return f"when to see a doctor for {symptom_str}"
            return f"{symptom_str} symptoms causes and home care"

        # If user asked about existing symptom directly (non-pronoun, non-new-symptom)
        if primary_symptom:
            if any(
                w in text_lower
                for w in [
                    "emergency",
                    "doctor",
                    "kab dikhaye",
                    "severe",
                    "दिखाना",
                    "डॉक्टर",
                    "dikhana",
                ]
            ):
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
