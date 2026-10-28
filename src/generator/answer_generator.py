"""Evidence-bound structured medical information answer generator."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Protocol

import requests
import yaml
from pydantic import ValidationError

from src.common.logging import get_logger
from src.common.schemas import GeneratedAnswer, RetrievedChunk
from src.generator.prompt_templates import GENERATOR_SYSTEM_PROMPT

logger = get_logger("generator.answer")

SAFE_ABSTENTION = (
    "I do not have enough reliable information in the available sources to answer that safely."
)
DEFAULT_DISCLAIMER = "This is general health information, not a diagnosis."
UNSAFE_MEDICINE_PATTERN = re.compile(
    r"\b\d+(?:\.\d+)?\s*(?:mg|ml|mcg|micrograms?|milligrams?|iu|units?|tablets?|capsules?)\b",
    re.IGNORECASE,
)
DIAGNOSIS_PATTERN = re.compile(
    r"\b(?:you have|you are diagnosed with|diagnosis is|this is definitely)\b",
    re.IGNORECASE,
)


class TextGenerationClient(Protocol):
    def generate(self, system_prompt: str, user_prompt: str) -> str: ...


class OllamaGeneratorClient:
    """Minimal client for a locally served Ollama model."""

    def __init__(
        self,
        model_name: str = "qwen2.5:3b",
        base_url: str = "http://127.0.0.1:11434",
        timeout: float = 60.0,
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
        if not isinstance(payload, dict) or not isinstance(payload.get("response"), str):
            raise TypeError("Ollama response must be an object containing response text")
        return payload["response"]


class MedicalAnswerGenerator:
    """Generate a structured answer using only the supplied retrieved passages."""

    def __init__(self, client: TextGenerationClient, max_retries: int = 1):
        self.client = client
        self.max_retries = max(0, max_retries)

    @classmethod
    def from_config(
        cls,
        config_path: Path = Path("configs/pipeline.yaml"),
        base_url: str = "http://127.0.0.1:11434",
        timeout: float = 60.0,
        max_retries: int = 1,
    ) -> MedicalAnswerGenerator:
        """Construct the local Ollama generator from the existing pipeline config."""
        with config_path.open("r", encoding="utf-8") as config_file:
            config = yaml.safe_load(config_file) or {}
        model_name = config.get("stages", {}).get("generator", {}).get("model_name", "qwen2.5:3b")
        return cls(
            OllamaGeneratorClient(
                model_name=model_name,
                base_url=base_url,
                timeout=timeout,
            ),
            max_retries=max_retries,
        )

    @staticmethod
    def _abstain(reason: str = SAFE_ABSTENTION) -> GeneratedAnswer:
        return GeneratedAnswer(
            disclaimer=DEFAULT_DISCLAIMER,
            abstained=True,
            abstention_reason=reason,
        )

    def generate(
        self,
        question: str,
        ledger_summary: str,
        chunks: list[RetrievedChunk],
        language: str = "en",
        script: str = "en",
        last_turns: list[dict[str, str]] | None = None,
    ) -> GeneratedAnswer:
        if not question.strip() or not chunks:
            return self._abstain()

        payload = {
            "question": question.strip(),
            "language": language,
            "script": script,
            "ledger_summary": ledger_summary,
            "last_turns": (last_turns or [])[-2:],
            "evidence": [
                {
                    "chunk_id": chunk.chunk_id,
                    "source": chunk.source,
                    "url": chunk.url,
                    "section": chunk.section,
                    "text": chunk.text,
                }
                for chunk in chunks
            ],
        }
        allowed_chunk_ids = {chunk.chunk_id for chunk in chunks}

        for attempt in range(self.max_retries + 1):
            try:
                raw = self.client.generate(
                    GENERATOR_SYSTEM_PROMPT,
                    json.dumps(payload, ensure_ascii=False),
                )
                result = GeneratedAnswer.model_validate_json(raw)
                if result.abstained:
                    return result.model_copy(
                        update={"disclaimer": result.disclaimer or DEFAULT_DISCLAIMER}
                    )
                rendered = result.to_text()
                if UNSAFE_MEDICINE_PATTERN.search(rendered) or DIAGNOSIS_PATTERN.search(rendered):
                    raise ValueError("Generated response violates the no-dose/no-diagnosis policy")
                if not result.citations or not set(result.citations).issubset(allowed_chunk_ids):
                    raise ValueError("Answer citations must refer to supplied evidence chunks")
                if any(f"[{chunk_id}]" not in rendered for chunk_id in result.citations):
                    raise ValueError("Each listed citation must appear inline in the answer")
                if not result.disclaimer.strip():
                    raise ValueError("A patient-facing answer requires a disclaimer")
                return result
            except (
                json.JSONDecodeError,
                ValidationError,
                ValueError,
                TypeError,
                requests.RequestException,
            ) as error:
                logger.warning(
                    "Generator response rejected on attempt %s (%s); retrying or abstaining.",
                    attempt + 1,
                    type(error).__name__,
                )

        return self._abstain()
