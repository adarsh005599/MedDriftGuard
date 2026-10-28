"""Claim-level evidence verifier and patient-answer policy checks."""

from __future__ import annotations

import json
import re
from typing import Protocol

import requests
import yaml
from pydantic import ValidationError

from src.common.logging import get_logger
from src.common.schemas import Claim, PipelineTrace, RetrievedChunk, VerifiedAnswer

logger = get_logger("verifier.claim")

CLAIM_SPLITTER_PROMPT = """Split the answer into atomic factual claims. Exclude headings,
disclaimers, and advice that is purely procedural. Do not add or rewrite facts. Preserve any
inline [chunk_id] citations that support each claim. Return a JSON array of strings only."""

SAFE_ABSTENTION = "I cannot safely verify enough of this information from the available sources. Please ask a doctor or pharmacist."


class ClaimSplitter(Protocol):
    def split(self, answer: str) -> list[str]: ...


class EntailmentScorer(Protocol):
    def score(self, claim: str, evidence: str) -> tuple[str, float]: ...


class Regenerator(Protocol):
    def __call__(self, unsupported_claims: list[str]) -> str: ...


class OllamaClaimSplitter:
    """Use a local Ollama model to split answer text into atomic claims."""

    def __init__(
        self,
        model_name: str = "qwen2.5:3b",
        base_url: str = "http://127.0.0.1:11434",
        timeout: float = 30.0,
    ):
        self.model_name = model_name
        self.endpoint = f"{base_url.rstrip('/')}/api/generate"
        self.timeout = timeout

    def split(self, answer: str) -> list[str]:
        response = requests.post(
            self.endpoint,
            json={
                "model": self.model_name,
                "system": CLAIM_SPLITTER_PROMPT,
                "prompt": answer,
                "stream": False,
                "format": "json",
            },
            timeout=self.timeout,
        )
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, dict) or not isinstance(payload.get("response"), str):
            raise TypeError("Ollama response must contain generated response text")
        claims = json.loads(payload["response"])
        if not isinstance(claims, list) or not all(
            isinstance(claim, str) and claim.strip() for claim in claims
        ):
            raise ValueError("Claim splitter must return a JSON array of non-empty strings")
        return [claim.strip() for claim in claims]


class LocalNLIEntailmentScorer:
    """Load a local Hugging Face NLI classifier on CPU for premise/hypothesis scoring."""

    def __init__(self, model_name: str):
        from transformers import pipeline

        self.classifier = pipeline(
            "text-classification",
            model=model_name,
            tokenizer=model_name,
            device=-1,
            top_k=None,
        )

    def score(self, claim: str, evidence: str) -> tuple[str, float]:
        predictions = self.classifier({"text": evidence, "text_pair": claim})[0]
        scores = {str(item["label"]).lower(): float(item["score"]) for item in predictions}
        entailment = max(
            (score for label, score in scores.items() if "entail" in label), default=0.0
        )
        contradiction = max(
            (score for label, score in scores.items() if "contrad" in label), default=0.0
        )
        if entailment >= contradiction:
            return "supported", entailment
        return "contradicted", contradiction


class ClaimVerifier:
    """Verify claims against retrieved passages and enforce final answer policies."""

    DOSAGE_PATTERN = re.compile(
        r"\b\d+(?:\.\d+)?\s*(?:mg|ml|mcg|micrograms?|milligrams?|iu|units?|tablets?|capsules?)\b",
        re.IGNORECASE,
    )
    DIAGNOSIS_PATTERN = re.compile(
        r"\b(?:you have|you are diagnosed with|this is definitely|diagnosis is)\b",
        re.IGNORECASE,
    )

    def __init__(
        self,
        splitter: ClaimSplitter,
        scorer: EntailmentScorer,
        supported_threshold: float = 0.70,
        contradicted_threshold: float = 0.30,
        abstain_threshold_ratio: float = 0.50,
        max_regenerations: int = 1,
        config_path: str | None = None,
    ):
        self.splitter = splitter
        self.scorer = scorer
        self.supported_threshold = supported_threshold
        self.contradicted_threshold = contradicted_threshold
        self.abstain_threshold_ratio = abstain_threshold_ratio
        self.max_regenerations = max(0, min(max_regenerations, 1))
        if config_path:
            self._load_config(config_path)

    def _load_config(self, config_path: str) -> None:
        with open(config_path, encoding="utf-8") as config_file:
            section = (yaml.safe_load(config_file) or {}).get("verifier", {})
        self.supported_threshold = float(
            section.get("threshold_supported", self.supported_threshold)
        )
        self.contradicted_threshold = float(
            section.get("threshold_contradicted", self.contradicted_threshold)
        )
        self.abstain_threshold_ratio = float(
            section.get("abstain_threshold_ratio", self.abstain_threshold_ratio)
        )
        self.max_regenerations = max(
            0, min(int(section.get("max_regenerations", self.max_regenerations)), 1)
        )

    @classmethod
    def from_config(
        cls,
        config_path: str = "configs/verifier.yaml",
        generator_model: str = "qwen2.5:3b",
    ) -> ClaimVerifier:
        with open(config_path, encoding="utf-8") as config_file:
            config = (yaml.safe_load(config_file) or {}).get("verifier", {})
        return cls(
            splitter=OllamaClaimSplitter(model_name=generator_model),
            scorer=LocalNLIEntailmentScorer(
                config.get("model_name", "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli")
            ),
            config_path=config_path,
        )

    @staticmethod
    def _safe_abstention(claims: list[Claim], unsupported_count: int) -> VerifiedAnswer:
        return VerifiedAnswer(
            final_text=SAFE_ABSTENTION,
            claims=claims,
            unsupported_count=unsupported_count,
            abstained=True,
            citations=[],
        )

    @staticmethod
    def _record_trace(
        trace: PipelineTrace | None,
        result: VerifiedAnswer,
        regeneration_count: int,
    ) -> VerifiedAnswer:
        if trace is not None:
            trace.atomic_claims = result.claims
            trace.verifier_verdicts = [
                {
                    "claim_id": claim.claim_id,
                    "label": claim.label,
                    "confidence": claim.confidence,
                    "supported": claim.supported,
                    "cited_chunk_ids": claim.cited_chunk_ids,
                }
                for claim in result.claims
            ]
            trace.regeneration_count = regeneration_count
            trace.final_answer = result.final_text
            trace.citations = result.citations
            trace.abstained = result.abstained
        return result

    def _verify_once(
        self, answer: str, chunks: list[RetrievedChunk]
    ) -> tuple[list[Claim], list[Claim]]:
        if not chunks:
            raise ValueError("Cannot verify claims without retrieved evidence")
        allowed_ids = {chunk.chunk_id for chunk in chunks}
        inline_ids = set(re.findall(r"\[([^\]]+)\]", answer))
        if not inline_ids.intersection(allowed_ids):
            raise ValueError("Answer must cite at least one supplied evidence chunk")
        if not any(
            marker in answer.casefold()
            for marker in ("disclaimer", "not a diagnosis", "general health information")
        ):
            raise ValueError("Answer must include the health information disclaimer")
        claim_texts = self.splitter.split(answer)
        if not claim_texts:
            return [], []

        chunk_texts = {chunk.chunk_id: chunk.text for chunk in chunks}
        verified: list[Claim] = []
        unsupported: list[Claim] = []

        for index, claim_text in enumerate(claim_texts, start=1):
            cited_ids = [
                chunk_id
                for chunk_id in re.findall(r"\[([^\]]+)\]", claim_text)
                if chunk_id in allowed_ids
            ]
            candidate_ids = list(
                dict.fromkeys(cited_ids + [chunk.chunk_id for chunk in chunks[:3]])
            )
            passage_results = [
                (chunk_id, *self.scorer.score(claim_text, chunk_texts[chunk_id]))
                for chunk_id in candidate_ids
            ]
            supported_passages = [
                (chunk_id, confidence)
                for chunk_id, label, confidence in passage_results
                if label == "supported" and confidence >= self.supported_threshold
            ]
            contradicted_passages = [
                (label, confidence)
                for _, label, confidence in passage_results
                if label == "contradicted" and confidence >= self.contradicted_threshold
            ]
            is_supported = bool(supported_passages)
            confidence = max((score for _, score in supported_passages), default=0.0)
            if not is_supported and contradicted_passages:
                verdict = "contradicted"
                confidence = max(score for _, score in contradicted_passages)
            elif is_supported:
                verdict = "supported"
            else:
                verdict = "not_enough_info"
            if verdict == "contradicted":
                verdict = "contradicted"
            supported_ids = [chunk_id for chunk_id, _ in supported_passages]
            evidence_text = "\n".join(chunk_texts[chunk_id] for chunk_id in supported_ids)
            claim = Claim(
                claim_id=f"claim_{index:03d}",
                text=claim_text,
                supported=is_supported,
                confidence=confidence,
                cited_chunk_ids=supported_ids,
                justification=evidence_text[:500] if is_supported else None,
                label=verdict,
            )
            (verified if is_supported else unsupported).append(claim)

        return verified, unsupported

    def verify(
        self,
        answer: str,
        chunks: list[RetrievedChunk],
        regenerate: Regenerator | None = None,
        trace: PipelineTrace | None = None,
    ) -> VerifiedAnswer:
        if not answer.strip() or not chunks:
            return self._record_trace(trace, self._safe_abstention([], 0), 0)

        try:
            verified, unsupported = self._verify_once(answer, chunks)
        except (
            ValueError,
            TypeError,
            json.JSONDecodeError,
            ValidationError,
            requests.RequestException,
        ) as error:
            logger.warning("Claim verification failed safely (%s).", type(error).__name__)
            return self._record_trace(trace, self._safe_abstention([], 0), 0)

        current_answer = answer
        regeneration_count = 0
        if unsupported and regenerate is not None and self.max_regenerations:
            try:
                regenerated_answer = regenerate([claim.text for claim in unsupported])
                verified, unsupported = self._verify_once(regenerated_answer, chunks)
                current_answer = regenerated_answer
                regeneration_count = 1
            except (
                ValueError,
                TypeError,
                json.JSONDecodeError,
                ValidationError,
                requests.RequestException,
            ) as error:
                logger.warning("Regenerated answer rejected safely (%s).", type(error).__name__)
                return self._record_trace(
                    trace,
                    self._safe_abstention(verified + unsupported, len(unsupported)),
                    1,
                )

        unsupported_count = len(unsupported)
        total_count = len(verified) + unsupported_count
        if not verified or (
            total_count and unsupported_count / total_count > self.abstain_threshold_ratio
        ):
            return self._record_trace(
                trace,
                self._safe_abstention(verified + unsupported, unsupported_count),
                regeneration_count,
            )

        supported_texts = []
        for claim in verified:
            claim_text = re.sub(r"\s*\[[^\]]+\]", "", claim.text).strip()
            citation_text = " ".join(f"[{chunk_id}]" for chunk_id in claim.cited_chunk_ids)
            supported_texts.append(f"{claim_text} {citation_text}".strip())
        disclaimer_lines = [
            line.strip()
            for line in current_answer.splitlines()
            if line.strip()
            and (
                "disclaimer" in line.casefold()
                or "not a diagnosis" in line.casefold()
                or "general health information" in line.casefold()
            )
        ]
        final_text = "\n".join(supported_texts + disclaimer_lines).strip()

        if self.DOSAGE_PATTERN.search(final_text) or self.DIAGNOSIS_PATTERN.search(final_text):
            return self._record_trace(
                trace,
                self._safe_abstention(verified + unsupported, unsupported_count),
                regeneration_count,
            )
        if (
            "disclaimer" not in final_text.casefold()
            and "not a diagnosis" not in final_text.casefold()
        ):
            return self._record_trace(
                trace,
                self._safe_abstention(verified + unsupported, unsupported_count),
                regeneration_count,
            )

        allowed_ids = {chunk.chunk_id for chunk in chunks}
        citations = sorted(
            {
                chunk_id
                for claim in verified
                for chunk_id in claim.cited_chunk_ids
                if chunk_id in allowed_ids
            }
        )
        if not citations or not re.search(r"\[[^\]]+\]", final_text):
            return self._record_trace(
                trace,
                self._safe_abstention(verified + unsupported, unsupported_count),
                regeneration_count,
            )

        return self._record_trace(
            trace,
            VerifiedAnswer(
                final_text=final_text,
                claims=verified + unsupported,
                unsupported_count=unsupported_count,
                abstained=False,
                citations=citations,
            ),
            regeneration_count,
        )
