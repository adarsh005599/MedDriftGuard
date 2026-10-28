"""Simple terminal demo for the existing MedDriftGuard P0-P5 components.

Default mode uses the real local KB/retriever plus deterministic mocked generation and
verification, so the end-to-end flow can be demonstrated without Ollama or NLI weights.
"""

from __future__ import annotations

import argparse
import re
from collections.abc import Callable
from dataclasses import dataclass

from src.common.schemas import PipelineTrace, RetrievedChunk, VerifiedAnswer
from src.generator.answer_generator import (
    DIAGNOSIS_PATTERN,
    UNSAFE_MEDICINE_PATTERN,
    MedicalAnswerGenerator,
)
from src.ledger.patient_ledger import PatientLedger
from src.retriever.hybrid_retriever import HybridRetriever
from src.rewriter.standalone_rewriter import StandaloneRewriter
from src.verifier.claim_verifier import ClaimVerifier


@dataclass
class DemoTurn:
    original_question: str
    standalone_query: str
    chunks: list[RetrievedChunk]
    generated_answer: str
    verified_answer: VerifiedAnswer
    evidence_missing_simulated: bool
    detected_language: str
    detected_script: str


class EvidenceExcerptGenerator:
    """Deterministic demo client that returns one sentence from the first evidence chunk."""

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        import json

        request = json.loads(user_prompt)
        evidence = request["evidence"]
        selected_chunk = None
        selected_sentence = None
        for chunk in evidence:
            body = chunk["text"].split("\n", maxsplit=1)[-1].strip()
            for sentence in re.split(r"(?<=[.!?])\s+", body):
                candidate = sentence.strip()
                # The demo must obey the same no-dose/no-diagnosis policy as a
                # real generator. For example, quoted source wording such as
                # "You have a fever" must not be displayed as a diagnosis.
                if (
                    candidate
                    and not UNSAFE_MEDICINE_PATTERN.search(candidate)
                    and not DIAGNOSIS_PATTERN.search(candidate)
                ):
                    selected_chunk = chunk
                    selected_sentence = candidate
                    break
            if selected_sentence:
                break
        if not selected_chunk or not selected_sentence:
            return json.dumps(
                {
                    "short_answer": "",
                    "general_information": "",
                    "self_care": "",
                    "when_to_see_doctor": "",
                    "citations": [],
                    "disclaimer": "This is general health information, not a diagnosis.",
                    "abstained": True,
                    "abstention_reason": "The retrieved passage has no usable text.",
                }
            )
        cited_sentence = f"{selected_sentence} [{selected_chunk['chunk_id']}]"
        return json.dumps(
            {
                "short_answer": cited_sentence,
                "general_information": "",
                "self_care": "",
                "when_to_see_doctor": "",
                "citations": [selected_chunk["chunk_id"]],
                "disclaimer": "This is general health information, not a diagnosis.",
                "abstained": False,
                "abstention_reason": None,
            }
        )


class ExactEvidenceSplitter:
    """Split the demo excerpt into claims without treating its disclaimer as medical evidence."""

    def split(self, answer: str) -> list[str]:
        return [
            line.strip()
            for line in answer.splitlines()
            if line.strip()
            and "not a diagnosis" not in line.casefold()
            and "general health information" not in line.casefold()
        ]


class ExactEvidenceScorer:
    """Demo-only scorer: supports a claim only if its wording appears in supplied passages."""

    def score(self, claim: str, evidence: str) -> tuple[str, float]:
        plain_claim = re.sub(r"\s*\[[^\]]+\]", "", claim).strip().rstrip(".").casefold()
        normalized_evidence = evidence.casefold()
        if plain_claim and plain_claim in normalized_evidence:
            return "supported", 1.0
        return "not_enough_info", 0.0


def build_offline_demo_components():
    generator = MedicalAnswerGenerator(EvidenceExcerptGenerator(), max_retries=0)
    verifier = ClaimVerifier(
        splitter=ExactEvidenceSplitter(),
        scorer=ExactEvidenceScorer(),
        max_regenerations=0,
    )
    return generator, verifier


def process_question(
    question: str,
    retriever,
    generator: MedicalAnswerGenerator,
    verifier: ClaimVerifier,
    ledger: PatientLedger | None = None,
    simulate_missing_evidence: bool = False,
    turn_id: int = 1,
) -> DemoTurn:
    ledger = ledger or PatientLedger()
    rewrite = StandaloneRewriter().rewrite(question, ledger=ledger, turn_id=turn_id)
    ledger.apply_updates(rewrite.ledger_updates, turn_id=turn_id)
    chunks = retriever.retrieve(rewrite.standalone_query_en, k=5)
    if simulate_missing_evidence:
        chunks = []

    trace = PipelineTrace(
        trace_id=f"demo-turn-{turn_id}",
        conv_id="demo-session",
        turn_id=turn_id,
        user_message=question,
        detected_language=rewrite.language,
        detected_script=rewrite.detected_script,
        rewriter_output=rewrite.to_dict(),
        retrieval_query=rewrite.standalone_query_en,
        ledger_state_after=ledger.get_active_facts(),
        retrieved_chunks=chunks,
    )
    generated = generator.generate(
        question=question,
        ledger_summary=ledger.to_summary_string(),
        chunks=chunks,
        language=rewrite.language,
        script=rewrite.detected_script,
        last_turns=[],
    )
    trace.generator_input = {
        "ledger_summary": ledger.to_summary_string(),
        "chunk_ids": [chunk.chunk_id for chunk in chunks],
    }
    trace.raw_generated_answer = generated.to_text()
    verified = verifier.verify(generated.to_text(), chunks, trace=trace)

    return DemoTurn(
        original_question=question,
        standalone_query=rewrite.standalone_query_en,
        chunks=chunks,
        generated_answer=generated.to_text(),
        verified_answer=verified,
        evidence_missing_simulated=simulate_missing_evidence,
        detected_language=rewrite.language,
        detected_script=rewrite.detected_script,
    )


def render_turn(turn: DemoTurn, write: Callable[[str], None] = print) -> None:
    write(f"You: {turn.original_question}")
    write(f"Retrieval query: {turn.standalone_query}")
    if turn.evidence_missing_simulated:
        write("Evidence: none (empty-evidence scenario)")
    else:
        write("Evidence chunks: " + (", ".join(c.chunk_id for c in turn.chunks) or "none"))
    write("Generated answer: " + turn.generated_answer)
    if turn.verified_answer.abstained:
        write("Verification: ABSTAINED safely")
    else:
        write(
            "Verification: "
            + f"{sum(claim.supported is True for claim in turn.verified_answer.claims)} supported claim(s); "
            + f"{turn.verified_answer.unsupported_count} unsupported claim(s)"
        )
    write("Final answer: " + turn.verified_answer.final_text)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the MedDriftGuard P0-P5 terminal demo.")
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Ask one question using real local retrieval and deterministic mocked answer/verifier clients.",
    )
    parser.add_argument(
        "--query",
        default="What home care is mentioned for a cough?",
        help="Question used by the built-in safe example or with --interactive.",
    )
    args = parser.parse_args()

    print("MedDriftGuard offline demo (real local retrieval; mocked generation and claim scoring).")
    retriever = HybridRetriever()
    generator, verifier = build_offline_demo_components()

    if args.interactive:
        question = input("Enter a general health information question: ").strip()
        if not question:
            print("No question entered; nothing to do.")
            return
        render_turn(process_question(question, retriever, generator, verifier))
        return

    print("\n--- Safe evidence-backed example ---")
    render_turn(process_question(args.query, retriever, generator, verifier))
    print("\n--- Missing-evidence example ---")
    render_turn(
        process_question(
            "How does a warp-drive engine work?",
            retriever,
            generator,
            verifier,
            simulate_missing_evidence=True,
        )
    )


if __name__ == "__main__":
    main()
