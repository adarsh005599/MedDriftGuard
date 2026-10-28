import json

from src.common.schemas import GeneratedAnswer, LedgerFact, PipelineTrace, RetrievedChunk
from src.generator.answer_generator import MedicalAnswerGenerator
from src.ledger.patient_ledger import PatientLedger
from src.verifier.claim_verifier import ClaimVerifier


def make_chunk(chunk_id="kb_fever_01", text="Fluids help prevent dehydration during fever."):
    return RetrievedChunk(
        chunk_id=chunk_id,
        source="MedlinePlus",
        url="https://example.com/fever",
        date_accessed="2026-10-26",
        section="Care",
        language="en",
        topic="fever",
        license="Public Domain",
        text=text,
    )


class MockGeneratorClient:
    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.prompts = []

    def generate(self, system_prompt, user_prompt):
        self.prompts.append((system_prompt, json.loads(user_prompt)))
        output = self.outputs.pop(0)
        if isinstance(output, Exception):
            raise output
        return json.dumps(output)


def good_generation(**overrides):
    data = {
        "short_answer": "Fluids can help prevent dehydration [kb_fever_01].",
        "general_information": "",
        "self_care": "",
        "when_to_see_doctor": "",
        "citations": ["kb_fever_01"],
        "disclaimer": "This is general health information, not a diagnosis.",
        "abstained": False,
        "abstention_reason": None,
    }
    data.update(overrides)
    return data


def test_generator_uses_evidence_and_limits_context_to_last_two_turns():
    client = MockGeneratorClient([good_generation()])
    ledger = PatientLedger()
    ledger._facts = [LedgerFact(fact_id="f1", fact_type="symptom", value="fever", turn_id=1)]
    turns = [
        {"role": "user", "content": "old turn"},
        {"role": "assistant", "content": "old answer"},
        {"role": "user", "content": "current follow-up"},
    ]

    answer = MedicalAnswerGenerator(client).generate(
        question="How can I avoid dehydration?",
        ledger_summary=ledger.to_summary_string(),
        chunks=[make_chunk()],
        language="en",
        script="en",
        last_turns=turns,
    )

    payload = client.prompts[0][1]
    assert isinstance(answer, GeneratedAnswer)
    assert not answer.abstained
    assert "[kb_fever_01]" in answer.to_text()
    assert payload["last_turns"] == turns[-2:]
    assert payload["evidence"][0]["chunk_id"] == "kb_fever_01"
    assert "fever" in payload["ledger_summary"]


def test_generator_abstains_without_evidence_or_for_invalid_citation():
    unused_client = MockGeneratorClient([])
    generator = MedicalAnswerGenerator(unused_client)
    assert generator.generate("question", "", []).abstained
    assert unused_client.prompts == []

    invalid = good_generation(citations=["not-a-real-chunk"])
    answer = MedicalAnswerGenerator(MockGeneratorClient([invalid, invalid])).generate(
        "question", "", [make_chunk()]
    )
    assert answer.abstained


def test_generator_preserves_model_requested_abstention():
    response = good_generation(
        short_answer="",
        abstained=True,
        abstention_reason="The supplied passages do not answer that question.",
        citations=[],
    )
    answer = MedicalAnswerGenerator(MockGeneratorClient([response])).generate(
        "question", "", [make_chunk()]
    )

    assert answer.abstained
    assert "do not answer" in answer.to_text()
    assert answer.citations == []


def test_generator_rejects_dose_and_diagnosis_responses():
    for text in (
        "Take 5 mg daily [kb_fever_01].",
        "You have dengue [kb_fever_01].",
    ):
        response = good_generation(short_answer=text)
        answer = MedicalAnswerGenerator(MockGeneratorClient([response, response])).generate(
            "question", "", [make_chunk()]
        )
        assert answer.abstained


class MockSplitter:
    def __init__(self, results):
        self.results = list(results)

    def split(self, answer):
        return self.results.pop(0)


class MockScorer:
    def __init__(self, results):
        self.results = list(results)
        self.calls = []

    def score(self, claim, evidence):
        self.calls.append((claim, evidence))
        return self.results.pop(0)


def test_verifier_accepts_supported_claim_and_records_pipeline_trace():
    chunk = make_chunk()
    answer = "Fluids can help prevent dehydration [kb_fever_01].\nThis is general health information, not a diagnosis."
    trace = PipelineTrace(trace_id="trace1", conv_id="conv1", turn_id=1, user_message="question")
    verifier = ClaimVerifier(
        splitter=MockSplitter([["Fluids can help prevent dehydration [kb_fever_01]."]]),
        scorer=MockScorer([("supported", 0.92)]),
    )

    result = verifier.verify(answer, [chunk], trace=trace)

    assert not result.abstained
    assert result.unsupported_count == 0
    assert result.citations == ["kb_fever_01"]
    assert "not a diagnosis" in result.final_text
    assert result.final_text.count("[kb_fever_01]") == 1
    assert trace.atomic_claims[0].label == "supported"
    assert trace.verifier_verdicts[0]["confidence"] == 0.92


def test_verifier_abstains_on_unsupported_claim():
    verifier = ClaimVerifier(
        splitter=MockSplitter([["This passage does not support claim [kb_fever_01]."]]),
        scorer=MockScorer([("not_enough_info", 0.91)]),
    )

    result = verifier.verify(
        "This passage does not support claim [kb_fever_01].\nThis is general health information, not a diagnosis.",
        [make_chunk()],
    )

    assert result.abstained
    assert result.unsupported_count == 1
    assert result.citations == []


def test_verifier_regenerates_once_and_verifies_new_answer():
    chunk = make_chunk()
    regenerated = "Fluids can help prevent dehydration [kb_fever_01].\nThis is general health information, not a diagnosis."
    verifier = ClaimVerifier(
        splitter=MockSplitter(
            [
                ["Unsupported statement [kb_fever_01]."],
                ["Fluids can help prevent dehydration [kb_fever_01]."],
            ]
        ),
        scorer=MockScorer([("not_enough_info", 0.9), ("supported", 0.95)]),
    )
    regen_calls = []

    result = verifier.verify(
        "Unsupported statement [kb_fever_01].\nThis is general health information, not a diagnosis.",
        [chunk],
        regenerate=lambda claims: regen_calls.append(claims) or regenerated,
    )

    assert not result.abstained
    assert len(regen_calls) == 1
    assert result.final_text.startswith("Fluids can help")


def test_verifier_rejects_diagnosis_dosage_missing_disclaimer_and_no_citation():
    cases = [
        ("You have fever [kb_fever_01].", "You have fever [kb_fever_01].", True),
        ("Take 5 mg [kb_fever_01].", "Take 5 mg [kb_fever_01].", True),
        (
            "Fluids are useful [kb_fever_01].",
            "Fluids are useful [kb_fever_01].",
            True,
        ),
        (
            "Fluids are useful [kb_fever_01].",
            "Fluids are useful [kb_fever_01].\nThis is general health information.",
            True,
        ),
        (
            "Fluids are useful.",
            "Fluids are useful.\nThis is general health information, not a diagnosis.",
            True,
        ),
    ]
    for claim_text, response, expected_abstain in cases:
        verifier = ClaimVerifier(
            splitter=MockSplitter([[claim_text]]),
            scorer=MockScorer([("supported", 0.99)]),
        )
        result = verifier.verify(response, [make_chunk()])
        assert result.abstained is expected_abstain
