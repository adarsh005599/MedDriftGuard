from fastapi.testclient import TestClient

from src.api.chat_service import DemoChatService, evidence_is_relevant, input_kind
from src.api.main import app, get_chat_service


class FakeChatService:
    def reply(self, message, conversation_id):
        return {
            "conversation_id": conversation_id,
            "answer": f"Evidence-backed demo reply to: {message} [chunk_01]",
            "citations": ["chunk_01"],
            "sources": [
                {
                    "chunk_id": "chunk_01",
                    "source": "Test source",
                    "url": "https://example.com/source",
                    "section": "Overview",
                }
            ],
            "abstained": False,
            "detected_language": "hi",
            "detected_script": "roman_hindi",
            "demo_mode": True,
            "mode_note": "DEMO/MOCK MODE",
        }


def test_chat_endpoint_returns_citations_and_demo_mode():
    app.dependency_overrides[get_chat_service] = lambda: FakeChatService()
    try:
        response = TestClient(app).post(
            "/chat", json={"message": "Mujhe khansi hai", "conversation_id": "test-conversation"}
        )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    payload = response.json()
    assert payload["demo_mode"] is True
    assert payload["citations"] == ["chunk_01"]
    assert payload["sources"][0]["chunk_id"] == "chunk_01"
    assert payload["detected_script"] == "roman_hindi"


def test_chat_endpoint_rejects_empty_message():
    response = TestClient(app).post("/chat", json={"message": "", "conversation_id": "test"})
    assert response.status_code == 422


def test_input_gate_handles_greetings_and_meaningless_text():
    assert input_kind("Namaste") == "greeting"
    assert input_kind("ttt") == "clarify"
    assert input_kind("   ") == "clarify"
    assert input_kind("I have a cough") == "query"


def test_relevance_gate_requires_query_evidence_match():
    class Chunk:
        topic = "cough"
        section = "Home care"
        text = "Warm fluids may soothe an irritated throat."

    assert evidence_is_relevant("cough home care", [Chunk()])
    assert not evidence_is_relevant("warp drive engine", [Chunk()])


def test_greeting_and_meaningless_text_do_not_invoke_retrieval():
    service = DemoChatService(retriever=None, generator=None, verifier=None)

    greeting = service.reply("Hello", "conversation")
    meaningless = service.reply("ttt", "conversation")

    assert greeting["abstained"] is True
    assert greeting["citations"] == []
    assert "general health-information questions" in greeting["answer"]
    assert meaningless["abstained"] is True
    assert meaningless["sources"] == []
    assert "meaningful health-information question" in meaningless["answer"]
