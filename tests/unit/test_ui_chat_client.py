import pytest
import requests

from src.ui.chat_client import ChatApiError, send_chat_message


class FakeResponse:
    def raise_for_status(self):
        return None

    def json(self):
        return {"answer": "A cited demo answer [chunk_01]", "demo_mode": True}


def test_ui_client_posts_to_chat_endpoint(monkeypatch):
    captured = {}

    def fake_post(url, json, timeout):
        captured.update({"url": url, "json": json, "timeout": timeout})
        return FakeResponse()

    monkeypatch.setattr("src.ui.chat_client.requests.post", fake_post)
    result = send_chat_message("http://127.0.0.1:8000/", "What is cough?", "conversation-1")

    assert captured["url"] == "http://127.0.0.1:8000/chat"
    assert captured["json"]["conversation_id"] == "conversation-1"
    assert result["demo_mode"] is True


def test_ui_client_shows_friendly_error_when_api_is_unavailable(monkeypatch):
    def unavailable(*args, **kwargs):
        raise requests.ConnectionError("offline")

    monkeypatch.setattr("src.ui.chat_client.requests.post", unavailable)
    with pytest.raises(ChatApiError, match="Could not reach"):
        send_chat_message("http://127.0.0.1:8000", "Hello", "conversation-1")
