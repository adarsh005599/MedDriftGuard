"""Small HTTP client used by the Streamlit presentation interface."""

from __future__ import annotations

from typing import Any

import requests


class ChatApiError(RuntimeError):
    """Raised when the local chat API cannot provide a valid reply."""


def send_chat_message(api_base_url: str, message: str, conversation_id: str) -> dict[str, Any]:
    """Send one chat turn to the local API and return its structured response."""
    try:
        response = requests.post(
            f"{api_base_url.rstrip('/')}/chat",
            json={"message": message, "conversation_id": conversation_id},
            timeout=90,
        )
        response.raise_for_status()
        payload = response.json()
    except (requests.RequestException, ValueError) as error:
        raise ChatApiError("Could not reach the local MedDriftGuard API.") from error
    if not isinstance(payload, dict) or not isinstance(payload.get("answer"), str):
        raise ChatApiError("The local API returned an invalid chat response.")
    return payload
