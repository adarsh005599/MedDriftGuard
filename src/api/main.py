"""
FastAPI application for MedDriftGuard.
Provides health check and conversational RAG endpoints.
"""

from functools import lru_cache

from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.api.chat_service import DemoChatService

app = FastAPI(
    title="MedDriftGuard API",
    description="Medical information assistant preventing multi-turn conversational hallucination drift in English, Hindi, and Hinglish",
    version="0.1.0",
)


class HealthResponse(BaseModel):
    status: str
    service: str
    version: str


class ChatRequest(BaseModel):
    """One general-health-information chat turn for the local demo service."""

    message: str = Field(min_length=1, max_length=4000)
    conversation_id: str = Field(min_length=1, max_length=128)


class CitationSource(BaseModel):
    chunk_id: str
    source: str
    source_id: str | None = None
    source_title: str | None = None
    url: str
    section: str
    page: int | None = None
    topic: str | None = None
    license: str | None = None
    excerpt: str | None = None


class ChatResponse(BaseModel):
    conversation_id: str
    answer: str
    citations: list[str]
    sources: list[CitationSource]
    abstained: bool
    detected_language: str
    detected_script: str
    demo_mode: bool
    mode_note: str


@lru_cache(maxsize=1)
def get_chat_service() -> DemoChatService:
    """Create the local components once, when the first chat request arrives."""
    return DemoChatService.create_default()


@app.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Return system operational status."""
    return HealthResponse(
        status="healthy",
        service="MedDriftGuard",
        version="0.1.0",
    )


@app.post("/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest, service: DemoChatService = Depends(get_chat_service)  # noqa: B008
) -> ChatResponse:
    """Reply using local retrieval and explicitly deterministic demo components."""
    try:
        return ChatResponse(**service.reply(request.message.strip(), request.conversation_id))
    except Exception as error:
        # The browser receives a friendly failure, while the server keeps the
        # original exception chained for local debugging.
        raise HTTPException(
            status_code=503, detail="The local chat service is unavailable."
        ) from error
