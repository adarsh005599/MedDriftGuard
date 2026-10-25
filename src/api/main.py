"""
FastAPI application for MedDriftGuard.
Provides health check and conversational RAG endpoints.
"""

from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(
    title="MedDriftGuard API",
    description="Medical information assistant preventing multi-turn conversational hallucination drift in English, Hindi, and Hinglish",
    version="0.1.0",
)


class HealthResponse(BaseModel):
    status: str
    service: str
    version: str


@app.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Return system operational status."""
    return HealthResponse(
        status="healthy",
        service="MedDriftGuard",
        version="0.1.0",
    )
