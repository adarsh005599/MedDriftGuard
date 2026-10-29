"""Provenance-aware PDF extraction for optional approved-source ingestion.

This adapter is derived from the legacy PDF RAG project's extraction approach,
but deliberately does not import LangChain, call an LLM, or treat an arbitrary
upload as trusted medical evidence.  Callers must provide approved provenance.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

from src.common.schemas import RetrievedChunk


class PDFIngestError(RuntimeError):
    """Raised when a PDF cannot be safely extracted or validated."""


@dataclass(frozen=True)
class PDFSource:
    """Approved provenance required before a PDF becomes a retrievable chunk."""

    source_id: str
    title: str
    url: str
    license: str
    topic: str
    date_accessed: str
    approved: bool = False

    def validate(self) -> None:
        if not self.approved:
            raise PDFIngestError("PDF source is not approved for trusted medical evidence")
        if not all(
            value.strip()
            for value in (
                self.source_id,
                self.title,
                self.url,
                self.license,
                self.topic,
                self.date_accessed,
            )
        ):
            raise PDFIngestError("Approved PDF provenance is incomplete")
        if not re.match(r"^https?://", self.url, re.IGNORECASE):
            raise PDFIngestError("Approved PDF source URL must use http or https")


def _split_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    words = text.split()
    if not words:
        return []
    if chunk_size <= 0 or overlap < 0 or overlap >= chunk_size:
        raise ValueError("chunk_size must be positive and overlap must be smaller than chunk_size")
    chunks: list[str] = []
    start = 0
    step = chunk_size - overlap
    while start < len(words):
        body = " ".join(words[start : start + chunk_size]).strip()
        if body:
            chunks.append(body)
        if start + chunk_size >= len(words):
            break
        start += step
    return chunks


def ingest_pdf_bytes(
    pdf_bytes: bytes,
    source: PDFSource,
    *,
    chunk_size: int = 220,
    overlap: int = 35,
) -> list[RetrievedChunk]:
    """Extract an approved PDF into deterministic, provenance-rich chunks.

    PyMuPDF is optional.  The default application does not require PDF support;
    install the ``pdf`` extra before calling this function.
    """
    source.validate()
    if not pdf_bytes:
        return []
    try:
        import fitz  # type: ignore
    except ImportError as error:
        raise PDFIngestError("PDF ingestion requires the optional 'pdf' dependency") from error

    try:
        document = fitz.open(stream=pdf_bytes, filetype="pdf")
    except Exception as error:
        raise PDFIngestError("Could not open PDF bytes") from error

    document_id = hashlib.sha256(pdf_bytes).hexdigest()[:16]
    result: list[RetrievedChunk] = []
    try:
        for page_number, page in enumerate(document, start=1):
            text = page.get_text("text") or ""
            for position, body in enumerate(_split_text(text, chunk_size, overlap)):
                chunk_key = f"{source.source_id}:{document_id}:{page_number}:{position}:{body}"
                chunk_id = f"pdf_{hashlib.sha256(chunk_key.encode('utf-8')).hexdigest()[:16]}"
                result.append(
                    RetrievedChunk(
                        chunk_id=chunk_id,
                        source=source.title,
                        source_id=source.source_id,
                        source_title=source.title,
                        url=source.url,
                        date_accessed=source.date_accessed,
                        section=f"PDF page {page_number}",
                        topic=source.topic,
                        license=source.license,
                        text=f"[{source.title} | PDF page {page_number}]\n{body}",
                        document_id=document_id,
                        page=page_number,
                        chunk_position=position,
                    )
                )
    finally:
        document.close()
    return result
