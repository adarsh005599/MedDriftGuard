"""Optional document ingestion adapters for approved evidence sources."""

from src.ingest.pdf_processor import PDFIngestError, PDFSource, ingest_pdf_bytes

__all__ = ["PDFIngestError", "PDFSource", "ingest_pdf_bytes"]
