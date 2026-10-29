# Legacy PDF RAG integration note

## Reused

- The old project's PyMuPDF-based native PDF text extraction approach.
- Its configurable word-window chunking idea.
- Its use of FAISS as an optional semantic index backend.

## Adapted

- PDF extraction is now in `src/ingest/pdf_processor.py` and returns MedDriftGuard `RetrievedChunk` records.
- Every imported chunk receives a deterministic ID and source/document/page/chunk-position metadata.
- FAISS persistence is in `src/retriever/faiss_index.py`, with a JSON metadata sidecar and count validation.
- Provenance approval is required before PDF content can enter the trusted evidence shape.

## Intentionally not integrated

- LangChain `ConversationBufferMemory`: it would mix assistant-generated text with user facts and duplicate the patient-state ledger.
- The direct Gemini/LangChain answer chain: it bypasses MedDriftGuard claim verification, fixed triage handling, and no-diagnosis/no-dosage policies.
- The old Streamlit upload UI: the current premium UI and approved local KB remain the safe default. Upload support should be added only with an explicit source-review workflow.

## Verification

The integration boundary is covered by `tests/unit/test_pdf_integration.py`. The existing full suite remains the primary regression check. Live PDF/FAISS execution requires the optional dependencies and an approved source record.
