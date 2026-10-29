# MedDriftGuard architecture

## Runtime flow

The trusted application path is:

`user message -> input gate -> standalone rewriter -> patient ledger -> hybrid retrieval -> evidence-bound generator -> claim verifier -> cited response`

The retriever receives only the current standalone query. Conversation history is used by the rewriter when resolving follow-ups; assistant output is never copied into the patient ledger as a fact.

## Evidence and provenance

The checked-in approved-source registry in `kb/SOURCES.md` remains the default medical evidence boundary. `kb/ingest.py` cleans and chunks approved HTML sources into `RetrievedChunk` records, and `src/retriever/` provides BM25, dense, reciprocal-rank-fusion, and optional reranking behavior.

The legacy PDF project contributed an optional adapter in `src/ingest/pdf_processor.py`. It extracts native PDF text with PyMuPDF and emits stable IDs plus source, document, page, and chunk metadata. A PDF is rejected unless the caller supplies approved provenance; arbitrary uploads are not automatically trusted medical evidence.

`src/retriever/faiss_index.py` provides optional FAISS persistence. It writes the vector index and a JSON metadata sidecar, never a pickle sidecar. The current default demo continues to use the existing local hybrid retriever and checked-in KB. FAISS can be used for an explicitly built approved-source index after installing the `pdf` extra.

## Safety boundaries

- Emergency handling must remain deterministic and occur before model generation.
- Generated claims are checked against retrieved evidence and the system abstains when evidence is absent or insufficient.
- Dosages and diagnosis claims are rejected by the existing safety filters and generator/verifier pipeline.
- API conversation state is keyed by `conversation_id`; raw health details are not written to normal logs.
- The configured red-flag triage policy is not currently wired into the demo runtime; emergency handling must be implemented and tested before this is presented as an emergency-safe clinical system.
- The old LangChain `ConversationBufferMemory` and direct Gemini chain were intentionally not integrated because they bypass the ledger/verifier contract and require an external API key.
