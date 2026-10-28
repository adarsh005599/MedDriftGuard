# MedDriftGuard Submission Status

Audit date: 28 October 2026. This document records what was observed in the
repository and what was run locally during this audit. It does not treat planned
features or historical documentation as completed work.

## Current prototype

The repository contains a local, terminal-based prototype for general health
information. Its demonstrated path is:

1. A rule-based standalone rewriter detects English, Devanagari Hindi, or Roman
   Hinglish and updates an in-memory patient-state ledger.
2. A local 26-chunk knowledge base is searched with BM25, dense retrieval, and
   reciprocal-rank fusion.
3. The terminal demo uses a deterministic evidence-excerpt generator and an
   exact-text verifier. It displays inline chunk IDs and abstains for a simulated
   empty-evidence case.

The terminal demo is not a live medical LLM or NLI demonstration. It explicitly
uses mock generation and mock claim scoring.

## Completed and evidenced work

| Area | Evidence from this audit | Status |
|---|---|---|
| Schemas, KB ingestion, ledger, rewriter, retrieval, generator and verifier interfaces | Source code and unit tests are present. | Implemented with important limitations below. |
| Knowledge base | `kb/chunks.jsonl` contains 26 local English chunks from seven local HTML files. | Present, but very small. |
| Retrieval evaluation | `python eval/retrieval_eval.py` completed on 28 October 2026. | Measured locally; see metrics below. |
| Terminal demo | `python demo.py` completed on 28 October 2026. It retrieved real local chunks, returned a cited deterministic excerpt, and safely abstained in the simulated no-evidence example. | Demonstrable offline prototype. |
| FastAPI + Streamlit UI | `GET /health`, demo-mode `POST /chat`, and a local Streamlit entry point are present. | Presentation prototype; see limitations. |
| M1 data preparation | Synthetic-data generator, validator, and evaluator scripts exist. | Not trained or evaluated with an adapter. |

## Retrieval metrics actually measured

The evaluation used 104 development questions and the 26-chunk local KB.

| Mode | Recall@5 | Recall@10 | MRR |
|---|---:|---:|---:|
| BM25 | 0.7788 | 0.8750 | 0.6612 |
| Dense | 0.8846 | 0.9519 | 0.7504 |
| Hybrid (RRF) | 0.8750 | 0.9519 | 0.7277 |
| Hybrid + reranker stage | 0.8750 | 0.9519 | 0.7470 |

The dense backend loaded `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`
through FastEmbed. The configured neural reranker was unavailable, so the
hybrid-rerank result used the deterministic contextual fallback. Dense-only was
the best Recall@5 result in this run; hybrid did not exceed it.

Recall@5 by script was: BM25 English 1.0000, Roman Hinglish 0.8182,
Devanagari 0.2609; dense English 0.9583, Roman Hinglish 0.6970, Devanagari
1.0000; hybrid and hybrid-rerank English 1.0000, Roman Hinglish 0.8485,
Devanagari 0.6522. These are retrieval-only measurements, not clinical-safety,
answer-quality, or multi-turn-drift results.

## Test and demo evidence

`python -m pytest --collect-only -q` lists 92 tests after the input-safety and
UI/API additions. On 28 October 2026, all 92 passed when run in complete test-file
groups (79 non-retrieval tests, 2 retrieval-evaluation tests, and 11 retriever
tests). The M1 validator was corrected so it checks unsafe user text
before stale cross-field data, while still reporting an invalid ledger fact type
as a schema error. This preserves the training-data safety rule without
weakening the test.

The two terminal-demo tests also pass, and `python demo.py` completed
successfully. On this Windows environment, FastEmbed can leave worker processes
attached to a terminal after retrieval tests; that affects terminal shutdown,
not the 13 passing retriever/evaluation test assertions.

Final API smoke checks also confirmed: a cough question returned citation
`med_cough_0002`; an out-of-scope warp-drive question safely abstained with no
sources; and `ttt` safely requested a meaningful question with no sources.

## Final pre-commit review — 28 October 2026

- `git diff --check` reported no whitespace errors.
- The complete 92-test suite was verified in test-file groups because the local
  FastEmbed backend can keep Windows worker processes attached after retrieval
  tests. The groups were 79 non-retrieval tests, 2 retrieval-evaluation tests,
  and 11 retriever tests; all passed.
- `python demo.py` completed with a cited local cough answer and a safe
  no-evidence abstention.
- Streamlit 1.66.1 was installed, and the website started locally in headless
  verification mode. The visible DEMO MODE label is accurate: retrieval is real
  and local, while generation and verification are deterministic mocks.
- Credential-shaped secret patterns were not found in commit candidates. The
  PII-pattern scan matched only public KB/source text and intentionally
  synthetic validation fixtures; no real patient records were identified.
- `.gitignore` excludes `.env`, virtual environments, Python caches, pytest and
  Ruff caches, local indexes, and model binaries. No large commit-candidate
  files were found.
- Standard setup commands now install the `ui` optional dependency, so the
  README, `Makefile`, `make.bat`, and `requirements.txt` agree on local website
  requirements.

## Mocked, unverified, and incomplete components

- The demo's generator and verifier are deterministic mocks. Live Ollama
  generation, live Ollama claim splitting, and the configured DeBERTa NLI model
  have not been verified in this audit.
- The rewriter has rule-based behavior and an optional Ollama client. Its live
  Ollama path has not been run; the rule lexicon covers only configured topics.
- Dense vectors are built in memory. Despite configuration references, no
  Qdrant or FAISS persisted index is implemented here.
- The neural cross-encoder reranker was not loaded in the measured evaluation.
- `src/triage`, `src/pipeline`, and `src/ui` contain package stubs, not an
  integrated triage gate, end-to-end service pipeline, or user interface.
- The FastAPI chat endpoint and Streamlit page are presentation-demo adapters.
  They use the same deterministic generator and verifier as `demo.py`, not a
  live medical model.
- The M1 notebook has one unexecuted Markdown instruction cell. It contains no
  runnable training cells, and `results/` contains no trained adapter or model
  evaluation artifact.
- The benchmark, 300-conversation HHDB target, human study, drift curve,
  triage classifier, and model-training claims are not evidenced by artifacts
  in this repository.

## Known limitations for a submission

- The KB is seven local English source pages / 26 chunks, not the planned
  broad multilingual corpus. Existing chunk lengths are below the documented
  250–350-word target.
- Development-question gold labels were not independently reviewed in this
  audit, and the set has 104 questions rather than the planned larger set.
- Retrieval metrics do not establish medical correctness, safe triage, answer
  faithfulness, or reduced conversational drift.
- The product is for general health information only; it must not be presented
  as a diagnosis or dosing tool.
- `write_phase3_doc.py` is a one-off untracked document generator whose output
  already exists in `docs/phase_3.md`; retain it locally or remove it later, but
  do not include it in the proposed commit.

## Exact local commands

From the project root in PowerShell:

```powershell
# Install the dependencies used by tests, CPU retrieval, and the local UI
python -m pip install -e ".[dev,retrieval,ui]"

# Run the complete suite
python -m pytest tests -q

# Run the two terminal-demo tests
python -m pytest tests/unit/test_demo.py -q

# Run the terminal prototype
python demo.py
python demo.py --interactive

# Reproduce the retrieval report and metrics
python eval/retrieval_eval.py

# Terminal 1: start the API (keep this terminal open)
uvicorn src.api.main:app --reload --host 127.0.0.1 --port 8000

# Terminal 2: start the local website
streamlit run src/ui/streamlit_app.py
```

Open `http://127.0.0.1:8000/docs` to inspect `GET /health` and the demo-mode
`POST /chat` endpoint. Streamlit normally opens at `http://localhost:8501`.

## Website prototype implemented

1. `POST /chat` keeps a conversation-scoped in-memory ledger and calls the
   existing retrieval plus deterministic demo pipeline. Its response includes
   source metadata, citations, language/script, abstention state, and an
   explicit `demo_mode: true` flag.
2. `src/ui/streamlit_app.py` provides a dark local chat interface. It accepts
   English, Hindi, and Roman Hinglish text; renders source cards; permanently
   displays the medical-information disclaimer and DEMO/MOCK MODE label; and
   provides loading, API-error, empty, and safe-abstention states.
3. API and UI-client contract tests cover citations, the demo-mode label, empty
   input rejection, request routing, and a friendly API-unavailable error.
4. The terminal demo remains unchanged in behavior. Live Ollama/NLI mode is
   still unverified and is not exposed as a working website feature.
5. The website now uses a dark clinical design system, sidebar workspace,
   language selector, new-conversation control, example prompts, responsive
   chat layout, source-detail panels, copy-ready answer panel, and an explicit
   demo badge. It removes raw citation IDs from the reading view but maps every
   displayed source card to a current retrieved source.
6. A small input/relevance gate returns a greeting for greetings, requests a
   meaningful question for blank or repeated-character text such as `ttt`, and
   safely abstains when no meaningful rewritten-query term appears in the
   retrieved evidence. It does not replace retrieval with hardcoded health
   answers.
7. The initial Hinglish fever/cough example was traced end to end. Rewriting
   and retrieval worked, but the deterministic demo excerpt quoted source text
   beginning “You have a fever”, which the existing no-diagnosis policy correctly
   rejected. The demo selector now chooses a different safe sentence from the
   same retrieved evidence; policy thresholds were not relaxed. The example
   chips now use KB-backed cough, Hinglish cough, Hindi headache, and acidity
   questions. The out-of-scope question remains available for the abstention
   demonstration but is not promoted as an example chip.

The UI package was installed and the Streamlit app successfully started locally
on port 8502 during the final verification. Its request client and the API/UI
integration contract are also covered by automated tests.

## Evidence to capture for the presentation

1. The welcome screen, DEMO MODE badge, disclaimer, and the “How it works”
   panel.
2. A cough question with its answer and expandable evidence card.
3. An out-of-scope question showing safe abstention and no sources.
4. A short Hinglish or Devanagari question, showing that the input is accepted.
5. `http://127.0.0.1:8000/docs` showing the local `/chat` contract.

This is the smallest path to a polished presentation prototype while preserving
the current Python backend and its limitations honestly.
