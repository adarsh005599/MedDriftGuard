# MedDriftGuard 🛡️🩺

> **Stopping Conversational Hallucination Drift in Hinglish Patient-Facing Health RAG**  
> *Major Project Phase-II & Research Paper*  
> **Team:** IILM University, B.Tech CSE · **Guide:** Ms. Bhumika Pal

---

## 🌟 Overview

Conversational medical chatbots built on standard Retrieval-Augmented Generation (RAG) suffer from a critical, documented failure mode: **as conversations become longer, the risk of medical hallucination rises significantly**. 

In multilingual and code-mixed settings—such as Indian patients communicating in **Hinglish** (Roman-script Hindi) or **Devanagari Hindi** alongside English—the failure is amplified:
1. Standard query concatenation pollutes the retrieval context with obsolete symptoms or conversational chit-chat.
2. The language model blends past conversational topics into current clinical guidance.
3. Language switching causes retrieval drop-offs (the "script gap").

**MedDriftGuard** solves this problem by introducing a novel, decoupled architecture:
- **Patient-State Ledger**: An explicit, versioned, and editable record of user-stated facts (symptoms, duration, allergies), isolated from the medical evidence store.
- **Standalone Current-Turn Rewriter**: Reformulates the user's latest query into a focused English search query without history bloat.
- **Deterministic Red-Flag Triage Gate**: Immediately intercepts medical emergencies (chest pain, stroke signs, severe trauma) before any LLM processing.
- **Evidence-Bound Generator & Claim Verifier**: Constrains generation strictly to retrieved authoritative evidence and verifies every atomic claim with an NLI entailment model.

---

## 📐 Architecture Diagram

```mermaid
flowchart TD
    classDef emergency fill:#ffebee,stroke:#c62828,stroke-width:2px,color:#b71c1c;
    classDef safety fill:#fff3e0,stroke:#ef6c00,stroke-width:2px,color:#e65100;
    classDef core fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px,color:#1b5e20;
    classDef store fill:#e1f5fe,stroke:#0277bd,stroke-width:2px,color:#01579b;
    classDef output fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px,color:#4a148c;

    UserMsg["👤 User Message<br/>(English / Devanagari / Roman Hinglish)"]:::core
    
    Triage{"🚨 Red-Flag Triage Gate<br/>(Deterministic Lexicon + Classifier)"}:::emergency
    EmergencyAlert["🚑 Safe Emergency Response<br/>(Call 112 / Ambulance 108 & Stop)"]:::emergency

    Rewriter["🔄 Query Rewriter & Script Detector<br/>(Produces Standalone English Query)"]:::core
    
    subgraph StateAndStore ["Decoupled Memory & Evidence"]
        Ledger["📋 Patient-State Ledger<br/>(Versioned User Facts, Retractions)"]:::store
        KBIndex["📚 Curated Knowledge Base<br/>(MedlinePlus, WHO, MoHFW, ICMR)"]:::store
    end

    Retriever["🔍 Hybrid Retriever<br/>(Multilingual ONNX Dense + BM25 + Reranker)"]:::core
    Generator["✍️ Evidence-Bound Generator<br/>(Inputs: Standalone Query + Top Chunks + Ledger)"]:::core
    
    Verifier{"🔬 Claim-Level Verifier<br/>(Atomic Claim NLI Entailment)"}:::safety
    Regen["Regenerate with Forbidden Unsupported Claims"]:::safety
    Abstain["Safely Abstain & Advise Consulting Clinician"]:::safety

    FinalResponse["💬 Verified Response<br/>(User Script + Citations + When to See Doctor + Disclaimer)"]:::output

    UserMsg --> Triage
    Triage -- "Emergency Detected" --> EmergencyAlert
    Triage -- "Safe / Routine" --> Rewriter
    
    Rewriter --> Ledger
    Rewriter -->|"Standalone Query Only (No Raw History)"| Retriever
    
    Retriever <--> KBIndex
    Retriever -->|"Retrieved Chunks [chunk_id]"| Generator
    Ledger -->|"User-Stated Facts"| Generator
    
    Generator --> Verifier
    Verifier -- "All Claims Supported" --> FinalResponse
    Verifier -- "Unsupported Claim Detected" --> Regen
    Regen --> Verifier
    Verifier -- "Still Unsupported" --> Abstain
```

---

## 🔬 Core Contributions

1. **C1 — DriftGuard Architecture**: Decoupled patient-state ledger, current-turn standalone retrieval, atomic claim verification, and deterministic emergency triage.
2. **C2 — HHDB Benchmark (Hinglish Health Drift Benchmark)**: 300 curated multi-turn clinical conversations (3–15 turns) across 3 script variants (English, Devanagari Hindi, Roman Hinglish) with built-in drift traps (topic switching, pronoun follow-ups, user corrections, leak tests).
3. **C3 — Scientific Drift Evaluation**: Rigorous evaluation measuring the **Unsupported-Claim Rate (UCR) vs. Turn Depth**, demonstrating that MedDriftGuard maintains flat error rates across extended conversations where traditional RAG fails.

---

## 🛡️ Medical Safety & Ethics Guardrails

- **Information Only**: MedDriftGuard is an informational awareness assistant. It **never diagnoses** medical conditions.
- **Zero Drug Dosing**: The knowledge base and models are strictly prohibited from indexing, training on, or generating pharmaceutical drug dosages or titration schedules.
- **Deterministic Triage**: Emergency signs (e.g. chest pain, stroke signs, difficulty breathing) trigger instantaneous hard-coded emergency instructions (112 / 108) without querying the LLM.
- **Privacy by Design**: No real patient data is ever used. No personally identifiable information (PII) is recorded in logs or evaluation traces.

---

## 📂 Repository Structure

```text
meddriftguard/
├── .agent/                          # Antigravity agent configuration
│   ├── rules/project.md             # Engineering and safety rules
│   └── workflows/                   # Saved workflows (/run-eval, /safety-check, /phase-report)
├── configs/                         # Declarative YAML configurations
│   ├── pipeline.yaml                # Master pipeline settings
│   ├── retriever.yaml               # Hybrid search & reranker settings
│   ├── models.yaml                  # Open model definitions
│   ├── triage.yaml                  # Red-flag emergency lexicon & emergency numbers
│   ├── verifier.yaml                # NLI entailment threshold configurations
│   ├── eval.yaml                    # Evaluation benchmarks & metrics
│   ├── train_m1.yaml                # Rewriter QLoRA training parameters
│   └── train_m2.yaml                # Multilingual triage classifier training parameters
├── kb/                              # Curated Knowledge Base
│   ├── SOURCES.md                   # Approved source URLs, licenses, and scopes
│   ├── raw/                         # Raw downloaded files
│   ├── clean/                       # Cleaned text sections
│   └── index/                       # Dense & sparse retrieval indices
├── src/                             # Source code
│   ├── triage/                      # Deterministic red-flag triage gate
│   ├── rewriter/                    # Conversational query normalizer & rewriter
│   ├── ledger/                      # Patient-state ledger (versioned facts)
│   ├── retriever/                   # Hybrid retriever (multilingual dense + BM25 + RRF)
│   ├── generator/                   # Evidence-bound response generator
│   ├── verifier/                    # Atomic claim-level NLI verifier
│   ├── pipeline/                    # End-to-end orchestration pipeline
│   ├── api/                         # FastAPI service application
│   ├── ui/                          # Streamlit conversational web interface
│   └── common/                      # Shared schemas, logging, seeds, i18n
├── data/                            # Datasets & Benchmarks
│   ├── synthetic/                   # Synthetic training samples (behaviour only)
│   ├── hhdb/                        # Hinglish Health Drift Benchmark (train/dev/test)
│   ├── redteam/                     # Red-team adversarial safety suite
│   ├── review_sheets/               # Medical reviewer spot-check sheets
│   └── human_eval/                  # Human rater evaluation protocol
├── training/                        # Training scripts & notebooks
├── eval/                            # Evaluation scripts & metric calculators
├── results/                         # Experiment outputs, metrics, and plots
├── docs/                            # Documentation & phase reports
│   ├── hypotheses.md                # Pre-registered scientific hypotheses
│   └── phase_0.md                   # Phase 0 scaffold verification report
├── tests/                           # Automated test suite
│   ├── unit/                        # Unit tests for schemas and modules
│   ├── integration/                 # End-to-end pipeline integration tests
│   └── fixtures/                    # Test conversation fixtures
├── pyproject.toml                   # Project dependencies and tool settings
├── Makefile                         # Automation commands (setup, test, lint, eval)
├── make.bat                         # Windows runner script
└── .env.example                     # Environment configuration template
```

---

## 🚀 Quick Start

### 1. Requirements
- Python 3.11+
- Virtual environment (recommended)

### 2. Setup
Clone the repository and install dependencies:
```bash
# Using pip (tests, CPU retrieval backend, and local website)
pip install -e ".[dev,retrieval,ui]"

# On Linux/macOS
make setup

# On Windows
.\make.bat setup
```

### 3. Running Unit Tests
```bash
# On Linux/macOS
make test

# On Windows
.\make.bat test

# Or directly using pytest
python -m pytest tests/unit -v
```

### 4. Running the API Server
```bash
# Launch FastAPI development server
uvicorn src.api.main:app --reload --host 127.0.0.1 --port 8000
```
Open your browser to `http://127.0.0.1:8000/health` to confirm the API is running.

### 5. Running the presentation website (Windows)

Open two PowerShell terminals in the project folder. In the first terminal run:

```powershell
python -m uvicorn src.api.main:app --reload --host 127.0.0.1 --port 8000
```

In the second terminal run:

```powershell
python -m streamlit run src\ui\streamlit_app.py
```

Open `http://localhost:8501`. The website is explicitly **DEMO MODE**: it uses
real local retrieval but deterministic/mock generation and verification. Try a
cough question to inspect real retrieved sources, then try an out-of-scope
question such as `How does a warp-drive engine work?` to see safe abstention.

### 6. Running the Terminal Demo (Windows)

From the project folder in the VS Code terminal:

```powershell
python -m pip install -e ".[dev,retrieval]"
.\make.bat demo
```

The demo runs a safe cough example and an empty-evidence abstention example. To enter your own question interactively:

```powershell
python demo.py --interactive
```

Retrieval uses the checked-in local KB and configured multilingual embedding model. To keep the demo runnable without model servers, answer generation and claim scoring use deterministic mock clients that demonstrate the P4/P5 flow; this is not a live LLM/NLI evaluation. The demo prints this limitation before running. The built-in examples were run successfully on 2026-10-27: the cough answer cited `med_cough_0002` and passed the mock check; the empty-evidence example abstained.

---

## 📋 Implementation Roadmap

| Phase | Description | Status |
|:---:|---|:---:|
| **P0** | Project scaffolding, configuration, Pydantic schemas, unit tests, and README | ⚠️ Core verified; 62 tests pass; dependency pins and README rendering not verified |
| **P1** | Approved-source knowledge-base ingestion, section chunking, metadata, and safety filter | ⚠️ Pipeline verified locally; 26 chunks generated below target size |
| **P2** | Multilingual dense retrieval, BM25, RRF, and reranking | ⚠️ Retrieval verified; reranker used deterministic fallback in the tested environment |
| **P3** | Patient-state ledger and prompted local rewriter | ⚠️ Rule baseline and mocked JSON path verified; live Ollama model not tested |

### Verified local checks

- Current automated suite: `python -m pytest tests -q` (92 tests; see `docs/submission_status.md` for the dated verification record)
- P3 fixture diagnostic: `python debug_p3.py`
- Retrieval benchmark: `python -c "from eval.retrieval_eval import run_evaluation; run_evaluation()"`
- The retrieval benchmark has 104 questions and the current local knowledge base has 26 chunks. See `docs/retrieval_results.md` for measured results and backend details.
- The local Ollama rewriter can be created with `StandaloneRewriter.from_config()`; when the local service is unavailable or returns invalid JSON, it falls back to deterministic rules.
- Acceptance gaps remain: the dense vectors are held in memory rather than persisted to Qdrant/FAISS; the supplied dev set has 104 questions, not the planned 140; the neural reranker and live Ollama model have not been verified.
- The neural reranker is optional and can be installed with `pip install -e ".[reranker]"`; without it, retrieval uses the documented deterministic fallback.
- The presentation website is local-only and explicitly marked DEMO MODE. It uses real local retrieval, but its generator and verifier are deterministic mocks.

---

## 📜 License
Apache-2.0 License. See `LICENSE` for details.
