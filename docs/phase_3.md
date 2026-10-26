# Phase 3 Completion Report: Patient-State Ledger & Standalone Rewriter

## Verification Update — 2026-10-26

This update supersedes the historical claims below, especially the claim that the prompted model had not been implemented and the old test count.

- Ledger supports all seven documented fact types and versioned `active`, `corrected`, and `retracted` states. Symptom extraction handles the tested English, Devanagari, and Roman Hinglish corrections, avoids recording informational questions as patient symptoms, and resolves follow-ups from the ledger or at most the last two turns.
- The prompted baseline has an optional local Ollama client via `StandaloneRewriter.from_config()`. It validates the documented JSON response, requires verbatim evidence from the current user message for updates, retries malformed/unsafe outputs, and falls back to deterministic rules. Unit tests use a mock; no paid API is required.
- Commands run: `python -m pytest tests/unit/test_ledger.py tests/unit/test_rewriter.py -q` — **20 passed**; `python debug_p3.py` — **0 fixture failures**; `make.bat test-all` — **62 passed, 0 failed** on Python 3.12.0.
- Regression coverage now checks every fixture turn's active symptoms, all fact types, malformed JSON retry, unsupported fact rejection, empty input, script detection, ambiguity, topic changes, and standalone-query-only retriever use.
- Limitation: live Ollama service/model execution has not been tested. Pattern-based rule fallback recognizes only its configured lexicon and cannot understand arbitrary medical topics.
- Acceptance status: implementation and offline tests pass, but live local-model behavior remains unverified; the deterministic fallback does not extract every documented fact type on its own.

## 1. What Was Built

### Patient-State Ledger (`src/ledger/patient_ledger.py`)
A versioned, in-memory memory book that records **only what the patient explicitly stated**.

- **`LedgerFact`** objects carry `fact_id`, `fact_type`, `value`, `turn_id`, `status` (`active | corrected | retracted`), and an optional `supersedes` pointer linking to the prior fact they replaced.
- Supported operations (via `apply_update(LedgerUpdate, turn_id)`):
  - **`add`** — records a new user-stated fact; deduplicates against existing active facts.
  - **`retract`** — marks the target fact as `retracted` and appends a retraction entry (supersedes chain preserved).
  - **`correct`** — marks the old fact as `corrected`, activates a new fact with `supersedes` pointer.
- **`get_active_facts()`** — returns only currently live, user-stated facts.
- **`get_retracted_facts()`** — returns all `retracted` and `corrected` facts for audit.
- **`to_summary_string()`** — produces a compact, privacy-safe summary (e.g. `Symptom: dry cough; Duration: 3 days`) used as grounding context for the generator — without exposing raw chat history.

The ledger **never** stores inferred diagnoses. Only user-stated facts enter the ledger.

---

### Standalone Rewriter (`src/rewriter/`)

#### `rule_extractor.py` — Deterministic Baseline
A zero-dependency, offline pattern extractor providing 100% baseline coverage:

- **`SYMPTOM_LEXICON`** — 40+ multilingual term -> canonical English mappings across fever, cough, headache, acidity, diarrhoea, hypertension, and dengue in English, Roman Hindi, and Devanagari.
- **`NEGATION_PATTERNS`** — 6 regex patterns covering English negations (`no`, `not`, `no longer`), Roman Hinglish (`nahi ... nahi hai`), and Devanagari.
- **`DURATION_PATTERNS`** — English, Roman Hindi (`2 din se`, `kal se`), and Devanagari duration extraction.
- **`TOPIC_SHIFT_PATTERNS`** — English (`forget that`, `by the way`), Hinglish (`chalo ye chhodo`), and Devanagari.
- **`PRONOUN_FOLLOWUP_PATTERNS`** — English (`its treatment`), Hinglish (`iska ilaj`, `uski home care`), and Devanagari.
- **`RuleBasedExtractor.extract_updates()`** — returns `List[LedgerUpdate]` with `add`, `retract`, and `correct` operations without any LLM dependency.

#### `prompt_templates.py` — LLM Prompt Specification
- **`REWRITER_SYSTEM_PROMPT`** — strict system prompt covering pronoun resolution rules, JSON-only output contract, and safety constraints (no diagnoses, no dosages).
- **`FEW_SHOT_EXAMPLES`** — 3 canonical multi-turn Hinglish examples demonstrating fever -> correction -> follow-up pronoun resolution.

#### `standalone_rewriter.py` — Core Rewriter Engine
`StandaloneRewriter.rewrite(user_message, ledger, last_turns, turn_id)` returns a `RewriterResult` with:

| Field | Description |
|---|---|
| `standalone_query_en` | Clean English retrieval query, free of raw conversation history |
| `detected_script` | `en / devanagari / roman_hindi` |
| `language` | `en / hi` |
| `topic_shift` | `True` if user explicitly changed subject |
| `ledger_updates` | `List[LedgerUpdate]` to apply to the ledger |

**Key algorithms:**
- **`_pick_primary_symptom()`** — specificity-ranked selection (prefers `migraine > headache`, `dry cough > cough`, `high fever > fever`, `diarrhoea > dehydration`) ensuring the most medically specific query.
- **Retraction-aware ledger lookup** — when generating the query for a retraction turn, symptoms retracted in the *current* turn are excluded from the primary symptom lookup so the query shifts to the remaining active symptom.
- **Pronoun resolution** — identifies the intent type (care vs. causes vs. doctor) using a multilingual keyword bank covering English, Roman Hindi, and Devanagari Hindi.

---

## 2. Architecture

```
User Message (EN / Roman Hinglish / Devanagari)
        |
        v
  +---------------------+
  |  i18n.detect_script |  -> script: en | devanagari | roman_hindi
  +---------------------+
        |
        v
  +-------------------------+
  |  RuleBasedExtractor     |
  |  +-- detect_topic_shift |
  |  +-- is_pronoun_followup|
  |  +-- extract_updates()  |  -> [LedgerUpdate(op, fact_type, value)]
  +-------------------------+
        |
        v
  +---------------------------------------------+
  |  StandaloneRewriter._formulate_query()      |
  |                                             |
  |  Pronoun follow-up? -->  resolve from ledger|
  |  New adds?          -->  symptom_str query  |
  |  Retraction turn?   -->  remaining active   |
  |  Direct inquiry?    -->  symptom + intent   |
  +---------------------------------------------+
        |
        v
  standalone_query_en   <- clean English string
  (passed to retriever -- zero raw history)
        |
        v
  +------------------+
  |  PatientLedger   | <- apply_updates() updates versioned fact store
  |  apply_updates() |
  +------------------+
```

---

## 3. Current Test Results

### Unit Tests

Current full-suite result: **62 passed, 0 failed**. The previous 40-test snapshot was from the earlier phase state and is superseded.

```
tests/unit/test_api.py                 [  1 passed ]
tests/unit/test_kb_ingest.py           [ 10 passed ]
tests/unit/test_ledger.py              [  6 passed ]
tests/unit/test_retrieval_eval.py      [  2 passed ]
tests/unit/test_retriever.py           [ 11 passed ]
tests/unit/test_rewriter.py            [ 14 passed ]
tests/unit/test_schemas.py             [ 18 passed ]
                                         -----------
                                TOTAL:   62 passed, 0 failed
```

**P3 Ledger tests (`test_ledger.py`):**

| Test | Coverage |
|---|---|
| `test_ledger_add_fact` | Fact addition, ID assignment, active status |
| `test_ledger_retract_fact` | Retraction, supersedes pointer, empty active set |
| `test_ledger_correct_fact_chain` | Correction supersedes chain, old fact marked corrected |
| `test_ledger_summary_string` | Summary formatting by fact_type |
| `test_ledger_never_stores_unmentioned_info` | Privacy guard: only user-stated facts |

**P3 Rewriter tests (`test_rewriter.py`):**

| Test | Coverage |
|---|---|
| `test_rule_extractor_negation_and_retraction` | Hinglish negation -> retract+add |
| `test_topic_shift_detection` | EN/Hinglish/Devanagari topic shift patterns |
| `test_pronoun_followup_resolution` | Pronoun ("its home care") resolved to ledger symptom |
| `test_30_conversations_acceptance_suite` | All 30 fixture conversations — 100% pass; active facts checked on every turn |

### Acceptance Test: 30 Hand-Written Conversations

| Language | Conversations | Turns | All Pass |
|---|:---:|:---:|:---:|
| English | 10 | 21 | 100% |
| Roman Hinglish | 10 | 21 | 100% |
| Devanagari Hindi | 10 | 21 | 100% |
| **Total** | **30** | **63** | **100%** |

**Verified per-turn:**
- Topic shift detected when expected
- Retracted symptoms absent from active ledger facts
- Standalone queries contain required terms (e.g. "dry cough", "home care", "dengue", "causes", "when to see a doctor")
- Standalone queries do not contain forbidden terms (e.g. "fever" after retraction)

---

## 4. Acceptance Criteria — Verification

| Criterion | Status |
|---|:---:|
| Patient-State Ledger accurately tracks active vs. retracted facts with superseding links | PASS |
| Retracted facts never appear in `get_active_facts()` | PASS |
| Ledger stores only user-stated facts (no inferred diagnoses) | PASS |
| Rewriter outputs structured `RewriterResult` with clean English `standalone_query_en` | PASS |
| 30 hand-written test conversations in `tests/fixtures/conversations_p3.json` pass 100% | PASS |
| Acceptance test `test_30_conversations_acceptance_suite` passes | PASS |
| All 62 cumulative tests pass (`make.bat test-all`) | PASS |
| `docs/phase_3.md` generated | PASS |

---

## 5. Design Decisions

### Why specificity-ranked `_pick_primary_symptom()`?
When a ledger contains both "headache" and "migraine", the more specific term ("migraine") should anchor the retrieval query. This reduces semantic dilution in the knowledge-base search.

### Why exclude retracted symptoms from ledger lookup during query formulation?
When a user says "No longer have high fever, fever is gone" (without stating a new symptom), the query must target the *remaining* concern (e.g. dehydration), not the just-retracted one. The `retracted_in_turn` exclusion set achieves this without modifying the ledger prematurely.

### Why `bachaav` maps to care, not causes?
"Bachaav" means prevention/protection — semantically closer to home-care guidance than root-cause explanation. Placing it in the care keyword bucket produces queries like "home care and hydration management for dengue awareness" which aligns with what a patient asking "iski home care aur bachaav kaise karein?" actually needs.

### Why Devanagari keywords in the English query formulator?
The formulator receives the raw user text and must classify intent before translating to English. Adding Devanagari markers (doctor, doctor-visit, actions) to the keyword buckets ensures intent detection works regardless of input script, without requiring a separate translation step.

---

## 6. Known Limitations & Next Steps

- The rule extractor covers the 8 symptom domains in the knowledge base. Extension to new domains (e.g. diabetes, asthma) requires adding entries to `SYMPTOM_LEXICON`.
- LLM-augmented mode (`prompted_baseline`) currently uses the rule extractor as the sole engine. Plugging in a Gemini API call with `REWRITER_SYSTEM_PROMPT` and `FEW_SHOT_EXAMPLES` would upgrade to LLM-quality reformulation with graceful fallback to rules if the API is unavailable.
- Ready for **Phase 4**: Integrating the rewriter into the full pipeline (`src/pipeline/`) and connecting ledger state across conversation sessions.
