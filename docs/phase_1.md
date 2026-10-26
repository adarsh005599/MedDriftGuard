# Phase 1 Completion Report: Knowledge-Base Ingestion Pipeline

## Verification Update — 2026-10-26

This update supersedes historical test counts and any blanket dosage/license compliance claims below.

- Ingestion now enforces the `Approved` source registry, exact registry license, and allowed source URL host. Unregistered and mismatched sources are skipped.
- Command run: `python -m pytest tests/unit/test_kb_ingest.py -q` — **10 passed**.
- Real ingestion of the seven local HTML files into a temporary directory produced **26 unique, schema-valid chunks** from MedlinePlus, MoHFW, and WHO. The implemented dosage patterns found no dosage strings in those outputs.
- Acceptance gap: these chunks measured about **60–134 whitespace words**, below the 250–350 target; short input sections were not padded or joined across headings.
- Limitations: this is only seven local English pages; no live download was tested. License terms were read from the registry but not independently legally audited. Pattern checks do not prove all possible dosing instructions are absent.

## 1. What Was Built
- **Ingestion Pipeline (`kb/ingest.py`)**: End-to-end pipeline reading approved medical documents, cleaning HTML, enforcing license validation, chunking, deduplicating, and saving validated records to `kb/chunks.jsonl`.
- **Noise & Clutter Cleaner (`kb/cleaner.py`)**: BeautifulSoup-based parser stripping `<script>`, `<style>`, `<nav>`, `<footer>`, `<aside>`, and ads while partitioning text by headings (`h1`–`h4`).
- **Zero-Dosage Safety Filter (`kb/safety_filter.py`)**: Rule-based detection and stripping of pharmaceutical drug amounts, schedules, and titration instructions.
- **Section-Aware Chunker (`kb/chunker.py`)**: Chunks text with 15% boundary overlap, attaches topic/section context headings, creates deterministic IDs (`{source}_{topic}_{index:04d}`), and deduplicates via SHA-256.
- **Auditing & Statistics (`docs/kb_stats.md`)**: Full statistical distribution across topics and sources with 20 shortest/longest chunks.
- **Unit Tests (`tests/unit/test_kb_ingest.py`)**: 6 comprehensive unit tests validating cleaning, dosage filtration, overlap, deduplication, and license enforcement.

## 2. Key Design Decisions
- **Deterministic Chunk Identifiers**: Format `{source_code}_{topic_slug}_{idx:04d}` (e.g., `med_fever_0001`, `who_dengue_awareness_0001`) ensures citations are permanently reproducible and traceable.
- **Preserved Topic and Section Preambles**: Prepending `[Topic | Section]` inside each chunk gives the dense embedding model and generator precise grounding context.
- **Mandatory License Gate**: Any source in `kb/SOURCES.md` lacking an explicit public-domain or open-access license is skipped automatically with a logged warning.

## 3. Historical P1 Snapshot (Superseded)

The original count and blanket dosage statement below describe an earlier test set. Current verification above checks the implemented patterns but does not prove all medication instructions are absent. Current chunks are below the planned target size because the local source pages are short.
Acceptance command executed:
```bash
python -m pytest tests/ -v
```
**Test Summary:**
- Total tests collected: 25 (19 from P0 + 6 from P1)
- Total tests passed: 25 (100% pass rate)
- Failures: 0
- Execution duration: ~0.39s

**Knowledge Base Metrics (`docs/kb_stats.md`):**
- Total chunks generated: 26
- Sources: MedlinePlus (10), MoHFW (8), WHO (8)
- Topics: fever, cough, headache, dengue awareness, hypertension awareness, acidity and gerd, diarrhoea and dehydration
- Prohibited drug dosages detected in final chunks: **0 (100% compliant)**

## 4. Known Limitations & Next Steps
- **Ready for Phase 2**: The generated `kb/chunks.jsonl` is ready for indexing with BGE-M3 embeddings, BM25 sparse keyword index, and reciprocal-rank fusion in Phase 2.
