# Phase 2 Completion Report: Hybrid Retriever & Benchmark Evaluation

## Verification Update — 2026-10-26

This update supersedes the older BGE-M3 model and retrieval metrics below.

- Retriever configuration now selects FastEmbed-supported multilingual `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`; BM25 and empty/malformed index behavior have regression tests.
- Command run: `python -m pytest tests/unit/test_retriever.py tests/unit/test_retrieval_eval.py -q` — **13 passed, 0 failed** (rerun during final review).
- Full evaluation command ran on **104 questions** and **26 chunks**. Measured Recall@5 / Recall@10 / MRR: BM25 **0.7788 / 0.8750 / 0.6612**; dense **0.8846 / 0.9519 / 0.7504**; hybrid **0.8750 / 0.9519 / 0.7277**; hybrid plus reranker stage **0.8750 / 0.9519 / 0.7470**.
- The test environment had no `sentence-transformers`; reranking used the deterministic contextual fallback, not a neural cross-encoder. The evaluation report labels this accurately.
- Limitations: the 104 questions do not meet the planned 140-question design, and their gold labels were not independently reviewed in this run. The small KB makes scores diagnostic only.

## 1. What Was Built
- **BM25 Keyword Search (`src/retriever/bm25_search.py`)**: Sparse lexical retrieval engine using `rank_bm25.BM25Okapi` over tokenized knowledge-base chunks.
- **Dense Semantic Vector Search (`src/retriever/dense_search.py`)**: Vector retrieval using FastEmbed's ONNX-accelerated BGE multilingual embedding model with L2-normalized cosine similarity.
- **Reciprocal Rank Fusion (`src/retriever/fusion.py`)**: Rank-combining algorithm merging sparse and dense candidates using $RRF(d) = \sum \frac{1}{60 + \text{rank}}$.
- **Cross-Encoder Contextual Reranker (`src/retriever/reranker.py`)**: Re-scoring stage utilizing query-document term coverage and heading alignment to refine candidate rankings.
- **Master Retriever (`src/retriever/hybrid_retriever.py`)**: Interface supporting `bm25_only`, `dense_only`, `hybrid`, and `hybrid_rerank`.
- **Diagnostic Benchmark (`data/dev_retrieval.jsonl`)**: 104 diagnostic test questions across English, Devanagari Hindi, and Roman Hinglish mapped to gold chunk IDs.
- **Evaluation Harness (`eval/retrieval_eval.py`)**: Automated evaluator generating [docs/retrieval_results.md](file:///c:/Users/happy/Downloads/Major-pro/docs/retrieval_results.md).

## 2. Key Design Decisions
- **Strict No-History Guardrail**: The retriever interface enforces that `standalone_query` must be a string, raising a `TypeError` if conversation history objects (e.g. lists or dicts) are passed.
- **Zero Heavy-GPU Dependencies**: Leveraged FastEmbed and ONNX Runtime for CPU execution without requiring multi-gigabyte PyTorch or CUDA installations.
- **Script-Gap Mitigation**: Hybrid fusion combines exact clinical terminology from BM25 with dense semantic invariance, boosting Roman Hinglish Recall@5 from 0.7273 to 0.8788.

## 3. Historical P2 Snapshot (Superseded)

The model, test count, and retrieval scores below are from an earlier configuration. Use the measured metrics and backend details in the verification update at the top of this report and in `docs/retrieval_results.md`.
Acceptance command executed:
```bash
python -m pytest tests/ -v
```
**Test Summary:**
- Total tests: 31 (19 from P0 + 6 from P1 + 6 from P2)
- Total passed: 31 (100% pass rate)
- Failures: 0
- Test execution time: ~1.54s

**Retrieval Performance (`docs/retrieval_results.md` on 104 queries):**
| Mode | Recall@5 | Recall@10 | MRR | English R@5 | Hinglish R@5 | Script Gap (EN - Roman) |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| `bm25_only` | 0.7788 | 0.8750 | 0.6612 | 1.0000 | 0.8182 | +0.1818 |
| `dense_only` | 0.7404 | 0.7788 | 0.6534 | 1.0000 | 0.7273 | +0.2727 |
| `hybrid` (RRF) | **0.7885** | **0.8558** | **0.6805** | 1.0000 | **0.8788** | **+0.1212** |
| `hybrid_rerank` | **0.7885** | **0.8558** | **0.6805** | 1.0000 | **0.8788** | **+0.1212** |

## 4. Known Limitations & Next Steps
- Ready for Phase 3: Implementing the decoupled Patient-State Ledger and prompted rewriter to handle multi-turn user corrections, follow-up questions, and topic shifts.
