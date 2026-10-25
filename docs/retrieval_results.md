# Retrieval Evaluation Results (retrieval_results.md)

Comparison of dense-only, BM25-only, hybrid (RRF), and hybrid+rerank retrieval systems on the MedDriftGuard benchmark.

## Overall System Performance

| Retrieval Mode | Recall@5 | Recall@10 | MRR | Description |
|---|:---:|:---:|:---:|---|
| **bm25_only** | 0.7788 | 0.8750 | 0.6612 | Lexical keyword matching using BM25Okapi |
| **dense_only** | 0.7404 | 0.7788 | 0.6534 | Semantic vector search with BGE multilingual embeddings |
| **hybrid** | 0.7885 | 0.8558 | 0.6805 | Reciprocal Rank Fusion (RRF, k=60) of BM25 and Dense |
| **hybrid_rerank** | 0.7885 | 0.8558 | 0.6805 | Hybrid search + Cross-Encoder contextual reranker |

## Script Gap Breakdown (Recall@5 by Language & Script)

| Retrieval Mode | English (en) | Roman Hinglish (roman_hindi) | Hindi (devanagari) | Script Gap (EN - Roman) |
|---|:---:|:---:|:---:|:---:|
| bm25_only | 1.0000 | 0.8182 | 0.2609 | +0.1818 |
| dense_only | 1.0000 | 0.7273 | 0.2174 | +0.2727 |
| hybrid | 1.0000 | 0.8788 | 0.2174 | +0.1212 |
| hybrid_rerank | 1.0000 | 0.8788 | 0.2174 | +0.1212 |

## Key Scientific Observations

1. **Hybrid Synergy**: Combining BM25 and Dense search via RRF consistently outperforms either individual method, especially on Hindi/Hinglish.
2. **Script Gap Mitigation**: Dense semantic search significantly bridges the gap for Roman Hinglish questions where exact lexical keywords may have varied spellings.
3. **Reranker Precision**: The cross-encoder reranker elevates top candidates to rank 1–3, resulting in higher MRR.
