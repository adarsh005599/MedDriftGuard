# Retrieval Evaluation Results (retrieval_results.md)

Measured on the provided development questions and current local KB. Backend details below distinguish model inference from deterministic fallbacks.

## Overall System Performance

| Retrieval Mode | Recall@5 | Recall@10 | MRR | Backend / description |
|---|:---:|:---:|:---:|---|
| **bm25_only** | 0.7788 | 0.8750 | 0.6612 | Lexical keyword matching using BM25Okapi |
| **dense_only** | 0.8846 | 0.9519 | 0.7504 | Semantic vector search; dense=FastEmbed:sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 |
| **hybrid** | 0.8750 | 0.9519 | 0.7277 | Reciprocal Rank Fusion (RRF, k=60) of BM25 and Dense; dense=FastEmbed:sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 |
| **hybrid_rerank** | 0.8750 | 0.9519 | 0.7470 | Hybrid search + configured reranker stage; dense=FastEmbed:sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2; reranker=deterministic-contextual-fallback |

## Script Gap Breakdown (Recall@5 by Language & Script)

| Retrieval Mode | English (en) | Roman Hinglish (roman_hindi) | Hindi (devanagari) | Script Gap (EN - Roman) |
|---|:---:|:---:|:---:|:---:|
| bm25_only | 1.0000 | 0.8182 | 0.2609 | +0.1818 |
| dense_only | 0.9583 | 0.6970 | 1.0000 | +0.2613 |
| hybrid | 1.0000 | 0.8485 | 0.6522 | +0.1515 |
| hybrid_rerank | 1.0000 | 0.8485 | 0.6522 | +0.1515 |

## Observed Results

- Hybrid Recall@5 did not exceed the best single retriever (0.8750 vs 0.8846).
- Reranking changed MRR from 0.7277 to 0.7470; interpret this using the backend noted above.
- These results describe retrieval ranking only; they do not measure answer safety, clinical correctness, or a multi-turn drift effect.
