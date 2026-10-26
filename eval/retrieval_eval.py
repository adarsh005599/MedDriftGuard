"""
Retrieval evaluation harness for MedDriftGuard.

Evaluates dense_only, bm25_only, hybrid, and hybrid_rerank retrieval modes
across English, Devanagari Hindi, and Roman Hinglish test sets in data/dev_retrieval.jsonl.
Computes Recall@5, Recall@10, and Mean Reciprocal Rank (MRR), outputting docs/retrieval_results.md.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

# Ensure project root is in sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.common.logging import get_logger
from src.retriever.hybrid_retriever import HybridRetriever, RetrieverMode

logger = get_logger("eval.retrieval")


def load_dev_questions(path: Path) -> list[dict[str, Any]]:
    """Load evaluation questions from JSONL file."""
    questions = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                questions.append(json.loads(line))
    return questions


def evaluate_mode(
    retriever: HybridRetriever,
    questions: list[dict[str, Any]],
    mode: RetrieverMode,
) -> dict[str, Any]:
    """
    Evaluate a specific retriever mode over questions.
    Computes Recall@5, Recall@10, and MRR overall and broken down by script.
    """
    total_queries = len(questions)
    hits_at_5 = 0
    hits_at_10 = 0
    reciprocal_ranks = []

    # Breakdown by script
    script_stats: dict[str, dict[str, Any]] = {}

    for q in questions:
        query_text = q["query"]
        gold_id = q["gold_chunk_id"]
        script = q.get("script", "unknown")

        if script not in script_stats:
            script_stats[script] = {"total": 0, "hits_at_5": 0, "hits_at_10": 0, "rr": []}
        script_stats[script]["total"] += 1

        # Retrieve top 10 chunks using specified mode
        retrieved_chunks = retriever.retrieve(query_text, k=10, mode_override=mode)
        retrieved_ids = [c.chunk_id for c in retrieved_chunks]

        # Calculate hits
        in_top_5 = gold_id in retrieved_ids[:5]
        in_top_10 = gold_id in retrieved_ids[:10]

        if in_top_5:
            hits_at_5 += 1
            script_stats[script]["hits_at_5"] += 1

        if in_top_10:
            hits_at_10 += 1
            script_stats[script]["hits_at_10"] += 1

        # Calculate reciprocal rank
        if gold_id in retrieved_ids:
            rank = retrieved_ids.index(gold_id) + 1
            rr = 1.0 / rank
        else:
            rr = 0.0

        reciprocal_ranks.append(rr)
        script_stats[script]["rr"].append(rr)

    overall_results = {
        "mode": mode,
        "total_queries": total_queries,
        "recall_at_5": round(hits_at_5 / total_queries, 4) if total_queries else 0.0,
        "recall_at_10": round(hits_at_10 / total_queries, 4) if total_queries else 0.0,
        "mrr": round(sum(reciprocal_ranks) / total_queries, 4) if total_queries else 0.0,
        "by_script": {},
    }

    for script, data in script_stats.items():
        n = data["total"]
        overall_results["by_script"][script] = {
            "total": n,
            "recall_at_5": round(data["hits_at_5"] / n, 4) if n else 0.0,
            "recall_at_10": round(data["hits_at_10"] / n, 4) if n else 0.0,
            "mrr": round(sum(data["rr"]) / n, 4) if n else 0.0,
        }

    return overall_results


def generate_markdown_report(results: list[dict[str, Any]], report_path: Path) -> None:
    """Generate docs/retrieval_results.md summarizing all retrieval mode evaluations."""
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8") as f:
        f.write("# Retrieval Evaluation Results (retrieval_results.md)\n\n")
        f.write(
            "Measured on the provided development questions and current local KB. "
            "Backend details below distinguish model inference from deterministic fallbacks.\n\n"
        )

        # Overall Table
        f.write("## Overall System Performance\n\n")
        f.write("| Retrieval Mode | Recall@5 | Recall@10 | MRR | Backend / description |\n")
        f.write("|---|:---:|:---:|:---:|---|\n")

        descriptions = {
            "bm25_only": "Lexical keyword matching using BM25Okapi",
            "dense_only": "Semantic vector search",
            "hybrid": "Reciprocal Rank Fusion (RRF, k=60) of BM25 and Dense",
            "hybrid_rerank": "Hybrid search + configured reranker stage",
        }

        for r in results:
            mode = r["mode"]
            backend = descriptions.get(mode, "")
            if mode in {"dense_only", "hybrid", "hybrid_rerank"}:
                backend += f"; dense={r.get('dense_backend', 'unknown')}"
            if mode == "hybrid_rerank":
                backend += f"; reranker={r.get('reranker_backend', 'unknown')}"
            f.write(
                f"| **{mode}** | {r['recall_at_5']:.4f} | {r['recall_at_10']:.4f} | "
                f"{r['mrr']:.4f} | {backend} |\n"
            )

        # Script Breakdown Table
        f.write("\n## Script Gap Breakdown (Recall@5 by Language & Script)\n\n")
        f.write(
            "| Retrieval Mode | English (en) | Roman Hinglish (roman_hindi) | Hindi (devanagari) | Script Gap (EN - Roman) |\n"
        )
        f.write("|---|:---:|:---:|:---:|:---:|\n")

        for r in results:
            mode = r["mode"]
            by_s = r["by_script"]
            en_r5 = by_s.get("en", {}).get("recall_at_5", 0.0)
            roman_r5 = by_s.get("roman_hindi", {}).get("recall_at_5", 0.0)
            dev_r5 = by_s.get("devanagari", {}).get("recall_at_5", 0.0)
            gap = en_r5 - roman_r5
            f.write(f"| {mode} | {en_r5:.4f} | {roman_r5:.4f} | {dev_r5:.4f} | {gap:+.4f} |\n")

        bm25 = next((r for r in results if r["mode"] == "bm25_only"), None)
        dense = next((r for r in results if r["mode"] == "dense_only"), None)
        hybrid = next((r for r in results if r["mode"] == "hybrid"), None)
        reranked = next((r for r in results if r["mode"] == "hybrid_rerank"), None)
        f.write("\n## Observed Results\n\n")
        if bm25 and dense and hybrid:
            best_single = max(bm25["recall_at_5"], dense["recall_at_5"])
            outcome = "exceeded" if hybrid["recall_at_5"] > best_single else "did not exceed"
            f.write(
                f"- Hybrid Recall@5 {outcome} the best single retriever "
                f"({hybrid['recall_at_5']:.4f} vs {best_single:.4f}).\n"
            )
        if hybrid and reranked:
            f.write(
                f"- Reranking changed MRR from {hybrid['mrr']:.4f} to "
                f"{reranked['mrr']:.4f}; interpret this using the backend noted above.\n"
            )
        f.write(
            "- These results describe retrieval ranking only; they do not measure answer safety, "
            "clinical correctness, or a multi-turn drift effect.\n"
        )


def run_evaluation(
    dev_path: Path = Path("data/dev_retrieval.jsonl"),
    report_path: Path = Path("docs/retrieval_results.md"),
) -> list[dict[str, Any]]:
    """Runs full evaluation across all four retrieval modes."""
    questions = load_dev_questions(dev_path)
    logger.info(f"Loaded {len(questions)} dev questions from {dev_path}.")

    retriever = HybridRetriever()
    dense_backend = (
        f"FastEmbed:{retriever.dense.model_name}"
        if retriever.dense.encoder is not None
        else "deterministic-character-trigram-fallback"
    )
    reranker_backend = (
        f"sentence-transformers:{retriever.reranker.model_name}"
        if retriever.reranker.reranker is not None
        else "deterministic-contextual-fallback"
    )

    modes: list[RetrieverMode] = ["bm25_only", "dense_only", "hybrid", "hybrid_rerank"]
    all_results = []

    for mode in modes:
        logger.info(f"Evaluating retrieval mode: {mode}...")
        res = evaluate_mode(retriever, questions, mode)
        res["dense_backend"] = dense_backend
        res["reranker_backend"] = reranker_backend
        all_results.append(res)
        logger.info(
            f"[{mode}] Recall@5: {res['recall_at_5']} | Recall@10: {res['recall_at_10']} | MRR: {res['mrr']}"
        )

    generate_markdown_report(all_results, report_path)
    return all_results


if __name__ == "__main__":
    run_evaluation()
