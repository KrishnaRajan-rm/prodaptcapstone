"""Retrieval Evaluation and Benchmark Suite.
Measures latency and compares BM25-only, Dense-only, Fused RRF, and Cross-Encoder Reranked
across the benchmark query suite.
"""

import os
import sys
import time
from pathlib import Path
import numpy as np

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.retrieval.hybrid import get_retriever

if sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

BENCHMARK_QUERIES = [
    {"query": "waited an hour for food", "aspect": "wait_time"},
    {"query": "slow service", "aspect": "wait_time"},
    {"query": "rude staff", "aspect": "staff_behavior"},
    {"query": "food was cold", "aspect": "food_quality"},
    {"query": "restaurant cleanliness", "aspect": "cleanliness"},
    {"query": "overpriced small portions", "aspect": "price_value"},
    {"query": "wrong order delivered", "aspect": "order_accuracy"},
    {"query": "noisy loud ambience", "aspect": "ambience_noise"}
]


def run_evaluation():
    print("=" * 80)
    print("HYBRID RAG RETRIEVAL EVALUATION & BENCHMARK SUITE")
    print("=" * 80)

    retriever = get_retriever()

    latencies_bm25 = []
    latencies_dense = []
    latencies_fusion = []
    latencies_rerank = []
    latencies_total = []

    print("\n--- Latency and Method Comparison (Top-5 Chunks) ---")
    print(f"{'Query':<30} | {'BM25 (ms)':<10} | {'Dense (ms)':<10} | {'RRF (ms)':<10} | {'Rerank (ms)':<12} | {'Total (ms)':<10}")
    print("-" * 90)

    for item in BENCHMARK_QUERIES:
        q = item["query"]
        resp = retriever.search(q, top_k=5, use_reranker=True)

        t_bm25 = resp.timings_ms.get("bm25_ms", 0.0)
        t_dense = resp.timings_ms.get("dense_ms", 0.0)
        t_fusion = resp.timings_ms.get("fusion_ms", 0.0)
        t_rerank = resp.timings_ms.get("rerank_ms", 0.0)
        t_tot = resp.timings_ms.get("total_ms", 0.0)

        latencies_bm25.append(t_bm25)
        latencies_dense.append(t_dense)
        latencies_fusion.append(t_fusion)
        latencies_rerank.append(t_rerank)
        latencies_total.append(t_tot)

        print(f"{q:<30} | {t_bm25:>9.1f} | {t_dense:>9.1f} | {t_fusion:>9.1f} | {t_rerank:>11.1f} | {t_tot:>9.1f}")

    print("-" * 90)
    print(f"{'AVERAGE':<30} | {np.mean(latencies_bm25):>9.1f} | {np.mean(latencies_dense):>9.1f} | {np.mean(latencies_fusion):>9.1f} | {np.mean(latencies_rerank):>11.1f} | {np.mean(latencies_total):>9.1f}")

    # Inspect top match per query
    print("\n--- Top Ranked Chunks by Query ---")
    for item in BENCHMARK_QUERIES[:5]:
        q = item["query"]
        resp = retriever.search(q, top_k=1, use_reranker=True)
        top_match = resp.results[0]
        print(f"\nQuery: '{q}'")
        print(f"  • Match: {top_match.restaurant_name} ({top_match.city}, {top_match.state}) - {top_match.stars}★")
        print(f"  • Rerank Score: {top_match.rerank_score:.3f} | RRF: {top_match.rrf_score:.4f} | BM25 Rank: {top_match.bm25_rank} | Dense Rank: {top_match.dense_rank}")
        print(f"  • Excerpt: \"{top_match.text_masked[:120]}...\"")

    print("\n" + "=" * 80)
    print("EVALUATION COMPLETED SUCCESSFULLY")
    print("=" * 80)


if __name__ == "__main__":
    run_evaluation()
