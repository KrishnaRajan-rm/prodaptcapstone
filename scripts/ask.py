"""Executable CLI for asking questions to the Hybrid RAG Assistant.
Usage:
    python scripts/ask.py "waited an hour for food"
    python scripts/ask.py --city "Nashville" "bar service issues"
    python scripts/ask.py --interactive
    python scripts/ask.py --retrieval-only "dirty tables"
"""

import argparse
import sys
from pathlib import Path

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.rag_agent import RAGAssistant
from app.retrieval.hybrid import get_retriever

if sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="AI Restaurant Operations RAG Assistant CLI")
    parser.add_argument("query", nargs="?", type=str, help="Search query or question")
    parser.add_argument("--city", type=str, default=None, help="Filter by city (e.g. Philadelphia, Nashville)")
    parser.add_argument("--state", type=str, default=None, help="Filter by state (e.g. PA, TN)")
    parser.add_argument("--top-k", type=int, default=5, help="Number of evidence chunks to retrieve")
    parser.add_argument("--retrieval-only", action="store_true", help="Only run hybrid retrieval without LLM generation")
    parser.add_argument("--no-rerank", action="store_true", help="Disable cross-encoder reranking")
    parser.add_argument("--interactive", action="store_true", help="Start an interactive query session")

    args = parser.parse_args()

    filters = {}
    if args.city:
        filters["city"] = args.city
    if args.state:
        filters["state"] = args.state
    if not filters:
        filters = None

    if args.interactive:
        print("\n" + "=" * 70)
        print("AI RESTAURANT OPERATIONS RAG ASSISTANT - INTERACTIVE MODE")
        print("Type your question and press Enter. Type 'exit' or 'quit' to stop.")
        print("=" * 70 + "\n")
        assistant = RAGAssistant()

        while True:
            try:
                user_q = input("\nEnter query > ").strip()
                if not user_q:
                    continue
                if user_q.lower() in {"exit", "quit", "q"}:
                    print("Exiting. Have a great day!")
                    break

                rag_ans = assistant.answer_query(
                    query=user_q,
                    top_k=args.top_k,
                    filters=filters,
                    use_reranker=not args.no_rerank
                )
                rag_ans.display()
            except KeyboardInterrupt:
                print("\nInterrupted. Exiting.")
                break
            except Exception as e:
                print(f"Error processing query: {e}")
        return

    if not args.query:
        # Default test query if none provided
        args.query = "waited an hour for food"
        print(f"No query specified. Running default demonstration query: '{args.query}'\n")

    if args.retrieval_only:
        retriever = get_retriever()
        print(f"\n[Running Hybrid Retrieval Only for: '{args.query}']")
        resp = retriever.search(
            query=args.query,
            top_k=args.top_k,
            filters=filters,
            use_reranker=not args.no_rerank
        )
        print(f"Found {len(resp.results)} chunks in {resp.timings_ms.get('total_ms', 0):.1f}ms (BM25: {resp.timings_ms.get('bm25_ms', 0):.1f}ms, Dense: {resp.timings_ms.get('dense_ms', 0):.1f}ms, Rerank: {resp.timings_ms.get('rerank_ms', 0):.1f}ms)")
        print("-" * 70)
        for i, r in enumerate(resp.results, start=1):
            score_desc = f"Rerank Score: {r.rerank_score:.4f}" if r.rerank_score is not None else f"RRF Score: {r.rrf_score:.4f}"
            bm25_str = f"Score: {r.bm25_score:.3f}" if r.bm25_score is not None else "N/A"
            dense_str = f"Sim: {r.dense_similarity:.3f}" if r.dense_similarity is not None else "N/A"
            print(f"    BM25 Rank: {r.bm25_rank} ({bm25_str}) | Dense Rank: {r.dense_rank} ({dense_str})")
            print(f"    Text: \"{r.text_masked[:120]}...\"\n")
        return

    assistant = RAGAssistant()
    print(f"\n[Retrieving evidence & generating answer for: '{args.query}']...")
    rag_ans = assistant.answer_query(
        query=args.query,
        top_k=args.top_k,
        filters=filters,
        use_reranker=not args.no_rerank
    )
    rag_ans.display()


if __name__ == "__main__":
    main()
