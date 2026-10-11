"""End-to-End RAG Assistant with OpenRouter Multi-Model Failover.
Integrates HybridRetriever with OpenRouter API using free models and automatic fallbacks.
"""

import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional
import requests
from dotenv import load_dotenv

# Ensure project root is in sys.path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from app.core.logging import logger
from app.retrieval.hybrid import HybridRetriever, SearchResult, get_retriever

load_dotenv()

# List of prioritized active free models on OpenRouter for robust failover
DEFAULT_FREE_MODELS = [
    os.getenv("OPENROUTER_MODEL", "liquid/lfm-2.5-2.6b:free"),
    "nvidia/nemotron-3-super-120b-a12b:free",
    "apodex/apodex-1.1-mini:free",
    "poolside/laguna-s-2.1:free",
    "liquid/lfm-2.5-2.6b:free",
]
# Deduplicate while preserving order
FREE_MODEL_FAILOVERS: List[str] = []
for m in DEFAULT_FREE_MODELS:
    if m and m not in FREE_MODEL_FAILOVERS:
        FREE_MODEL_FAILOVERS.append(m)


@dataclass
class RAGAnswer:
    query: str
    answer: str
    model_used: str
    retrieved_chunks: List[SearchResult]
    timings_ms: Dict[str, float] = field(default_factory=dict)
    filters: Optional[Dict[str, Any]] = None

    def display(self):
        print("\n" + "=" * 75)
        print(f"QUERY: {self.query}")
        print("=" * 75)
        print(f"MODEL USED: {self.model_used}")
        print(f"LATENCY: Retrieval = {self.timings_ms.get('retrieval_ms', 0):.1f}ms | LLM Generation = {self.timings_ms.get('llm_ms', 0):.1f}ms | Total = {self.timings_ms.get('total_ms', 0):.1f}ms")
        if self.filters:
            print(f"FILTERS APPLIED: {self.filters}")

        print(f"\n--- RETRIEVED EVIDENCE CHUNKS ({len(self.retrieved_chunks)} Chunks) ---")
        for idx, c in enumerate(self.retrieved_chunks, 1):
            score_str = f"Rerank Score: {c.rerank_score:.3f}" if c.rerank_score is not None else f"RRF Score: {c.rrf_score:.4f}"
            bm25_info = f"Rank {c.bm25_rank} (Score {c.bm25_score:.2f})" if c.bm25_rank is not None else "Not in BM25 top"
            dense_info = f"Rank {c.dense_rank} (Cosine {c.dense_similarity:.3f})" if c.dense_rank is not None else "Not in Dense top"
            print(f"[{idx}] {c.restaurant_name} ({c.city}, {c.state} | {c.stars}★) | Chunk: {c.chunk_id}")
            print(f"    • Scores: {score_str} | BM25: {bm25_info} | Dense: {dense_info}")
            print(f"    • Aspects: {c.aspects}")
            print(f"    • Review Text: \"{c.text_masked}\"\n")

        print("=" * 75)
        print("--- GENERATED ANSWER ---")
        print(self.answer)
        print("=" * 75 + "\n")


class RAGAssistant:
    """RAG Assistant combining HybridRetriever and OpenRouter LLM generation."""

    def __init__(
        self,
        retriever: Optional[HybridRetriever] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None
    ):
        self.retriever = retriever or get_retriever()
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        self.base_url = (base_url or os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")).rstrip("/")
        if not self.api_key:
            logger.warning("No OPENROUTER_API_KEY found in environment or .env file.")

    def _call_openrouter_with_failover(self, messages: List[Dict[str, str]], max_tokens: Optional[int] = None) -> tuple[str, str]:
        """Calls OpenRouter with automatic sequential failover across free models and continuation on length limit."""
        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY is not set.")

        effective_max_tokens = max_tokens or int(os.getenv("OPENROUTER_MAX_TOKENS", "2500"))

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "http://localhost:8000",
            "X-Title": "RestaurantOperationsAssistant"
        }

        errors = []
        for model in FREE_MODEL_FAILOVERS:
            payload = {
                "model": model,
                "messages": messages,
                "max_tokens": effective_max_tokens,
                "temperature": 0.2
            }
            try:
                logger.info(f"Attempting LLM generation with model: '{model}' (max_tokens={effective_max_tokens})...")
                resp = requests.post(
                    f"{self.base_url}/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=55
                )

                if resp.status_code == 200:
                    data = resp.json()
                    choice = data["choices"][0]
                    content = (choice.get("message", {}).get("content") or "").strip()
                    finish_reason = choice.get("finish_reason")

                    # If response was truncated due to output token limit, request clean conclusion
                    if finish_reason == "length" and content:
                        logger.info(f"Model '{model}' reached length limit. Requesting concluding completion...")
                        cont_payload = {
                            "model": model,
                            "messages": messages + [
                                {"role": "assistant", "content": content},
                                {"role": "user", "content": "Please continue and conclude your answer concisely from where you left off."}
                            ],
                            "max_tokens": 1000,
                            "temperature": 0.2
                        }
                        try:
                            c_resp = requests.post(
                                f"{self.base_url}/chat/completions",
                                headers=headers,
                                json=cont_payload,
                                timeout=30
                            )
                            if c_resp.status_code == 200:
                                c_text = (c_resp.json()["choices"][0]["message"].get("content") or "").strip()
                                content = content + "\n\n" + c_text
                        except Exception as c_err:
                            logger.warning(f"Continuation request failed: {c_err}")

                    logger.info(f"Successfully generated answer using '{model}'.")
                    return content, model
                else:
                    err_msg = f"Model '{model}' returned HTTP {resp.status_code}: {resp.text[:120]}"
                    logger.warning(err_msg)
                    errors.append(err_msg)
            except Exception as e:
                err_msg = f"Model '{model}' failed with exception: {e}"
                logger.warning(err_msg)
                errors.append(err_msg)

        raise RuntimeError(f"All free OpenRouter candidate models failed. Errors: {'; '.join(errors)}")

    def answer_query(
        self,
        query: str,
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
        use_reranker: bool = True
    ) -> RAGAnswer:
        """Retrieves evidence chunks and generates a grounded response."""
        t_total_start = time.perf_counter()

        # 1. Retrieval
        retrieval_resp = self.retriever.search(
            query=query,
            top_k=top_k,
            filters=filters,
            use_reranker=use_reranker
        )
        retrieval_ms = retrieval_resp.timings_ms.get("total_ms", 0.0)

        # 2. Build Context Prompt
        context_blocks = []
        for i, chunk in enumerate(retrieval_resp.results, start=1):
            context_blocks.append(
                f"[Source {i}] Restaurant: {chunk.restaurant_name} | City: {chunk.city}, {chunk.state} | Rating: {chunk.stars} stars | Chunk ID: {chunk.chunk_id}\n"
                f"Review Excerpt: \"{chunk.text_masked}\"\n"
            )

        context_str = "\n".join(context_blocks) if context_blocks else "No relevant review evidence found in data store."

        system_prompt = (
            "You are an expert AI Restaurant Operations & Review Assistant.\n"
            "Your task is to answer the user's question based strictly on the provided review excerpts and restaurant evidence.\n"
            "Guidelines:\n"
            "1. Ground every claim in the provided sources. Cite the restaurant name and chunk ID whenever mentioning a review or incident.\n"
            "2. Identify actionable operational insights (e.g. wait times, food quality, staff behavior, cleanliness, prices).\n"
            "3. If the retrieved evidence does not contain enough information to answer, state so honestly.\n"
            "4. Keep your answer concise, professional, well-structured with bullet points, and always conclude your answer completely without trailing off or cutting off mid-sentence."
        )

        user_prompt = (
            f"User Question: {query}\n\n"
            f"Retrieved Evidence:\n{context_str}\n\n"
            "Please provide a comprehensive, grounded answer with citations."
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        # 3. LLM Generation with automatic failover
        t_llm_start = time.perf_counter()
        answer_text, model_used = self._call_openrouter_with_failover(messages)
        llm_ms = round((time.perf_counter() - t_llm_start) * 1000, 2)
        total_ms = round((time.perf_counter() - t_total_start) * 1000, 2)

        timings = dict(retrieval_resp.timings_ms)
        timings["retrieval_ms"] = retrieval_ms
        timings["llm_ms"] = llm_ms
        timings["total_ms"] = total_ms

        return RAGAnswer(
            query=query,
            answer=answer_text,
            model_used=model_used,
            retrieved_chunks=retrieval_resp.results,
            timings_ms=timings,
            filters=filters
        )


def ask(query: str, top_k: int = 5, city: Optional[str] = None) -> RAGAnswer:
    """Quick helper function to ask a question to the assistant."""
    assistant = RAGAssistant()
    filters = {"city": city} if city else None
    return assistant.answer_query(query, top_k=top_k, filters=filters)


def main():
    import argparse

    if sys.stdout.encoding.lower() != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description="AI Restaurant Operations RAG Assistant")
    parser.add_argument("query", nargs="?", type=str, help="Search query or question")
    parser.add_argument("--city", type=str, default=None, help="Filter by city (e.g. Philadelphia, Nashville)")
    parser.add_argument("--top-k", type=int, default=5, help="Number of evidence chunks to retrieve")
    parser.add_argument("--interactive", action="store_true", help="Start an interactive query session")
    args = parser.parse_args()

    assistant = RAGAssistant()

    query = args.query
    if not query:
        print("\n" + "=" * 75)
        print("AI RESTAURANT OPERATIONS RAG ASSISTANT")
        print("=" * 75)
        try:
            query = input("Enter your question or operational query (or 'exit' to quit): ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")
            return

    if not query or query.lower() in {"exit", "quit", "q"}:
        print("No query provided. Exiting.")
        return

    filters = {"city": args.city} if args.city else None
    print(f"\n[Retrieving evidence & generating answer for query: '{query}']...")
    ans = assistant.answer_query(query, top_k=args.top_k, filters=filters)
    ans.display()

    # Continuous prompt loop for subsequent questions
    while True:
        try:
            followup = input("Ask another query (or press Enter / 'exit' to quit): ").strip()
            if not followup or followup.lower() in {"exit", "quit", "q"}:
                print("Session finished.")
                break
            print(f"\n[Retrieving evidence & generating answer for query: '{followup}']...")
            ans = assistant.answer_query(followup, top_k=args.top_k, filters=filters)
            ans.display()
        except (KeyboardInterrupt, EOFError):
            print("\nSession finished.")
            break


if __name__ == "__main__":
    main()

