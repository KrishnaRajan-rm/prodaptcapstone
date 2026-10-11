"""End-to-End RAG Assistant with OpenRouter Multi-Model Failover.
Integrates HybridRetriever with OpenRouter API using free models and automatic fallbacks.
"""

import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import requests
from dotenv import load_dotenv

from app.core.logging import logger
from app.retrieval.hybrid import HybridRetriever, SearchResult, get_retriever

load_dotenv()

# List of prioritized free models on OpenRouter for robust failover
DEFAULT_FREE_MODELS = [
    os.getenv("OPENROUTER_MODEL", "liquid/lfm-2.5-2.6b:free"),
    "liquid/lfm-2.5-2.6b:free",
    "meta-llama/llama-3.3-70b-instruct:free",
    "google/gemini-2.0-flash-exp:free",
    "meta-llama/llama-3.2-3b-instruct:free",
    "meta-llama/llama-3.1-8b-instruct:free",
    "mistralai/mistral-small-24b-instruct-2501:free",
    "qwen/qwen-2.5-coder-32b-instruct:free",
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
        print("\n" + "=" * 70)
        print(f"QUERY: {self.query}")
        print("=" * 70)
        print(f"MODEL: {self.model_used}")
        print(f"TIMINGS: Retrieval = {self.timings_ms.get('retrieval_ms', 0):.1f}ms | LLM = {self.timings_ms.get('llm_ms', 0):.1f}ms | Total = {self.timings_ms.get('total_ms', 0):.1f}ms")
        print("\n--- RETRIEVED EVIDENCE CHUNKS (Top 3) ---")
        for idx, c in enumerate(self.retrieved_chunks[:3], 1):
            score_str = f"Rerank: {c.rerank_score:.3f}" if c.rerank_score is not None else f"RRF: {c.rrf_score:.4f}"
            print(f"[{idx}] {c.restaurant_name} ({c.city}, {c.state} - {c.stars}★) | Chunk: {c.chunk_id} | {score_str}")
            print(f"    \"{c.text_masked[:120]}...\"\n")
        print("--- GENERATED ANSWER ---")
        print(self.answer)
        print("=" * 70 + "\n")


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

    def _call_openrouter_with_failover(self, messages: List[Dict[str, str]], max_tokens: int = 800) -> tuple[str, str]:
        """Calls OpenRouter with automatic sequential failover across free models."""
        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY is not set.")

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
                "max_tokens": max_tokens,
                "temperature": 0.2
            }
            try:
                logger.info(f"Attempting LLM generation with model: '{model}'...")
                resp = requests.post(
                    f"{self.base_url}/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=45
                )

                if resp.status_code == 200:
                    data = resp.json()
                    content = data["choices"][0]["message"]["content"]
                    logger.info(f"Successfully generated answer using '{model}'.")
                    return content, model
                else:
                    err_msg = f"Model '{model}' returned HTTP {resp.status_code}: {resp.text}"
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
            "4. Keep your answer concise, professional, and well-structured with bullet points."
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
