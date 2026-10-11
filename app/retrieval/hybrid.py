"""Unified Hybrid RAG Retrieval Engine.
Orchestrates BM25 keyword retrieval, dense semantic retrieval,
Reciprocal Rank Fusion (RRF), and Cross-Encoder reranking with timing telemetry.
"""

import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.core.config import settings
from app.core.logging import logger
from app.retrieval.bm25 import BM25Retriever
from app.retrieval.dense import DenseRetriever
from app.retrieval.fusion import reciprocal_rank_fusion
from app.retrieval.rerank import CrossEncoderReranker


@dataclass
class SearchResult:
    chunk_id: str
    final_rank: int
    rrf_rank: int
    review_id: str
    business_id: str
    restaurant_name: str
    city: str
    state: str
    categories: Optional[str]
    price_range: Optional[int]
    is_open: int
    stars: float
    business_stars: float
    date: str
    year_month: str
    credibility: float
    text_masked: str
    aspects: str
    n_tokens: int
    bm25_score: Optional[float]
    bm25_rank: Optional[int]
    dense_similarity: Optional[float]
    dense_rank: Optional[int]
    rrf_score: float
    rerank_score: Optional[float]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "final_rank": self.final_rank,
            "rrf_rank": self.rrf_rank,
            "review_id": self.review_id,
            "business_id": self.business_id,
            "restaurant_name": self.restaurant_name,
            "city": self.city,
            "state": self.state,
            "categories": self.categories,
            "price_range": self.price_range,
            "is_open": self.is_open,
            "stars": self.stars,
            "business_stars": self.business_stars,
            "date": self.date,
            "year_month": self.year_month,
            "credibility": self.credibility,
            "text_masked": self.text_masked,
            "aspects": self.aspects,
            "n_tokens": self.n_tokens,
            "bm25_score": self.bm25_score,
            "bm25_rank": self.bm25_rank,
            "dense_similarity": self.dense_similarity,
            "dense_rank": self.dense_rank,
            "rrf_score": self.rrf_score,
            "rerank_score": self.rerank_score
        }


@dataclass
class HybridRetrievalResponse:
    query: str
    results: List[SearchResult]
    timings_ms: Dict[str, float] = field(default_factory=dict)
    filters_applied: Optional[Dict[str, Any]] = None
    reranker_used: bool = False

    def __iter__(self):
        return iter(self.results)

    def __len__(self):
        return len(self.results)

    def __getitem__(self, index):
        return self.results[index]


class HybridRetriever:
    """Unified hybrid retrieval pipeline combining BM25, Dense, RRF, and CrossEncoder reranking."""

    def __init__(
        self,
        db_path: Optional[Path] = None,
        vectors_path: Optional[Path] = None,
        ids_path: Optional[Path] = None,
        embedding_model_name: Optional[str] = None,
        reranker_model_name: Optional[str] = None
    ):
        self.bm25_retriever = BM25Retriever(db_path=db_path)
        self.dense_retriever = DenseRetriever(
            vectors_path=vectors_path,
            ids_path=ids_path,
            db_path=db_path,
            model_name=embedding_model_name
        )
        self.reranker = CrossEncoderReranker(model_name=reranker_model_name)

    def search(
        self,
        query: str,
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
        use_reranker: bool = True,
        bm25_depth: int = 100,
        dense_depth: int = 100,
        rerank_depth: int = 40,
        rrf_k: int = 60,
        bm25_weight: float = 1.0,
        dense_weight: float = 1.0
    ) -> HybridRetrievalResponse:
        """Executes full hybrid search pipeline:
        1. BM25 keyword retrieval (depth = bm25_depth)
        2. Dense vector retrieval (depth = dense_depth)
        3. Reciprocal Rank Fusion (RRF constant = rrf_k)
        4. Optional Cross-Encoder reranking (depth = rerank_depth) -> top_k
        """
        if not query or not re.search(r'[a-zA-Z0-9]', query):
            return HybridRetrievalResponse(
                query=query or "",
                results=[],
                timings_ms={"total_ms": 0.0},
                filters_applied=filters,
                reranker_used=False
            )

        timings: Dict[str, float] = {}
        t_total_start = time.perf_counter()

        # Step 1: BM25 retrieval
        t0 = time.perf_counter()
        bm25_candidates = self.bm25_retriever.search(query=query, top_k=bm25_depth, filters=filters)
        timings["bm25_ms"] = round((time.perf_counter() - t0) * 1000, 2)

        # Step 2: Dense retrieval
        t0 = time.perf_counter()
        dense_candidates = self.dense_retriever.search(query=query, top_k=dense_depth, filters=filters)
        timings["dense_ms"] = round((time.perf_counter() - t0) * 1000, 2)

        # Step 3: Reciprocal Rank Fusion
        t0 = time.perf_counter()
        fused_candidates = reciprocal_rank_fusion(
            bm25_candidates=bm25_candidates,
            dense_candidates=dense_candidates,
            k=rrf_k,
            bm25_weight=bm25_weight,
            dense_weight=dense_weight,
            top_k=max(rerank_depth, top_k) if use_reranker else top_k
        )
        timings["fusion_ms"] = round((time.perf_counter() - t0) * 1000, 2)

        # Step 4: Reranking
        t0 = time.perf_counter()
        reranked_candidates = []
        reranker_actually_used = False

        if use_reranker and fused_candidates:
            rerank_pool = fused_candidates[:rerank_depth]
            reranked_candidates = self.reranker.rerank(query=query, candidates=rerank_pool, top_k=top_k)
            # Check if actual rerank scores were produced
            if reranked_candidates and reranked_candidates[0].rerank_score is not None:
                reranker_actually_used = True
        else:
            # No reranker; convert fused candidates directly
            for rank, c in enumerate(fused_candidates[:top_k], start=1):
                reranked_candidates.append(
                    type("RerankedCandidateStub", (), {
                        "chunk_id": c.chunk_id,
                        "final_rank": rank,
                        "rerank_score": None,
                        "rrf_score": c.rrf_score,
                        "rrf_rank": c.rrf_rank,
                        "bm25_score": c.bm25_score,
                        "bm25_rank": c.bm25_rank,
                        "dense_similarity": c.dense_similarity,
                        "dense_rank": c.dense_rank,
                        "review_id": c.review_id,
                        "business_id": c.business_id,
                        "restaurant_name": c.restaurant_name,
                        "city": c.city,
                        "state": c.state,
                        "categories": c.categories,
                        "price_range": c.price_range,
                        "is_open": c.is_open,
                        "stars": c.stars,
                        "business_stars": c.business_stars,
                        "date": c.date,
                        "year_month": c.year_month,
                        "credibility": c.credibility,
                        "text_masked": c.text_masked,
                        "aspects": c.aspects,
                        "n_tokens": c.n_tokens
                    })()
                )

        timings["rerank_ms"] = round((time.perf_counter() - t0) * 1000, 2)
        timings["total_ms"] = round((time.perf_counter() - t_total_start) * 1000, 2)

        # Convert to final SearchResult objects
        results: List[SearchResult] = []
        for rc in reranked_candidates:
            results.append(SearchResult(
                chunk_id=rc.chunk_id,
                final_rank=rc.final_rank,
                rrf_rank=getattr(rc, "rrf_rank", rc.final_rank),
                review_id=rc.review_id,
                business_id=rc.business_id,
                restaurant_name=rc.restaurant_name,
                city=rc.city,
                state=rc.state,
                categories=rc.categories,
                price_range=rc.price_range,
                is_open=rc.is_open,
                stars=rc.stars,
                business_stars=rc.business_stars,
                date=rc.date,
                year_month=rc.year_month,
                credibility=rc.credibility,
                text_masked=rc.text_masked,
                aspects=rc.aspects,
                n_tokens=rc.n_tokens,
                bm25_score=rc.bm25_score,
                bm25_rank=rc.bm25_rank,
                dense_similarity=rc.dense_similarity,
                dense_rank=rc.dense_rank,
                rrf_score=rc.rrf_score,
                rerank_score=rc.rerank_score
            ))

        return HybridRetrievalResponse(
            query=query,
            results=results,
            timings_ms=timings,
            filters_applied=filters,
            reranker_used=reranker_actually_used
        )


# Global singleton instance for high-performance reuse
_global_hybrid_retriever: Optional[HybridRetriever] = None


def get_retriever() -> HybridRetriever:
    global _global_hybrid_retriever
    if _global_hybrid_retriever is None:
        _global_hybrid_retriever = HybridRetriever()
    return _global_hybrid_retriever


def hybrid_search(
    query: str,
    top_k: int = 10,
    filters: Optional[Dict[str, Any]] = None,
    use_reranker: bool = True
) -> HybridRetrievalResponse:
    """Convenience functional API for hybrid search."""
    retriever = get_retriever()
    return retriever.search(
        query=query,
        top_k=top_k,
        filters=filters,
        use_reranker=use_reranker
    )
