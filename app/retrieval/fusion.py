"""Reciprocal Rank Fusion (RRF) for Combining BM25 and Dense Retrieval.
Implements: RRF score = sum(w_m / (k + rank_m)) with deduplication by stable chunk_id.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional

from app.retrieval.bm25 import BM25Candidate
from app.retrieval.dense import DenseCandidate


@dataclass
class FusedCandidate:
    chunk_id: str
    rrf_score: float
    rrf_rank: int
    bm25_score: Optional[float]
    bm25_rank: Optional[int]
    dense_similarity: Optional[float]
    dense_rank: Optional[int]
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


def reciprocal_rank_fusion(
    bm25_candidates: List[BM25Candidate],
    dense_candidates: List[DenseCandidate],
    k: int = 60,
    bm25_weight: float = 1.0,
    dense_weight: float = 1.0,
    top_k: Optional[int] = None
) -> List[FusedCandidate]:
    """Combines BM25 and Dense candidates using weighted Reciprocal Rank Fusion (RRF).
    
    Formula:
        Score(d) = sum_{m in {bm25, dense}} w_m / (k + rank_m(d))
    using 1-based ranks.
    """
    scores: Dict[str, float] = {}
    meta_map: Dict[str, dict] = {}
    bm25_info: Dict[str, tuple] = {}
    dense_info: Dict[str, tuple] = {}

    # Process BM25 candidates
    for cand in bm25_candidates:
        cid = cand.chunk_id
        rrf_contrib = bm25_weight / (k + cand.bm25_rank)
        scores[cid] = scores.get(cid, 0.0) + rrf_contrib
        bm25_info[cid] = (cand.bm25_score, cand.bm25_rank)

        if cid not in meta_map:
            meta_map[cid] = {
                "review_id": cand.review_id,
                "business_id": cand.business_id,
                "restaurant_name": cand.restaurant_name,
                "city": cand.city,
                "state": cand.state,
                "categories": cand.categories,
                "price_range": cand.price_range,
                "is_open": cand.is_open,
                "stars": cand.stars,
                "business_stars": cand.business_stars,
                "date": cand.date,
                "year_month": cand.year_month,
                "credibility": cand.credibility,
                "text_masked": cand.text_masked,
                "aspects": cand.aspects,
                "n_tokens": cand.n_tokens
            }

    # Process Dense candidates
    for cand in dense_candidates:
        cid = cand.chunk_id
        rrf_contrib = dense_weight / (k + cand.dense_rank)
        scores[cid] = scores.get(cid, 0.0) + rrf_contrib
        dense_info[cid] = (cand.dense_similarity, cand.dense_rank)

        if cid not in meta_map:
            meta_map[cid] = {
                "review_id": cand.review_id,
                "business_id": cand.business_id,
                "restaurant_name": cand.restaurant_name,
                "city": cand.city,
                "state": cand.state,
                "categories": cand.categories,
                "price_range": cand.price_range,
                "is_open": cand.is_open,
                "stars": cand.stars,
                "business_stars": cand.business_stars,
                "date": cand.date,
                "year_month": cand.year_month,
                "credibility": cand.credibility,
                "text_masked": cand.text_masked,
                "aspects": cand.aspects,
                "n_tokens": cand.n_tokens
            }

    # Sort descending by fused score
    sorted_chunk_ids = sorted(scores.keys(), key=lambda cid: scores[cid], reverse=True)
    if top_k is not None:
        sorted_chunk_ids = sorted_chunk_ids[:top_k]

    fused_candidates: List[FusedCandidate] = []
    for rank, cid in enumerate(sorted_chunk_ids, start=1):
        meta = meta_map[cid]
        bm25_s, bm25_r = bm25_info.get(cid, (None, None))
        dense_s, dense_r = dense_info.get(cid, (None, None))

        fused_candidates.append(FusedCandidate(
            chunk_id=cid,
            rrf_score=float(scores[cid]),
            rrf_rank=rank,
            bm25_score=bm25_s,
            bm25_rank=bm25_r,
            dense_similarity=dense_s,
            dense_rank=dense_r,
            review_id=meta["review_id"],
            business_id=meta["business_id"],
            restaurant_name=meta["restaurant_name"],
            city=meta["city"],
            state=meta["state"],
            categories=meta["categories"],
            price_range=meta["price_range"],
            is_open=meta["is_open"],
            stars=meta["stars"],
            business_stars=meta["business_stars"],
            date=meta["date"],
            year_month=meta["year_month"],
            credibility=meta["credibility"],
            text_masked=meta["text_masked"],
            aspects=meta["aspects"],
            n_tokens=meta["n_tokens"]
        ))

    return fused_candidates
