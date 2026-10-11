"""Cross-Encoder Reranking Module using cross-encoder/ms-marco-MiniLM-L-6-v2.
Provides reranking over fused candidate pools with graceful fallback.
"""

from dataclasses import dataclass
from typing import List, Optional
from sentence_transformers import CrossEncoder

from app.core.config import settings
from app.core.logging import logger
from app.retrieval.fusion import FusedCandidate


@dataclass
class RerankedCandidate:
    chunk_id: str
    final_rank: int
    rerank_score: Optional[float]
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


class CrossEncoderReranker:
    """Reranks fused candidate chunks using a pretrained cross-encoder."""

    _models_cache: dict = {}

    def __init__(self, model_name: Optional[str] = None):
        self.model_name = model_name or settings.CROSS_ENCODER_MODEL_NAME

    @classmethod
    def _get_model(cls, model_name: str) -> Optional[CrossEncoder]:
        if model_name in cls._models_cache:
            return cls._models_cache[model_name]

        try:
            logger.info(f"Loading CrossEncoder model '{model_name}' on CPU...")
            model = CrossEncoder(model_name, device="cpu")
            cls._models_cache[model_name] = model
            return model
        except Exception as e:
            logger.warning(f"Failed to load CrossEncoder '{model_name}': {e}. Reranking fallback active.")
            cls._models_cache[model_name] = None
            return None

    def rerank(
        self,
        query: str,
        candidates: List[FusedCandidate],
        top_k: Optional[int] = None
    ) -> List[RerankedCandidate]:
        """Reranks candidates by cross-encoder score. Falls back to fused order if model is unavailable."""
        if not candidates:
            return []

        model = self._get_model(self.model_name)

        # Fallback if model could not be loaded
        if model is None:
            logger.info("Reranker unavailable; falling back to RRF ranking.")
            return [
                RerankedCandidate(
                    chunk_id=c.chunk_id,
                    final_rank=idx,
                    rerank_score=None,
                    rrf_score=c.rrf_score,
                    rrf_rank=getattr(c, "rrf_rank", getattr(c, "final_rank", idx)),
                    bm25_score=c.bm25_score,
                    bm25_rank=c.bm25_rank,
                    dense_similarity=c.dense_similarity,
                    dense_rank=c.dense_rank,
                    review_id=c.review_id,
                    business_id=c.business_id,
                    restaurant_name=c.restaurant_name,
                    city=c.city,
                    state=c.state,
                    categories=c.categories,
                    price_range=c.price_range,
                    is_open=c.is_open,
                    stars=c.stars,
                    business_stars=c.business_stars,
                    date=c.date,
                    year_month=c.year_month,
                    credibility=c.credibility,
                    text_masked=c.text_masked,
                    aspects=c.aspects,
                    n_tokens=c.n_tokens
                )
                for idx, c in enumerate(candidates[:top_k] if top_k else candidates, start=1)
            ]

        # Form query-document pairs
        pairs = [(query, c.text_masked) for c in candidates]
        scores = model.predict(pairs)

        # Sort candidates descending by reranker score
        scored_candidates = list(zip(candidates, scores))
        scored_candidates.sort(key=lambda x: x[1], reverse=True)

        if top_k is not None:
            scored_candidates = scored_candidates[:top_k]

        reranked_list: List[RerankedCandidate] = []
        for rank, (c, score) in enumerate(scored_candidates, start=1):
            reranked_list.append(RerankedCandidate(
                chunk_id=c.chunk_id,
                final_rank=rank,
                rerank_score=float(score),
                rrf_score=c.rrf_score,
                rrf_rank=getattr(c, "rrf_rank", getattr(c, "final_rank", rank)),
                bm25_score=c.bm25_score,
                bm25_rank=c.bm25_rank,
                dense_similarity=c.dense_similarity,
                dense_rank=c.dense_rank,
                review_id=c.review_id,
                business_id=c.business_id,
                restaurant_name=c.restaurant_name,
                city=c.city,
                state=c.state,
                categories=c.categories,
                price_range=c.price_range,
                is_open=c.is_open,
                stars=c.stars,
                business_stars=c.business_stars,
                date=c.date,
                year_month=c.year_month,
                credibility=c.credibility,
                text_masked=c.text_masked,
                aspects=c.aspects,
                n_tokens=c.n_tokens
            ))

        return reranked_list
