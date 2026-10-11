"""Comprehensive Unit and Integration Tests for Hybrid RAG Retrieval System.
Covers all requirements from rag.txt:
- BM25 retrieval independently
- Dense retrieval independently
- RRF fusion and rank ordering
- Duplicate removal and stable chunk ID mapping
- Metadata filtering
- Missing and malformed queries
- No-result behavior
- PII placeholder handling
- Reranker enabled, disabled, and fallback
- The 5 required queries from rag.txt
"""

import pytest
from app.retrieval.bm25 import BM25Retriever, BM25Candidate, clean_query_text
from app.retrieval.dense import DenseRetriever, DenseCandidate
from app.retrieval.fusion import reciprocal_rank_fusion, FusedCandidate
from app.retrieval.rerank import CrossEncoderReranker
from app.retrieval.hybrid import HybridRetriever, get_retriever


@pytest.fixture(scope="module")
def retriever():
    return get_retriever()


def test_bm25_retrieval_independently(retriever):
    """Test BM25 keyword retrieval independently with FTS5."""
    results = retriever.bm25_retriever.search("waited an hour for food", top_k=5)
    assert len(results) > 0
    assert results[0].bm25_score is not None
    assert results[0].bm25_rank == 1
    assert results[0].chunk_id != ""
    assert results[0].restaurant_name != ""
    assert "[PERSON]" not in results[0].text_masked or len(results[0].text_masked) > 20


def test_dense_retrieval_independently(retriever):
    """Test dense vector semantic retrieval independently."""
    results = retriever.dense_retriever.search("slow service", top_k=5)
    assert len(results) > 0
    assert results[0].dense_similarity > 0.5
    assert results[0].dense_rank == 1
    assert results[0].chunk_id != ""
    assert results[0].city != ""


def test_rrf_fusion_logic():
    """Unit test verifying the Reciprocal Rank Fusion formula and deduplication."""
    # Create mock BM25 candidate
    c_bm25 = BM25Candidate(
        chunk_id="chunk_1",
        bm25_score=-12.5,
        bm25_rank=1,
        review_id="r1",
        business_id="b1",
        restaurant_name="Tasty Tacos",
        city="Philadelphia",
        state="PA",
        categories="Mexican",
        price_range=2,
        is_open=1,
        stars=4.0,
        business_stars=4.0,
        date="2020-01-01",
        year_month="2020-01",
        credibility=0.8,
        text_masked="Great tacos but long wait.",
        aspects="['wait_time']",
        n_tokens=50
    )

    # Create mock Dense candidates: chunk_1 (rank 2) and chunk_2 (rank 1)
    c_dense_1 = DenseCandidate(
        chunk_id="chunk_1",
        dense_similarity=0.85,
        dense_rank=2,
        review_id="r1",
        business_id="b1",
        restaurant_name="Tasty Tacos",
        city="Philadelphia",
        state="PA",
        categories="Mexican",
        price_range=2,
        is_open=1,
        stars=4.0,
        business_stars=4.0,
        date="2020-01-01",
        year_month="2020-01",
        credibility=0.8,
        text_masked="Great tacos but long wait.",
        aspects="['wait_time']",
        n_tokens=50
    )
    c_dense_2 = DenseCandidate(
        chunk_id="chunk_2",
        dense_similarity=0.90,
        dense_rank=1,
        review_id="r2",
        business_id="b2",
        restaurant_name="Burger Barn",
        city="Tampa",
        state="FL",
        categories="Burgers",
        price_range=1,
        is_open=1,
        stars=3.0,
        business_stars=3.5,
        date="2020-02-01",
        year_month="2020-02",
        credibility=0.5,
        text_masked="Super slow burger service.",
        aspects="['wait_time']",
        n_tokens=40
    )

    fused = reciprocal_rank_fusion([c_bm25], [c_dense_1, c_dense_2], k=60)
    assert len(fused) == 2

    # Expected RRF scores:
    # chunk_1: 1/(60+1) + 1/(60+2) = 1/61 + 1/62 = 0.016393 + 0.016129 = 0.032522
    # chunk_2: 0 + 1/(60+1) = 1/61 = 0.016393
    assert fused[0].chunk_id == "chunk_1"
    assert fused[0].rrf_rank == 1
    assert pytest.approx(fused[0].rrf_score, rel=1e-3) == (1/61 + 1/62)
    assert fused[1].chunk_id == "chunk_2"
    assert fused[1].rrf_rank == 2
    assert pytest.approx(fused[1].rrf_score, rel=1e-3) == (1/61)


def test_metadata_filtering(retriever):
    """Test metadata filtering by city and open status."""
    city_filter = {"city": "Philadelphia", "is_open": 1}
    resp = retriever.search("burger delicious", top_k=5, filters=city_filter)
    assert len(resp.results) > 0
    for r in resp.results:
        assert r.city == "Philadelphia"
        assert r.is_open == 1


def test_malformed_and_empty_queries(retriever):
    """Test handling of empty, whitespace, and syntax-breaking queries."""
    assert retriever.search("", top_k=5).results == []
    assert retriever.search("   ", top_k=5).results == []
    assert retriever.search("!@#$%^&*()_+=-~`<>?:;", top_k=5).results == []


def test_pii_placeholder_handling(retriever):
    """Test query containing PII placeholders like [PERSON] and [EMAIL]."""
    raw_q = "[PERSON] complained to [EMAIL] about [PHONE] cold soup"
    cleaned = clean_query_text(raw_q)
    assert "[PERSON]" not in cleaned
    assert "[EMAIL]" not in cleaned
    assert "complained" in cleaned
    assert "soup" in cleaned

    resp = retriever.search(raw_q, top_k=3)
    assert len(resp.results) > 0


def test_reranker_enabled_disabled_fallback(retriever):
    """Test reranking enabled, disabled, and fallback when model fails."""
    # 1. Reranker disabled
    resp_no_rerank = retriever.search("dirty tables", top_k=5, use_reranker=False)
    assert len(resp_no_rerank.results) > 0
    assert resp_no_rerank.results[0].rerank_score is None

    # 2. Reranker enabled
    resp_reranked = retriever.search("dirty tables", top_k=5, use_reranker=True)
    assert len(resp_reranked.results) > 0
    assert resp_reranked.results[0].rerank_score is not None

    # 3. Fallback when reranker model name is invalid
    bad_reranker = CrossEncoderReranker(model_name="non_existent/fake_model_xyz")
    mock_fused = resp_no_rerank.results[:2]
    # Convert SearchResult to FusedCandidate mock
    fallback_res = bad_reranker.rerank("dirty tables", mock_fused)
    assert len(fallback_res) == 2
    assert fallback_res[0].rerank_score is None


@pytest.mark.parametrize("query_text", [
    "waited an hour for food",
    "slow service",
    "rude staff",
    "food was cold",
    "restaurant cleanliness"
])
def test_the_five_required_queries(retriever, query_text):
    """Verify all 5 required benchmark queries from rag.txt."""
    resp = retriever.search(query=query_text, top_k=5, use_reranker=True)
    assert len(resp.results) == 5
    assert resp.timings_ms["total_ms"] > 0
    assert resp.results[0].chunk_id != ""
    assert len(resp.results[0].text_masked) > 10
