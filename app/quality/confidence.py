"""Phase 4 Explainable Deterministic Confidence Scoring.
Computes a transparent composite score [0-100] based on evidence relevance, source traceability,
cross-agent agreement, and data freshness penalties.
"""

from typing import Any, Dict, List, Optional
from app.quality.schemas import (
    ConfidenceScoreResult,
    ContradictionCheckResult,
    GroundingCheckResult,
    StatisticalCheckResult,
)


def compute_confidence_score(
    grounding: GroundingCheckResult,
    statistical: StatisticalCheckResult,
    contradictions: ContradictionCheckResult,
    retrieved_count: int,
    query_expects_fresh: bool = False,
) -> ConfidenceScoreResult:
    """Computes explainable confidence score following the documented formula."""
    weights = {
        "evidence_relevance": 0.35,
        "source_traceability": 0.25,
        "agent_agreement": 0.20,
        "staleness_penalty": 0.20,
    }

    # 1. Evidence relevance [0.0 - 1.0]
    if retrieved_count <= 0:
        c_relevance = 0.0
    else:
        # Based on verified findings excerpt overlap
        if grounding.findings:
            overlaps = [f.excerpt_overlap for f in grounding.findings if f.exists]
            c_relevance = sum(overlaps) / len(overlaps) if overlaps else 0.85
        else:
            c_relevance = 1.0

    # 2. Source traceability [0.0 - 1.0]
    if grounding.total_citations > 0:
        c_traceability = grounding.verified_citations / float(grounding.total_citations)
    else:
        # If evidence was retrieved and used in the analysis pipeline
        c_traceability = 1.0 if retrieved_count > 0 else 0.0

    # 3. Agent agreement [0.0 - 1.0]
    if contradictions.has_contradictions:
        c_agreement = max(0.0, 1.0 - (0.35 * len(contradictions.contradictions)))
    else:
        c_agreement = 1.0

    # Dataset age is reported as a limitation, not treated as a failure of
    # evidence quality. Only missing disclosure reduces the score.
    c_staleness = 0.0 if statistical.stale_historical_data_disclosed else 0.5

    # Composite calculation
    raw_score = (
        weights["evidence_relevance"] * c_relevance +
        weights["source_traceability"] * c_traceability +
        weights["agent_agreement"] * c_agreement -
        weights["staleness_penalty"] * c_staleness
    )
    bounded_score = round(max(0.0, min(1.0, raw_score)) * 100.0, 1)
    max_possible = 80.0
    normalized_score = round(min(100.0, max(0.0, (bounded_score / max_possible) * 100.0)), 1)

    # Determine assessment category
    if retrieved_count == 0 and grounding.total_citations == 0:
        assessment = "unknown"
    elif bounded_score >= 65.0 or normalized_score >= 80.0:
        assessment = "high"
    elif bounded_score >= 45.0 or normalized_score >= 55.0:
        assessment = "medium"
    else:
        assessment = "low"

    explanation = (
        f"Score {bounded_score}/80.0 ({normalized_score}% Grounded Baseline • {assessment}) derived from: "
        f"evidence relevance ({c_relevance:.2f} * {weights['evidence_relevance']}), "
        f"source traceability ({c_traceability:.2f} * {weights['source_traceability']}), "
        f"agent agreement ({c_agreement:.2f} * {weights['agent_agreement']}), "
        f"staleness penalty ({c_staleness:.2f} * {weights['staleness_penalty']}). "
        f"Note: 80.0 represents the theoretical ceiling (100% grounded baseline) under the composite formula."
    )

    return ConfidenceScoreResult(
        score=bounded_score,
        max_possible_score=max_possible,
        normalized_score=normalized_score,
        assessment=assessment,
        weights=weights,
        component_scores={
            "evidence_relevance": round(c_relevance, 3),
            "source_traceability": round(c_traceability, 3),
            "agent_agreement": round(c_agreement, 3),
            "staleness_penalty": round(c_staleness, 3),
        },
        explanation=explanation,
    )
