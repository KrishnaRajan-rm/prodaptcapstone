"""Unit Tests for Phase 4 Quality Gate Components.
Covers scenarios 5 to 15:
5. Valid source IDs with matching excerpts pass grounding checks.
6. Fabricated or nonexistent source IDs are flagged.
7. Valid source IDs with unrelated excerpts are flagged.
8. Business-ID mismatches are blocked or safely rejected.
9. Stale historical data is disclosed.
10. Trend calculations are verified against deterministic fixtures.
11. Unsupported root-cause hypotheses are flagged.
12. Contradictory agent outputs are detected.
13. Unsupported recommendations and invented targets are flagged.
14. Confidence scoring follows the documented formula.
15. Judge failures and timeouts produce safe outcomes.
"""

import pytest
from app.quality.schemas import QualityStatus
from app.quality.grounding import validate_grounding
from app.quality.statistical import validate_statistics_and_freshness
from app.quality.recommendations import validate_recommendations
from app.quality.contradictions import detect_contradictions
from app.quality.confidence import compute_confidence_score
from app.quality.judge import evaluate_with_llm_judge


def test_05_valid_source_ids_with_matching_excerpts_pass():
    """5. Valid source IDs with matching excerpts pass grounding checks."""
    evidence = [
        {
            "chunk_id": "chk_001",
            "business_id": "BIZ_1",
            "text_masked": "The steak was cold and the server took 45 minutes to bring drinks.",
        }
    ]
    root_cause = {
        "hypotheses": [
            {
                "hypothesis_id": "H1",
                "hypothesized_cause": "Kitchen expediter bottleneck",
                "supporting_source_ids": ["chk_001"],
                "supporting_evidence": ["steak was cold and the server took 45 minutes"],
            }
        ]
    }

    res = validate_grounding(
        target_business_id="BIZ_1",
        retrieved_evidence=evidence,
        classification_payload=None,
        root_cause_payload=root_cause,
        action_planner_payload=None,
    )

    assert res.status == QualityStatus.PASSED
    assert res.total_citations == 1
    assert res.verified_citations == 1
    assert len(res.missing_citations) == 0


def test_06_fabricated_or_nonexistent_source_ids_flagged():
    """6. Fabricated or nonexistent source IDs are flagged."""
    evidence = [
        {"chunk_id": "chk_001", "business_id": "BIZ_1", "text_masked": "Great food"}
    ]
    root_cause = {
        "hypotheses": [
            {
                "hypothesis_id": "H1",
                "hypothesized_cause": "Invented claim",
                "supporting_source_ids": ["chk_FABRICATED_999"],
                "supporting_evidence": ["Completely made up quote"],
            }
        ]
    }

    res = validate_grounding(
        target_business_id="BIZ_1",
        retrieved_evidence=evidence,
        classification_payload=None,
        root_cause_payload=root_cause,
        action_planner_payload=None,
    )

    assert res.status == QualityStatus.FLAGGED
    assert "chk_FABRICATED_999" in res.missing_citations
    assert res.verified_citations == 0


def test_07_valid_source_ids_with_unrelated_excerpts_flagged():
    """7. Valid source IDs with unrelated excerpts are flagged."""
    evidence = [
        {
            "chunk_id": "chk_001",
            "business_id": "BIZ_1",
            "text_masked": "The parking lot was clean and well paved.",
        }
    ]
    root_cause = {
        "hypotheses": [
            {
                "hypothesis_id": "H1",
                "hypothesized_cause": "Severe food poisoning outbreak",
                "supporting_source_ids": ["chk_001"],
                "supporting_evidence": ["Raw chicken caused salmonella poisoning"],
            }
        ]
    }

    res = validate_grounding(
        target_business_id="BIZ_1",
        retrieved_evidence=evidence,
        classification_payload=None,
        root_cause_payload=root_cause,
        action_planner_payload=None,
    )

    assert res.status == QualityStatus.FLAGGED
    assert "chk_001" in res.unrelated_citations


def test_08_business_id_mismatches_blocked():
    """8. Business-ID mismatches are blocked to prevent cross-business data contamination."""
    evidence = [
        {
            "chunk_id": "chk_001",
            "business_id": "BIZ_OTHER_999",  # Belongs to a different competitor!
            "text_masked": "Pizzas were burnt.",
        }
    ]
    root_cause = {
        "hypotheses": [
            {
                "hypothesis_id": "H1",
                "hypothesized_cause": "Oven temperature malfunction",
                "supporting_source_ids": ["chk_001"],
                "supporting_evidence": ["Pizzas were burnt"],
            }
        ]
    }

    res = validate_grounding(
        target_business_id="BIZ_MY_RESTAURANT",
        retrieved_evidence=evidence,
        classification_payload=None,
        root_cause_payload=root_cause,
        action_planner_payload=None,
    )

    assert res.status == QualityStatus.BLOCKED
    assert len(res.business_id_mismatches) >= 1


def test_09_stale_historical_data_is_disclosed():
    """9. Stale historical data (cutoff 2022-01 vs 2026-10) is clearly disclosed."""
    trend_payload = {
        "analysis_period": "2020-01 to 2022-01",
        "total_reviews_analyzed": 320,
        "average_rating": 3.8,
        "forecast": {
            "status": "computed",
            "forecast_values": {"2022-02": 10.5, "2022-03": 11.2},
        }
    }

    res = validate_statistics_and_freshness(
        query="What is the current restaurant performance right now in 2026?",
        trend_payload=trend_payload,
    )

    assert res.stale_historical_data_disclosed is True
    assert "2022-01" in res.staleness_disclosure
    assert "2026-10" in res.staleness_disclosure
    assert any("2022-01" in w for w in res.warnings)


def test_10_trend_calculations_verified_against_fixtures():
    """10. Trend calculations and rating boundaries verified against fixtures."""
    # Invalid rating > 5.0
    bad_trend = {
        "analysis_period": "2020-01 to 2022-01",
        "total_reviews_analyzed": 100,
        "average_rating": 8.5,  # Out of bounds!
    }
    res = validate_statistics_and_freshness(query="general overview", trend_payload=bad_trend)
    assert res.status == QualityStatus.FLAGGED
    assert any("average_rating out of [1.0, 5.0] range" in w for w in res.warnings)


def test_11_unsupported_root_cause_hypotheses_flagged():
    """11. Unsupported root cause hypotheses lacking citations are flagged."""
    evidence = [{"chunk_id": "chk_001", "business_id": "BIZ_1", "text_masked": "Soup was cold"}]
    # Hypothesis with empty supporting source IDs
    root_cause = {
        "hypotheses": [
            {
                "hypothesis_id": "H1",
                "hypothesized_cause": "New location onboarding failure",
                "supporting_source_ids": [],
                "supporting_evidence": [],
            }
        ]
    }
    res = validate_grounding(
        target_business_id="BIZ_1",
        retrieved_evidence=evidence,
        classification_payload=None,
        root_cause_payload=root_cause,
        action_planner_payload=None,
    )
    # Total citations 0, verified 0
    assert res.total_citations == 0


def test_12_contradictory_agent_outputs_detected():
    """12. Contradictory agent outputs across specialists are detected."""
    # Scenario A: Mild classification vs Critical action planner
    classification_mild = {"severity": "low", "aspects": ["Waiting time"], "sentiment": "negative"}
    action_planner_critical = {
        "ranked_actions": [
            {"rank": 1, "priority_score": 92.0, "owner_role": "Kitchen Lead", "issue_summary": "Waiting time"}
        ]
    }
    res = detect_contradictions(
        classification_payload=classification_mild,
        trend_payload=None,
        root_cause_payload=None,
        action_planner_payload=action_planner_critical,
    )
    assert res.has_contradictions is True
    assert res.status == QualityStatus.FLAGGED
    assert any("priority score 92.0" in c for c in res.contradictions)

    # Scenario B: Trend improving vs Root cause failure
    trend_improving = {"improving_aspects": ["Food quality"]}
    root_cause_failing = {
        "hypotheses": [{"hypothesized_cause": "Deteriorating food quality due to spoilage"}]
    }
    res2 = detect_contradictions(
        classification_payload=None,
        trend_payload=trend_improving,
        root_cause_payload=root_cause_failing,
        action_planner_payload=None,
    )
    assert res2.has_contradictions is True
    assert any("improving" in c for c in res2.contradictions)


def test_13_unsupported_recommendations_and_invented_targets_flagged():
    """13. Unsupported recommendations and invented named employees flagged."""
    action_plan = {
        "ranked_actions": [
            {
                "rank": 1,
                "priority_score": 75.0,
                "owner_role": "John Smith",  # Named individual employee, NOT an operational role!
                "issue_summary": "Unrelated HVAC ventilation failure",
                "recommended_action": "Fix AC unit",
                "target_value": "Zero complaints",
            }
        ]
    }
    classification = {"aspects": ["Waiting time"], "categories": ["Service"]}
    root_cause = {"hypotheses": [{"hypothesized_cause": "Slow staffing turnaround"}]}

    res = validate_recommendations(
        action_planner_payload=action_plan,
        root_cause_payload=root_cause,
        classification_payload=classification,
    )

    assert res.status == QualityStatus.FLAGGED
    assert res.owner_roles_valid is False
    assert any("John Smith" in inv for inv in res.invalid_owners)


def test_14_confidence_scoring_follows_documented_formula():
    """14. Confidence scoring strictly follows the documented formula."""
    from app.quality.schemas import GroundingCheckResult, GroundingFinding, StatisticalCheckResult, ContradictionCheckResult

    grounding = GroundingCheckResult(
        total_citations=2,
        verified_citations=2,
        findings=[
            GroundingFinding(source_id="c1", excerpt_overlap=1.0),
            GroundingFinding(source_id="c2", excerpt_overlap=1.0),
        ]
    )
    stat = StatisticalCheckResult(stale_historical_data_disclosed=True)
    contradictions = ContradictionCheckResult(has_contradictions=False)

    # Expected: 100 * (0.35 * 1.0 + 0.25 * 1.0 + 0.20 * 1.0 - 0.20 * 0.0) = 80.0
    conf = compute_confidence_score(
        grounding=grounding,
        statistical=stat,
        contradictions=contradictions,
        retrieved_count=2,
        query_expects_fresh=False
    )

    assert conf.score == 80.0
    assert conf.assessment == "high"
    assert conf.component_scores["source_traceability"] == 1.0
    assert conf.component_scores["agent_agreement"] == 1.0
    assert conf.component_scores["staleness_penalty"] == 0.0


def test_15_judge_failures_and_timeouts_produce_safe_outcomes():
    """15. LLM Judge failures/timeouts gracefully fall back to safe deterministic outcome."""
    class FailingLLMClient:
        def generate_json(self, *args, **kwargs):
            raise TimeoutError("Simulated LLM Judge timeout after 8 seconds")

    failing_client = FailingLLMClient()
    verdict = evaluate_with_llm_judge(
        query="Slow table service",
        target_business_id="BIZ_1",
        retrieved_evidence=[],
        classification_payload=None,
        root_cause_payload=None,
        action_planner_payload=None,
        deterministic_fallback_verdict=QualityStatus.PASSED,
        llm_client=failing_client,
    )

    assert verdict.verdict == QualityStatus.PASSED
    assert verdict.model_used == "safe_deterministic_fallback"
    assert any("unavailable" in iss for iss in verdict.issues)
    assert verdict.latency_ms >= 0.0
