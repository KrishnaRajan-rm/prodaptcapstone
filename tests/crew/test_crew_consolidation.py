"""Tests for Phase 4 CrewAI Multi-Agent Drafting and Validation Layer."""

import pytest
from app.crew.schemas import CrewConsolidatedReport, CrewValidationResult
from app.crew.consolidation import (
    build_compact_evidence_summary,
    deterministic_synthesize_report,
    run_crewai_draft_and_validate,
)
from app.orchestration.schemas import OrchestrationRequest
from app.orchestration.graph import run_orchestration_workflow
from tests.orchestration.test_langgraph_workflow import MockAsyncA2AClient


def test_build_compact_evidence_summary():
    evidence = [
        {"chunk_id": "c1", "text_masked": "Very slow service [PERSON_1] waited 40 mins."},
        {"chunk_id": "c2", "text_masked": "Lukewarm pasta served."},
    ]
    classification = {
        "categories": [{"category": "Speed of Service", "aspects": [{"aspect": "Wait Time"}]}]
    }
    trends = {
        "trends": [{"category": "Speed of Service", "direction": "increasing", "slope": 1.2, "is_forecast": False}]
    }
    root_cause = {
        "hypotheses": [
            {"hypothesized_cause": "Shortage of kitchen staff", "distinction_type": "chronic", "supporting_source_ids": ["c1"]}
        ]
    }
    action_plan = {
        "ranked_actions": [
            {"recommended_action": "Add expediter", "problem_addressed": "Wait Time", "owner_role": "kitchen_manager", "kpi": "Table turnover"}
        ]
    }

    summary = build_compact_evidence_summary(evidence, classification, trends, root_cause, action_plan)
    assert len(summary["evidence_snippets"]) == 2
    assert "Speed of Service" in summary["categories_summary"][0]
    assert "direction=increasing" in summary["trends_summary"][0]
    assert "Shortage of kitchen staff" in summary["root_causes_summary"][0]
    assert "Add expediter" in summary["actions_summary"][0]


def test_deterministic_synthesize_report():
    report, val = deterministic_synthesize_report(
        query="Why are customers waiting so long?",
        business_id="biz_target",
        evidence=[{"chunk_id": "c1", "text": "Delay in kitchen"}],
        classification={"categories": [{"category": "Speed of Service"}]},
        trends={"trends": [{"category": "Speed of Service", "direction": "increasing"}]},
        root_cause={"hypotheses": [{"hypothesized_cause": "Line bottleneck", "distinction_type": "acute", "supporting_source_ids": ["c1"]}]},
        action_plan={"ranked_actions": [{"recommended_action": "Reorganize line", "owner_role": "manager", "kpi": "Wait time"}]},
    )

    assert isinstance(report, CrewConsolidatedReport)
    assert "biz_target" in report.executive_summary or "Why are customers" in report.executive_summary
    assert len(report.key_findings) > 0
    assert len(report.prioritized_recommendations) > 0
    assert report.prioritized_recommendations[0].owner_role == "manager"
    assert "January 2022" in report.data_limitations_and_cutoffs[0]

    assert isinstance(val, CrewValidationResult)
    assert val.is_valid is True
    assert val.severity == "none"
    assert val.staleness_disclosed is True
    assert val.pii_leakage_detected is False


def test_run_crewai_draft_and_validate_mock():
    report, val, status = run_crewai_draft_and_validate(
        query="Analyze food quality issues",
        business_id="biz_food",
        evidence=[{"chunk_id": "c_food_1", "text": "Cold soup"}],
        classification={"categories": [{"category": "Food Quality"}]},
        trends=None,
        root_cause=None,
        action_plan=None,
    )
    assert status in ("success", "fallback")
    assert isinstance(report, CrewConsolidatedReport)
    assert isinstance(val, CrewValidationResult)
    assert len(report.key_findings) > 0


def test_fallback_action_does_not_repeat_raw_query():
    report, _, _ = run_crewai_draft_and_validate(
        query="Why is service slow?",
        business_id="biz_test",
        evidence=[{"chunk_id": "c1", "text": "slow service"}],
        classification=None,
        trends=None,
        root_cause=None,
        action_plan=None,
    )

    first = report.prioritized_recommendations[0]
    assert "Why is service slow?" not in first.problem_addressed
    assert "service" in first.problem_addressed.lower() or "operational" in first.problem_addressed.lower()


@pytest.mark.asyncio
async def test_workflow_executes_with_crewai_consolidation():
    """Verify that full LangGraph orchestration includes CrewAI drafting and validation."""
    mock_client = MockAsyncA2AClient()
    req = OrchestrationRequest(
        request_id="req-crew-test",
        business_id="biz_test",
        query="Investigate chronic dining delays and recommend operational fixes",
        viewer_role="manager",
    )
    res = await run_orchestration_workflow(request=req, client=mock_client)

    assert res.workflow_status in ("success", "partial_success")
    assert res.crew_report is not None
    assert res.executive_summary is not None
    assert "executive_summary" in res.crew_report
    assert res.crew_validation is not None
    assert res.crew_validation.get("is_valid") is True
    assert res.provenance.get("crew_ai") is not None
