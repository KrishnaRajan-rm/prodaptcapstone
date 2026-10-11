"""Integration tests for Phase 4 LangGraph Workflow with Quality Gate and Role-based Decoding.
Covers scenarios 16, 19, 20:
16. Passed, flagged, and blocked routes behave correctly in the LangGraph workflow.
19. Existing LangGraph workflow continues to pass with new gates integrated.
20. Concurrency telemetry accurately matches actual overlapping execution between Classification and Trends.
"""

import asyncio
import time
from types import SimpleNamespace
from typing import Any, Dict, List, Optional
import pytest

from app.api.routes import build_execution_trace
from app.orchestration.schemas import OrchestrationRequest
from app.orchestration.graph import run_orchestration_workflow
from app.orchestration.a2a_async_client import AsyncA2AAgentClient
from app.agents.schemas import AgentResponseEnvelope, EvidenceRef


class QualityMockA2AClient(AsyncA2AAgentClient):
    """Mock A2A client for testing LangGraph Quality Gate execution routes."""

    def __init__(
        self,
        classification_delay: float = 0.05,
        trend_delay: float = 0.05,
        cited_source_id: str = "chk_valid_1",
        owner_role: str = "Kitchen Lead",
    ):
        super().__init__()
        self.classification_delay = classification_delay
        self.trend_delay = trend_delay
        self.cited_source_id = cited_source_id
        self.owner_role = owner_role

    async def classify_complaint(self, text: str, review_id: str = "rev_1", business_id: Optional[str] = None) -> AgentResponseEnvelope:
        await asyncio.sleep(self.classification_delay)
        return AgentResponseEnvelope(
            agent_id="classification_agent",
            status="success",
            payload={
                "request_id": review_id,
                "categories": ["Service", "Food quality"],
                "aspects": ["Waiting time", "Food temperature"],
                "sentiment": "negative",
                "severity": "medium",
                "severity_reason": "Extended delay",
                "evidence_snippets": ["Waited 45 minutes"],
            },
        )

    async def analyze_trends(self, business_id: Optional[str] = None, city: Optional[str] = None, horizon: int = 3) -> AgentResponseEnvelope:
        await asyncio.sleep(self.trend_delay)
        return AgentResponseEnvelope(
            agent_id="trend_agent",
            status="success",
            payload={
                "business_id": business_id,
                "analysis_period": "2020-01 to 2022-01",
                "total_reviews_analyzed": 250,
                "average_rating": 3.5,
                "worsening_aspects": ["Waiting time"],
                "improving_aspects": ["Cleanliness"],
                "forecast": {
                    "status": "computed",
                    "forecast_values": {"2022-02": 8.5},
                },
                "satisfaction_prediction": {
                    "status": "computed",
                    "predicted_score": 3.4,
                    "predicted_label": "Moderate",
                },
            },
        )

    async def analyze_root_cause(
        self,
        problem_statement: str,
        business_id: Optional[str] = None,
        top_k: int = 5,
        classification_findings: Optional[Dict[str, Any]] = None,
        trend_findings: Optional[Dict[str, Any]] = None,
    ) -> AgentResponseEnvelope:
        return AgentResponseEnvelope(
            agent_id="root_cause_agent",
            status="success",
            payload={
                "problem_statement": problem_statement,
                "business_id": business_id,
                "retrieved_chunk_count": 1,
                "hypotheses": [
                    {
                        "hypothesis_id": "H1",
                        "hypothesized_cause": "Line cook bottleneck during peak shift",
                        "distinction_type": "inferred_explanation",
                        "confidence": "high",
                        "supporting_source_ids": [self.cited_source_id],
                        "supporting_evidence": ["Waited 45 minutes for order"],
                        "additional_info_needed": "Kitchen POS ticket timestamps",
                    }
                ],
            },
        )

    async def generate_action_plan(
        self,
        business_id: Optional[str] = None,
        problem_statement: Optional[str] = None,
        classification_findings: Optional[Dict[str, Any]] = None,
        trend_findings: Optional[Dict[str, Any]] = None,
        root_cause_findings: Optional[Dict[str, Any]] = None,
    ) -> AgentResponseEnvelope:
        return AgentResponseEnvelope(
            agent_id="action_planner_agent",
            status="success",
            payload={
                "business_id": business_id,
                "scoring_methodology": "Standard composite ranking",
                "total_issues_evaluated": 1,
                "ranked_actions": [
                    {
                        "rank": 1,
                        "priority_score": 75.0,
                        "issue_summary": "Waiting time line cook bottleneck",
                        "recommended_action": "Add dedicated floater cook on weekend evenings",
                        "owner_role": self.owner_role,
                        "measurable_kpi": "Table turn time",
                        "target_value": "< 20 min (proposed)",
                    }
                ],
            },
        )


MOCK_EVIDENCE_FIXTURE = [
    {
        "chunk_id": "chk_valid_1",
        "business_id": "BIZ_PASSED_100",
        "text_masked": "Waited 45 minutes for order from server [PERSON_1]",
    }
]


def test_build_execution_trace_uses_real_evidence_without_placeholder_pii():
    """Execution trace should reflect actual retrieved evidence rather than mock customer placeholders."""
    res = SimpleNamespace(
        business_id="BIZ_TRACE_001",
        retrieved_evidence=[
            {
                "chunk_id": "chk_real_1",
                "review_id": "rev_real_1",
                "text_masked": "Waited 45 minutes for our order and the soup arrived cold.",
                "text": "Waited 45 minutes for our order and the soup arrived cold.",
            },
            {
                "chunk_id": "chk_real_2",
                "review_id": "rev_real_2",
                "text_masked": "Bar service was disorganized during peak hours.",
                "text": "Bar service was disorganized during peak hours.",
            },
        ],
        classification={"categories": ["Service"], "aspects": ["Waiting time"]},
        trends={"worsening_aspects": ["Waiting time"]},
        root_cause={"hypotheses": [{"hypothesized_cause": "Peak-period staffing gap"}]},
        action_plan={"ranked_actions": [{"recommended_action": "Add a dedicated runner during lunch"}]},
        crew_report={"executive_summary": "Summary with evidence-backed actions."},
        crew_validation={"status": "passed"},
        quality_decision="passed",
        confidence_score=80.0,
        warnings=[],
        workflow_status="success",
        agent_timings={
            "classification": {"duration_ms": 145.0},
            "trends": {"duration_ms": 132.0},
            "root_cause": {"duration_ms": 118.0},
            "action_planner": {"duration_ms": 121.0},
        },
        stage_timings={
            "validate_request": {"started_at": 1.0, "finished_at": 1.01, "duration_ms": 10.0, "status": "completed"},
            "retrieve_context": {"started_at": 1.01, "finished_at": 1.02, "duration_ms": 10.0, "status": "completed"},
            "pii_gate_2": {"started_at": 1.02, "finished_at": 1.03, "duration_ms": 10.0, "status": "completed"},
        },
    )

    events = build_execution_trace(request_id="req_placeholder_guard", query="Slow wait times", res=res, t_start=0.0)

    trace_text = "\n".join(str(event.model_dump()) for event in events)
    assert "Sarah Miller" not in trace_text
    assert "smiller@gmail.com" not in trace_text
    assert "555-0144" not in trace_text
    assert "chk_real_1" in trace_text
    assert "Waited 45 minutes for our order and the soup arrived cold." in trace_text
    assert all(event.started_at is not None and event.duration_ms == 10.0 for event in events)
    assert len(events) == 3


def test_execution_trace_keeps_quality_verdict_separate_from_confidence():
    res = SimpleNamespace(
        business_id="BIZ_TRACE_002",
        retrieved_evidence=[],
        quality_decision="flagged",
        confidence_score=80.0,
        quality_gate={"decision": "flagged", "warnings": ["Recommendation is unsupported."]},
        stage_timings={
            "quality_gate": {"started_at": 2.0, "finished_at": 2.02, "duration_ms": 20.0, "status": "flagged"}
        },
    )

    events = build_execution_trace(request_id="req_quality_verdict", query="Check service issues", res=res, t_start=0.0)

    assert len(events) == 1
    assert events[0].status == "flagged"
    assert "80.0/80" in events[0].output_summary
    assert events[0].output_payload["decision"] == "flagged"


@pytest.mark.asyncio
async def test_16_workflow_passed_route_with_analyst_and_manager():
    """16. Workflow passed route: verifies role-based decoding (masked for analyst, decoded for manager)."""
    mock_client = QualityMockA2AClient(cited_source_id="chk_valid_1", owner_role="Kitchen Lead")

    # Request 1: Viewer is ANALYST -> content must remain masked
    req_analyst = OrchestrationRequest(
        query="Customer Alice at alice@mail.com waited 45 minutes",
        business_id="BIZ_PASSED_100",
        viewer_role="analyst",
        skip_llm_judge=True,
        filters={"preloaded_evidence": MOCK_EVIDENCE_FIXTURE},
    )

    resp_analyst = await run_orchestration_workflow(req_analyst, client=mock_client)

    assert resp_analyst.workflow_status in ("success", "partial_success")
    assert {"classification", "trends", "retrieve_context"}.issubset(resp_analyst.stage_timings)
    assert resp_analyst.stage_timings["decode_pii_role"]["status"] == "skipped"
    assert resp_analyst.pii_decoded is False
    assert "[EMAIL_1]" in resp_analyst.query or "alice@mail.com" not in resp_analyst.query

    # Request 2: Viewer is MANAGER -> content decoded at response boundary
    req_manager = OrchestrationRequest(
        query="Customer Bob at bob@mail.com waited 45 minutes",
        business_id="BIZ_PASSED_100",
        viewer_role="manager",
        skip_llm_judge=True,
        filters={"preloaded_evidence": MOCK_EVIDENCE_FIXTURE},
    )
    resp_manager = await run_orchestration_workflow(req_manager, client=mock_client)

    assert resp_manager.workflow_status in ("success", "partial_success")
    assert resp_manager.pii_decoded is True
    assert "bob@mail.com" in resp_manager.query


@pytest.mark.asyncio
async def test_16_workflow_flagged_route_for_invented_roles():
    """16b. Workflow flagged route: non-operational employee owner triggers flagged status."""
    mock_client = QualityMockA2AClient(cited_source_id="chk_valid_1", owner_role="David Miller")
    evidence = [
        {
            "chunk_id": "chk_valid_1",
            "business_id": "BIZ_FLAGGED_200",
            "text_masked": "Waited 45 minutes for order",
        }
    ]

    req = OrchestrationRequest(
        query="Cold steak and long wait",
        business_id="BIZ_FLAGGED_200",
        viewer_role="analyst",
        skip_llm_judge=True,
        filters={"preloaded_evidence": evidence},
    )
    resp = await run_orchestration_workflow(req, client=mock_client)

    assert resp.quality_decision == "flagged"
    assert any("David Miller" in str(w) or "Flagged" in str(w) for w in resp.warnings)


@pytest.mark.asyncio
async def test_16_workflow_blocked_route_for_business_id_mismatch():
    """16c. Workflow blocked route: cross-business attribution triggers safe restricted result."""
    # Chunk belongs to competitor BIZ_COMPETITOR_999, but request is for BIZ_TARGET_300
    competitor_evidence = [
        {
            "chunk_id": "chk_competitor_999",
            "business_id": "BIZ_COMPETITOR_999",
            "text_masked": "Pizzas were burnt and undercooked.",
        }
    ]
    mock_client = QualityMockA2AClient(cited_source_id="chk_competitor_999", owner_role="Kitchen Lead")

    req = OrchestrationRequest(
        query="Slow table service",
        business_id="BIZ_TARGET_300",
        viewer_role="analyst",
        skip_llm_judge=True,
        filters={"preloaded_evidence": competitor_evidence},
    )
    resp = await run_orchestration_workflow(req, client=mock_client)

    assert resp.quality_decision == "blocked"
    assert resp.workflow_status == "failed"
    assert resp.action_plan is None  # Sanitized by safe_blocked_node


@pytest.mark.asyncio
async def test_20_concurrency_telemetry_accurate_overlap():
    """20. Concurrency telemetry accurately matches actual overlapping execution."""
    mock_client = QualityMockA2AClient(classification_delay=0.08, trend_delay=0.08)
    evidence = [
        {
            "chunk_id": "chk_valid_1",
            "business_id": "BIZ_CONCURRENT",
            "text_masked": "Waited 45 minutes for order",
        }
    ]

    req = OrchestrationRequest(
        query="Parallel test query",
        business_id="BIZ_CONCURRENT",
        skip_llm_judge=True,
        filters={"preloaded_evidence": evidence},
    )
    resp = await run_orchestration_workflow(req, client=mock_client)

    c_time = resp.agent_timings.get("classification")
    t_time = resp.agent_timings.get("trends")

    assert c_time is not None
    assert t_time is not None
    assert c_time.status == "success"
    assert t_time.status == "success"

    # Verify concurrency: max(starts) < min(ends)
    is_overlapping = max(c_time.started_at, t_time.started_at) < min(c_time.ended_at, t_time.ended_at)
    assert is_overlapping is True, (
        f"Expected overlapping execution: Classification ({c_time.started_at:.3f} -> {c_time.ended_at:.3f}), "
        f"Trends ({t_time.started_at:.3f} -> {t_time.ended_at:.3f})"
    )
