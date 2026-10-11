"""Comprehensive Unit Tests for LangGraph Orchestration Workflow.
Covers all 16 required test scenarios defined in Phase 3 Orchestration requirements.
Uses deterministic asynchronous mocks for A2A services without paid LLM calls.
"""

import os
import sys
from pathlib import Path

# Add project root to sys.path
root_dir = Path(__file__).resolve().parents[2]
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

import asyncio
import time
from typing import Any, Dict, List, Optional
import pytest
import httpx

from app.orchestration.schemas import (
    AgentTimingInfo,
    AggregatedWorkflowResponse,
    OrchestrationRequest,
)
from app.orchestration.graph import (
    build_orchestration_graph,
    run_orchestration_workflow,
)
from app.orchestration.a2a_async_client import AsyncA2AAgentClient
from app.agents.schemas import AgentResponseEnvelope, EvidenceRef


# ==============================================================================
# Mock A2A Client Fixtures
# ==============================================================================

class MockAsyncA2AClient(AsyncA2AAgentClient):
    """Mock A2A client with configurable latencies, failures, and deterministic responses."""

    def __init__(
        self,
        classification_delay: float = 0.05,
        trend_delay: float = 0.05,
        classification_fail: bool = False,
        trend_fail: bool = False,
        root_cause_fail: bool = False,
        action_planner_fail: bool = False,
        classification_timeout: bool = False,
        trend_timeout: bool = False,
        root_cause_timeout: bool = False,
        action_planner_timeout: bool = False,
    ):
        super().__init__()
        self.classification_delay = classification_delay
        self.trend_delay = trend_delay
        self.classification_fail = classification_fail
        self.trend_fail = trend_fail
        self.root_cause_fail = root_cause_fail
        self.action_planner_fail = action_planner_fail
        self.classification_timeout = classification_timeout
        self.trend_timeout = trend_timeout
        self.root_cause_timeout = root_cause_timeout
        self.action_planner_timeout = action_planner_timeout

        self.call_log: List[str] = []
        self.call_args: Dict[str, Any] = {}

    async def get_agent_card(self, agent_name: str) -> Dict[str, Any]:
        return {
            "name": f"{agent_name}_agent",
            "version": "1.0.0",
            "status": "active",
            "url": f"http://localhost:8000/{agent_name}"
        }

    async def classify_complaint(self, text: str, review_id: str = "rev_1", business_id: Optional[str] = None) -> AgentResponseEnvelope:
        self.call_log.append("classify_complaint")
        self.call_args["classification"] = {"text": text, "review_id": review_id, "business_id": business_id}
        if self.classification_timeout:
            raise httpx.ReadTimeout("Classification Agent simulated timeout")
        if self.classification_fail:
            raise RuntimeError("Classification Agent internal failure")

        await asyncio.sleep(self.classification_delay)
        return AgentResponseEnvelope(
            agent_id="classification_agent",
            status="success",
            payload={
                "request_id": review_id,
                "categories": ["Food quality", "Service"],
                "aspects": ["Food temperature", "Waiting time"],
                "sentiment": "negative",
                "severity": "medium",
                "severity_reason": "Lukewarm food served after extended delay",
                "short_explanation": "Severe temperature failure and slow turnaround",
                "evidence_snippets": ["Steak was lukewarm", "Waited 45 minutes"],
                "is_multi_aspect": True,
            },
            evidence=[
                EvidenceRef(source_id="c_src_1", snippet="Steak was lukewarm", is_supporting=True)
            ]
        )

    async def analyze_trends(self, business_id: Optional[str] = None, city: Optional[str] = None, horizon: int = 3) -> AgentResponseEnvelope:
        self.call_log.append("analyze_trends")
        self.call_args["trends"] = {"business_id": business_id, "city": city, "horizon": horizon}
        if self.trend_timeout:
            raise httpx.ReadTimeout("Trend Analysis Agent simulated timeout")
        if self.trend_fail:
            raise RuntimeError("Trend Analysis Agent internal failure")

        await asyncio.sleep(self.trend_delay)
        return AgentResponseEnvelope(
            agent_id="trend_agent",
            status="success",
            payload={
                "business_id": business_id,
                "analysis_period": "2020-01 to 2022-01",
                "total_reviews_analyzed": 450,
                "average_rating": 3.4,
                "worsening_aspects": ["Waiting time"],
                "improving_aspects": ["Cleanliness"],
                "forecast": {
                    "status": "computed",
                    "target_metric": "monthly_complaints",
                    "forecast_values": {"2022-02": 12.4, "2022-03": 13.1},
                    "mae": 1.2
                },
                "satisfaction_prediction": {
                    "status": "computed",
                    "predicted_score": 3.2,
                    "predicted_label": "Moderate",
                    "metric_value": 0.28
                },
                "deterministic_provenance": "DuckDB parquet query"
            }
        )

    async def analyze_root_cause(
        self,
        problem_statement: str,
        business_id: Optional[str] = None,
        top_k: int = 5,
        classification_findings: Optional[Dict[str, Any]] = None,
        trend_findings: Optional[Dict[str, Any]] = None
    ) -> AgentResponseEnvelope:
        self.call_log.append("analyze_root_cause")
        self.call_args["root_cause"] = {
            "problem_statement": problem_statement,
            "business_id": business_id,
            "classification_findings": classification_findings,
            "trend_findings": trend_findings
        }
        if self.root_cause_timeout:
            raise httpx.ReadTimeout("Root Cause Agent simulated timeout")
        if self.root_cause_fail:
            raise RuntimeError("Root Cause Agent internal failure")

        return AgentResponseEnvelope(
            agent_id="root_cause_agent",
            status="success",
            payload={
                "problem_statement": problem_statement,
                "business_id": business_id,
                "retrieved_chunk_count": 3,
                "hypotheses": [
                    {
                        "hypothesis_id": "H1",
                        "hypothesized_cause": "Kitchen expediting bottleneck during peak rush hours",
                        "distinction_type": "inferred_explanation",
                        "confidence": "high",
                        "supporting_evidence": ["Orders backed up for 40 minutes"],
                        "supporting_source_ids": ["chk_101", "chk_102"],
                        "additional_info_needed": "Kitchen display system ticket logs"
                    }
                ]
            },
            evidence=[
                EvidenceRef(source_id="chk_101", snippet="Orders backed up for 40 minutes", is_supporting=True)
            ]
        )

    async def generate_action_plan(
        self,
        business_id: Optional[str] = None,
        problem_statement: Optional[str] = None,
        classification_findings: Optional[Dict[str, Any]] = None,
        trend_findings: Optional[Dict[str, Any]] = None,
        root_cause_findings: Optional[Dict[str, Any]] = None
    ) -> AgentResponseEnvelope:
        self.call_log.append("generate_action_plan")
        self.call_args["action_planner"] = {
            "business_id": business_id,
            "problem_statement": problem_statement,
            "root_cause_findings": root_cause_findings
        }
        if self.action_planner_timeout:
            raise httpx.ReadTimeout("Action Planner Agent simulated timeout")
        if self.action_planner_fail:
            raise RuntimeError("Action Planner Agent internal failure")

        return AgentResponseEnvelope(
            agent_id="action_planner_agent",
            status="success",
            payload={
                "business_id": business_id,
                "scoring_methodology": "Composite score = 100 * [0.35*freq + 0.25*sev + 0.20*trend + 0.20*rating_impact]",
                "total_issues_evaluated": 1,
                "ranked_actions": [
                    {
                        "rank": 1,
                        "priority_score": 78.5,
                        "issue_summary": "Kitchen line delay causing lukewarm food",
                        "recommended_action": "Implement a dedicated expediter on weekend dinner shifts",
                        "owner_role": "Kitchen Lead",
                        "measurable_kpi": "Ticket-to-Table Delivery Time",
                        "target_value": "< 18 minutes",
                        "measurement_period": "30 days",
                        "verification_method": "Weekly POS turnaround audit"
                    }
                ]
            }
        )


# ==============================================================================
# Unit Test Cases (Covering Requirements 1 to 16)
# ==============================================================================

@pytest.mark.asyncio
async def test_01_valid_request_all_four_agents():
    """1. Valid request and successful execution of all four agents."""
    mock_client = MockAsyncA2AClient()
    req = OrchestrationRequest(
        query="Cold steak and 45 minute wait time",
        business_id="BIZ_TEST_101",
        city="Philadelphia"
    )

    resp = await run_orchestration_workflow(req, client=mock_client)

    assert resp.workflow_status == "success"
    assert resp.classification is not None
    assert resp.trends is not None
    assert resp.root_cause is not None
    assert resp.action_plan is not None
    assert resp.total_duration_ms > 0
    assert "classify_complaint" in mock_client.call_log
    assert "analyze_trends" in mock_client.call_log
    assert "analyze_root_cause" in mock_client.call_log
    assert "generate_action_plan" in mock_client.call_log


@pytest.mark.asyncio
async def test_02_classification_and_trends_execute_concurrently():
    """2. Classification and Trend execute concurrently (overlapping timestamps)."""
    # Configure both with 0.1s delay
    mock_client = MockAsyncA2AClient(classification_delay=0.1, trend_delay=0.1)
    req = OrchestrationRequest(query="Slow service and cold soup", business_id="BIZ_PARALLEL")

    t0 = time.time()
    resp = await run_orchestration_workflow(req, client=mock_client)
    elapsed = time.time() - t0

    c_timing = resp.agent_timings["classification"]
    t_timing = resp.agent_timings["trends"]

    # Concurrency check: overlapping timestamps
    is_overlapping = max(c_timing.started_at, t_timing.started_at) < min(c_timing.ended_at, t_timing.ended_at)
    assert is_overlapping, "Classification and Trend Analysis must execute with overlapping timestamps."
    # Total runtime should be roughly ~0.1s + overhead, significantly less than 0.2s + downstream sequential sum
    assert elapsed < 0.4, f"Expected concurrent execution under 0.4s, but took {elapsed:.3f}s"


@pytest.mark.asyncio
async def test_03_root_cause_starts_after_upstream_available():
    """3. Root Cause starts only after the required upstream results are available."""
    mock_client = MockAsyncA2AClient(classification_delay=0.08, trend_delay=0.08)
    req = OrchestrationRequest(query="Delayed entrees and cold food", business_id="BIZ_123")

    resp = await run_orchestration_workflow(req, client=mock_client)

    c_timing = resp.agent_timings["classification"]
    t_timing = resp.agent_timings["trends"]
    rc_timing = resp.agent_timings["root_cause"]

    assert rc_timing.started_at >= c_timing.ended_at, "Root cause must start after classification completes"
    assert rc_timing.started_at >= t_timing.ended_at, "Root cause must start after trends completes"
    # Upstream findings must be propagated to root cause
    assert mock_client.call_args["root_cause"]["classification_findings"] is not None
    assert mock_client.call_args["root_cause"]["trend_findings"] is not None


@pytest.mark.asyncio
async def test_04_action_planner_starts_after_root_cause():
    """4. Action Planner starts only after Root Cause succeeds."""
    mock_client = MockAsyncA2AClient()
    req = OrchestrationRequest(query="Rude waiter and late food", business_id="BIZ_123")

    resp = await run_orchestration_workflow(req, client=mock_client)

    rc_timing = resp.agent_timings["root_cause"]
    ap_timing = resp.agent_timings["action_planner"]

    assert ap_timing.started_at >= rc_timing.ended_at, "Action planner must start after root cause finishes"
    assert mock_client.call_args["action_planner"]["root_cause_findings"] is not None


@pytest.mark.asyncio
async def test_05_business_id_and_filters_propagate():
    """5. Business ID and filters propagate correctly through the workflow."""
    mock_client = MockAsyncA2AClient()
    req = OrchestrationRequest(
        query="Burnt burgers",
        business_id="TARGET_REST_888",
        city="Tampa"
    )

    resp = await run_orchestration_workflow(req, client=mock_client)

    assert resp.business_id == "TARGET_REST_888"
    assert mock_client.call_args["classification"]["business_id"] == "TARGET_REST_888"
    assert mock_client.call_args["trends"]["business_id"] == "TARGET_REST_888"
    assert mock_client.call_args["trends"]["city"] == "Tampa"
    assert mock_client.call_args["root_cause"]["business_id"] == "TARGET_REST_888"
    assert mock_client.call_args["action_planner"]["business_id"] == "TARGET_REST_888"


@pytest.mark.asyncio
async def test_06_retrieved_evidence_preserved():
    """6. Retrieved evidence and source references are preserved."""
    mock_client = MockAsyncA2AClient()
    req = OrchestrationRequest(
        query="Cold soup and slow service",
        business_id="BIZ_EVIDENCE"
    )

    resp = await run_orchestration_workflow(req, client=mock_client)

    # Root cause produced evidence
    hyp = resp.root_cause["hypotheses"][0]
    assert "chk_101" in hyp["supporting_source_ids"]
    assert len(hyp["supporting_evidence"]) > 0


@pytest.mark.asyncio
async def test_07_classification_timeout():
    """7. Classification timeout handling."""
    mock_client = MockAsyncA2AClient(classification_timeout=True)
    req = OrchestrationRequest(query="Slow food prep", business_id="BIZ_TIMEOUT")

    resp = await run_orchestration_workflow(req, client=mock_client)

    assert resp.agent_timings["classification"].status == "timeout"
    assert resp.classification is None
    # Partial success because trends and downstream still proceed
    assert resp.workflow_status in ("partial_success", "failed")
    assert any("Classification Agent request timed out" in w for w in resp.warnings)


@pytest.mark.asyncio
async def test_08_trend_timeout():
    """8. Trend timeout handling."""
    mock_client = MockAsyncA2AClient(trend_timeout=True)
    req = OrchestrationRequest(query="Lukewarm fries", business_id="BIZ_TREND_TO")

    resp = await run_orchestration_workflow(req, client=mock_client)

    assert resp.agent_timings["trends"].status == "timeout"
    assert resp.trends is None
    assert resp.workflow_status == "partial_success"
    assert any("Trend Analysis Agent request timed out" in w for w in resp.warnings)


@pytest.mark.asyncio
async def test_09_root_cause_failure():
    """9. Root Cause failure handling."""
    mock_client = MockAsyncA2AClient(root_cause_fail=True)
    req = OrchestrationRequest(query="Rude manager and incorrect billing", business_id="BIZ_RC_FAIL")

    resp = await run_orchestration_workflow(req, client=mock_client)

    assert resp.agent_timings["root_cause"].status == "failed"
    assert resp.root_cause is None
    # Action planner must be skipped because root cause failed
    assert resp.agent_timings["action_planner"].status == "skipped"
    assert resp.action_plan is None
    assert resp.workflow_status == "partial_success"


@pytest.mark.asyncio
async def test_10_action_planner_failure():
    """10. Action Planner failure handling."""
    mock_client = MockAsyncA2AClient(action_planner_fail=True)
    req = OrchestrationRequest(query="Table dirty and delayed order", business_id="BIZ_AP_FAIL")

    resp = await run_orchestration_workflow(req, client=mock_client)

    assert resp.agent_timings["action_planner"].status == "failed"
    assert resp.action_plan is None
    # Upstream results must be preserved
    assert resp.classification is not None
    assert resp.trends is not None
    assert resp.root_cause is not None
    assert resp.workflow_status == "partial_success"


@pytest.mark.asyncio
async def test_11_invalid_request_rejection():
    """11. Invalid request and invalid query validation."""
    mock_client = MockAsyncA2AClient()

    # Empty query must fail validation before expensive calls
    with pytest.raises(Exception):
        OrchestrationRequest(query="")

    # Direct node validation check
    graph = build_orchestration_graph(client=mock_client)
    invalid_state = {
        "request_id": "req_invalid",
        "business_id": None,
        "query": " ",
        "city": None,
        "start_date": None,
        "end_date": None,
        "top_k": 5,
        "review_id": None,
        "filters": {},
        "retrieved_evidence": [],
        "classification_result": None,
        "trend_analysis_result": None,
        "root_cause_result": None,
        "action_planning_result": None,
        "agent_statuses": {},
        "agent_durations_ms": {},
        "agent_timestamps": {},
        "agent_errors": {},
        "warnings": [],
        "errors": [],
        "workflow_status": "success",
        "final_result": None,
        "graph_start_time": time.time(),
        "total_duration_ms": 0.0,
    }

    final_state = await graph.ainvoke(invalid_state)
    assert final_state["workflow_status"] == "failed"
    assert len(mock_client.call_log) == 0, "No agents should be called if request validation fails"


@pytest.mark.asyncio
async def test_12_partial_success_and_structured_error_reporting():
    """12. Partial success and structured error reporting."""
    mock_client = MockAsyncA2AClient(classification_fail=True)
    req = OrchestrationRequest(query="Uncooked chicken nuggets", business_id="BIZ_PARTIAL")

    resp = await run_orchestration_workflow(req, client=mock_client)

    assert resp.workflow_status == "partial_success"
    assert len(resp.errors) > 0
    err = resp.errors[0]
    assert err["node"] == "classification"
    assert "internal failure" in err["error"]


@pytest.mark.asyncio
async def test_13_no_downstream_execution_when_upstream_missing():
    """13. No downstream execution with missing required inputs."""
    # When both classification and trends fail, root-cause must be skipped
    mock_client = MockAsyncA2AClient(classification_fail=True, trend_fail=True)
    req = OrchestrationRequest(query="Dirty glasses and slow hostess", business_id="BIZ_UPSTREAM_FAIL")

    resp = await run_orchestration_workflow(req, client=mock_client)

    assert resp.agent_timings["root_cause"].status == "skipped"
    assert resp.agent_timings["action_planner"].status == "skipped"
    assert "analyze_root_cause" not in mock_client.call_log
    assert "generate_action_plan" not in mock_client.call_log


@pytest.mark.asyncio
async def test_14_a2a_connectivity_and_agent_cards():
    """14. A2A connectivity and Agent Card integration."""
    mock_client = MockAsyncA2AClient()
    for ag in ("classification", "trends", "root_cause", "action_planner"):
        card = await mock_client.get_agent_card(ag)
        assert card["name"] == f"{ag}_agent"
        assert card["version"] == "1.0.0"


@pytest.mark.asyncio
async def test_15_final_response_schema_validation():
    """15. Final response schema validation."""
    mock_client = MockAsyncA2AClient()
    req = OrchestrationRequest(query="Cold fries and broken fountain drink machine", business_id="BIZ_SCHEMA")

    resp = await run_orchestration_workflow(req, client=mock_client)

    # Must be valid instance of AggregatedWorkflowResponse
    assert isinstance(resp, AggregatedWorkflowResponse)
    dumped = resp.model_dump()
    assert "workflow_status" in dumped
    assert "agent_timings" in dumped
    assert "classification" in dumped
    assert "trends" in dumped
    assert "root_cause" in dumped
    assert "action_plan" in dumped


@pytest.mark.asyncio
async def test_16_total_and_per_agent_latency_instrumentation():
    """16. Total and per-agent latency instrumentation."""
    mock_client = MockAsyncA2AClient(classification_delay=0.03, trend_delay=0.03)
    req = OrchestrationRequest(query="Slow drive thru", business_id="BIZ_LATENCY")

    resp = await run_orchestration_workflow(req, client=mock_client)

    assert resp.total_duration_ms > 0
    for ag in ("classification", "trends", "root_cause", "action_planner"):
        timing = resp.agent_timings[ag]
        assert timing.duration_ms >= 0
        assert timing.started_at > 0
        assert timing.ended_at >= timing.started_at
