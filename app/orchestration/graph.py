"""LangGraph Workflow Graph Definition for Restaurant Operations Review Assistant.
Constructs, compiles, and routes the multi-agent workflow with PII Gate 2, Quality Gate,
and conditional decision edges.
"""

import time
from typing import Any, Dict, Optional
from langgraph.graph import StateGraph, START, END

from app.core.logging import logger
from app.orchestration.schemas import (
    AggregatedWorkflowResponse,
    OrchestrationRequest,
)
from app.orchestration.state import OrchestrationState
from app.orchestration.nodes import create_orchestration_nodes
from app.orchestration.a2a_async_client import AsyncA2AAgentClient


def route_after_validation(state: OrchestrationState) -> str:
    """Conditional routing: skips expensive retrieval and agents if input validation fails."""
    if state.get("workflow_status") == "failed":
        logger.info("Validation failed, routing directly to aggregate_results.")
        return "aggregate_results"
    return "retrieve_context"


def route_after_pii_gate(state: OrchestrationState) -> str:
    """Conditional routing: fails closed to safe_blocked if PII Gate 2 detects unmasked sensitive data."""
    if state.get("pii_blocked"):
        logger.warning("PII Gate 2 blocked request; routing to safe_blocked.")
        return "safe_blocked"
    return "proceed_parallel"


def route_after_root_cause(state: OrchestrationState) -> str:
    """Conditional routing: skips Action Planner if Root Cause Analysis did not succeed."""
    root_status = state.get("agent_statuses", {}).get("root_cause")
    if root_status == "success":
        return "action_planner"
    logger.info(f"Root cause status is '{root_status}', skipping action_planner to crewai_consolidation.")
    return "crewai_consolidation"


def route_after_quality_gate(state: OrchestrationState) -> str:
    """Conditional routing: bifurcates based on Quality Gate decision (passed/flagged vs blocked)."""
    q_status = state.get("quality_gate_status", "passed")
    if q_status == "blocked":
        logger.warning("Quality Gate BLOCKED report; routing to safe_blocked.")
        return "safe_blocked"
    logger.info(f"Quality Gate status '{q_status}'; routing to decode_pii_role.")
    return "decode_pii_role"


def build_orchestration_graph(client: Optional[AsyncA2AAgentClient] = None) -> Any:
    """Builds and compiles the executable LangGraph workflow graph with Phase 4 gates."""
    nodes = create_orchestration_nodes(client=client)

    builder = StateGraph(OrchestrationState)

    # Register nodes
    builder.add_node("validate_request", nodes["validate_request"])
    builder.add_node("retrieve_context", nodes["retrieve_context"])
    builder.add_node("pii_gate_2", nodes["pii_gate_2"])
    builder.add_node("classification", nodes["classification"])
    builder.add_node("trends", nodes["trends"])
    builder.add_node("root_cause", nodes["root_cause"])
    builder.add_node("action_planner", nodes["action_planner"])
    builder.add_node("crewai_consolidation", nodes["crewai_consolidation"])
    builder.add_node("quality_gate", nodes["quality_gate"])
    builder.add_node("safe_blocked", nodes["safe_blocked"])
    builder.add_node("decode_pii_role", nodes["decode_pii_role"])
    builder.add_node("aggregate_results", nodes["aggregate_results"])

    # Define edges & workflow control flow
    builder.add_edge(START, "validate_request")

    # Conditional routing after validation
    builder.add_conditional_edges(
        "validate_request",
        route_after_validation,
        {
            "retrieve_context": "retrieve_context",
            "aggregate_results": "aggregate_results",
        }
    )

    # Context retrieval leads into PII Gate 2
    builder.add_edge("retrieve_context", "pii_gate_2")

    # Conditional routing after PII Gate 2
    builder.add_conditional_edges(
        "pii_gate_2",
        route_after_pii_gate,
        {
            "proceed_parallel": "classification",
            "safe_blocked": "safe_blocked",
        }
    )

    # Parallel branch: pii_gate_2 forks to both classification and trends
    builder.add_edge("pii_gate_2", "trends")

    # Parallel join: both classification and trends lead to root_cause
    builder.add_edge("classification", "root_cause")
    builder.add_edge("trends", "root_cause")

    # Conditional routing after root-cause
    builder.add_conditional_edges(
        "root_cause",
        route_after_root_cause,
        {
            "action_planner": "action_planner",
            "crewai_consolidation": "crewai_consolidation",
        }
    )

    # Action planner leads into CrewAI consolidation
    builder.add_edge("action_planner", "crewai_consolidation")

    # CrewAI consolidation leads into Quality Gate
    builder.add_edge("crewai_consolidation", "quality_gate")

    # Conditional routing after Quality Gate
    builder.add_conditional_edges(
        "quality_gate",
        route_after_quality_gate,
        {
            "safe_blocked": "safe_blocked",
            "decode_pii_role": "decode_pii_role",
        }
    )

    # Blocked and Decoded routes both merge into final aggregate node
    builder.add_edge("safe_blocked", "aggregate_results")
    builder.add_edge("decode_pii_role", "aggregate_results")
    builder.add_edge("aggregate_results", END)

    compiled_graph = builder.compile()
    return compiled_graph


# Cached default graph instance
_default_graph = None

def get_orchestration_graph() -> Any:
    """Returns singleton compiled LangGraph instance using default A2A client."""
    global _default_graph
    if _default_graph is None:
        _default_graph = build_orchestration_graph()
    return _default_graph


async def run_orchestration_workflow(
    request: OrchestrationRequest,
    client: Optional[AsyncA2AAgentClient] = None,
    graph: Optional[Any] = None,
) -> AggregatedWorkflowResponse:
    """Executes the complete orchestrated LangGraph workflow for a request."""
    workflow_graph = graph or (build_orchestration_graph(client) if client else get_orchestration_graph())

    initial_state: OrchestrationState = {
        "request_id": request.request_id,
        "business_id": request.business_id,
        "query": request.query,
        "city": request.city,
        "start_date": request.start_date,
        "end_date": request.end_date,
        "top_k": request.top_k,
        "review_id": request.review_id,
        "filters": request.filters,
        "viewer_role": request.viewer_role,
        "skip_llm_judge": request.skip_llm_judge,
        "retrieved_evidence": [],
        "pii_vault": None,
        "pii_blocked": False,
        "pii_tokens_count": 0,
        "pii_decoded": False,
        "classification_result": None,
        "trend_analysis_result": None,
        "root_cause_result": None,
        "action_planning_result": None,
        "quality_gate_decision": None,
        "quality_gate_status": None,
        "agent_statuses": {},
        "agent_durations_ms": {},
        "agent_timestamps": {},
        "agent_errors": {},
        "stage_timings": {},
        "warnings": [],
        "errors": [],
        "workflow_status": "success",
        "final_result": None,
        "graph_start_time": time.time(),
        "total_duration_ms": 0.0,
    }

    final_state = await workflow_graph.ainvoke(initial_state)

    if final_state.get("final_result"):
        response = AggregatedWorkflowResponse.model_validate(final_state["final_result"])
        response.stage_timings = final_state.get("stage_timings", {})
        return response

    issues = final_state.get("errors") or final_state.get("warnings") or []
    raise RuntimeError(
        f"Orchestration graph returned no final result for request {request.request_id}. Issues: {issues}"
    )
