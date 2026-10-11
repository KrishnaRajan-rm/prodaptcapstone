"""LangGraph typed state definition for the Restaurant Operations Orchestrator.
Includes custom reducers for parallel branch state convergence.
"""

import operator
from typing import Annotated, Any, Dict, List, Literal, Optional
from typing_extensions import TypedDict


def merge_dicts(left: Optional[Dict[str, Any]], right: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Reducer that merges two dictionaries without losing keys."""
    res = dict(left) if left else {}
    if right:
        res.update(right)
    return res


def merge_errors(left: Optional[List[Dict[str, Any]]], right: Optional[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    """Reducer that combines structured error records."""
    res = list(left) if left else []
    if right:
        res.extend(right)
    return res


class OrchestrationState(TypedDict):
    """Shared typed state carrying request, context, evidence, privacy vault, and specialist agent outputs."""

    # Input request & filtering metadata
    request_id: str
    business_id: Optional[str]
    query: str
    city: Optional[str]
    start_date: Optional[str]
    end_date: Optional[str]
    top_k: int
    review_id: Optional[str]
    filters: Dict[str, Any]
    viewer_role: str
    skip_llm_judge: bool

    # Shared context & retrieved evidence
    retrieved_evidence: List[Dict[str, Any]]

    # Phase 4 PII Gate 2
    pii_vault: Optional[Any]
    pii_blocked: bool
    pii_tokens_count: int
    pii_decoded: bool

    # Specialist agent results
    classification_result: Optional[Dict[str, Any]]
    trend_analysis_result: Optional[Dict[str, Any]]
    root_cause_result: Optional[Dict[str, Any]]
    action_planning_result: Optional[Dict[str, Any]]

    # Phase 4 CrewAI Multi-Agent Drafting & Validation
    crew_report: Optional[Dict[str, Any]]
    crew_validation: Optional[Dict[str, Any]]
    crew_status: Optional[str]  # "success", "fallback", "skipped", "failed"
    crew_duration_ms: Optional[float]

    # Phase 4 Quality Gate
    quality_gate_decision: Optional[Dict[str, Any]]
    quality_gate_status: Optional[str]  # "passed", "flagged", "blocked"

    # Parallel branch telemetry reducers
    agent_statuses: Annotated[Dict[str, str], merge_dicts]
    agent_durations_ms: Annotated[Dict[str, float], merge_dicts]
    agent_timestamps: Annotated[Dict[str, Dict[str, float]], merge_dicts]
    agent_errors: Annotated[Dict[str, str], merge_dicts]
    stage_timings: Annotated[Dict[str, Dict[str, Any]], merge_dicts]

    # Observability and diagnostics
    warnings: Annotated[List[str], operator.add]
    errors: Annotated[List[Dict[str, Any]], merge_errors]

    # Lifecycle and aggregation
    workflow_status: Literal["success", "partial_success", "failed"]
    final_result: Optional[Dict[str, Any]]
    graph_start_time: float
    total_duration_ms: float
