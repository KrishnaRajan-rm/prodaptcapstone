"""Pydantic schemas for the FastAPI backend gateway."""

from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field


class ExecutionTraceEvent(BaseModel):
    """Structured telemetry event representing an individual pipeline execution stage."""
    event_id: str
    request_id: str
    stage: str
    component: str
    status: Literal["completed", "failed", "skipped", "in_progress", "flagged", "blocked"]
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    duration_ms: Optional[float] = None
    input_summary: str = ""
    output_summary: str = ""
    input_payload: Optional[Any] = None
    output_payload: Optional[Any] = None
    source_ids: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    error_summary: Optional[str] = None


class AnalysisRequest(BaseModel):
    """Payload for POST /api/v1/analysis."""
    query: str = Field(..., min_length=2, description="Manager operational question")
    mode: Literal["auto", "agent"] = Field(default="auto", description="'auto' for LangGraph workflow or 'agent' for direct single agent")
    agent: Optional[Literal["classification", "trends", "root_cause", "action_planner"]] = Field(
        default=None,
        description="Required if mode='agent'"
    )
    business_id: Optional[str] = Field(default=None, description="Optional restaurant business identifier")
    start_date: Optional[str] = Field(default=None, description="Optional YYYY-MM or YYYY-MM-DD filter")
    end_date: Optional[str] = Field(default=None, description="Optional YYYY-MM or YYYY-MM-DD filter")
    city: Optional[str] = Field(default=None, description="Optional city filter")
    filters: Dict[str, Any] = Field(default_factory=dict, description="Metadata filters")
    viewer_role: str = Field(default="manager", description="Viewer role for access and decoding")
    skip_llm_judge: bool = Field(default=False, description="Whether to bypass LLM judge in Quality Gate")


class AnalysisResponse(BaseModel):
    """Comprehensive, frontend-ready response returned by POST /api/v1/analysis."""
    request_id: str
    query: str
    mode: str
    selected_agent: Optional[str] = None
    business_id: Optional[str] = None
    workflow_status: Literal["success", "partial_success", "failed"]
    total_duration_ms: float = 0.0

    # Executive Summary and CrewAI consolidation
    executive_summary: Optional[str] = None
    crew_report: Optional[Dict[str, Any]] = None
    crew_validation: Optional[Dict[str, Any]] = None
    crew_status: Optional[str] = None

    # Specialist Agent Outputs
    classification: Optional[Dict[str, Any]] = None
    trends: Optional[Dict[str, Any]] = None
    root_cause: Optional[Dict[str, Any]] = None
    action_plan: Optional[Dict[str, Any]] = None

    # Evidence & Quality Gate
    retrieved_evidence: List[Dict[str, Any]] = Field(default_factory=list)
    quality_gate: Optional[Dict[str, Any]] = None
    quality_decision: Optional[str] = None  # "passed", "flagged", "blocked"
    confidence_score: Optional[float] = None
    staleness_disclosure: Optional[str] = None

    # Telemetry and Execution Trace
    agent_timings: Dict[str, Any] = Field(default_factory=dict)
    execution_trace: List[ExecutionTraceEvent] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    errors: List[Dict[str, Any]] = Field(default_factory=list)
    provenance: Dict[str, Any] = Field(default_factory=dict)


class AgentStatusInfo(BaseModel):
    agent_id: str
    name: str
    description: str
    port: int
    url: str
    status: Literal["available", "unavailable", "unknown"]
    latency_ms: Optional[float] = None
    card: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class AgentStatusResponse(BaseModel):
    agents: Dict[str, AgentStatusInfo]
    total_available: int
    total_configured: int
    last_checked: str


class DashboardStatsResponse(BaseModel):
    total_reviews: int
    total_complaints: int
    average_rating: Optional[float] = None
    cities_covered: int
    first_review_month: Optional[str] = None
    last_review_month: Optional[str] = None
    top_complaint_category: Optional[str] = None
    source: str


class ScenarioItem(BaseModel):
    id: str
    title: str
    description: str
    sample_query: str
    mode: Literal["auto", "agent"]
    agent: Optional[str] = None
    business_id: Optional[str] = None
    business_name: Optional[str] = None
    city: Optional[str] = None
    suggested_start_date: Optional[str] = None
    suggested_end_date: Optional[str] = None


class ScenariosResponse(BaseModel):
    scenarios: List[ScenarioItem]


class FeedbackCreate(BaseModel):
    request_id: str
    rating: int = Field(..., ge=1, le=5, description="1 to 5 rating scale (or 1=not useful, 5=useful)")
    useful: Optional[bool] = Field(default=None, description="Useful / Not Useful boolean")
    flags: Optional[str] = Field(default="", description="Error categories: hallucination, ungrounded, wrong_agent, stale")
    comment: Optional[str] = Field(default="", description="Manager feedback notes")
    role: str = Field(default="manager")


class FeedbackResponse(BaseModel):
    feedback_id: str
    request_id: str
    status: str
    message: str


class FeedbackSummaryResponse(BaseModel):
    total_feedback: int
    average_rating: float
    useful_count: int
    not_useful_count: int
    common_flags: Dict[str, int]


class EvaluationSummaryResponse(BaseModel):
    benchmark_name: str = "DeepEval Recommendation & Groundedness Benchmark"
    status: Literal["operational", "pending", "unavailable"]
    faithfulness_score: Optional[float] = None
    relevancy_score: Optional[float] = None
    passed: bool = True
    test_cases_evaluated: int = 2
    evaluated_at: str = ""
    details: Dict[str, Any] = Field(default_factory=dict)
