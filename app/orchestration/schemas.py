"""Pydantic schemas for LangGraph Orchestration request, response, and telemetry."""

import re
import uuid
from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field, field_validator


class OrchestrationRequest(BaseModel):
    """Input request model for the LangGraph Orchestrator workflow."""
    request_id: str = Field(default_factory=lambda: f"req_{uuid.uuid4().hex[:10]}")
    query: str = Field(..., min_length=2, description="Natural language complaint, question, or operational prompt")
    business_id: Optional[str] = Field(default=None, description="Optional target business/restaurant identifier")
    city: Optional[str] = Field(default=None, description="Optional city filter (e.g. 'Philadelphia')")
    start_date: Optional[str] = Field(default=None, description="Start date filter (YYYY-MM or YYYY-MM-DD)")
    end_date: Optional[str] = Field(default=None, description="End date filter (YYYY-MM or YYYY-MM-DD)")
    top_k: int = Field(default=5, ge=1, le=50, description="Evidence chunks to retrieve")
    review_id: Optional[str] = Field(default=None, description="Optional specific review identifier")
    filters: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata filters")
    viewer_role: str = Field(default="manager", description="Role requesting report: default is 'manager'")
    skip_llm_judge: bool = Field(default=False, description="Optionally bypass LLM judge for fast deterministic evaluation")

    @field_validator("query")
    @classmethod
    def validate_query_text(cls, v: str) -> str:
        cleaned = v.strip()
        if not cleaned:
            raise ValueError("Query cannot be empty or solely whitespace.")
        return cleaned

    @field_validator("start_date", "end_date")
    @classmethod
    def validate_date_format(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return None
        v = v.strip()
        if not v:
            return None
        # Must match YYYY-MM or YYYY-MM-DD
        if not re.match(r"^\d{4}-\d{2}(-\d{2})?$", v):
            raise ValueError(f"Invalid date format '{v}'. Expected YYYY-MM or YYYY-MM-DD.")
        return v


class AgentTimingInfo(BaseModel):
    """Telemetry for an individual agent execution within the graph."""
    status: Literal["success", "failed", "timeout", "skipped"] = "skipped"
    duration_ms: float = 0.0
    started_at: float = 0.0
    ended_at: float = 0.0
    error_message: Optional[str] = None


class AggregatedWorkflowResponse(BaseModel):
    """Consolidated, schema-validated end-to-end response returned by the orchestrator."""
    request_id: str
    business_id: Optional[str] = None
    query: str
    workflow_status: Literal["success", "partial_success", "failed"]
    total_duration_ms: float = 0.0
    classification: Optional[Dict[str, Any]] = None
    trends: Optional[Dict[str, Any]] = None
    root_cause: Optional[Dict[str, Any]] = None
    action_plan: Optional[Dict[str, Any]] = None
    retrieved_evidence: List[Dict[str, Any]] = Field(default_factory=list)
    agent_timings: Dict[str, AgentTimingInfo] = Field(default_factory=dict)
    warnings: List[str] = Field(default_factory=list)
    errors: List[Dict[str, Any]] = Field(default_factory=list)
    provenance: Dict[str, Any] = Field(default_factory=dict)

    # Phase 4 CrewAI Multi-Agent Drafting & Validation additions
    executive_summary: Optional[str] = None
    crew_report: Optional[Dict[str, Any]] = None
    crew_validation: Optional[Dict[str, Any]] = None
    crew_status: Optional[str] = None

    # Phase 4 Quality & Privacy additions
    quality_gate: Optional[Dict[str, Any]] = None
    quality_decision: Optional[str] = None  # "passed", "flagged", "blocked"
    confidence_score: Optional[float] = None
    staleness_disclosure: Optional[str] = None
    pii_masked: bool = True
    pii_decoded: bool = False
    viewer_role: str = "manager"
