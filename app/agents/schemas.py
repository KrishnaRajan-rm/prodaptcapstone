"""Shared Pydantic schemas and standard response envelopes for the 4 Specialist Agents.
Follows Google ADK and A2A protocol interoperability requirements.
"""

from enum import Enum
from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field, field_validator


# ==============================================================================
# Common Evidence & Response Envelopes
# ==============================================================================

class EvidenceRef(BaseModel):
    """Traceable evidence pointer to a source chunk, review, or data record."""
    source_id: str = Field(description="Unique identifier of the evidence item")
    chunk_id: Optional[str] = Field(default=None, description="Chunk ID if retrieved from RAG chunks")
    review_id: Optional[str] = Field(default=None, description="Review ID if linked to an original review")
    business_id: Optional[str] = Field(default=None, description="Business ID linked to the record")
    snippet: str = Field(default="", description="Relevant excerpt or factual claim")
    relevance_score: Optional[float] = Field(default=None, description="Retrieval or similarity score")
    is_supporting: bool = Field(default=True, description="True if supports claim, False if contradictory")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional context metadata")


class AgentResponseEnvelope(BaseModel):
    """Standard A2A response envelope returned by all specialist agents."""
    agent_id: str = Field(description="Identifier of the executing agent (e.g. classification_agent)")
    status: Literal["success", "insufficient_data", "error"] = Field(
        default="success",
        description="Overall lifecycle execution status"
    )
    payload: Dict[str, Any] = Field(
        default_factory=dict,
        description="Validated domain-specific payload produced by the agent"
    )
    evidence: List[EvidenceRef] = Field(
        default_factory=list,
        description="Traceable references to underlying source reviews/records"
    )
    warnings: List[str] = Field(
        default_factory=list,
        description="Non-fatal warnings or uncertainty disclosures"
    )
    limitations: List[str] = Field(
        default_factory=list,
        description="Documented boundaries and data coverage caveats"
    )
    execution_metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Metadata including latency, model used, and timestamps"
    )


# ==============================================================================
# 1. Classification Agent Schemas (Port 8001)
# ==============================================================================

class ComplaintCategory(str, Enum):
    SERVICE_SPEED = "Service speed"
    FOOD_QUALITY = "Food quality"
    CLEANLINESS = "Cleanliness"
    PRICING = "Pricing"
    STAFF_BEHAVIOUR = "Staff behaviour"
    OTHER = "Other / uncategorized"


class AspectTag(str, Enum):
    WAITING_TIME = "Waiting time"
    FOOD_TEMPERATURE = "Food temperature"
    TASTE_FRESHNESS = "Taste and freshness"
    HYGIENE = "Hygiene"
    STAFF_RESPONSIVENESS = "Staff responsiveness"
    PORTION_SIZE = "Portion size"
    VALUE_FOR_MONEY = "Value for money"
    OTHER = "Other"


class SentimentType(str, Enum):
    POSITIVE = "positive"
    NEUTRAL = "neutral"
    NEGATIVE = "negative"
    MIXED = "mixed"


class SeverityLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ClassificationRequest(BaseModel):
    """Input payload for Classification Agent."""
    review_id: Optional[str] = Field(default="req_0", description="Review or request ID")
    text: str = Field(description="Review text or complaint statement to analyze")
    business_id: Optional[str] = Field(default=None, description="Optional restaurant ID")
    chunk_id: Optional[str] = Field(default=None, description="Optional chunk reference")


class ClassificationPayload(BaseModel):
    """Structured output from Classification Agent."""
    request_id: str
    categories: List[str] = Field(description="Identified complaint categories")
    aspects: List[str] = Field(description="Specific operational aspects mentioned")
    sentiment: SentimentType = Field(description="Overall customer sentiment")
    severity: SeverityLevel = Field(description="Operational issue severity assessment")
    severity_reason: str = Field(description="Documented rationale for the severity assessment")
    short_explanation: str = Field(description="Concise analytical explanation")
    evidence_snippets: List[str] = Field(default_factory=list, description="Direct excerpts from input text")
    is_multi_aspect: bool = Field(default=False, description="True if review spans multiple distinct issues")

    @field_validator("categories", "aspects", "evidence_snippets", mode="before")
    @classmethod
    def coerce_list_fields(cls, v):
        if v is None:
            return []
        if isinstance(v, str):
            v = v.strip()
            return [v] if v else []
        if isinstance(v, list):
            return [str(x) for x in v if x is not None]
        return [str(v)]


# ==============================================================================
# 2. Trend Analysis Agent Schemas (Port 8002)
# ==============================================================================

class TrendDirection(str, Enum):
    INCREASING = "increasing"
    DECREASING = "decreasing"
    STABLE = "stable"


class AspectTrend(BaseModel):
    aspect: str
    direction: TrendDirection
    total_mentions: int
    total_complaints: int
    complaint_rate_pct: float
    period_change_pct: float
    monthly_series: Dict[str, int] = Field(default_factory=dict, description="year_month -> count")


class ForecastResult(BaseModel):
    status: Literal["computed", "insufficient_data"] = "insufficient_data"
    target_metric: str = "monthly_complaints"
    horizon_months: int = 3
    historical_periods_used: int = 0
    forecast_values: Dict[str, float] = Field(default_factory=dict, description="Future year_month -> predicted count")
    model_type: str = "linear_trend_baseline"
    mae: Optional[float] = None
    rmse: Optional[float] = None
    evaluation_notes: str = ""


class SatisfactionPredictionResult(BaseModel):
    status: Literal["computed", "insufficient_data"] = "insufficient_data"
    target_metric: str = "average_star_rating"
    predicted_score: Optional[float] = None
    predicted_label: Optional[str] = None
    evaluation_metric: str = "MAE"
    metric_value: Optional[float] = None
    train_samples: int = 0
    test_samples: int = 0
    features_used: List[str] = Field(default_factory=list)
    evaluation_notes: str = ""


class TrendAnalysisRequest(BaseModel):
    """Input payload for Trend Analysis Agent."""
    business_id: Optional[str] = Field(default=None, description="Target restaurant ID")
    city: Optional[str] = Field(default=None, description="Optional city filter")
    aspects: Optional[List[str]] = Field(default=None, description="Aspects to analyze (defaults to all)")
    start_year_month: Optional[str] = Field(default=None, description="e.g. '2019-01'")
    end_year_month: Optional[str] = Field(default=None, description="e.g. '2021-12'")
    forecast_horizon: int = Field(default=3, description="Months ahead to forecast")


class TrendAnalysisPayload(BaseModel):
    """Structured output from Trend Analysis Agent."""
    business_id: Optional[str]
    city: Optional[str]
    analysis_period: str
    total_reviews_analyzed: int
    average_rating: float
    aspect_trends: List[AspectTrend]
    worsening_aspects: List[str]
    improving_aspects: List[str]
    forecast: ForecastResult
    satisfaction_prediction: SatisfactionPredictionResult
    deterministic_provenance: str = Field(description="DuckDB/Parquet query source")


# ==============================================================================
# 3. Root-Cause Analysis Agent Schemas (Port 8003)
# ==============================================================================

class DistinctionType(str, Enum):
    OBSERVED_FACT = "observed_fact"
    STATISTICAL_ASSOCIATION = "statistical_association"
    INFERRED_EXPLANATION = "inferred_explanation"


class RootCauseHypothesis(BaseModel):
    hypothesis_id: str
    hypothesized_cause: str = Field(description="Plausible operational root-cause explanation")
    distinction_type: DistinctionType = Field(description="Clear epistemic category of the claim")
    supporting_evidence: List[str] = Field(default_factory=list, description="Grounding excerpts from reviews or records")
    supporting_source_ids: List[str] = Field(default_factory=list, description="Chunk/review IDs supporting cause")
    contradictory_evidence: List[str] = Field(default_factory=list, description="Opposing review excerpts if any")
    contradictory_source_ids: List[str] = Field(default_factory=list, description="Chunk/review IDs contradicting cause")
    alternative_explanations: List[str] = Field(default_factory=list, description="Plausible competing explanations")
    confidence: Literal["low", "medium", "high"] = Field(description="Calibrated confidence level")
    additional_info_needed: str = Field(description="Operational records required to verify beyond doubt")

    @field_validator(
        "supporting_evidence",
        "supporting_source_ids",
        "contradictory_evidence",
        "contradictory_source_ids",
        "alternative_explanations",
        mode="before"
    )
    @classmethod
    def coerce_hypothesis_lists(cls, v):
        if v is None:
            return []
        if isinstance(v, str):
            v = v.strip()
            return [v] if v else []
        if isinstance(v, list):
            return [str(x) for x in v if x is not None]
        return [str(v)]


class RootCauseRequest(BaseModel):
    """Input payload for Root-Cause Analysis Agent."""
    problem_statement: str = Field(description="Description of recurring operational failure")
    business_id: Optional[str] = Field(default=None, description="Target restaurant ID")
    aspect: Optional[str] = Field(default=None, description="Specific operational aspect")
    classification_findings: Optional[Dict[str, Any]] = Field(default=None, description="Optional upstream Agent 1 output")
    trend_findings: Optional[Dict[str, Any]] = Field(default=None, description="Optional upstream Agent 2 output")
    top_k_evidence: int = Field(default=5, description="Number of evidence chunks to retrieve via HybridRetriever")


class RootCausePayload(BaseModel):
    """Structured output from Root-Cause Analysis Agent."""
    problem_statement: str
    business_id: Optional[str]
    hypotheses: List[RootCauseHypothesis]
    retrieved_chunk_count: int
    factual_observations: List[str] = Field(default_factory=list)


# ==============================================================================
# 4. Action Planner Agent Schemas (Port 8004)
# ==============================================================================

class PriorityScoreBreakdown(BaseModel):
    """Explicitly documented priority score formula factors (0-100 scale)."""
    frequency_score: float = Field(description="Normalized frequency [0-1]")
    severity_score: float = Field(description="Normalized severity [0-1]")
    trend_score: float = Field(description="Normalized worsening trend rate [0-1]")
    rating_impact_score: float = Field(description="Normalized customer rating impact [0-1]")
    composite_score: float = Field(description="100 * (0.35*freq + 0.25*sev + 0.20*trend + 0.20*rating_impact)")
    formula_weights: str = "0.35*freq + 0.25*sev + 0.20*trend + 0.20*rating_impact"


class ActionRecommendation(BaseModel):
    rank: int = Field(description="Priority ranking (1 = highest urgency)")
    priority_score: float = Field(description="Composite score (0 - 100)")
    scoring_details: PriorityScoreBreakdown
    issue_summary: str = Field(description="Specific operational issue summary")
    evidence_rationale: str = Field(description="Why this received its priority rank based on evidence")
    recommended_action: str = Field(description="Concrete actionable corrective procedure")
    owner_role: str = Field(description="Specific suggested owner role (e.g. Kitchen Lead, Shift Manager)")
    measurable_kpi: str = Field(description="Objective metric to evaluate progress")
    baseline_value: Optional[str] = Field(default=None, description="Historical observed baseline from data if known")
    target_value: str = Field(description="Quantified target value (explicitly marked if proposed)")
    is_proposed_target: bool = Field(default=True, description="True if target is a recommended target proposal")
    measurement_period: str = Field(default="30 days", description="Evaluation timeframe")
    dependencies: List[str] = Field(default_factory=list, description="Prerequisites and operational constraints")
    expected_outcome: str = Field(description="Anticipated operational outcome")
    verification_method: str = Field(description="How management will verify implementation success")

    @field_validator("dependencies", mode="before")
    @classmethod
    def coerce_dependencies_list(cls, v):
        if v is None:
            return []
        if isinstance(v, str):
            v = v.strip()
            return [v] if v else []
        if isinstance(v, list):
            return [str(x) for x in v if x is not None]
        return [str(v)]


class ActionPlannerRequest(BaseModel):
    """Input payload for Action Planner Agent."""
    business_id: Optional[str] = Field(default=None, description="Target restaurant ID")
    problem_statement: Optional[str] = Field(default=None, description="Operational problem summary")
    classification_findings: Optional[Dict[str, Any]] = Field(default=None, description="Classification Agent output")
    trend_findings: Optional[Dict[str, Any]] = Field(default=None, description="Trend Analysis Agent output")
    root_cause_findings: Optional[Dict[str, Any]] = Field(default=None, description="Root-Cause Agent output")


class ActionPlannerPayload(BaseModel):
    """Structured output from Action Planner Agent."""
    business_id: Optional[str]
    ranked_actions: List[ActionRecommendation]
    scoring_methodology: str = Field(description="Documented normalization formula and weight breakdown")
    total_issues_evaluated: int
