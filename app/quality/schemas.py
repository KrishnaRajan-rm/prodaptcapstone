"""Schemas and Data Models for Phase 4 Quality Gate and Evidence Grounding."""

from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class QualityStatus(str, Enum):
    PASSED = "passed"
    FLAGGED = "flagged"
    BLOCKED = "blocked"


class GroundingFinding(BaseModel):
    source_id: str
    exists: bool = True
    excerpt_overlap: float = Field(default=1.0, description="Normalized string overlap ratio [0.0, 1.0]")
    business_id_match: bool = True
    supported: bool = True
    note: str = ""


class GroundingCheckResult(BaseModel):
    status: QualityStatus = QualityStatus.PASSED
    total_citations: int = 0
    verified_citations: int = 0
    missing_citations: List[str] = Field(default_factory=list)
    unrelated_citations: List[str] = Field(default_factory=list)
    business_id_mismatches: List[str] = Field(default_factory=list)
    findings: List[GroundingFinding] = Field(default_factory=list)


class StatisticalCheckResult(BaseModel):
    status: QualityStatus = QualityStatus.PASSED
    dataset_cutoff_date: str = "2022-01"
    current_reference_date: str = "2026-10"
    stale_historical_data_disclosed: bool = True
    staleness_disclosure: str = ""
    forecast_properly_labeled: bool = True
    verified_trends: List[Dict[str, Any]] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)


class RecommendationCheckResult(BaseModel):
    status: QualityStatus = QualityStatus.PASSED
    mapped_to_findings: bool = True
    priority_consistent: bool = True
    owner_roles_valid: bool = True
    targets_labeled_proposed: bool = True
    unsupported_assumptions: List[str] = Field(default_factory=list)
    invalid_owners: List[str] = Field(default_factory=list)


class ContradictionCheckResult(BaseModel):
    status: QualityStatus = QualityStatus.PASSED
    has_contradictions: bool = False
    contradictions: List[str] = Field(default_factory=list)


class ConfidenceScoreResult(BaseModel):
    score: float = Field(default=80.0, description="Deterministic confidence score [0.0, 100.0]")
    max_possible_score: float = Field(default=80.0, description="Theoretical maximum ceiling of formula")
    normalized_score: float = Field(default=100.0, description="Score normalized to 0-100 percentage")
    assessment: str = Field(default="high", description="high, medium, low, unknown")
    formula: str = (
        "score = 100 * (0.35 * evidence_relevance + 0.25 * source_traceability + "
        "0.20 * agent_agreement - 0.20 * staleness_penalty)"
    )
    weights: Dict[str, float] = Field(
        default_factory=lambda: {
            "evidence_relevance": 0.35,
            "source_traceability": 0.25,
            "agent_agreement": 0.20,
            "staleness_penalty": 0.20,
        }
    )
    component_scores: Dict[str, float] = Field(default_factory=dict)
    explanation: str = ""


class LLMJudgeVerdict(BaseModel):
    verdict: QualityStatus = QualityStatus.PASSED
    faithfulness_score: float = Field(default=1.0, ge=0.0, le=1.0)
    relevance_score: float = Field(default=1.0, ge=0.0, le=1.0)
    issues: List[str] = Field(default_factory=list)
    remediation_notes: List[str] = Field(default_factory=list)
    latency_ms: float = 0.0
    model_used: str = "llm_judge"


class QualityGateDecision(BaseModel):
    decision: QualityStatus = QualityStatus.PASSED
    summary: str = ""
    grounding: GroundingCheckResult = Field(default_factory=GroundingCheckResult)
    statistical: StatisticalCheckResult = Field(default_factory=StatisticalCheckResult)
    recommendations: RecommendationCheckResult = Field(default_factory=RecommendationCheckResult)
    contradictions: ContradictionCheckResult = Field(default_factory=ContradictionCheckResult)
    confidence: ConfidenceScoreResult = Field(default_factory=ConfidenceScoreResult)
    llm_judge: Optional[LLMJudgeVerdict] = None
    telemetry_ms: Dict[str, float] = Field(default_factory=dict)
