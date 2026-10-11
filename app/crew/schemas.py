"""Schemas for CrewAI Operations Report Drafting and Validation."""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class RecommendedActionReport(BaseModel):
    priority_rank: int = Field(default=1, description="Priority ranking 1-N")
    problem_addressed: str = Field(default="", description="Operational problem targeted")
    recommended_action: str = Field(default="", description="Concrete operational action")
    rationale: str = Field(default="", description="Operational justification")
    owner_role: str = Field(default="kitchen_manager", description="Designated role")
    kpi: str = Field(default="", description="Measurement metric")
    target: str = Field(default="proposed", description="Proposed target or SLA")
    source_ids: List[str] = Field(default_factory=list, description="Supporting review or chunk IDs")


class CrewConsolidatedReport(BaseModel):
    """Structured report produced by the Operations Report Drafter."""
    executive_summary: str = Field(
        default="",
        description="Concise 2-4 paragraph executive summary synthesized across all specialist findings."
    )
    key_findings: List[str] = Field(
        default_factory=list,
        description="Bulleted core operational findings with supporting evidence references."
    )
    operational_problems: List[str] = Field(
        default_factory=list,
        description="Identified operational bottlenecks and customer dissatisfaction drivers."
    )
    evidence_backed_causes: List[str] = Field(
        default_factory=list,
        description="Likely root causes, explicitly distinguishing hypotheses from proven facts."
    )
    prioritized_recommendations: List[RecommendedActionReport] = Field(
        default_factory=list,
        description="Ranked actionable recommendations with owners and measurable KPIs."
    )
    data_limitations_and_cutoffs: List[str] = Field(
        default_factory=list,
        description="Explicit disclosures regarding dataset cutoff dates (Jan 2022) and scope."
    )
    evidence_source_ids: List[str] = Field(
        default_factory=list,
        description="List of cited chunk and review identifiers."
    )


class CrewValidationResult(BaseModel):
    """Structured audit produced by the Operations Report Validator."""
    is_valid: bool = Field(default=True, description="Whether the draft meets accuracy standards")
    severity: str = Field(default="none", description="'none', 'minor', or 'major'")
    supported_claims_count: int = Field(default=0, description="Count of evidence-backed claims")
    flagged_claims_count: int = Field(default=0, description="Count of unsupported or speculative claims")
    issues: List[str] = Field(default_factory=list, description="Identified inconsistencies or unsupported claims")
    remediation_notes: List[str] = Field(default_factory=list, description="Corrective edits applied or recommended")
    pii_leakage_detected: bool = Field(default=False, description="Whether any raw PII was reintroduced")
    staleness_disclosed: bool = Field(default=True, description="Whether historical data cutoff is disclosed")
