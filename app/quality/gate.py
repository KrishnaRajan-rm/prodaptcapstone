"""Master Quality Gate Orchestrator.
Combines deterministic evidence-grounding, statistical validation, recommendation sanity,
contradiction detection, explainable confidence scoring, and LLM-as-judge evaluation.
"""

import time
from typing import Any, Dict, List, Optional

from app.core.logging import logger
from app.agents.llm import LLMClient
from app.quality.schemas import (
    QualityGateDecision,
    QualityStatus,
)
from app.quality.grounding import validate_grounding
from app.quality.statistical import validate_statistics_and_freshness
from app.quality.recommendations import validate_recommendations
from app.quality.contradictions import detect_contradictions
from app.quality.confidence import compute_confidence_score
from app.quality.judge import evaluate_with_llm_judge


def evaluate_quality_gate(
    query: str,
    target_business_id: Optional[str],
    retrieved_evidence: List[Dict[str, Any]],
    classification_payload: Optional[Dict[str, Any]],
    trend_payload: Optional[Dict[str, Any]],
    root_cause_payload: Optional[Dict[str, Any]],
    action_planner_payload: Optional[Dict[str, Any]],
    skip_llm_judge: bool = False,
    llm_client: Optional[LLMClient] = None,
) -> QualityGateDecision:
    """Executes the comprehensive Phase 4 Quality Gate suite."""
    t_start = time.time()

    # 1. Deterministic Grounding Checks
    t_det_start = time.time()
    grounding_res = validate_grounding(
        target_business_id=target_business_id,
        retrieved_evidence=retrieved_evidence,
        classification_payload=classification_payload,
        root_cause_payload=root_cause_payload,
        action_planner_payload=action_planner_payload,
    )

    # 2. Statistical & Historical Staleness Checks
    stat_res = validate_statistics_and_freshness(
        query=query,
        trend_payload=trend_payload,
    )

    # 3. Recommendation Sanity Checks
    rec_res = validate_recommendations(
        action_planner_payload=action_planner_payload,
        root_cause_payload=root_cause_payload,
        classification_payload=classification_payload,
    )

    # 4. Cross-Agent Contradiction Detection
    contra_res = detect_contradictions(
        classification_payload=classification_payload,
        trend_payload=trend_payload,
        root_cause_payload=root_cause_payload,
        action_planner_payload=action_planner_payload,
    )

    # 5. Explainable Confidence Scoring
    conf_res = compute_confidence_score(
        grounding=grounding_res,
        statistical=stat_res,
        contradictions=contra_res,
        retrieved_count=len(retrieved_evidence or []),
        query_expects_fresh=("2026" in (query or "").lower() or "today" in (query or "").lower()),
    )
    det_duration_ms = (time.time() - t_det_start) * 1000

    # 6. Preliminary Deterministic Verdict
    if (
        grounding_res.status == QualityStatus.BLOCKED
        or stat_res.status == QualityStatus.BLOCKED
        or rec_res.status == QualityStatus.BLOCKED
        or contra_res.status == QualityStatus.BLOCKED
    ):
        preliminary_verdict = QualityStatus.BLOCKED
    elif (
        grounding_res.status == QualityStatus.FLAGGED
        or stat_res.status == QualityStatus.FLAGGED
        or rec_res.status == QualityStatus.FLAGGED
        or contra_res.status == QualityStatus.FLAGGED
        or conf_res.score < 60.0
    ):
        preliminary_verdict = QualityStatus.FLAGGED
    else:
        preliminary_verdict = QualityStatus.PASSED

    # 7. LLM-as-Judge Evaluation (if not explicitly skipped)
    judge_verdict = None
    if not skip_llm_judge:
        judge_verdict = evaluate_with_llm_judge(
            query=query,
            target_business_id=target_business_id,
            retrieved_evidence=retrieved_evidence,
            classification_payload=classification_payload,
            root_cause_payload=root_cause_payload,
            action_planner_payload=action_planner_payload,
            deterministic_fallback_verdict=preliminary_verdict,
            llm_client=llm_client,
        )

    # 8. Final Consolidated Decision Synthesis
    # Hard rule: deterministic BLOCKED cannot be downgraded by LLM judge
    if preliminary_verdict == QualityStatus.BLOCKED:
        final_decision = QualityStatus.BLOCKED
        summary = "Blocked: Critical failure in data provenance, cross-business isolation, or evidence fabrication."
    elif preliminary_verdict == QualityStatus.FLAGGED:
        if judge_verdict and judge_verdict.verdict == QualityStatus.BLOCKED:
            final_decision = QualityStatus.BLOCKED
            summary = "Blocked: Evaluator detected critical ungrounded or contradictory findings."
        else:
            final_decision = QualityStatus.FLAGGED
            summary = "Flagged: Report contains warnings, unverified assumptions, or historical data staleness."
    else:
        if judge_verdict and judge_verdict.verdict == QualityStatus.BLOCKED:
            final_decision = QualityStatus.BLOCKED
            summary = "Blocked by quality evaluator."
        elif judge_verdict and judge_verdict.verdict == QualityStatus.FLAGGED:
            final_decision = QualityStatus.FLAGGED
            summary = "Flagged: Evaluator identified potential inconsistencies requiring review."
        else:
            final_decision = QualityStatus.PASSED
            summary = "Passed: All operational recommendations, source citations, and privacy boundaries verified."

    total_duration_ms = (time.time() - t_start) * 1000

    telemetry = {
        "deterministic_checks_ms": det_duration_ms,
        "llm_judge_ms": judge_verdict.latency_ms if judge_verdict else 0.0,
        "total_quality_gate_ms": total_duration_ms,
    }

    return QualityGateDecision(
        decision=final_decision,
        summary=summary,
        grounding=grounding_res,
        statistical=stat_res,
        recommendations=rec_res,
        contradictions=contra_res,
        confidence=conf_res,
        llm_judge=judge_verdict,
        telemetry_ms=telemetry,
    )
