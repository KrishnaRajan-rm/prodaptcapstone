"""Phase 4 Cross-Agent Contradiction Detection."""

from typing import Any, Dict, List, Optional
from app.quality.schemas import ContradictionCheckResult, QualityStatus


def detect_contradictions(
    classification_payload: Optional[Dict[str, Any]],
    trend_payload: Optional[Dict[str, Any]],
    root_cause_payload: Optional[Dict[str, Any]],
    action_planner_payload: Optional[Dict[str, Any]],
) -> ContradictionCheckResult:
    """Identifies contradictory claims across specialist agent outputs."""
    result = ContradictionCheckResult()
    contradictions: List[str] = []

    # 1. Severity mismatch: Mild classification vs Critical action planner
    if classification_payload and action_planner_payload:
        class_sev = str(classification_payload.get("severity", "")).lower()
        actions = action_planner_payload.get("ranked_actions", [])
        if class_sev in ("low", "negligible", "minor") and actions:
            top_action = actions[0]
            priority = top_action.get("priority_score", 0.0)
            if priority >= 85.0:
                contradictions.append(
                    f"Contradiction: Classification severity is '{class_sev}', but Action Planner assigned "
                    f"critical priority score {priority:.1f} to top action without documented escalation."
                )

    # 2. Trend direction mismatch: Trend improving vs Root Cause claiming failure
    if trend_payload and (root_cause_payload or action_planner_payload):
        improving = [str(a).lower() for a in trend_payload.get("improving_aspects", [])]
        worsening = [str(a).lower() for a in trend_payload.get("worsening_aspects", [])]

        if root_cause_payload:
            for hyp in root_cause_payload.get("hypotheses", []):
                cause_lower = str(hyp.get("hypothesized_cause", "")).lower()
                for imp in improving:
                    if imp in cause_lower and "deteriorat" in cause_lower:
                        contradictions.append(
                            f"Contradiction: Trend Analysis reports '{imp}' is improving, but Root Cause asserts "
                            f"deteriorating failure for the same aspect."
                        )

    # 3. Sentiment vs Severity contradiction within classification
    if classification_payload:
        sent = str(classification_payload.get("sentiment", "")).lower()
        sev = str(classification_payload.get("severity", "")).lower()
        if sent == "positive" and sev in ("high", "critical"):
            contradictions.append(
                f"Contradiction: Classification reports 'positive' sentiment but '{sev}' severity."
            )

    if contradictions:
        result.has_contradictions = True
        result.contradictions = contradictions
        result.status = QualityStatus.FLAGGED

    return result
