"""Phase 4 Recommendation Validation for Action Planner Outputs."""

import re
from typing import Any, Dict, List, Optional, Set
from app.quality.schemas import QualityStatus, RecommendationCheckResult

# Reasonable operational role terms (case-insensitive substring or token check)
STANDARD_ROLE_TERMS: Set[str] = {
    "manager", "lead", "supervisor", "director", "head", "chef", "cook",
    "expediter", "server", "host", "bartender", "staff", "crew", "team",
    "operations", "operator", "qa", "trainer", "officer"
}

# Regex for detecting specific individual person names (e.g. "John Smith", "Dave", "Mary Jane")
PERSON_NAME_PATTERN = re.compile(r'^[A-Z][a-z]+\s+[A-Z][a-z]+$')


def validate_recommendations(
    action_planner_payload: Optional[Dict[str, Any]],
    root_cause_payload: Optional[Dict[str, Any]],
    classification_payload: Optional[Dict[str, Any]],
) -> RecommendationCheckResult:
    """Validates operational roles, priority alignment, and problem mapping for action plans."""
    result = RecommendationCheckResult()

    if not action_planner_payload or not isinstance(action_planner_payload, dict):
        return result

    ranked_actions = action_planner_payload.get("ranked_actions", [])
    if not ranked_actions:
        return result

    # Gather known problems from upstream agents
    known_aspects = set()
    if classification_payload and isinstance(classification_payload, dict):
        for asp in classification_payload.get("aspects", []) or []:
            known_aspects.add(str(asp).lower())
        for cat in classification_payload.get("categories", []) or []:
            known_aspects.add(str(cat).lower())

    known_hypotheses = set()
    if root_cause_payload and isinstance(root_cause_payload, dict):
        for hyp in root_cause_payload.get("hypotheses", []) or []:
            cause = hyp.get("hypothesized_cause", "")
            if cause:
                known_hypotheses.add(str(cause).lower())

    for action in ranked_actions:
        owner = str(action.get("owner_role", "")).strip()
        summary = str(action.get("issue_summary", "")).strip()
        rec = str(action.get("recommended_action", "")).strip()
        target = str(action.get("target_value", "")).strip()

        # 1. Role validation: Must be an operational role, not a named person
        owner_lower = owner.lower()
        has_role_term = any(term in owner_lower for term in STANDARD_ROLE_TERMS)
        is_named_person = bool(PERSON_NAME_PATTERN.match(owner)) and not has_role_term

        if is_named_person or not has_role_term:
            result.owner_roles_valid = False
            result.invalid_owners.append(f"Invalid named employee or non-operational role '{owner}' for action '{rec[:30]}'")
            result.status = QualityStatus.FLAGGED

        # 2. Check problem mapping
        if known_aspects or known_hypotheses:
            summary_lower = summary.lower()
            rec_lower = rec.lower()
            matches_known = any(asp in summary_lower or asp in rec_lower for asp in known_aspects) or \
                            any(hyp in summary_lower or hyp in rec_lower for hyp in known_hypotheses)
            # Check basic keyword overlap if direct match fails
            if not matches_known:
                words = set(re.findall(r'\b\w{4,}\b', summary_lower))
                upstream_text = " ".join(known_aspects | known_hypotheses)
                if not any(w in upstream_text for w in words):
                    result.unsupported_assumptions.append(
                        f"Action '{rec[:40]}' references issue '{summary}' not identified in upstream findings."
                    )
                    result.status = QualityStatus.FLAGGED

        # 3. Check target labeling
        if target and not any(tag in target.lower() for tag in ["proposed", "<", ">", "%", "target", "min", "max", "baseline"]):
            result.targets_labeled_proposed = False
            result.unsupported_assumptions.append(f"Target '{target}' is presented as absolute rather than proposed.")
            result.status = QualityStatus.FLAGGED

    return result
