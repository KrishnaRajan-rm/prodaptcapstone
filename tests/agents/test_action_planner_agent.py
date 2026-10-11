"""Unit tests for Agent 4 — Action Planner Agent (Port 8004)."""

import os
import pytest

os.environ["MOCK_LLM"] = "1"

from app.agents.action_planner.agent import ActionPlannerAgent
from app.agents.schemas import ActionPlannerRequest


@pytest.fixture
def planner():
    return ActionPlannerAgent()


def test_priority_scoring_formula_and_ranking(planner):
    req = ActionPlannerRequest(
        problem_statement="Severe table wait times and lukewarm food arrival"
    )
    res = planner.analyze(req)
    assert res.status == "success"
    actions = res.payload["ranked_actions"]
    assert len(actions) > 0

    # Verify ranking monotonicity: priority_score of rank 1 >= rank 2
    for i in range(len(actions) - 1):
        assert actions[i]["priority_score"] >= actions[i + 1]["priority_score"]
        assert actions[i]["rank"] == i + 1

    # Verify score calculation formula integrity
    first = actions[0]
    sc = first["scoring_details"]
    expected_comp = round(100.0 * (
        0.35 * sc["frequency_score"] +
        0.25 * sc["severity_score"] +
        0.20 * sc["trend_score"] +
        0.20 * sc["rating_impact_score"]
    ), 1)
    assert abs(first["priority_score"] - expected_comp) < 0.2


def test_actionable_recommendations_structure(planner):
    req = ActionPlannerRequest(
        problem_statement="Dining room cleanliness and sanitization compliance"
    )
    res = planner.analyze(req)
    assert res.status == "success"
    actions = res.payload["ranked_actions"]
    assert len(actions) > 0

    first = actions[0]
    assert len(first["owner_role"]) > 0
    assert len(first["measurable_kpi"]) > 0
    assert len(first["target_value"]) > 0
    assert first["is_proposed_target"] is True
    assert len(first["dependencies"]) > 0
    assert len(first["verification_method"]) > 0


def test_ranking_consistency(planner):
    req = ActionPlannerRequest(problem_statement="Food temperature inconsistency")
    res1 = planner.analyze(req)
    res2 = planner.analyze(req)
    assert res1.payload["ranked_actions"][0]["priority_score"] == res2.payload["ranked_actions"][0]["priority_score"]
    assert res1.payload["ranked_actions"][0]["issue_summary"] == res2.payload["ranked_actions"][0]["issue_summary"]
