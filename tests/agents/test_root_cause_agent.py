"""Unit tests for Agent 3 — Root-Cause Analysis Agent (Port 8003)."""

import os
import pytest

os.environ["MOCK_LLM"] = "1"

from app.agents.root_cause.agent import RootCauseAgent
from app.agents.schemas import RootCauseRequest, DistinctionType


@pytest.fixture
def root_cause_agent():
    return RootCauseAgent()


def test_grounded_hypothesis_generation(root_cause_agent):
    req = RootCauseRequest(
        problem_statement="Customers repeatedly complaining about slow service and 45-minute wait times",
        top_k_evidence=3
    )
    res = root_cause_agent.analyze(req)
    assert res.status == "success"
    payload = res.payload
    assert len(payload["hypotheses"]) > 0

    hyp = payload["hypotheses"][0]
    assert len(hyp["hypothesized_cause"]) > 0
    assert hyp["distinction_type"] in [d.value for d in DistinctionType]
    assert len(hyp["supporting_evidence"]) > 0
    assert len(hyp["supporting_source_ids"]) > 0
    assert len(hyp["additional_info_needed"]) > 0
    assert len(res.evidence) > 0


def test_epistemic_distinction_integrity(root_cause_agent):
    req = RootCauseRequest(
        problem_statement="Food arriving cold at tables during evening dinner rush",
        top_k_evidence=3
    )
    res = root_cause_agent.analyze(req)
    assert res.status == "success"
    hyp = res.payload["hypotheses"][0]
    assert hyp["distinction_type"] == DistinctionType.INFERRED_EXPLANATION.value
    assert len(hyp["alternative_explanations"]) > 0


def test_empty_problem_statement(root_cause_agent):
    req = RootCauseRequest(problem_statement="")
    res = root_cause_agent.analyze(req)
    assert res.status == "insufficient_data"
    assert len(res.warnings) > 0
