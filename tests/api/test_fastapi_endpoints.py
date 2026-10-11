"""Tests for FastAPI Gateway Endpoints."""

import pytest
from fastapi.testclient import TestClient
from app.main import app
from tests.orchestration.test_langgraph_workflow import MockAsyncA2AClient


@pytest.fixture
def client(monkeypatch):
    """Test client with mocked A2A client to avoid requiring running microservices."""
    mock_a2a = MockAsyncA2AClient()
    monkeypatch.setattr("app.api.routes.get_a2a_client", lambda: mock_a2a)
    with TestClient(app) as test_client:
        yield test_client


def test_health_check(client):
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "service" in data


def test_list_scenarios(client):
    response = client.get("/api/v1/scenarios")
    assert response.status_code == 200
    data = response.json()
    assert "scenarios" in data
    assert len(data["scenarios"]) >= 6
    scenario = data["scenarios"][0]
    assert "title" in scenario
    assert "sample_query" in scenario
    assert "mode" in scenario


def test_agents_status(client):
    response = client.get("/api/v1/agents/status")
    assert response.status_code == 200
    data = response.json()
    assert "agents" in data
    assert len(data["agents"]) == 4
    for key in ("classification", "trends", "root_cause", "action_planner"):
        assert key in data["agents"]
        assert "status" in data["agents"][key]


def test_feedback_submission_and_summary(client):
    fb_payload = {
        "request_id": "req-fb-test-123",
        "rating": 5,
        "useful": True,
        "flags": "none",
        "comment": "Accurate root cause and actionable KPIs.",
        "role": "manager",
    }
    resp = client.post("/api/v1/feedback", json=fb_payload)
    assert resp.status_code == 200
    res_data = resp.json()
    assert res_data["status"] == "recorded"
    assert "feedback_id" in res_data

    # Check summary
    summary_resp = client.get("/api/v1/feedback/summary")
    assert summary_resp.status_code == 200
    sum_data = summary_resp.json()
    assert sum_data["total_feedback"] >= 1
    assert sum_data["average_rating"] >= 1.0


def test_analysis_auto_mode(client):
    req_payload = {
        "query": "Investigate chronic dining delays and recommend operational fixes",
        "mode": "auto",
        "business_id": "biz_demo",
        "viewer_role": "manager",
        "skip_llm_judge": True,
    }
    resp = client.post("/api/v1/analysis", json=req_payload)
    assert resp.status_code == 200
    data = resp.json()

    assert data["request_id"].startswith("req_")
    assert data["workflow_status"] in ("success", "partial_success")
    assert data["mode"] == "auto"
    assert data["executive_summary"] is not None
    assert data["crew_report"] is not None
    assert data["quality_decision"] is not None
    assert data["confidence_score"] is not None
    assert len(data["execution_trace"]) >= 4

    # Verify persistence by reading back
    req_id = data["request_id"]
    get_resp = client.get(f"/api/v1/analysis/{req_id}")
    assert get_resp.status_code == 200
    persisted = get_resp.json()
    assert persisted["request_id"] == req_id


def test_analysis_single_agent_mode(client):
    req_payload = {
        "query": "Classify severe dining complaint regarding food temperature",
        "mode": "agent",
        "agent": "classification",
        "business_id": "biz_demo",
    }
    resp = client.post("/api/v1/analysis", json=req_payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["mode"] == "agent"
    assert data["selected_agent"] == "classification"
    assert data["classification"] is not None
    assert data["classification"].get("categories")
    assert data["classification"].get("aspects")
    assert "Single Agent Execution" not in data["executive_summary"]
    assert data["trends"] is None
    assert len(data["execution_trace"]) == 1


def test_evaluation_summary(client):
    resp = client.get("/api/v1/evaluation/summary")
    assert resp.status_code == 200
    data = resp.json()
    assert data["benchmark_name"] is not None
    assert data["faithfulness_score"] is not None
    assert data["relevancy_score"] is not None
    assert data["passed"] is True
