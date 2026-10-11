"""Service tests for all 4 A2A Specialist Agents (Ports 8001-8004)."""

import os
import pytest
from starlette.testclient import TestClient

os.environ["MOCK_LLM"] = "1"

from app.agents.classification.service import create_classification_app
from app.agents.trends.service import create_trend_app
from app.agents.root_cause.service import create_root_cause_app
from app.agents.action_planner.service import create_action_planner_app


def test_classification_service_a2a():
    app = create_classification_app(port=8001)
    with TestClient(app) as client:
        # Agent Card resolution
        r_card = client.get("/.well-known/agent-card.json")
        assert r_card.status_code == 200
        assert r_card.json()["name"] == "classification_agent"

        # Health endpoint
        r_health = client.get("/health")
        assert r_health.status_code == 200
        assert r_health.json()["status"] == "healthy"

        # Classification execution
        r_post = client.post("/classify", json={"text": "The waiter was extremely rude and ignored us for 30 minutes."})
        assert r_post.status_code == 200
        data = r_post.json()
        assert data["agent_id"] == "classification_agent"
        assert data["status"] == "success"
        assert "Staff behaviour" in data["payload"]["categories"]


def test_trend_service_a2a():
    app = create_trend_app(port=8002)
    with TestClient(app) as client:
        # Agent Card resolution
        r_card = client.get("/.well-known/agent-card.json")
        assert r_card.status_code == 200
        assert r_card.json()["name"] == "trend_analysis_agent"

        # Health endpoint
        r_health = client.get("/health")
        assert r_health.status_code == 200

        # Trends execution
        r_post = client.post("/trends", json={"business_id": "PP3BBaVxZLcJU54uP_wL6Q"})
        assert r_post.status_code == 200
        data = r_post.json()
        assert data["agent_id"] == "trend_agent"
        assert data["payload"]["total_reviews_analyzed"] > 0


def test_root_cause_service_a2a():
    app = create_root_cause_app(port=8003)
    with TestClient(app) as client:
        # Agent Card resolution
        r_card = client.get("/.well-known/agent-card.json")
        assert r_card.status_code == 200
        assert r_card.json()["name"] == "root_cause_agent"

        # Health endpoint
        r_health = client.get("/health")
        assert r_health.status_code == 200

        # Root-cause execution
        r_post = client.post("/root-cause", json={"problem_statement": "Cold food delivery", "top_k_evidence": 2})
        assert r_post.status_code == 200
        data = r_post.json()
        assert data["agent_id"] == "root_cause_agent"
        assert len(data["payload"]["hypotheses"]) > 0


def test_action_planner_service_a2a():
    app = create_action_planner_app(port=8004)
    with TestClient(app) as client:
        # Agent Card resolution
        r_card = client.get("/.well-known/agent-card.json")
        assert r_card.status_code == 200
        assert r_card.json()["name"] == "action_planner_agent"

        # Health endpoint
        r_health = client.get("/health")
        assert r_health.status_code == 200

        # Action-plan execution
        r_post = client.post("/action-plan", json={"problem_statement": "Kitchen expediting delay"})
        assert r_post.status_code == 200
        data = r_post.json()
        assert data["agent_id"] == "action_planner_agent"
        assert len(data["payload"]["ranked_actions"]) > 0


def test_agent_independence():
    """Verify each service is isolated and can instantiate without others running."""
    app1 = create_classification_app(port=8001)
    app4 = create_action_planner_app(port=8004)
    with TestClient(app1) as c1, TestClient(app4) as c4:
        assert c1.get("/health").status_code == 200
        assert c4.get("/health").status_code == 200
