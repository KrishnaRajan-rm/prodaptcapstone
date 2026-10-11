import os
import sys
from pathlib import Path

# Add project root to sys.path
root_dir = Path(__file__).resolve().parents[2]
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

os.environ["MOCK_LLM"] = "1"

import pytest
from app.orchestration.schemas import OrchestrationRequest
from app.orchestration.graph import run_orchestration_workflow
from app.orchestration.a2a_async_client import AsyncA2AAgentClient
from app.agents.classification.service import create_classification_app
from app.agents.trends.service import create_trend_app
from app.agents.root_cause.service import create_root_cause_app
from app.agents.action_planner.service import create_action_planner_app


@pytest.mark.asyncio
async def test_full_orchestration_real_service_pipeline():
    """Tests complete LangGraph workflow against real specialist A2A Starlette apps."""
    # Build live Starlette applications
    app_registry = {
        "classification": create_classification_app(port=8001),
        "trends": create_trend_app(port=8002),
        "root_cause": create_root_cause_app(port=8003),
        "action_planner": create_action_planner_app(port=8004),
    }

    client = AsyncA2AAgentClient(app_registry=app_registry)

    req = OrchestrationRequest(
        query="The wait time for our entrees was over 45 minutes and the soup arrived cold.",
        business_id="PP3BBaVxZLcJU54uP_wL6Q",
        city="Philadelphia"
    )

    try:
        resp = await run_orchestration_workflow(req, client=client)

        assert resp.workflow_status in ("success", "partial_success")
        assert resp.total_duration_ms > 0

        # Check telemetry recorded for all 4 agents
        for ag in ("classification", "trends", "root_cause", "action_planner"):
            assert ag in resp.agent_timings
            assert resp.agent_timings[ag].duration_ms >= 0

        # Classification must have aspect and category
        if resp.classification:
            assert "categories" in resp.classification
            assert "aspects" in resp.classification

        # Trends must have analysis period
        if resp.trends:
            assert "analysis_period" in resp.trends

    finally:
        await client.aclose()


if __name__ == "__main__":
    import asyncio
    asyncio.run(test_full_orchestration_real_service_pipeline())
    print("Integration test passed successfully!")
