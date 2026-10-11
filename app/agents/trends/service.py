"""Trend Analysis Agent A2A Service (Port 8002).
Serves the Google ADK Trend Analysis Agent over A2A protocol.
Provides:
  - Agent Card at `/.well-known/agent-card.json` (and `/.well-known/agent.json`, `/agent-card`)
  - A2A JSON-RPC 2.0 endpoint at `/`
  - REST endpoint at `/trends`
"""

import sys
import os
from pathlib import Path

# Ensure project root is in sys.path regardless of execution directory
root_dir = Path(__file__).resolve().parents[3]
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route
from google.adk.a2a.utils.agent_to_a2a import to_a2a

from app.core.logging import logger
from app.agents.trends.agent import TrendAnalysisAgent
from app.agents.schemas import TrendAnalysisRequest


AGENT_CARD_DATA = {
    "name": "trend_analysis_agent",
    "description": "Specialist Agent for deterministic historical trends, evaluated complaint forecasting, and customer satisfaction prediction.",
    "version": "1.0.0",
    "url": "http://localhost:8002/",
    "capabilities": {
        "historical_trend_analysis": True,
        "complaint_trend_prediction": True,
        "customer_satisfaction_prediction": True,
        "deterministic_sql_provenance": True,
        "chronological_evaluation": True,
    },
    "input_modes": ["application/json"],
    "output_modes": ["application/json"],
}


def create_trend_app(port: int = 8002, host: str = "localhost") -> Starlette:
    """Builds and configures the A2A Starlette service for Trend Analysis Agent."""
    trend_agent = TrendAnalysisAgent()

    # Base A2A application using Google ADK to_a2a
    base_app = to_a2a(
        trend_agent.adk_agent,
        host=host,
        port=port,
        protocol="http",
    )

    # REST endpoint for direct invocation
    async def trends_rest(request: Request) -> JSONResponse:
        try:
            body = await request.json()
            req = TrendAnalysisRequest.model_validate(body)
        except Exception as e:
            return JSONResponse({"status": "error", "error": str(e)}, status_code=400)
        try:
            result = trend_agent.analyze(req)
            return JSONResponse(result.model_dump())
        except Exception as e:
            logger.exception(f"Trend analysis execution failed: {e}")
            return JSONResponse({"status": "error", "error": str(e)}, status_code=500)

    # Agent Card endpoint aliases
    async def get_agent_card(request: Request) -> JSONResponse:
        return JSONResponse(AGENT_CARD_DATA)

    # Health check
    async def health(request: Request) -> JSONResponse:
        return JSONResponse({"status": "healthy", "agent": "trend_analysis_agent", "port": port})

    # Mount custom routes
    base_app.routes.append(Route("/trends", trends_rest, methods=["POST"]))
    base_app.routes.append(Route("/.well-known/agent.json", get_agent_card, methods=["GET", "HEAD"]))
    base_app.routes.append(Route("/agent-card", get_agent_card, methods=["GET", "HEAD"]))
    base_app.routes.append(Route("/health", health, methods=["GET"]))

    return base_app


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("TREND_AGENT_PORT", "8002"))
    app = create_trend_app(port=port)
    uvicorn.run(app, host="0.0.0.0", port=port)
