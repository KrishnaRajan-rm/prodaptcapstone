"""Classification Agent A2A Service (Port 8001).
Serves the Google ADK Classification Agent over A2A protocol.
Provides:
  - Agent Card at `/.well-known/agent-card.json` (and `/.well-known/agent.json`, `/agent-card`)
  - A2A JSON-RPC 2.0 endpoint at `/`
  - REST endpoint at `/classify`
"""

import sys
import os
import json
from pathlib import Path

# Ensure project root is in sys.path regardless of execution directory
root_dir = Path(__file__).resolve().parents[3]
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))
from typing import Optional
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.routing import Route
from google.adk.a2a.utils.agent_to_a2a import to_a2a
from a2a.helpers.agent_card import AgentCard

from app.core.logging import logger
from app.agents.classification.agent import ClassificationAgent
from app.agents.schemas import ClassificationRequest


AGENT_CARD_DATA = {
    "name": "classification_agent",
    "description": "Specialist Agent for restaurant review complaint classification, aspect tagging, sentiment, and severity assessment with grounded evidence.",
    "version": "1.0.0",
    "url": "http://localhost:8001/",
    "capabilities": {
        "complaint_classification": True,
        "aspect_analysis": True,
        "sentiment_analysis": True,
        "severity_assessment": True,
        "evidence_provenance": True,
    },
    "input_modes": ["application/json", "text/plain"],
    "output_modes": ["application/json"],
}


def create_classification_app(port: int = 8001, host: str = "localhost") -> Starlette:
    """Builds and configures the A2A Starlette service for Classification Agent."""
    classifier = ClassificationAgent()

    # Base A2A application using Google ADK to_a2a
    base_app = to_a2a(
        classifier.adk_agent,
        host=host,
        port=port,
        protocol="http",
    )

    # REST endpoint for direct invocation
    async def classify_rest(request: Request) -> JSONResponse:
        try:
            body = await request.json()
            req = ClassificationRequest.model_validate(body)
            result = classifier.analyze(req)
            return JSONResponse(result.model_dump())
        except Exception as e:
            logger.error(f"Error in classify REST endpoint: {e}")
            return JSONResponse({"status": "error", "error": str(e)}, status_code=400)

    # Agent Card endpoint aliases
    async def get_agent_card(request: Request) -> JSONResponse:
        return JSONResponse(AGENT_CARD_DATA)

    # Health check
    async def health(request: Request) -> JSONResponse:
        return JSONResponse({"status": "healthy", "agent": "classification_agent", "port": port})

    # Mount custom routes
    base_app.routes.append(Route("/classify", classify_rest, methods=["POST"]))
    base_app.routes.append(Route("/.well-known/agent.json", get_agent_card, methods=["GET", "HEAD"]))
    base_app.routes.append(Route("/agent-card", get_agent_card, methods=["GET", "HEAD"]))
    base_app.routes.append(Route("/health", health, methods=["GET"]))

    return base_app


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("CLASSIFICATION_AGENT_PORT", "8001"))
    app = create_classification_app(port=port)
    uvicorn.run(app, host="0.0.0.0", port=port)
