"""Root-Cause Analysis Agent A2A Service (Port 8003).
Serves the Google ADK Root-Cause Analysis Agent over A2A protocol.
Provides:
  - Agent Card at `/.well-known/agent-card.json` (and `/.well-known/agent.json`, `/agent-card`)
  - A2A JSON-RPC 2.0 endpoint at `/`
  - REST endpoint at `/root-cause`
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
from app.agents.root_cause.agent import RootCauseAgent
from app.agents.schemas import RootCauseRequest


AGENT_CARD_DATA = {
    "name": "root_cause_agent",
    "description": "Specialist Agent for investigating restaurant operational failures using Hybrid RAG evidence and grounded root-cause hypotheses.",
    "version": "1.0.0",
    "url": "http://localhost:8003/",
    "capabilities": {
        "hybrid_rag_integration": True,
        "root_cause_hypotheses": True,
        "epistemic_distinctions": True,
        "contradictory_evidence_detection": True,
        "evidence_provenance": True,
    },
    "input_modes": ["application/json"],
    "output_modes": ["application/json"],
}


def create_root_cause_app(port: int = 8003, host: str = "localhost") -> Starlette:
    """Builds and configures the A2A Starlette service for Root-Cause Analysis Agent."""
    rc_agent = RootCauseAgent()

    # Base A2A application using Google ADK to_a2a
    base_app = to_a2a(
        rc_agent.adk_agent,
        host=host,
        port=port,
        protocol="http",
    )

    # REST endpoint for direct invocation
    async def root_cause_rest(request: Request) -> JSONResponse:
        try:
            body = await request.json()
            req = RootCauseRequest.model_validate(body)
        except Exception as e:
            return JSONResponse({"status": "error", "error": str(e)}, status_code=400)
        try:
            result = rc_agent.analyze(req)
            return JSONResponse(result.model_dump())
        except Exception as e:
            logger.exception(f"Root-cause execution failed: {e}")
            return JSONResponse({"status": "error", "error": str(e)}, status_code=500)

    # Agent Card endpoint aliases
    async def get_agent_card(request: Request) -> JSONResponse:
        return JSONResponse(AGENT_CARD_DATA)

    # Health check
    async def health(request: Request) -> JSONResponse:
        return JSONResponse({"status": "healthy", "agent": "root_cause_agent", "port": port})

    # Mount custom routes
    base_app.routes.append(Route("/root-cause", root_cause_rest, methods=["POST"]))
    base_app.routes.append(Route("/.well-known/agent.json", get_agent_card, methods=["GET", "HEAD"]))
    base_app.routes.append(Route("/agent-card", get_agent_card, methods=["GET", "HEAD"]))
    base_app.routes.append(Route("/health", health, methods=["GET"]))

    return base_app


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("ROOT_CAUSE_AGENT_PORT", "8003"))
    app = create_root_cause_app(port=port)
    uvicorn.run(app, host="0.0.0.0", port=port)
