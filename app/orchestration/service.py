"""Orchestrator Service (Port 8000).
Exposes the LangGraph multi-agent orchestration pipeline over HTTP REST endpoints.
Endpoints:
  - POST /orchestrate: Executes full end-to-end multi-agent workflow
  - GET  /graph/mermaid: Returns Mermaid diagram of graph topology
  - GET  /health: Service liveness probe
  - GET  /agent-cards: Inspects availability and metadata of upstream specialist agents
"""

import sys
import os
from pathlib import Path

# Ensure project root is in sys.path
root_dir = Path(__file__).resolve().parents[2]
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from typing import Any, Dict, Optional
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, PlainTextResponse
from starlette.routing import Route

from app.core.logging import logger
from app.orchestration.schemas import (
    AggregatedWorkflowResponse,
    OrchestrationRequest,
)
from app.orchestration.graph import (
    build_orchestration_graph,
    run_orchestration_workflow,
)
from app.orchestration.a2a_async_client import AsyncA2AAgentClient


def create_orchestrator_app(
    client: Optional[AsyncA2AAgentClient] = None,
    host: str = "localhost",
    port: int = 8000,
) -> Starlette:
    """Creates Starlette application hosting the LangGraph workflow."""
    a2a_client = client or AsyncA2AAgentClient(base_host=host)
    compiled_graph = build_orchestration_graph(client=a2a_client)

    async def orchestrate_handler(request: Request) -> JSONResponse:
        try:
            body = await request.json()
            req = OrchestrationRequest.model_validate(body)
            response = await run_orchestration_workflow(
                request=req,
                client=a2a_client,
                graph=compiled_graph
            )
            return JSONResponse(response.model_dump())
        except Exception as e:
            logger.error(f"Error in /orchestrate: {e}")
            return JSONResponse({
                "workflow_status": "failed",
                "error": str(e),
            }, status_code=400)

    async def mermaid_handler(request: Request) -> PlainTextResponse:
        """Returns Mermaid flowchart diagram of the compiled LangGraph."""
        try:
            diagram = compiled_graph.get_graph().draw_mermaid()
            return PlainTextResponse(diagram)
        except Exception as e:
            return PlainTextResponse(f"Error generating mermaid diagram: {e}", status_code=500)

    async def agent_cards_handler(request: Request) -> JSONResponse:
        """Checks upstream A2A connectivity and fetches Agent Cards."""
        results = {}
        for ag in ("classification", "trends", "root_cause", "action_planner"):
            try:
                card = await a2a_client.get_agent_card(ag)
                results[ag] = {"status": "available", "card": card}
            except Exception as e:
                results[ag] = {"status": "unavailable", "error": str(e)}
        return JSONResponse(results)

    async def health_handler(request: Request) -> JSONResponse:
        return JSONResponse({
            "status": "healthy",
            "service": "langgraph_orchestrator",
            "port": port,
        })

    routes = [
        Route("/orchestrate", orchestrate_handler, methods=["POST"]),
        Route("/graph/mermaid", mermaid_handler, methods=["GET"]),
        Route("/agent-cards", agent_cards_handler, methods=["GET"]),
        Route("/health", health_handler, methods=["GET"]),
    ]

    app = Starlette(routes=routes)
    return app


def main():
    """Runs the LangGraph orchestrator service via Uvicorn."""
    import uvicorn
    port = int(os.getenv("ORCHESTRATOR_PORT", "8000"))
    host = os.getenv("ORCHESTRATOR_HOST", "0.0.0.0")
    print(f"Starting LangGraph Orchestrator Service on http://{host}:{port}...")
    app = create_orchestrator_app(port=port)
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
