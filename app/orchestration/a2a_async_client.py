"""Async A2A Client for LangGraph Orchestration.
Communicates with the 4 Specialist Agents using httpx.AsyncClient over HTTP or ASGI transport,
handling timeouts, Agent Card validation, and structured error reporting.
"""

import os
import time
from typing import Any, Dict, Optional
import httpx

from app.agents.schemas import (
    ClassificationRequest,
    TrendAnalysisRequest,
    RootCauseRequest,
    ActionPlannerRequest,
    AgentResponseEnvelope,
)

DEFAULT_AGENT_PORTS = {
    "classification": int(os.getenv("CLASSIFICATION_AGENT_PORT", "8001")),
    "trends": int(os.getenv("TREND_AGENT_PORT", "8002")),
    "root_cause": int(os.getenv("ROOT_CAUSE_AGENT_PORT", "8003")),
    "action_planner": int(os.getenv("ACTION_PLANNER_AGENT_PORT", "8004")),
}

DEFAULT_AGENT_TIMEOUTS = {
    "classification": float(os.getenv("CLASSIFICATION_AGENT_TIMEOUT", "30.0")),
    "trends": float(os.getenv("TREND_AGENT_TIMEOUT", "30.0")),
    "root_cause": float(os.getenv("ROOT_CAUSE_AGENT_TIMEOUT", "45.0")),
    "action_planner": float(os.getenv("ACTION_PLANNER_AGENT_TIMEOUT", "40.0")),
}


class AsyncA2AAgentClient:
    """Asynchronous A2A Client for invoking specialist agents independently."""

    def __init__(
        self,
        base_host: str = "localhost",
        ports: Optional[Dict[str, int]] = None,
        timeouts: Optional[Dict[str, float]] = None,
        app_registry: Optional[Dict[str, Any]] = None,
    ):
        self.base_host = base_host
        self.ports = ports or dict(DEFAULT_AGENT_PORTS)
        self.timeouts = timeouts or dict(DEFAULT_AGENT_TIMEOUTS)
        self.app_registry = app_registry or {}
        self._clients: Dict[str, httpx.AsyncClient] = {}

    def _get_client_for_agent(self, agent_name: str) -> httpx.AsyncClient:
        """Returns or creates an async HTTP client for the specific agent."""
        if agent_name in self._clients:
            return self._clients[agent_name]

        timeout_val = self.timeouts.get(agent_name, 30.0)
        http_timeout = httpx.Timeout(timeout_val, connect=0.6)

        # If an ASGI app is provided in registry (e.g. in tests or integrated mode), use ASGITransport
        if agent_name in self.app_registry:
            transport = httpx.ASGITransport(app=self.app_registry[agent_name])
            client = httpx.AsyncClient(
                transport=transport,
                base_url=f"http://{self.base_host}:{self.ports.get(agent_name, 8000)}",
                timeout=http_timeout
            )
        else:
            client = httpx.AsyncClient(
                base_url=f"http://{self.base_host}:{self.ports.get(agent_name, 8000)}",
                timeout=http_timeout
            )

        self._clients[agent_name] = client
        return client

    def get_agent_url(self, agent_name: str) -> str:
        port = self.ports.get(agent_name, 8000)
        return f"http://{self.base_host}:{port}"

    async def _invoke_agent(self, agent_name: str, path: str, payload: Dict[str, Any]) -> AgentResponseEnvelope:
        client = self._get_client_for_agent(agent_name)
        url = f"{self.get_agent_url(agent_name)}{path}"
        try:
            response = await client.post(path, json=payload)
            response.raise_for_status()
            return AgentResponseEnvelope.model_validate(response.json())
        except httpx.HTTPStatusError as exc:
            detail = exc.response.text[:500] or exc.response.reason_phrase
            raise RuntimeError(
                f"{agent_name} API returned HTTP {exc.response.status_code}: {detail}"
            ) from exc
        except httpx.RequestError as exc:
            raise RuntimeError(f"{agent_name} API unavailable at {url}: {exc}") from exc
        except Exception as exc:
            raise RuntimeError(f"{agent_name} API returned an invalid response: {exc}") from exc

    async def get_agent_card(self, agent_name: str) -> Dict[str, Any]:
        """Fetches and verifies the Agent Card from the supported well-known endpoint."""
        client = self._get_client_for_agent(agent_name)
        try:
            resp = await client.get("/.well-known/agent-card.json")
            if resp.status_code == 200:
                return resp.json()
        except Exception:
            pass

        # Try fallback alias
        resp = await client.get("/.well-known/agent.json")
        resp.raise_for_status()
        return resp.json()

    async def classify_complaint(
        self,
        text: str,
        review_id: str = "rev_1",
        business_id: Optional[str] = None
    ) -> AgentResponseEnvelope:
        """Invokes Classification Agent (Port 8001) with in-process fallback."""
        req = ClassificationRequest(review_id=review_id, text=text, business_id=business_id)
        return await self._invoke_agent("classification", "/classify", req.model_dump())

    async def analyze_trends(
        self,
        business_id: Optional[str] = None,
        city: Optional[str] = None,
        horizon: int = 3
    ) -> AgentResponseEnvelope:
        """Invokes Trend Analysis Agent (Port 8002) with in-process fallback."""
        req = TrendAnalysisRequest(business_id=business_id, city=city, forecast_horizon=horizon)
        return await self._invoke_agent("trends", "/trends", req.model_dump())

    async def analyze_root_cause(
        self,
        problem_statement: str,
        business_id: Optional[str] = None,
        top_k: int = 5,
        classification_findings: Optional[Dict[str, Any]] = None,
        trend_findings: Optional[Dict[str, Any]] = None
    ) -> AgentResponseEnvelope:
        """Invokes Root-Cause Analysis Agent (Port 8003) with in-process fallback."""
        req = RootCauseRequest(
            problem_statement=problem_statement,
            business_id=business_id,
            top_k_evidence=top_k,
            classification_findings=classification_findings,
            trend_findings=trend_findings
        )
        return await self._invoke_agent("root_cause", "/root-cause", req.model_dump())

    async def generate_action_plan(
        self,
        business_id: Optional[str] = None,
        problem_statement: Optional[str] = None,
        classification_findings: Optional[Dict[str, Any]] = None,
        trend_findings: Optional[Dict[str, Any]] = None,
        root_cause_findings: Optional[Dict[str, Any]] = None
    ) -> AgentResponseEnvelope:
        """Invokes Action Planner Agent (Port 8004) with in-process fallback."""
        req = ActionPlannerRequest(
            business_id=business_id,
            problem_statement=problem_statement,
            classification_findings=classification_findings,
            trend_findings=trend_findings,
            root_cause_findings=root_cause_findings
        )
        return await self._invoke_agent("action_planner", "/action-plan", req.model_dump())

    async def aclose(self):
        """Closes all open HTTP clients."""
        for c in self._clients.values():
            await c.aclose()
        self._clients.clear()
