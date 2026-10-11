"""Unified A2A Client Helper for interacting with the 4 Specialist Agents.
Connects over standard A2A protocol (JSON-RPC 2.0 & Agent Card resolution)
and provides high-level typed invocation methods.
"""

import os
from typing import Any, Dict, List, Optional
import httpx
from pydantic import BaseModel

from app.core.logging import logger
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


class A2AAgentClient:
    """Client for querying the four independently hosted A2A specialist agents."""

    def __init__(self, base_host: str = "localhost", ports: Optional[Dict[str, int]] = None):
        self.base_host = base_host
        self.ports = ports or DEFAULT_AGENT_PORTS
        self._http = httpx.Client(timeout=30.0)

    def get_agent_url(self, agent_name: str) -> str:
        port = self.ports.get(agent_name, 8000)
        return f"http://{self.base_host}:{port}"

    def get_agent_card(self, agent_name: str) -> Dict[str, Any]:
        """Fetches the Agent Card from the supported well-known endpoint."""
        url = f"{self.get_agent_url(agent_name)}/.well-known/agent-card.json"
        try:
            resp = self._http.get(url)
            if resp.status_code == 200:
                return resp.json()
        except Exception:
            pass

        # Try alias
        alias_url = f"{self.get_agent_url(agent_name)}/.well-known/agent.json"
        resp = self._http.get(alias_url)
        resp.raise_for_status()
        return resp.json()

    def send_a2a_jsonrpc(self, agent_name: str, message_text: str) -> Dict[str, Any]:
        """Sends an A2A JSON-RPC 2.0 message/send request to an agent."""
        url = self.get_agent_url(agent_name)
        payload = {
            "jsonrpc": "2.0",
            "id": "req_1",
            "method": "message/send",
            "params": {
                "message": {
                    "message_id": "msg_001",
                    "role": "user",
                    "parts": [{"text": message_text}]
                }
            }
        }
        resp = self._http.post(url, json=payload)
        resp.raise_for_status()
        return resp.json()

    # High-level typed agent invocations
    def classify_complaint(self, text: str, review_id: str = "rev_1", business_id: Optional[str] = None) -> AgentResponseEnvelope:
        """Invokes Classification Agent (Port 8001)."""
        url = f"{self.get_agent_url('classification')}/classify"
        req = ClassificationRequest(review_id=review_id, text=text, business_id=business_id)
        resp = self._http.post(url, json=req.model_dump())
        resp.raise_for_status()
        return AgentResponseEnvelope.model_validate(resp.json())

    def analyze_trends(self, business_id: Optional[str] = None, city: Optional[str] = None, horizon: int = 3) -> AgentResponseEnvelope:
        """Invokes Trend Analysis Agent (Port 8002)."""
        url = f"{self.get_agent_url('trends')}/trends"
        req = TrendAnalysisRequest(business_id=business_id, city=city, forecast_horizon=horizon)
        resp = self._http.post(url, json=req.model_dump())
        resp.raise_for_status()
        return AgentResponseEnvelope.model_validate(resp.json())

    def analyze_root_cause(
        self,
        problem_statement: str,
        business_id: Optional[str] = None,
        top_k: int = 5,
        classification_findings: Optional[Dict[str, Any]] = None,
        trend_findings: Optional[Dict[str, Any]] = None
    ) -> AgentResponseEnvelope:
        """Invokes Root-Cause Analysis Agent (Port 8003)."""
        url = f"{self.get_agent_url('root_cause')}/root-cause"
        req = RootCauseRequest(
            problem_statement=problem_statement,
            business_id=business_id,
            top_k_evidence=top_k,
            classification_findings=classification_findings,
            trend_findings=trend_findings
        )
        resp = self._http.post(url, json=req.model_dump())
        resp.raise_for_status()
        return AgentResponseEnvelope.model_validate(resp.json())

    def generate_action_plan(
        self,
        business_id: Optional[str] = None,
        problem_statement: Optional[str] = None,
        classification_findings: Optional[Dict[str, Any]] = None,
        trend_findings: Optional[Dict[str, Any]] = None,
        root_cause_findings: Optional[Dict[str, Any]] = None
    ) -> AgentResponseEnvelope:
        """Invokes Action Planner Agent (Port 8004)."""
        url = f"{self.get_agent_url('action_planner')}/action-plan"
        req = ActionPlannerRequest(
            business_id=business_id,
            problem_statement=problem_statement,
            classification_findings=classification_findings,
            trend_findings=trend_findings,
            root_cause_findings=root_cause_findings
        )
        resp = self._http.post(url, json=req.model_dump())
        resp.raise_for_status()
        return AgentResponseEnvelope.model_validate(resp.json())

    def close(self):
        self._http.close()


# Singleton client
_client: Optional[A2AAgentClient] = None

def get_a2a_client() -> A2AAgentClient:
    global _client
    if _client is None:
        _client = A2AAgentClient()
    return _client
