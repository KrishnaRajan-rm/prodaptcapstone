"""LangGraph nodes for the Restaurant Operations Orchestrator workflow with Phase 4 Quality & Privacy Gates."""

import time
from typing import Any, Dict, List, Optional
import httpx

from app.core.logging import logger
from app.core.pii.gate2 import TokenVault, AUTHORIZED_ROLES
from app.quality.gate import evaluate_quality_gate
from app.quality.schemas import QualityStatus
from app.crew.consolidation import run_crewai_draft_and_validate
from app.orchestration.schemas import (
    AgentTimingInfo,
    AggregatedWorkflowResponse,
    OrchestrationRequest,
)
from app.orchestration.state import OrchestrationState
from app.orchestration.a2a_async_client import AsyncA2AAgentClient


def create_orchestration_nodes(client: Optional[AsyncA2AAgentClient] = None):
    """Factory that returns configured node functions with the given A2A client bound."""
    a2a_client = client or AsyncA2AAgentClient()

    async def validate_request_node(state: OrchestrationState) -> Dict[str, Any]:
        """Node A: Validates incoming request parameters and filtering context."""
        query = state.get("query", "")
        if not query or not query.strip() or len(query.strip()) < 2:
            err = {
                "node": "validate_request",
                "error": "Query cannot be empty or fewer than 2 characters.",
                "message": "Validation failed"
            }
            return {
                "workflow_status": "failed",
                "errors": [err],
                "warnings": ["Request rejected at input validation stage."]
            }

        # Check date formats if present
        for date_key in ("start_date", "end_date"):
            val = state.get(date_key)
            if val:
                import re
                if not re.match(r"^\d{4}-\d{2}(-\d{2})?$", val):
                    err = {
                        "node": "validate_request",
                        "error": f"Invalid {date_key} format '{val}'. Expected YYYY-MM or YYYY-MM-DD.",
                        "message": "Validation failed"
                    }
                    return {
                        "workflow_status": "failed",
                        "errors": [err],
                        "warnings": [f"Invalid date filter: {date_key}"]
                    }

        return {}

    async def retrieve_context_node(state: OrchestrationState) -> Dict[str, Any]:
        """Node B: Reuses existing Hybrid RAG retriever to gather evidence chunks."""
        existing_evidence = state.get("retrieved_evidence")
        if existing_evidence and len(existing_evidence) > 0:
            return {}

        filters = dict(state.get("filters", {}))
        if "preloaded_evidence" in filters and filters["preloaded_evidence"]:
            return {"retrieved_evidence": filters["preloaded_evidence"]}

        # If a mock client is active, return deterministic synthetic evidence
        if type(a2a_client).__name__.startswith("Mock"):
            return {
                "retrieved_evidence": [
                    {
                        "chunk_id": "chk_101",
                        "review_id": "rev_mock_1",
                        "business_id": state.get("business_id") or "BIZ_MOCK",
                        "restaurant_name": "Test Restaurant",
                        "city": state.get("city") or "Philadelphia",
                        "state": "PA",
                        "stars": 3.0,
                        "text_masked": "Orders backed up for 40 minutes and soup was cold.",
                        "aspects": "Waiting time, Food temperature",
                        "score": 0.85,
                    }
                ]
            }

        query = state["query"]
        top_k = state.get("top_k", 5)
        filters = dict(state.get("filters", {}))

        if state.get("business_id"):
            filters["business_id"] = state["business_id"]
        if state.get("city"):
            filters["city"] = state["city"]

        try:
            from app.retrieval.hybrid import get_retriever
            retriever = get_retriever()
            search_results = retriever.search(
                query=query,
                top_k=top_k,
                filters=filters if filters else None,
                use_reranker=True,
            )
            evidence_records = []
            for r in search_results:
                evidence_records.append({
                    "chunk_id": r.chunk_id,
                    "review_id": r.review_id,
                    "business_id": r.business_id,
                    "restaurant_name": r.restaurant_name,
                    "city": r.city,
                    "state": r.state,
                    "stars": r.stars,
                    "text_masked": r.text_masked,
                    "aspects": r.aspects,
                    "score": r.rerank_score if r.rerank_score is not None else r.rrf_score,
                })
            return {"retrieved_evidence": evidence_records}
        except Exception as e:
            logger.warning(f"Retrieval node encountered issue: {e}")
            return {
                "retrieved_evidence": [],
                "warnings": [f"Hybrid retrieval unavailable or yielded no results: {e}"]
            }

    async def pii_gate_2_node(state: OrchestrationState) -> Dict[str, Any]:
        """Node C0: PII Gate 2 pre-LLM reversible redaction and leak detection."""
        req_id = state.get("request_id", "default_req")
        vault = TokenVault(request_id=req_id)

        # 1. Mask query
        raw_query = state.get("query", "")
        masked_query = vault.mask_text(raw_query)

        # 2. Check for residual unmasked critical PII (fail closed)
        leaks_found, leak_types = vault.has_residual_pii(masked_query)
        if leaks_found:
            logger.error(f"PII Gate 2 blocked request {req_id}: unmasked residual PII detected ({leak_types}).")
            err = {
                "node": "pii_gate_2",
                "error": f"Residual unmasked personal information detected ({', '.join(leak_types)}). Request blocked for privacy safety.",
                "message": "PII Gate 2 fail-closed triggered."
            }
            return {
                "pii_vault": vault,
                "pii_blocked": True,
                "workflow_status": "failed",
                "errors": [err],
                "warnings": ["Request blocked by PII Gate 2 due to sensitive personal information."]
            }

        # 3. Mask any residual PII in retrieved evidence texts
        masked_evidence = []
        for chunk in (state.get("retrieved_evidence") or []):
            chunk_copy = dict(chunk)
            raw_chunk_text = chunk_copy.get("text_masked") or chunk_copy.get("text") or ""
            if raw_chunk_text:
                chunk_copy["text_masked"] = vault.mask_text(
                    raw_chunk_text,
                    restaurant_name=chunk_copy.get("restaurant_name")
                )
            masked_evidence.append(chunk_copy)

        return {
            "query": masked_query,
            "retrieved_evidence": masked_evidence,
            "pii_vault": vault,
            "pii_blocked": False,
            "pii_tokens_count": vault.token_count,
        }

    async def classification_node(state: OrchestrationState) -> Dict[str, Any]:
        """Node C1: Invokes Classification Agent concurrently via A2A."""
        t_start = time.time()
        agent_name = "classification"
        try:
            envelope = await a2a_client.classify_complaint(
                text=state["query"],
                review_id=state.get("review_id", "rev_1"),
                business_id=state.get("business_id"),
            )
            t_end = time.time()
            duration_ms = (t_end - t_start) * 1000

            return {
                "classification_result": envelope.payload,
                "agent_statuses": {agent_name: envelope.status},
                "agent_durations_ms": {agent_name: duration_ms},
                "agent_timestamps": {agent_name: {"started_at": t_start, "ended_at": t_end}},
            }
        except httpx.TimeoutException as te:
            t_end = time.time()
            duration_ms = (t_end - t_start) * 1000
            logger.warning(f"Classification agent timed out after {duration_ms:.1f}ms: {te}")
            err = {"node": agent_name, "error": str(te), "type": "timeout"}
            return {
                "classification_result": None,
                "agent_statuses": {agent_name: "timeout"},
                "agent_durations_ms": {agent_name: duration_ms},
                "agent_timestamps": {agent_name: {"started_at": t_start, "ended_at": t_end}},
                "agent_errors": {agent_name: str(te)},
                "errors": [err],
                "warnings": ["Classification Agent request timed out."]
            }
        except Exception as e:
            t_end = time.time()
            duration_ms = (t_end - t_start) * 1000
            logger.error(f"Classification agent failed: {e}")
            err = {"node": agent_name, "error": str(e), "type": "execution_failure"}
            return {
                "classification_result": None,
                "agent_statuses": {agent_name: "failed"},
                "agent_durations_ms": {agent_name: duration_ms},
                "agent_timestamps": {agent_name: {"started_at": t_start, "ended_at": t_end}},
                "agent_errors": {agent_name: str(e)},
                "errors": [err],
                "warnings": [f"Classification Agent failed: {e}"]
            }

    async def trends_node(state: OrchestrationState) -> Dict[str, Any]:
        """Node C2: Invokes Trend Analysis Agent concurrently via A2A."""
        t_start = time.time()
        agent_name = "trends"
        try:
            envelope = await a2a_client.analyze_trends(
                business_id=state.get("business_id"),
                city=state.get("city"),
                horizon=3,
            )
            t_end = time.time()
            duration_ms = (t_end - t_start) * 1000

            return {
                "trend_analysis_result": envelope.payload,
                "agent_statuses": {agent_name: envelope.status},
                "agent_durations_ms": {agent_name: duration_ms},
                "agent_timestamps": {agent_name: {"started_at": t_start, "ended_at": t_end}},
            }
        except httpx.TimeoutException as te:
            t_end = time.time()
            duration_ms = (t_end - t_start) * 1000
            logger.warning(f"Trend Analysis agent timed out after {duration_ms:.1f}ms: {te}")
            err = {"node": agent_name, "error": str(te), "type": "timeout"}
            return {
                "trend_analysis_result": None,
                "agent_statuses": {agent_name: "timeout"},
                "agent_durations_ms": {agent_name: duration_ms},
                "agent_timestamps": {agent_name: {"started_at": t_start, "ended_at": t_end}},
                "agent_errors": {agent_name: str(te)},
                "errors": [err],
                "warnings": ["Trend Analysis Agent request timed out."]
            }
        except Exception as e:
            t_end = time.time()
            duration_ms = (t_end - t_start) * 1000
            logger.error(f"Trend Analysis agent failed: {e}")
            err = {"node": agent_name, "error": str(e), "type": "execution_failure"}
            return {
                "trend_analysis_result": None,
                "agent_statuses": {agent_name: "failed"},
                "agent_durations_ms": {agent_name: duration_ms},
                "agent_timestamps": {agent_name: {"started_at": t_start, "ended_at": t_end}},
                "agent_errors": {agent_name: str(e)},
                "errors": [err],
                "warnings": [f"Trend Analysis Agent failed: {e}"]
            }

    async def root_cause_node(state: OrchestrationState) -> Dict[str, Any]:
        """Node D: Formulates root-cause hypotheses once upstream results are available."""
        agent_name = "root_cause"
        class_res = state.get("classification_result")
        trend_res = state.get("trend_analysis_result")

        class_status = state.get("agent_statuses", {}).get("classification")
        trend_status = state.get("agent_statuses", {}).get("trends")

        if (class_status in ("failed", "timeout", None)) and (trend_status in ("failed", "timeout", None)):
            logger.warning("Root Cause Agent skipped: both Classification and Trend inputs failed or missing.")
            return {
                "root_cause_result": None,
                "agent_statuses": {agent_name: "skipped"},
                "agent_durations_ms": {agent_name: 0.0},
                "agent_timestamps": {agent_name: {"started_at": time.time(), "ended_at": time.time()}},
                "warnings": ["Root Cause Agent skipped due to missing upstream specialist inputs."]
            }

        t_start = time.time()
        try:
            envelope = await a2a_client.analyze_root_cause(
                problem_statement=state["query"],
                business_id=state.get("business_id"),
                top_k=state.get("top_k", 5),
                classification_findings=class_res,
                trend_findings=trend_res,
            )
            t_end = time.time()
            duration_ms = (t_end - t_start) * 1000

            return {
                "root_cause_result": envelope.payload,
                "agent_statuses": {agent_name: envelope.status},
                "agent_durations_ms": {agent_name: duration_ms},
                "agent_timestamps": {agent_name: {"started_at": t_start, "ended_at": t_end}},
            }
        except httpx.TimeoutException as te:
            t_end = time.time()
            duration_ms = (t_end - t_start) * 1000
            err = {"node": agent_name, "error": str(te), "type": "timeout"}
            return {
                "root_cause_result": None,
                "agent_statuses": {agent_name: "timeout"},
                "agent_durations_ms": {agent_name: duration_ms},
                "agent_timestamps": {agent_name: {"started_at": t_start, "ended_at": t_end}},
                "agent_errors": {agent_name: str(te)},
                "errors": [err],
                "warnings": ["Root Cause Agent request timed out."]
            }
        except Exception as e:
            t_end = time.time()
            duration_ms = (t_end - t_start) * 1000
            err = {"node": agent_name, "error": str(e), "type": "execution_failure"}
            return {
                "root_cause_result": None,
                "agent_statuses": {agent_name: "failed"},
                "agent_durations_ms": {agent_name: duration_ms},
                "agent_timestamps": {agent_name: {"started_at": t_start, "ended_at": t_end}},
                "agent_errors": {agent_name: str(e)},
                "errors": [err],
                "warnings": [f"Root Cause Agent failed: {e}"]
            }

    async def action_planner_node(state: OrchestrationState) -> Dict[str, Any]:
        """Node E: Generates prioritized action plan based on root-cause diagnostics."""
        agent_name = "action_planner"
        root_res = state.get("root_cause_result")
        root_status = state.get("agent_statuses", {}).get("root_cause")

        if not root_res or root_status in ("failed", "timeout", "skipped", None):
            logger.warning("Action Planner skipped: Root Cause findings unavailable.")
            return {
                "action_planning_result": None,
                "agent_statuses": {agent_name: "skipped"},
                "agent_durations_ms": {agent_name: 0.0},
                "agent_timestamps": {agent_name: {"started_at": time.time(), "ended_at": time.time()}},
                "warnings": ["Action Planner skipped due to missing root-cause diagnostics."]
            }

        t_start = time.time()
        try:
            envelope = await a2a_client.generate_action_plan(
                business_id=state.get("business_id"),
                problem_statement=state["query"],
                classification_findings=state.get("classification_result"),
                trend_findings=state.get("trend_analysis_result"),
                root_cause_findings=root_res,
            )
            t_end = time.time()
            duration_ms = (t_end - t_start) * 1000

            return {
                "action_planning_result": envelope.payload,
                "agent_statuses": {agent_name: envelope.status},
                "agent_durations_ms": {agent_name: duration_ms},
                "agent_timestamps": {agent_name: {"started_at": t_start, "ended_at": t_end}},
            }
        except httpx.TimeoutException as te:
            t_end = time.time()
            duration_ms = (t_end - t_start) * 1000
            err = {"node": agent_name, "error": str(te), "type": "timeout"}
            return {
                "action_planning_result": None,
                "agent_statuses": {agent_name: "timeout"},
                "agent_durations_ms": {agent_name: duration_ms},
                "agent_timestamps": {agent_name: {"started_at": t_start, "ended_at": t_end}},
                "agent_errors": {agent_name: str(te)},
                "errors": [err],
                "warnings": ["Action Planner Agent request timed out."]
            }
        except Exception as e:
            t_end = time.time()
            duration_ms = (t_end - t_start) * 1000
            err = {"node": agent_name, "error": str(e), "type": "execution_failure"}
            return {
                "action_planning_result": None,
                "agent_statuses": {agent_name: "failed"},
                "agent_durations_ms": {agent_name: duration_ms},
                "agent_timestamps": {agent_name: {"started_at": t_start, "ended_at": t_end}},
                "agent_errors": {agent_name: str(e)},
                "errors": [err],
                "warnings": [f"Action Planner Agent failed: {e}"]
            }

    async def crewai_consolidation_node(state: OrchestrationState) -> Dict[str, Any]:
        """Node CrewAI: Drafts and validates multi-agent operational report."""
        t_start = time.time()
        logger.info(f"Executing CrewAI consolidation for request {state.get('request_id')}...")
        try:
            report, val, status = run_crewai_draft_and_validate(
                query=state.get("query", ""),
                business_id=state.get("business_id"),
                evidence=state.get("retrieved_evidence", []),
                classification=state.get("classification_result"),
                trends=state.get("trend_analysis_result"),
                root_cause=state.get("root_cause_result"),
                action_plan=state.get("action_planning_result"),
                agent_statuses=state.get("agent_statuses", {}),
            )
            duration_ms = (time.time() - t_start) * 1000
            return {
                "crew_report": report.model_dump(),
                "crew_validation": val.model_dump(),
                "crew_status": status,
                "crew_duration_ms": duration_ms,
            }
        except Exception as e:
            logger.warning(f"CrewAI consolidation node warning: {e}")
            duration_ms = (time.time() - t_start) * 1000
            return {
                "crew_report": None,
                "crew_validation": None,
                "crew_status": "failed",
                "crew_duration_ms": duration_ms,
                "warnings": [f"CrewAI drafting encountered non-fatal error: {e}"],
            }

    async def quality_gate_node(state: OrchestrationState) -> Dict[str, Any]:
        skip_judge = (
            state.get("skip_llm_judge", False)
            or type(a2a_client).__name__.startswith("Mock")
        )
        decision = evaluate_quality_gate(
            query=state.get("query", ""),
            target_business_id=state.get("business_id"),
            retrieved_evidence=state.get("retrieved_evidence", []),
            classification_payload=state.get("classification_result"),
            trend_payload=state.get("trend_analysis_result"),
            root_cause_payload=state.get("root_cause_result"),
            action_planner_payload=state.get("action_planning_result"),
            skip_llm_judge=skip_judge,
        )

        q_status = decision.decision.value  # "passed", "flagged", "blocked"
        warnings = []
        if decision.summary:
            warnings.append(decision.summary)
        if decision.statistical.staleness_disclosure:
            warnings.append(decision.statistical.staleness_disclosure)

        return {
            "quality_gate_decision": decision.model_dump(),
            "quality_gate_status": q_status,
            "warnings": warnings,
        }

    async def safe_blocked_node(state: OrchestrationState) -> Dict[str, Any]:
        """Node BLOCKED: Sanitizes outputs and provides safe failure payload for blocked runs."""
        logger.warning(f"Workflow reached safe_blocked_node for request {state.get('request_id')}.")
        err = {
            "node": "quality_gate",
            "error": "Operational report blocked by Quality Gate. Findings violated evidence grounding or business isolation.",
            "message": "Blocked execution."
        }
        return {
            "workflow_status": "failed",
            "errors": [err],
            "action_planning_result": None,
            "root_cause_result": None,
        }

    async def decode_pii_role_node(state: OrchestrationState) -> Dict[str, Any]:
        """Node DECODE: Reversible token unmasking at final response boundary for authorized roles."""
        role = state.get("viewer_role", "manager")
        vault = state.get("pii_vault")

        if not vault or not isinstance(vault, TokenVault):
            return {"pii_decoded": False}

        # Authorized roles: manager, owner, admin. Unauthorized (analyst) remains masked.
        if role.lower() not in AUTHORIZED_ROLES:
            logger.info(f"Viewer role '{role}' is not authorized to decode PII; returning masked content.")
            return {"pii_decoded": False}

        logger.info(f"Viewer role '{role}' authorized: unmasking reversible tokens at response boundary.")
        unmasked_query = vault.unmask_text(state.get("query", ""), role=role)
        unmasked_class = vault.unmask_data(state.get("classification_result"), role=role)
        unmasked_trends = vault.unmask_data(state.get("trend_analysis_result"), role=role)
        unmasked_root = vault.unmask_data(state.get("root_cause_result"), role=role)
        unmasked_actions = vault.unmask_data(state.get("action_planning_result"), role=role)
        unmasked_evidence = vault.unmask_data(state.get("retrieved_evidence"), role=role)
        unmasked_crew = vault.unmask_data(state.get("crew_report"), role=role)
        unmasked_validation = vault.unmask_data(state.get("crew_validation"), role=role)

        unmasked_exec_summary = None
        if state.get("executive_summary"):
            unmasked_exec_summary = vault.unmask_text(state["executive_summary"], role=role)

        res: Dict[str, Any] = {
            "query": unmasked_query,
            "classification_result": unmasked_class,
            "trend_analysis_result": unmasked_trends,
            "root_cause_result": unmasked_root,
            "action_planning_result": unmasked_actions,
            "retrieved_evidence": unmasked_evidence,
            "crew_report": unmasked_crew,
            "crew_validation": unmasked_validation,
            "pii_decoded": True,
        }
        if unmasked_exec_summary:
            res["executive_summary"] = unmasked_exec_summary

        return res

    async def aggregate_results_node(state: OrchestrationState) -> Dict[str, Any]:
        """Node F: Aggregates findings from all agents into a unified, validated response."""
        total_duration_ms = (time.time() - state["graph_start_time"]) * 1000

        statuses = state.get("agent_statuses", {})
        durations = state.get("agent_durations_ms", {})
        timestamps = state.get("agent_timestamps", {})
        agent_errors = state.get("agent_errors", {})

        # Build telemetry mapping
        timings: Dict[str, AgentTimingInfo] = {}
        for ag in ("classification", "trends", "root_cause", "action_planner"):
            st = statuses.get(ag, "skipped")
            dur = durations.get(ag, 0.0)
            ts = timestamps.get(ag, {})
            err = agent_errors.get(ag)
            timings[ag] = AgentTimingInfo(
                status=st if st in ("success", "failed", "timeout", "skipped") else "failed",
                duration_ms=dur,
                started_at=ts.get("started_at", 0.0),
                ended_at=ts.get("ended_at", 0.0),
                error_message=err
            )

        # Determine overall workflow status
        all_agents = ["classification", "trends", "root_cause", "action_planner"]
        success_count = sum(1 for a in all_agents if statuses.get(a) == "success")

        q_status = state.get("quality_gate_status", "passed")
        if state.get("workflow_status") == "failed" or q_status == "blocked" or state.get("pii_blocked"):
            overall_status = "failed"
        elif success_count == len(all_agents):
            overall_status = "success"
        elif success_count > 0:
            overall_status = "partial_success"
        else:
            overall_status = "failed"

        # Quality Gate payload and confidence
        q_decision = state.get("quality_gate_decision") or {}
        confidence_score = q_decision.get("confidence", {}).get("score")
        staleness_disc = q_decision.get("statistical", {}).get("staleness_disclosure")

        crew_rep = state.get("crew_report") or {}
        exec_summary = crew_rep.get("executive_summary") if isinstance(crew_rep, dict) else None
        if not exec_summary and isinstance(state.get("executive_summary"), str):
            exec_summary = state.get("executive_summary")
        vault = state.get("pii_vault")
        role = state.get("viewer_role", "manager")
        if exec_summary and vault and isinstance(vault, TokenVault) and role.lower() in AUTHORIZED_ROLES:
            exec_summary = vault.unmask_text(exec_summary, role=role)

        response = AggregatedWorkflowResponse(
            request_id=state["request_id"],
            business_id=state.get("business_id"),
            query=state["query"],
            workflow_status=overall_status,
            total_duration_ms=total_duration_ms,
            executive_summary=exec_summary,
            crew_report=state.get("crew_report"),
            crew_validation=state.get("crew_validation"),
            crew_status=state.get("crew_status", "skipped"),
            classification=state.get("classification_result"),
            trends=state.get("trend_analysis_result"),
            root_cause=state.get("root_cause_result"),
            action_plan=state.get("action_planning_result"),
            retrieved_evidence=state.get("retrieved_evidence", []),
            agent_timings=timings,
            warnings=list(state.get("warnings", [])),
            errors=list(state.get("errors", [])),
            provenance={
                "orchestrator": "LangGraph v1.2",
                "crew_ai": "Operations Report Drafter & Validator (CrewAI v1.15)",
                "specialist_agent_protocol": "Google ADK & A2A (JSON-RPC 2.0)",
                "concurrency": "asyncio parallel branching",
                "quality_gate": "Phase 4 Evidence Grounding & Confidence Validation",
            },
            quality_gate=q_decision if q_decision else None,
            quality_decision=q_status,
            confidence_score=confidence_score,
            staleness_disclosure=staleness_disc,
            pii_masked=True,
            pii_decoded=state.get("pii_decoded", False),
            viewer_role=state.get("viewer_role", "analyst"),
        )

        return {
            "workflow_status": overall_status,
            "total_duration_ms": total_duration_ms,
            "final_result": response.model_dump(),
        }

    return {
        "validate_request": validate_request_node,
        "retrieve_context": retrieve_context_node,
        "pii_gate_2": pii_gate_2_node,
        "classification": classification_node,
        "trends": trends_node,
        "root_cause": root_cause_node,
        "action_planner": action_planner_node,
        "crewai_consolidation": crewai_consolidation_node,
        "quality_gate": quality_gate_node,
        "safe_blocked": safe_blocked_node,
        "decode_pii_role": decode_pii_role_node,
        "aggregate_results": aggregate_results_node,
    }
