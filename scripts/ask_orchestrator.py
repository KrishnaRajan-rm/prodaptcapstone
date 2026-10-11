"""Interactive CLI for LangGraph Restaurant Operations Orchestrator with Phase 4 Quality & Privacy Gates.
Allows querying the orchestrated multi-agent workflow from the command line.

Usage:
    # Default analyst mode (masked PII):
    python scripts/ask_orchestrator.py "Waiters ignore us and food is cold" --role analyst

    # Manager mode (reversible PII unmasked):
    python scripts/ask_orchestrator.py "Waiters ignore us and food is cold" --role manager

    # Direct query with business ID:
    python scripts/ask_orchestrator.py "cold steak" --business PP3BBaVxZLcJU54uP_wL6Q
"""

import sys
import os
import argparse
import asyncio
import time
from pathlib import Path

# Ensure project root is in sys.path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import httpx
from app.orchestration.schemas import OrchestrationRequest
from app.orchestration.graph import run_orchestration_workflow
from app.orchestration.a2a_async_client import AsyncA2AAgentClient, DEFAULT_AGENT_PORTS

# Lazy-load in-process Starlette ASGI apps if network services are not pre-running
from app.agents.classification.service import create_classification_app
from app.agents.trends.service import create_trend_app
from app.agents.root_cause.service import create_root_cause_app
from app.agents.action_planner.service import create_action_planner_app


async def test_or_build_client() -> AsyncA2AAgentClient:
    """Detects if background microservices are running on ports 8001-8004.
    If not, integrates in-process ASGI transports directly so CLI works with zero manual server launches.
    """
    probe_client = httpx.AsyncClient(timeout=0.6)
    port = DEFAULT_AGENT_PORTS.get("classification", 8001)
    try:
        resp = await probe_client.get(f"http://127.0.0.1:{port}/.well-known/agent-card.json")
        if resp.status_code == 200:
            print("[ORCHESTRATOR] Connected to live A2A microservices on ports 8001-8004.", flush=True)
            return AsyncA2AAgentClient()
    except Exception:
        pass
    finally:
        await probe_client.aclose()

    print("[ORCHESTRATOR] Live background services not detected on ports 8001-8004.", flush=True)
    print("[ORCHESTRATOR] Attaching in-process A2A ASGI service transports...", flush=True)
    app_registry = {
        "classification": create_classification_app(port=8001),
        "trends": create_trend_app(port=8002),
        "root_cause": create_root_cause_app(port=8003),
        "action_planner": create_action_planner_app(port=8004),
    }
    return AsyncA2AAgentClient(app_registry=app_registry)


def format_output(resp):
    """Prints a structured, manager/analyst-friendly terminal summary of the workflow results."""
    print("\n" + "=" * 80)
    print(f"LANGGRAPH ORCHESTRATION REPORT: {resp.workflow_status.upper()}")
    print("=" * 80)
    print(f"Request ID:      {resp.request_id}")
    print(f"Target Business: {resp.business_id or 'General Operational Scope'}")
    print(f"Query Prompt:    \"{resp.query}\"")
    print(f"Viewer Role:     {resp.viewer_role.upper()} (PII Decoded: {'YES' if resp.pii_decoded else 'NO - MASKED'})")
    print(f"Total Duration:  {resp.total_duration_ms:.1f}ms\n")

    # Phase 4 Quality Gate & Confidence Summary
    if resp.quality_gate:
        q = resp.quality_gate
        q_dec = (resp.quality_decision or "unknown").upper()
        conf = q.get("confidence", {})
        grounding = q.get("grounding", {})
        stat = q.get("statistical", {})

        print("-" * 80)
        print(f"PHASE 4 QUALITY GATE VERDICT: [{q_dec}]")
        print("-" * 80)
        print(f" • Gate Status:       {q_dec}")
        print(f" • Summary:           {q.get('summary', 'Validation complete.')}")
        print(f" • Confidence Score:  {conf.get('score', 0.0)}/100 ({str(conf.get('assessment')).upper()})")
        print(f" • Scoring Formula:   {conf.get('formula')}")
        print(f" • Citations Checked: {grounding.get('verified_citations', 0)} verified / {grounding.get('total_citations', 0)} cited")
        if stat.get("staleness_disclosure"):
            print(f" • Dataset Cutoff:    {stat.get('staleness_disclosure')}")

        tele = q.get("telemetry_ms", {})
        if tele:
            print(f" • Gate Latency:      Deterministic={tele.get('deterministic_checks_ms', 0):.1f}ms | Judge={tele.get('llm_judge_ms', 0):.1f}ms | Total={tele.get('total_quality_gate_ms', 0):.1f}ms")
        print()

    # Agent Timings & Concurrency Verification
    print("-" * 80)
    print("1. AGENT EXECUTION TELEMETRY & CONCURRENCY")
    print("-" * 80)
    c_timing = resp.agent_timings.get("classification")
    t_timing = resp.agent_timings.get("trends")

    for ag_name in ("classification", "trends", "root_cause", "action_planner"):
        ti = resp.agent_timings.get(ag_name)
        if ti:
            err_str = f" | Error: {ti.error_message}" if ti.error_message else ""
            print(f" • {ag_name.replace('_', ' ').title():<18}: Status={ti.status:<7} | Duration={ti.duration_ms:>7.1f}ms{err_str}")

    if c_timing and t_timing and c_timing.status == "success" and t_timing.status == "success":
        overlap = max(c_timing.started_at, t_timing.started_at) < min(c_timing.ended_at, t_timing.ended_at)
        print(f"\n Concurrency Check (Classification || Trends): {'TRUE (Overlapping Execution)' if overlap else 'COMPLETED'}")
        print(f"   Classification: {c_timing.started_at:.3f} -> {c_timing.ended_at:.3f}")
        print(f"   Trend Analysis: {t_timing.started_at:.3f} -> {t_timing.ended_at:.3f}")

    # Step 2: Classification Findings
    if resp.classification:
        c = resp.classification
        print("\n" + "-" * 80)
        print("2. COMPLAINT CLASSIFICATION & ASPECT TAGGING (Port 8001)")
        print("-" * 80)
        print(f" • Categories: {c.get('categories')}")
        print(f" • Aspects:    {c.get('aspects')}")
        print(f" • Sentiment:  {c.get('sentiment')} | Severity: {c.get('severity')}")
        print(f" • Rationale:  {c.get('severity_reason')}")
        if c.get("evidence_snippets"):
            print(f" • Evidence:   {c.get('evidence_snippets')}")

    # Step 3: Trend Analysis Findings
    if resp.trends:
        tr = resp.trends
        print("\n" + "-" * 80)
        print("3. HISTORICAL TRENDS & SATISFACTION PREDICTION (Port 8002)")
        print("-" * 80)
        print(f" • Historical Range: {tr.get('analysis_period')} ({tr.get('total_reviews_analyzed', 0):,} reviews, {tr.get('average_rating')} stars)")
        print(f" • Worsening Trends: {tr.get('worsening_aspects')}")
        print(f" • Improving Trends: {tr.get('improving_aspects')}")
        fc = tr.get("forecast", {})
        if fc:
            print(f" • 3-Month Forecast: {fc.get('forecast_values')} (MAE={fc.get('mae')})")
        sat = tr.get("satisfaction_prediction", {})
        if sat:
            print(f" • Predicted Rating: {sat.get('predicted_score')} / 5.0 ({sat.get('predicted_label')} tier | Test MAE={sat.get('metric_value')})")

    # Step 4: Root Cause Hypotheses
    if resp.root_cause:
        rc = resp.root_cause
        print("\n" + "-" * 80)
        print(f"4. ROOT-CAUSE DIAGNOSTIC HYPOTHESES (Port 8003 | {rc.get('retrieved_chunk_count', 0)} Chunks)")
        print("-" * 80)
        for h in rc.get("hypotheses", []):
            print(f" • [{h.get('hypothesis_id')}] {h.get('hypothesized_cause')}")
            print(f"   - Distinction Type:  {h.get('distinction_type')} | Confidence: {h.get('confidence')}")
            print(f"   - Supporting Chunks: {h.get('supporting_source_ids')}")
            if h.get("supporting_evidence"):
                print(f"   - Supporting Excerpt: \"{h.get('supporting_evidence')[0][:120]}...\"")
            print(f"   - Required Verification: {h.get('additional_info_needed')}\n")

    # Step 5: Prioritized Action Plan
    if resp.action_plan:
        ap = resp.action_plan
        print("-" * 80)
        print(f"5. PRIORITIZED CORRECTIVE ACTIONS & KPIS (Port 8004)")
        print("-" * 80)
        print(f"Formula: {ap.get('scoring_methodology')}\n")
        for act in ap.get("ranked_actions", []):
            print(f" [Rank {act.get('rank')}] Score: {act.get('priority_score')} / 100 | Owner: {act.get('owner_role')}")
            print(f"  • Issue:        {act.get('issue_summary')}")
            print(f"  • Action:       {act.get('recommended_action')}")
            print(f"  • KPI:          {act.get('measurable_kpi')} (Target: {act.get('target_value')})")
            print(f"  • Verification: {act.get('verification_method')}\n")

    if resp.warnings:
        print("-" * 80)
        print(f"WARNINGS: {resp.warnings}")

    if resp.errors:
        print("-" * 80)
        print(f"ERRORS: {resp.errors}")

    print("=" * 80 + "\n")


async def async_main():
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

    parser = argparse.ArgumentParser(description="LangGraph Restaurant Operations Orchestrator CLI")
    parser.add_argument("query", nargs="?", default=None, help="Operational complaint or query")
    parser.add_argument("--business", "-b", default=None, help="Target business ID (e.g. PP3BBaVxZLcJU54uP_wL6Q)")
    parser.add_argument("--city", "-c", default=None, help="Target city")
    parser.add_argument("--role", "-r", default="manager", choices=["manager", "analyst", "owner", "admin"], help="Viewer role (default: manager)")
    parser.add_argument("--skip-judge", action="store_true", help="Bypass LLM judge for rapid deterministic evaluation")
    args = parser.parse_args()

    client = await test_or_build_client()

    try:
        if args.query:
            req = OrchestrationRequest(
                query=args.query,
                business_id=args.business,
                city=args.city,
                viewer_role=args.role,
                skip_llm_judge=args.skip_judge
            )
            print(f"\n[ORCHESTRATOR] Executing workflow for query: \"{args.query}\"...")
            resp = await run_orchestration_workflow(req, client=client)
            format_output(resp)
            return

        print("\n" + "=" * 80)
        print("LANGGRAPH RESTAURANT OPERATIONS ORCHESTRATOR (CLI)")
        print("Coordinated Multi-Agent Workflow with PII Masking & Quality Gate")
        print("=" * 80)
        print("Enter a customer review or operational problem statement.")
        print("Type 'exit' or 'q' to quit.\n")

        while True:
            try:
                user_query = input("Enter operational query > ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nExiting.")
                break

            if not user_query or user_query.lower() in ("exit", "quit", "q"):
                print("Goodbye.")
                break

            user_biz = input("Enter business ID (leave blank for general) [e.g. PP3BBaVxZLcJU54uP_wL6Q]: ").strip() or None
            req = OrchestrationRequest(query=user_query, business_id=user_biz, viewer_role="manager")
            print(f"\n[ORCHESTRATOR] Executing workflow...")
            resp = await run_orchestration_workflow(req, client=client)
            format_output(resp)

    finally:
        await client.aclose()


def main():
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
