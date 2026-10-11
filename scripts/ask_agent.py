"""Interactive Multi-Agent Assistant CLI.
Allows testing any of the 4 specialist agents interactively on arbitrary user prompts.

Usage:
    # Interactive menu:
    python scripts/ask_agent.py

    # Direct query to a specific agent:
    python scripts/ask_agent.py --agent classification "Wait time was 45 minutes and staff was rude"
    python scripts/ask_agent.py --agent root_cause "Food arrives lukewarm during dinner rush"
    python scripts/ask_agent.py --agent trends "PP3BBaVxZLcJU54uP_wL6Q"
    python scripts/ask_agent.py --agent action_planner "Long table wait times and lukewarm food"
"""

import sys
import os
import argparse
from pathlib import Path

# Ensure project root is in sys.path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

from app.agents.classification.agent import ClassificationAgent
from app.agents.trends.agent import TrendAnalysisAgent
from app.agents.root_cause.agent import RootCauseAgent
from app.agents.action_planner.agent import ActionPlannerAgent
from app.agents.schemas import (
    ClassificationRequest,
    TrendAnalysisRequest,
    RootCauseRequest,
    ActionPlannerRequest,
)


def run_classification(agent: ClassificationAgent, user_prompt: str):
    print(f"\n[AGENT 1: CLASSIFICATION] Analyzing review text...")
    req = ClassificationRequest(review_id="cli_rev", text=user_prompt)
    env = agent.analyze(req)
    p = env.payload

    print("\n" + "=" * 76)
    print("CLASSIFICATION RESULTS")
    print("=" * 76)
    print(f"Categories: {p.get('categories')}")
    print(f"Aspects:    {p.get('aspects')}")
    print(f"Sentiment:  {p.get('sentiment')}")
    print(f"Severity:   {p.get('severity')} ({p.get('severity_reason')})")
    print(f"Summary:    {p.get('short_explanation')}")
    if p.get('evidence_snippets'):
        print("\nSupporting Excerpts:")
        for s in p.get('evidence_snippets', []):
            print(f"  • \"{s}\"")
    print(f"Execution:  Model={env.execution_metadata.get('model')} | Latency={env.execution_metadata.get('latency_ms', 0):.1f}ms")
    print("=" * 76 + "\n")
    return p


def run_trends(agent: TrendAnalysisAgent, user_prompt: str):
    target = user_prompt.strip() if user_prompt.strip() and user_prompt.strip() != "all" else None
    print(f"\n[AGENT 2: TREND ANALYSIS] Querying historical records for: {target or 'All Restaurants'}...")
    req = TrendAnalysisRequest(business_id=target, forecast_horizon=3)
    env = agent.analyze(req)

    print("\n" + "=" * 76)
    print("TREND ANALYSIS RESULTS")
    print("=" * 76)
    if env.status == "insufficient_data":
        print(f"Status: INSUFFICIENT DATA")
        print(f"Warnings: {env.warnings}")
        print("=" * 76 + "\n")
        return env.payload

    p = env.payload
    print(f"Analysis Period:        {p.get('analysis_period')}")
    print(f"Total Reviews Analyzed: {p.get('total_reviews_analyzed'):,}")
    print(f"Average Star Rating:    {p.get('average_rating')} / 5.0")
    print(f"Worsening Aspects:      {p.get('worsening_aspects')}")
    print(f"Improving Aspects:      {p.get('improving_aspects')}")

    fc = p.get('forecast', {})
    print(f"\n[3-Month Forecast ({fc.get('target_metric')})]")
    print(f" • Status: {fc.get('status')} | Projections: {fc.get('forecast_values')}")
    print(f" • Evaluation: Held-out MAE={fc.get('mae')}, RMSE={fc.get('rmse')}")

    sat = p.get('satisfaction_prediction', {})
    print(f"\n[Customer Satisfaction Prediction]")
    print(f" • Predicted Rating: {sat.get('predicted_score')} / 5.0 ({sat.get('predicted_label')} tier)")
    print(f" • Evaluation: Chronological Test MAE={sat.get('metric_value')}")
    print(f"Execution:  Latency={env.execution_metadata.get('latency_ms', 0):.1f}ms")
    print("=" * 76 + "\n")
    return p


def run_root_cause(agent: RootCauseAgent, user_prompt: str, classification_findings=None, trend_findings=None):
    print(f"\n[AGENT 3: ROOT-CAUSE ANALYSIS] Searching Hybrid RAG evidence for: '{user_prompt}'...")
    req = RootCauseRequest(
        problem_statement=user_prompt,
        top_k_evidence=4,
        classification_findings=classification_findings,
        trend_findings=trend_findings
    )
    env = agent.analyze(req)
    p = env.payload

    print("\n" + "=" * 76)
    print("ROOT-CAUSE HYPOTHESES")
    print("=" * 76)
    print(f"Retrieved Evidence Chunks: {p.get('retrieved_chunk_count', 0)}")
    for idx, hyp in enumerate(p.get("hypotheses", []), 1):
        print(f"\n[Hypothesis {idx}] {hyp.get('hypothesized_cause')}")
        print(f" • Type:       {hyp.get('distinction_type')} | Confidence: {hyp.get('confidence')}")
        print(f" • Source IDs: {hyp.get('supporting_source_ids')}")
        if hyp.get('supporting_evidence'):
            print(f" • Supporting Review Excerpts:")
            for s in hyp.get('supporting_evidence', []):
                print(f"    - \"{s}\"")
        if hyp.get('contradictory_evidence'):
            print(f" • Contradictory Evidence ({hyp.get('contradictory_source_ids')}):")
            for c in hyp.get('contradictory_evidence', []):
                print(f"    - \"{c}\"")
        if hyp.get('alternative_explanations'):
            print(f" • Alternative Causes: {', '.join(hyp.get('alternative_explanations', []))}")
        print(f" • Verification Needed: {hyp.get('additional_info_needed')}")
    print(f"\nExecution:  Model={env.execution_metadata.get('model')} | Latency={env.execution_metadata.get('latency_ms', 0):.1f}ms")
    print("=" * 76 + "\n")
    return p


def run_action_plan(agent: ActionPlannerAgent, user_prompt: str, classification_findings=None, trend_findings=None, root_cause_findings=None):
    print(f"\n[AGENT 4: ACTION PLANNER] Formulating prioritized recommendations for: '{user_prompt}'...")
    req = ActionPlannerRequest(
        problem_statement=user_prompt,
        classification_findings=classification_findings,
        trend_findings=trend_findings,
        root_cause_findings=root_cause_findings
    )
    env = agent.analyze(req)
    p = env.payload

    print("\n" + "=" * 76)
    print("PRIORITIZED OPERATIONAL ACTION PLAN")
    print("=" * 76)
    print(f"Formula: {p.get('scoring_methodology')}\n")
    for act in p.get("ranked_actions", []):
        print(f"[Rank {act.get('rank')}] Score: {act.get('priority_score')} / 100 | Owner: {act.get('owner_role')}")
        print(f" • Issue:        {act.get('issue_summary')}")
        print(f" • Action:       {act.get('recommended_action')}")
        print(f" • KPI:          {act.get('measurable_kpi')}")
        print(f" • Baseline:     {act.get('baseline_value')} -> Target: {act.get('target_value')}")
        print(f" • Verification: {act.get('verification_method')}\n")
    print(f"Execution:  Model={env.execution_metadata.get('model')} | Latency={env.execution_metadata.get('latency_ms', 0):.1f}ms")
    print("=" * 76 + "\n")
    return p


def interactive_menu():
    print("\n" + "=" * 76)
    print("RESTAURANT OPERATIONS ASSISTANT — SPECIALIST AGENTS INTERACTIVE TESTER")
    print("=" * 76)
    print("Select an agent to test with your custom prompt:")
    print("  1. Classification Agent (Port 8001) - Classify review complaints & severity")
    print("  2. Trend Analysis Agent (Port 8002)  - Historical trends, forecast & satisfaction")
    print("  3. Root-Cause Analysis Agent (Port 8003) - Evidence-backed root causes via RAG")
    print("  4. Action Planner Agent (Port 8004) - Prioritized operational action plans & KPIs")
    print("  5. Run All 4 Agents in Sequence (End-to-End Pipeline)")
    print("  Q. Quit\n")

    classifier = None
    trends_agent = None
    rc_agent = None
    planner = None

    while True:
        try:
            choice = input("Select option (1-5, or Q) > ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if choice in ("q", "quit", "exit"):
            print("Goodbye.")
            break

        if choice == "1":
            if not classifier:
                classifier = ClassificationAgent()
            prompt = input("\nEnter customer review to classify > ").strip()
            if prompt:
                run_classification(classifier, prompt)

        elif choice == "2":
            if not trends_agent:
                trends_agent = TrendAnalysisAgent()
            prompt = input("\nEnter business_id (or press Enter for overall dataset) > ").strip()
            run_trends(trends_agent, prompt)

        elif choice == "3":
            if not rc_agent:
                rc_agent = RootCauseAgent()
            prompt = input("\nEnter operational problem statement > ").strip()
            if prompt:
                run_root_cause(rc_agent, prompt)

        elif choice == "4":
            if not planner:
                planner = ActionPlannerAgent()
            prompt = input("\nEnter operational issue to plan actions for > ").strip()
            if prompt:
                run_action_plan(planner, prompt)

        elif choice == "5":
            if not classifier: classifier = ClassificationAgent()
            if not trends_agent: trends_agent = TrendAnalysisAgent()
            if not rc_agent: rc_agent = RootCauseAgent()
            if not planner: planner = ActionPlannerAgent()

            review_input = input("\nEnter customer review / problem statement > ").strip()
            if not review_input:
                review_input = "We waited 45 minutes for our table and the food arrived lukewarm with rude staff."
            business_input = input("Enter business_id (optional, press Enter for general) > ").strip()

            c_res = run_classification(classifier, review_input)
            t_res = run_trends(trends_agent, business_input or "PP3BBaVxZLcJU54uP_wL6Q")
            rc_res = run_root_cause(rc_agent, review_input, c_res, t_res)
            run_action_plan(planner, review_input, c_res, t_res, rc_res)

        else:
            print("Invalid option. Please enter 1, 2, 3, 4, 5, or Q.")


def main():
    parser = argparse.ArgumentParser(description="Test Specialist Agents with Custom Prompts")
    parser.add_argument("--agent", choices=["classification", "trends", "root_cause", "action_planner"], help="Specific agent")
    parser.add_argument("prompt", nargs="?", help="Input prompt or review text")
    args = parser.parse_args()

    if args.agent and args.prompt:
        if args.agent == "classification":
            run_classification(ClassificationAgent(), args.prompt)
        elif args.agent == "trends":
            run_trends(TrendAnalysisAgent(), args.prompt)
        elif args.agent == "root_cause":
            run_root_cause(RootCauseAgent(), args.prompt)
        elif args.agent == "action_planner":
            run_action_plan(ActionPlannerAgent(), args.prompt)
    else:
        interactive_menu()


if __name__ == "__main__":
    main()
