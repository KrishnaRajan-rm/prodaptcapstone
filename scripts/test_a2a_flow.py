"""End-to-End Demonstration Script for the 4 Specialist Agents.
Connects with all 4 agents over A2A protocol / services, running a complete analysis flow:
  1. Classification Agent (Port 8001): Analyzes customer complaint.
  2. Trend Analysis Agent (Port 8002): Analyzes historical trends, forecast, and satisfaction.
  3. Root-Cause Analysis Agent (Port 8003): Formulates evidence-backed hypotheses via Hybrid RAG.
  4. Action Planner Agent (Port 8004): Produces prioritized action plan with KPIs and owner roles.

Usage:
    python scripts/test_a2a_flow.py
"""

import sys
import os
import json
from pathlib import Path

root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from app.agents.classification.service import create_classification_app
from app.agents.trends.service import create_trend_app
from app.agents.root_cause.service import create_root_cause_app
from app.agents.action_planner.service import create_action_planner_app
from starlette.testclient import TestClient


def main():
    print("\n" + "=" * 78)
    print("AI-POWERED RESTAURANT OPERATIONS — 4 SPECIALIST AGENTS PIPELINE FLOW")
    print("=" * 78)

    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sample_review = (
        "We waited 45 minutes for our table and another 30 minutes for our entrees! "
        "When the food finally arrived, my steak was lukewarm and the fries were soggy. "
        "The server seemed overwhelmed and ignored us when we asked for water."
    )
    sample_business_id = "PP3BBaVxZLcJU54uP_wL6Q"

    # Step 1: Agent 1 — Classification Agent (Port 8001)
    print("\n[STEP 1] Running Classification Agent (Port 8001)...")
    app1 = create_classification_app(port=8001)
    with TestClient(app1) as client1:
        card = client1.get("/.well-known/agent-card.json").json()
        print(f"  • Connected to: {card['name']} (v{card['version']})")
        resp1 = client1.post("/classify", json={"text": sample_review, "review_id": "demo_rev_01"}).json()
        p1 = resp1["payload"]
        print(f"  • Categories: {p1['categories']}")
        print(f"  • Aspects: {p1['aspects']}")
        print(f"  • Sentiment: {p1['sentiment']} | Severity: {p1['severity']} ({p1['severity_reason']})")
        print(f"  • Evidence: {len(resp1['evidence'])} grounded excerpts")

    # Step 2: Agent 2 — Trend Analysis Agent (Port 8002)
    print("\n[STEP 2] Running Trend Analysis Agent (Port 8002)...")
    app2 = create_trend_app(port=8002)
    with TestClient(app2) as client2:
        card = client2.get("/.well-known/agent-card.json").json()
        print(f"  • Connected to: {card['name']} (v{card['version']})")
        resp2 = client2.post("/trends", json={"business_id": sample_business_id, "forecast_horizon": 3}).json()
        p2 = resp2["payload"]
        print(f"  • Analysis Period: {p2['analysis_period']} ({p2['total_reviews_analyzed']} reviews, {p2['average_rating']} stars)")
        print(f"  • Worsening Aspects: {p2['worsening_aspects']}")
        fc = p2["forecast"]
        print(f"  • Forecast ({fc['target_metric']}): {fc['forecast_values']} (MAE={fc['mae']})")
        sat = p2["satisfaction_prediction"]
        print(f"  • Satisfaction Prediction: {sat['predicted_score']} stars ({sat['predicted_label']} tier | Test MAE={sat['metric_value']})")

    # Step 3: Agent 3 — Root-Cause Analysis Agent (Port 8003)
    print("\n[STEP 3] Running Root-Cause Analysis Agent (Port 8003)...")
    app3 = create_root_cause_app(port=8003)
    with TestClient(app3) as client3:
        card = client3.get("/.well-known/agent-card.json").json()
        print(f"  • Connected to: {card['name']} (v{card['version']})")
        resp3 = client3.post("/root-cause", json={
            "problem_statement": "Excessive table wait times and lukewarm food delivery",
            "business_id": sample_business_id,
            "top_k_evidence": 3,
            "classification_findings": p1,
            "trend_findings": p2,
        }).json()
        p3 = resp3["payload"]
        print(f"  • Retrieved Chunks: {p3['retrieved_chunk_count']}")
        for hyp in p3["hypotheses"]:
            print(f"  • Hypothesis [{hyp['hypothesis_id']}]: {hyp['hypothesized_cause']}")
            print(f"    - Type: {hyp['distinction_type']} | Confidence: {hyp['confidence']}")
            print(f"    - Supporting Chunk IDs: {hyp['supporting_source_ids']}")
            print(f"    - Action Needed: {hyp['additional_info_needed']}")

    # Step 4: Agent 4 — Action Planner Agent (Port 8004)
    print("\n[STEP 4] Running Action Planner Agent (Port 8004)...")
    app4 = create_action_planner_app(port=8004)
    with TestClient(app4) as client4:
        card = client4.get("/.well-known/agent-card.json").json()
        print(f"  • Connected to: {card['name']} (v{card['version']})")
        resp4 = client4.post("/action-plan", json={
            "business_id": sample_business_id,
            "problem_statement": "Elevated table turnaround wait times and lukewarm food staging",
            "classification_findings": p1,
            "trend_findings": p2,
            "root_cause_findings": p3,
        }).json()
        p4 = resp4["payload"]
        print(f"  • Scoring Formula: {p4['scoring_methodology']}")
        print(f"  • Prioritized Actions ({len(p4['ranked_actions'])} items):")
        for act in p4["ranked_actions"]:
            print(f"\n    [Rank {act['rank']}] Score: {act['priority_score']} pts | Owner: {act['owner_role']}")
            print(f"     - Issue: {act['issue_summary']}")
            print(f"     - Action: {act['recommended_action']}")
            print(f"     - KPI: {act['measurable_kpi']} (Baseline: {act['baseline_value']} -> Target: {act['target_value']})")
            print(f"     - Verification: {act['verification_method']}")

    print("\n" + "=" * 78)
    print("SUCCESS: ALL 4 SPECIALIST AGENTS COMPLETED THE PIPELINE FLOW")
    print("=" * 78 + "\n")


if __name__ == "__main__":
    main()
