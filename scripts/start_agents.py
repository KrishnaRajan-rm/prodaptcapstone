"""Service Launcher for the 4 Specialist Agents.
Can launch all 4 agents concurrently on their dedicated ports (8001-8004),
or launch an individual agent service for focused testing.

Usage:
    # Run all 4 agents concurrently:
    python scripts/start_agents.py

    # Run a single agent:
    python scripts/start_agents.py --agent classification --port 8001
    python scripts/start_agents.py --agent trends --port 8002
    python scripts/start_agents.py --agent root_cause --port 8003
    python scripts/start_agents.py --agent action_planner --port 8004
"""

import sys
import os
import argparse
import multiprocessing
import signal
import time
from pathlib import Path

# Ensure project root is in sys.path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

import uvicorn
from app.core.logging import logger
from app.agents.classification.service import create_classification_app
from app.agents.trends.service import create_trend_app
from app.agents.root_cause.service import create_root_cause_app
from app.agents.action_planner.service import create_action_planner_app


SERVICES = {
    "classification": {
        "factory": create_classification_app,
        "default_port": int(os.getenv("CLASSIFICATION_AGENT_PORT", "8001")),
        "description": "Classification Agent (Port 8001)",
    },
    "trends": {
        "factory": create_trend_app,
        "default_port": int(os.getenv("TREND_AGENT_PORT", "8002")),
        "description": "Trend Analysis Agent (Port 8002)",
    },
    "root_cause": {
        "factory": create_root_cause_app,
        "default_port": int(os.getenv("ROOT_CAUSE_AGENT_PORT", "8003")),
        "description": "Root-Cause Analysis Agent (Port 8003)",
    },
    "action_planner": {
        "factory": create_action_planner_app,
        "default_port": int(os.getenv("ACTION_PLANNER_AGENT_PORT", "8004")),
        "description": "Action Planner Agent (Port 8004)",
    },
}


def run_single_service(agent_key: str, host: str, port: int):
    """Runs a single agent service using uvicorn."""
    info = SERVICES[agent_key]
    logger.info(f"Starting {info['description']} on {host}:{port}...")
    app = info["factory"](port=port, host=host)
    uvicorn.run(app, host=host, port=port, log_level="warning")


def main():
    parser = argparse.ArgumentParser(description="Start A2A Specialist Agents")
    parser.add_argument(
        "--agent",
        choices=["all", "classification", "trends", "root_cause", "action_planner"],
        default="all",
        help="Which agent service to run (default: all)"
    )
    parser.add_argument("--host", default="0.0.0.0", help="Host interface (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=None, help="Port override for single agent mode")
    args = parser.parse_args()

    if args.agent != "all":
        port = args.port or SERVICES[args.agent]["default_port"]
        run_single_service(args.agent, args.host, port)
        return

    # Start all 4 agents in parallel processes
    print("=" * 70)
    print("STARTING ALL 4 SPECIALIST AGENTS (Google ADK & A2A Protocol)")
    print("=" * 70)
    processes = []
    for key, info in SERVICES.items():
        port = info["default_port"]
        print(f" • {info['description']} -> http://localhost:{port}/")
        print(f"   Agent Card: http://localhost:{port}/.well-known/agent-card.json")
        p = multiprocessing.Process(
            target=run_single_service,
            args=(key, args.host, port),
            daemon=True
        )
        p.start()
        processes.append(p)

    print("=" * 70)
    print("All 4 services running in background. Press Ctrl+C to stop.")
    print("=" * 70)

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nShutting down all agent services...")
        for p in processes:
            p.terminate()
        for p in processes:
            p.join(timeout=2)
        print("All agent services terminated.")


if __name__ == "__main__":
    main()
