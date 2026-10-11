"""Root-level entrypoint for the AI Restaurant Operations RAG Assistant.
Enables running directly via:
    python rag_agent.py "your question here"
    python rag_agent.py --interactive
    python rag_agent.py --city "Philadelphia" "cleanliness issues"
"""

import sys
from pathlib import Path

# Ensure project root is in sys.path
root_dir = Path(__file__).resolve().parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from app.rag_agent import main

if __name__ == "__main__":
    main()
