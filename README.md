# AI-Powered Restaurant Operations & Review Assistant

An enterprise-grade, multi-agent AI operational intelligence system built on the **Yelp Open Dataset (2.2M+ customer reviews)**. It autonomously diagnoses restaurant friction points, classifies complaint aspects and severity, models historical trend trajectories, formulates evidence-grounded root causes, drafts executive operational reports via **CrewAI**, enforces strict **PII Gate 2** privacy boundaries, and validates recommendations through an authoritative **Quality Gate** and **DeepEval** evaluation benchmark.

---

## Architecture Diagram & Data Flow

```mermaid
flowchart TD
    User["Manager / Executive (React UI / Streamlit)"]
    API["FastAPI Gateway (Port 8000)"]
    
    subgraph Privacy_and_Retrieval ["Privacy & Retrieval Subsystem"]
        PII["PII Gate 2 (Regex + spaCy NER + TokenVault)"]
        Hybrid["Hybrid RAG Engine (BM25 + Dense MiniLM-L6-v2 + RRF)"]
        Parquet["Clean Parquet & Memmap Store (59.8k Vector Chunks)"]
    end

    subgraph LangGraph_Orchestrator ["LangGraph Workflow Coordinator"]
        GraphStart["Input Validation & Guardrails"]
        Fork["Parallel Async Branching"]
        Join["Join & Dependency Resolver"]
    end

    subgraph A2A_Mesh ["Google ADK & A2A Microservices (Ports 8001-8004)"]
        A1["Classification Agent (:8001)"]
        A2["Trend Analysis Agent (:8002)"]
        A3["Root-Cause Agent (:8003)"]
        A4["Action Planner Agent (:8004)"]
    end

    subgraph Consolidation_and_Quality ["Consolidation & Quality Assurance"]
        CrewDrafter["CrewAI Operations Report Drafter"]
        CrewValidator["CrewAI Draft Validator"]
        QGate["Authoritative Quality Gate (Grounding + Stats + Confidence)"]
        Judge["LLM-as-Judge Evaluator (Gemini 3.8 / Fallback)"]
    end

    subgraph Persistence ["Persistence & Feedback"]
        DB["SQLite Database (restaurant_ops.db)"]
    end

    User -->|POST /api/v1/analysis| API
    API --> GraphStart
    GraphStart --> Hybrid
    Hybrid <--> Parquet
    Hybrid --> PII
    PII --> Fork
    Fork --> A1
    Fork --> A2
    A1 & A2 --> Join
    Join --> A3
    A3 --> A4
    A4 --> CrewDrafter
    CrewDrafter --> CrewValidator
    CrewValidator --> QGate
    QGate <--> Judge
    QGate --> API
    API -->|Persist Traces & Feedback| DB
    API -->|Consolidated Response & Trace| User
```

---

## Key Features

1. **Hybrid RAG Semantic Retrieval**:
   - BM25 full-text keyword indexing + dense 384-dimensional sentence transformer embeddings (`all-MiniLM-L6-v2`).
   - Reciprocal Rank Fusion (RRF) and Cross-Encoder reranking over 59,894 indexed chunk vectors.
   - Metadata filtering across business IDs, cities (8 major metros), and star ratings.

2. **Specialist Agent Microservice Mesh**:
   - **Classification Agent** (Port 8001): Identifies categories (Speed of Service, Food Quality, Staff, Cleanliness, Pricing), aspects, sentiment, and severity.
   - **Trend Analysis Agent** (Port 8002): Analyzes historical complaint slopes (DuckDB parquet analytics) and satisfaction predictions.
   - **Root-Cause Analysis Agent** (Port 8003): Distinguishes acute vs chronic failure modes with explicit evidence attribution.
   - **Action Planner Agent** (Port 8004): Generates prioritized recommendations mapped to owner roles (`kitchen_manager`, `general_manager`) with measurable KPIs.

3. **CrewAI Drafting & Validation Layer**:
   - **Operations Report Drafter**: Synthesizes multi-agent outputs into an executive-ready operational brief with material claim citations and historical cutoff disclosures.
   - **Operations Report Validator**: Audits drafts against retrieved evidence chunks to catch hallucinations, numerical contradictions, or data leaks.

4. **PII Gate 2 & Privacy Safeguards**:
   - Request-scoped `TokenVault` redacting customer names, emails, phones, and card numbers into reversible tokens (`[PERSON_1]`, `[EMAIL_1]`, `[PHONE_1]`) prior to any external LLM prompt.
   - Residual leakage scanner actively blocks prompt transmission if unmasked PII is present.

5. **Authoritative Quality Gate**:
   - Evidence grounding verification, cross-business isolation checks, and contradiction detection.
   - Explainable deterministic confidence scoring formula ($0\text{--}100$):
     $$\text{Score} = 100 \times \max(0, \min(1, 0.35 R_{\text{ev}} + 0.25 T_{\text{src}} + 0.20 A_{\text{agree}} - 0.20 P_{\text{stale}}))$$
   - LLM-as-Judge evaluation with bounded timeout fallback.

6. **Dual Frontend Interfaces**:
   - **React Dashboard** (Vite + TypeScript + Lucide): Apple-inspired operations workspace with timeline execution traces, scenario picker, and interactive tabs.
   - **Streamlit Dashboard**: Lightweight Python UI for rapid demonstrations.

---

## Project Structure

```text
├── app/
│   ├── api/                     # FastAPI Gateway routes, schemas, and SQLite persistence
│   │   ├── routes.py            # Endpoints: /analysis, /agents/status, /scenarios, /feedback
│   │   ├── schemas.py           # Pydantic request/response models
│   │   ├── persistence.py       # SQLite database operations (traces, logs, feedback)
│   │   └── scenarios.py         # Grounded demo scenarios from Yelp dataset
│   ├── agents/                  # 4 Specialist Agent microservices (Ports 8001-8004)
│   │   ├── classification/      # Port 8001
│   │   ├── trends/              # Port 8002
│   │   ├── root_cause/          # Port 8003
│   │   └── action_planner/      # Port 8004
│   ├── crew/                    # CrewAI drafting & validation layer
│   │   ├── consolidation.py     # Operations Report Drafter & Validator agents
│   │   └── schemas.py           # Crew report & validation models
│   ├── orchestration/           # LangGraph workflow engine
│   │   ├── graph.py             # StateGraph definition and compiled workflow
│   │   ├── nodes.py             # LangGraph node execution functions
│   │   └── state.py             # Typed orchestration state with reducers
│   ├── quality/                 # Quality Gate & Privacy
│   │   ├── confidence.py        # Explainable confidence scoring formula
│   │   ├── gate.py              # Quality Gate orchestrator
│   │   ├── grounding.py         # Evidence overlap & isolation validator
│   │   ├── judge.py             # LLM-as-Judge evaluator
│   │   └── statistical.py       # Freshness & cutoff date disclosures
│   ├── retrieval/               # Hybrid RAG engine (BM25 + Dense + RRF)
│   ├── ui/                      # Streamlit alternative dashboard
│   │   └── streamlit_app.py
│   └── main.py                  # FastAPI Application entry point (Port 8000)
├── ui/                          # Apple-inspired React + TypeScript + Vite Dashboard
│   ├── src/
│   │   ├── components/          # Sidebar, Workspace, Results, Trace, Activity, Feedback
│   │   ├── api/client.ts        # Typed API client
│   │   └── types/index.ts       # Frontend models
│   └── package.json
├── evaluation/
│   └── deepeval_eval.py         # DeepEval recommendation evaluation benchmark
├── data_store/                  # Parquet aggregates, vector index, and SQLite DB
└── tests/                       # 101 automated pytest unit & integration tests
```

---

## Setup & Installation

### Prerequisites
- **Python**: 3.12+
- **Node.js**: v20+ (v25.9 tested) & npm

### 1. Python Environment Setup
```powershell
# Create and activate virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install required dependencies
pip install -r requirements.txt
```

### 2. React UI Setup
```powershell
cd ui
npm install
npm run build
cd ..
```

---

## Running the Application

### Option A: Launch All Microservices & Gateway
```powershell
# Terminal 1: Start the 4 Specialist Agents (Ports 8001, 8002, 8003, 8004)
python scripts/start_agents.py

# Terminal 2: Start the FastAPI Gateway (Port 8000)
python app/main.py
```

### Option B: Launch the React Dashboard
```powershell
# Terminal 3: Start Vite Dev Server (Port 3000)
cd ui
npm run dev
```
Open browser at `http://localhost:3000`.

### Option C: Launch the Streamlit Alternative Dashboard
```powershell
streamlit run app/ui/streamlit_app.py
```
Open browser at `http://localhost:8501`.

---

## API Documentation & Sample Usage

The interactive Swagger UI is available at `http://localhost:8000/docs`.

### 1. Common Multi-Agent Analysis (`POST /api/v1/analysis`)
```json
{
  "query": "Customers are repeatedly complaining about slow service and long wait times. What operational issues are causing this, and what should we fix first?",
  "mode": "auto",
  "business_id": "_ab50qdWOk0DdB6XOrBitw",
  "viewer_role": "manager"
}
```

### 2. Single Agent Direct Invocation (`POST /api/v1/analysis`)
```json
{
  "query": "Classify severe food quality and temperature complaints",
  "mode": "agent",
  "agent": "classification",
  "business_id": "_ab50qdWOk0DdB6XOrBitw"
}
```

### 3. Check Live Agent Status (`GET /api/v1/agents/status`)
Returns live connectivity and latency for ports 8001 through 8004.

### 4. Submit Manager Feedback (`POST /api/v1/feedback`)
```json
{
  "request_id": "req_12345",
  "rating": 5,
  "useful": true,
  "comment": "Accurately diagnosed dining delays and prioritized expediter staffing."
}
```

---

## Verification & Test Results

Run the complete automated test suite across all phases:
```powershell
$env:MOCK_LLM="1"
pytest tests -v
```

**Results: 101 passed, 0 failures across all suites**:
- `tests/phase_0` (Ingestion & Schema): 5 passed
- `tests/phase_1` (Preprocessing & Aggregates): 3 passed
- `tests/phase_2` (Vector Store & Hybrid RAG): 4 passed
- `tests/phase_3` (Specialist Agents & A2A Protocol): 4 passed
- `tests/phase_4` (Quality Gate & Grounding): 2 passed
- `tests/orchestration` (LangGraph & Concurrency): 17 passed
- `tests/quality` (PII Gate 2, Confidence, DeepEval): 22 passed
- `tests/crew` (CrewAI Drafter & Validator): 4 passed
- `tests/api` (FastAPI Endpoints & SQLite): 7 passed
- Microservice & Integration Tests: 33 passed
