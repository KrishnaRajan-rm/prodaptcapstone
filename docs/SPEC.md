# SPECIFICATION: AI-Powered Restaurant Operations & Review Assistant

Source: `Antigravity_Master_Prompt_Restaurant_Operations_Assistant.pdf` (Sections 1, 4, 5, 6)

---

## 1. MISSION

Build a semantic retrieval and analysis system on the Yelp Open Dataset that lets a restaurant manager type a natural-language question and get:
1. The most relevant historical reviews (hybrid keyword + semantic search, with metadata filters).
2. Complaint classification, prioritization, root-cause analysis, trend analysis and forecasts.
3. Recommended corrective actions, each backed by cited evidence.
4. A confidence score, quality checks, PII protection and a full execution trace.

Everything must run on free resources only: a free Google Gemini API key, free OpenRouter models, Google ADK, and local open-source models. No paid services. No Docker required.

### Required features (every item must exist and be demonstrable)
#### Requirement 1 (basic):
- RAG similarity search over restaurant reviews
- Hybrid search (keywords + semantic)
- Customer complaint classification agent
- Metadata filtering (restaurant, category, location, rating)
- Basic operational improvement recommendation
- Input validation guardrails
- Complaint prioritization
- All core functions exposed as API endpoints

#### Requirement 2 (advanced):
- DeepEval for recommendation quality evaluation
- Customer satisfaction prediction
- LLM-as-judge for complaint analysis validation
- Multi-agent restaurant operations collaboration
- Complaint trend prediction
- Review aspect and root-cause analysis
- Feedback loop for continuous improvement
- A simple front-end interface

Owner add-ons (also required): PII protection (gate 1 at ingest, gate 2 at the LLM boundary, role-based decode), groundedness and confidence pipeline, execution trace shown in the output.

Deliverables: architecture diagram, design document with trade-offs, full runnable code with README (setup + sample usage), and a 10-minute demo script.

---

## 4. LOCKED ARCHITECTURE DECISIONS

### 4.1 Two layers of data
- **Statistics layer (all restaurant data)**: Parquet + DuckDB. Every restaurant, every review, used for trends, peers, prioritization, prediction.
- **Semantic layer (embedded subset)**: only selected chunks are embedded. Full mode caps at `EMBED_MAX_CHUNKS`, chosen by stratified sampling: all reviews with stars <= 3 first, plus a sample of 4-5 star reviews (target 70/30), balanced across restaurants (each restaurant with >= 30 reviews represented), spread across years, deterministic seed. State this trade-off in `docs/DESIGN.md`.

### 4.2 Retrieval stack (chosen for reliability on a laptop with no Docker)
- **Metadata + keyword**: one SQLite database with a `chunks_meta` table (indexed on city, state, business_id, stars, year_month, is_open, categories) and an FTS5 virtual table for BM25 (`tokenize='porter unicode61'`). Escape and quote every user token before building the MATCH string. Placeholders like `[PERSON]` must be removed from the FTS query and excluded from the indexed text used for BM25.
- **Dense**: embeddings from `BAAI/bge-small-en-v1.5` (normalized, 384-d, float16 on disk) stored in a NumPy memmap file `vectors.f16.npy` plus a `vector_ids.parquet` mapping row index to `chunk_id`. Search = filter in SQLite to candidate row indexes, then one matrix-vector product on those rows, take top-k. Brute force is fast at this size (hundreds of thousands of vectors) and has no server or API drift. Put it behind a `DenseStore` interface so a Qdrant adapter could be added later. Do NOT use Qdrant local mode (not meant for this scale).
- **Fusion**: Reciprocal Rank Fusion (`k=60`) of BM25 top-100 and dense top-100.
- **Rerank**: cross-encoder `cross-encoder/ms-marco-MiniLM-L-6-v2` on the top 40, return top 10. If it cannot load, skip rerank and set a warning flag.
- **Extras**: query expansion (Phase 5), recency boost `score = 1 + 0.15 * exp(-age_days / 365)`, near-duplicate removal (normalized text hash, cosine > 0.97), per-reviewer cap of 2 results, bounded feedback boost (+-10%).
- Query text for bge gets the prefix: `Represent this sentence for searching relevant passages: `

### 4.3 PII design (two gates)
- **Gate 1 (ingest, only on the embedded subset)**: detect PERSON via spaCy `en_core_web_sm` (disable parser/lemmatizer), plus regex for email, phone, URL, credit card. Replace with typed placeholders `[PERSON]` `[EMAIL]` `[PHONE]` `[URL]` `[CARD]`. Allow-list: never mask business names, tokens from business names, dish and food vocabulary (`data_preprocessing/preprocessing/food_vocab.txt`, at least 300 terms), city names, or anything with RapidFuzz ratio >= 90 against the review's own restaurant name (protects names like dmitris, Milktooth, Shalimar, Kanella, Caviar & Bananas). Original text goes to the `vault` table in SQLite keyed by `chunk_id`. Embed and index only the masked text. Save masking statistics.
- **Gate 2 (LLM boundary)**: runs on the whole outgoing evidence pack and user query. Same detectors plus allow-list. New hits become numbered reversible tokens `<PERSON_1>`; the mapping lives in memory for that one request only. Defense in depth: an ADK/LLM-call safety-net scan that blocks any call whose payload still matches PII regexes.
- **Decode**: only after the quality gate, only in the API response layer, only for role manager. Evidence quotes for managers are fetched from the vault by `chunk_id`. Role analyst always sees masked text. Traces and logs always stay masked.
- **Experiment**: `evaluation/pii_experiment.py` compares retrieval on raw vs masked embeddings (1,000 reviews, 30 queries, top-10 overlap and recall) and writes `evaluation/results/pii_experiment.json`.

### 4.4 Aspect taxonomy and tagging
- **Aspects (multi-label)**: `wait_time`, `food_quality`, `staff_behavior`, `order_accuracy`, `cleanliness`, `price_value`, `portion_size`, `ambience_noise`, `takeout_delivery`, `health_safety`.
- **Severity weights**: `health_safety`: 1.0, `cleanliness`: 0.8, `staff_behavior`: 0.7, `food_quality`: 0.7, `order_accuracy`: 0.6, `wait_time`: 0.6, `price_value`: 0.5, `takeout_delivery`: 0.5, `portion_size`: 0.4, `ambience_noise`: 0.3.
- **All reviews (statistics layer)**: review-level aspect flags computed in DuckDB SQL with `regexp_matches` on lowercase text. At least 30 patterns per aspect in `data_preprocessing/preprocessing/aspect_lexicon.yaml`. A "complaint" = aspect flag AND (stars <= 3 OR negative cue regex hit).
- **Embedded chunks**: sentence-level aspect + sentiment (VADER compound <= -0.05 = negative), stored per chunk as `aspects` (list) and `aspect_sentiments` (JSON).
- Reviews with stars 4 that mention "service was very slow" must be captured, so do not filter complaints by stars alone.
- Audit script `scripts/aspect_audit.py` to export 150 random reviews with tags to CSV.

### 4.5 Agents (Google ADK, single framework; do not use CrewAI)
- Python code (not the LLM) performs retrieval and statistics first and builds an Evidence Pack (JSON). Agents never call tools; they receive the pack via ADK session state.
- Workflow: `SequentialAgent[ ParallelAgent[ClassifierAgent, TrendAnalystAgent], RootCauseAgent, ActionPlannerAgent ]`, then a `LoopAgent` (`max_iterations=2`) around quality-gate retry.
- Each agent has an `output_schema` (Pydantic) and `output_key`. Prompts in `app/agents/prompts/`.
- Priority (P1/P2/P3) is computed in pure code, never by the LLM.

### 4.6 Prioritization (pure code)
For each aspect of the filtered scope, over the last 90 days (configurable):
`priority_score = 100 * (0.35 * frequency + 0.25 * severity + 0.20 * trend + 0.20 * rating_impact)`
- Any `health_safety` complaint hit, in the window, forces P1.
- P1 >= 60, P2 40-59, P3 < 40.

### 4.7 Quality gate and confidence (Python)
1. Schema check. 2. Citation check. 3. Number check. 4. Groundedness. 5. Confidence 0-100.

### 4.8 LLM gateway (single choke point)
`app/core/llm/gateway.py`: LiteLLM underneath, ordered candidate lists, token bucket rate limiter, SQLite response cache, mock LLM for tests.

---

## 5. FOLDER STRUCTURE

```
dataset/
data_preprocessing/
  ingestion/
  cleaning/
  preprocessing/
    aspect_lexicon.yaml
    food_vocab.txt
  storage/
data_store/
  parquet/{filtered,clean,features,aggregates}/
  sqlite/
  vectors/
  models/
  progress/
app/
  core/
    config.py, logging.py, errors.py, schemas.py
    llm/
    pii/
    trace/
  guardrails/
  retrieval/
  analytics/
  agents/
  quality/
  feedback/
  api/
ui/
evaluation/
scripts/
tests/
docs/
```

---

## 6. DATA FACTS

- JSON Lines (one object per line): `yelp_academic_dataset_{business,review,user,tip,checkin}.json`.
- Restaurant definition: categories contain any of `Restaurants`, `Cafes`, `Coffee & Tea`, `Bakeries`, `Bars`, `Breweries`, `Brewpubs`, `Pubs`, `Food Trucks`, `Diners`, `Fast Food`, `Desserts`, `Ice Cream & Frozen Yogurt`, and do NOT contain `Grocery`, `Convenience Stores`.
- Pipeline modes:
  - `dev`: one city (`DEV_CITY`), `review_count` between `DEV_MIN_REVIEWS` and 1,000, top `DEV_MAX_RESTAURANTS` by `review_count`.
  - `full`: all restaurants everywhere.
- Resumable streaming data pipeline with batch processing.
