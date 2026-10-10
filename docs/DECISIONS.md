# ARCHITECTURAL DECISIONS LOG (docs/DECISIONS.md)

All design decisions, rationale, spikes, and empirical findings are documented here.

---

## Decision 1: Python Environment & PyTorch Setup
- **Context**: Host system is Windows 10/11 running Python 3.12 (64-bit), CPU only.
- **Decision**: Created virtual environment `.venv` using `--system-site-packages` to leverage pre-installed scientific computing dependencies (torch, transformers, sentence-transformers, pyarrow, pandas) without redundant disk space consumption (~21.5 GB free disk space available). Installed `duckdb` (v1.5.6), `spacy` (v3.8.16), `en_core_web_sm` (v3.8.0), `vaderSentiment`, `statsmodels`, `plotly`, `pytest`, `pytest-asyncio`, and `rapidfuzz`.
- **Status**: LOCKED.

## Decision 2: Batching & Memory Safety for Heavy Yelp Dataset
- **Context**: Yelp review dataset is ~5.34 GB uncompressed JSONL, user dataset is ~3.36 GB JSONL. System memory has ~4.2 GB free RAM.
- **Decision**: Strictly avoid loading full raw JSON or full review/user tables into Python memory or Pandas. Use DuckDB streaming reads with `read_ndjson` and explicit schema projection, writing directly to Snappy/zstd compressed Parquet in `data_store/parquet/`. Ingestion runs in staged, resumable passes tracked by `data_store/progress/ingestion.json` with temporary file atomic rename (`.tmp` -> `.parquet`).
- **Status**: LOCKED.

## Decision 3: Restaurant Category Filtering
- **Context**: Yelp dataset includes non-restaurant businesses (dentists, retail, car repair, salons) and grocery stores.
- **Decision**: Ingest business records where categories match the positive restaurant list (`Restaurants`, `Cafes`, `Coffee & Tea`, `Bakeries`, `Bars`, `Breweries`, `Brewpubs`, `Pubs`, `Food Trucks`, `Diners`, `Fast Food`, `Desserts`, `Ice Cream & Frozen Yogurt`) and do NOT contain negative terms (`Grocery`, `Convenience Stores`).
- **Status**: LOCKED.

## Decision 4: Two-Layer Architecture (Full vs Embedded Subset)
- **Context**: The full dataset contains millions of reviews. Embedding millions of reviews on CPU would take days.
- **Decision**:
  - Statistics layer: all restaurant reviews processed into Parquet and DuckDB for accurate trends, aggregations, peer analysis, and satisfaction models.
  - Semantic layer: stratified sampling for embedded chunks (stars <= 3 prioritized 70%, 4-5 stars 30%, capped by `EMBED_MAX_CHUNKS` or `DEV_MAX_RESTAURANTS` in dev mode), preserving representative coverage.
- **Status**: LOCKED.

## Decision 5: PII Protection Architecture
- **Context**: Review text contains personal names, phone numbers, emails, URLs, and credit card patterns.
- **Decision**: Gate 1 at ingest masks PERSON (via spaCy `en_core_web_sm` with parser/lemmatizer disabled for performance) and regexes for contact info, while preserving business names (and variants with RapidFuzz ratio >= 90) and food terms via `food_vocab.txt`. Original text is safely stored in a local SQLite `pii_vault` table for authorized manager-role retrieval.
- **Status**: LOCKED.

## Decision 6: Retrieval & Storage Stack
- **Context**: Must run completely locally on Windows with zero Docker containers and no cloud dependencies.
- **Decision**:
  - Metadata + BM25: SQLite FTS5 table with Porter tokenizer.
  - Dense Search: Normalized 384-dimensional float16 embeddings in memory-mapped NumPy file (`vectors.f16.npy`) with index mapping in `vector_ids.parquet`.
  - Fusion: Reciprocal Rank Fusion (k=60).
- **Status**: LOCKED.
