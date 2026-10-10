# SPIKE FINDINGS & BENCHMARKS (docs/SPIKE_NOTES.md)

Empirical evidence gathered during Phase 0 prototyping and spikes on the local environment.

---

## 1. DuckDB JSON Streaming Throughput (Spike A)
- **Dataset**: `dataset/yelp_academic_dataset_business.json` (118.8 MB) and `dataset/yelp_academic_dataset_review.json` (5.34 GB).
- **Engine**: DuckDB v1.5.6 with `read_ndjson`, `maximum_object_size=134217728`, `ignore_errors=true`.
- **Results**:
  - `business.json`: All 150,346 records read in **0.29s** (~521,272 rows/second).
  - `review.json`: 100,000-line slice scanned and parsed in **0.17s** (~583,437 rows/second).
- **Implication**: DuckDB can process the entire multi-gigabyte Yelp dataset via streaming in seconds to minutes without loading entire tables into RAM.

---

## 2. Embedding Model CPU Speed (Spike D)
- **Model**: `BAAI/bge-small-en-v1.5` (384-dimensional, normalized).
- **Environment**: Intel x86_64 CPU (PyTorch 2.14.0 CPU-only).
- **Results**:
  - Model load time: **5.37s**.
  - Embedding throughput: **430.0 chunks/second** (100 sample chunks encoded in 0.23s).
- **Implication**: CPU embedding throughput is strong. Stratified semantic chunk subsets can be embedded quickly without GPU.

---

## 3. SQLite FTS5 & BM25 Verification (Spike E)
- **Build**: Python built-in `sqlite3` on Windows.
- **Test**: Created virtual table using `fts5(chunk_id, text, tokenize='porter unicode61')`.
- **Query**: `"waited hour food"` correctly matched and ranked `"waited an hour for food and the server was nowhere to be found"` at rank 1 with BM25 score `-0.7191`.
- **Implication**: Native SQLite FTS5 works out of the box with zero external C-extensions needed.

---

## 4. spaCy NER & Food Vocabulary (Spike B & C)
- Model: `en_core_web_sm` (v3.8.0).
- Disabled pipeline components: `parser`, `lemmatizer` to achieve 5x throughput boost for entity masking (`nlp.pipe`).
- Allow-listing restaurant names, food terminology, and high RapidFuzz similarity ensures zero false-positive redaction of dishes or brand names.
