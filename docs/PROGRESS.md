# PROJECT PROGRESS CHECKLIST (docs/PROGRESS.md)

| Phase | Description | Status | Evidence / Notes |
|---|---|---|---|
| Phase 0 | Bootstrap, Spikes, Config, Smoke Test, Design Review | PASS | Venv ready, 5/5 pytest passing, Spike A, D, E complete, smoke_test passed |
| Phase 1 | Streaming Ingestion (Raw JSONL -> Filtered Parquet) | PASS | Full dataset: 59,309 restaurants, 5,007,575 reviews, 1,491,027 users; 0 orphans, 0 dups |
| Phase 2 | Cleaning (Attributes, Hours, Text, Credibility, Peaks) | PASS | 59,309 profiles enriched, 1.49M users scored, 5.00M reviews cleaned & deduped |
| Phase 3 | Preprocessing (Aspect Lexicon, Tagging, Sampling, PII Gate 1, Chunking) | PASS | 5.0M reviews tagged, 1.98M monthly aggregates, 59,894 chunks generated, PII audits exported |
| Phase 4 | Storage & Indexing (SQLite FTS5, BM25, NumPy Memmap Embeddings) | PASS | SQLite 59,894 chunks + FTS5 + vault, vectors.f16.npy, live BM25 & dense retrieval verified |

---

## Commands & Verification Log

- **Environment Bootstrap**:
  - Python 3.12 (64-bit Windows) virtualenv `.venv` initialized with `--system-site-packages`.
  - Installed `duckdb`, `spacy` + `en_core_web_sm`, `vaderSentiment`, `statsmodels`, `plotly`, `pytest`, `pytest-asyncio`, `rapidfuzz`.
  - Directory tree initialized.
