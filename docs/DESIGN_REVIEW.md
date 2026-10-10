# DESIGN REVIEW & RISK CRITIQUE (docs/DESIGN_REVIEW.md)

## Round 1: 12 Identified Weaknesses and System Risks

1. **Memory Exhaustion on 5.34 GB Review & 3.36 GB User JSON**: Loading either file completely into a Pandas DataFrame will cause immediate Out-Of-Memory (OOM) faults on a 16 GB machine with ~4.2 GB free RAM.
2. **Windows Pathing and Encoding Bugs**: Windows default code pages (CP1252) break on Unicode emojis, accented review text, and non-ASCII business names unless UTF-8 is strictly enforced everywhere.
3. **DuckDB Object Size Limit on User `friends` Array**: Raw `user.json` contains lines with up to 100,000 friend IDs causing standard JSON parsers to abort unless `maximum_object_size` is expanded (to 128 MB) and `friends` is dropped immediately during projection.
4. **Yelp Python 2 Stringified Attributes**: Attribute values such as `"{u'valet': False, u'lot': True}"` fail standard JSON parsing and will crash naive parsers unless parsed using safe AST literal evaluation.
5. **PII Masking False Positives on Restaurant & Food Entities**: Standard spaCy PERSON NER frequently flags French dish names, chef names that match restaurant titles (e.g. "Dmitri's", "Milktooth"), and menu items as persons.
6. **FTS5 Token Pollution by PII Redaction Tokens**: Indexing `[PERSON]` into SQLite FTS5 causes queries containing common words or pronouns to artificially hit masked review tokens.
7. **Dense Vector Dimension Mismatch and Storage Drift**: Inconsistent normalization or vector size mismatch degrades cosine similarity search; float16 vs float32 precision conversions must maintain parity.
8. **Skewed Aspect Tagging from Star Ratings Alone**: A 4-star review mentioning "food was great but wait time was an hour" is still an operational complaint about service speed.
9. **Zero-Division and Small-Sample Noise in Trend Forecasting**: Restaurants with fewer than 10 reviews in a 3-month window will produce erratic volatility in percentage trends.
10. **Data Leakage in Satisfaction Modeling**: Training a review satisfaction classifier using the review's own star rating or test-split reviews from the future creates severe target leakage.
11. **Concurrency and Database Locks in SQLite**: Default SQLite database locking fails under concurrent reads/writes unless WAL (Write-Ahead Logging) mode is activated.
12. **CPU Embedding Latency on Massive Chunk Sets**: Embedding millions of reviews on CPU without GPU acceleration would stall the pipeline for days if unconstrained.

---

## Round 2: Mitigation Decisions & Strategy

1. **DuckDB Streaming & Batching**: Confirmed in Spike A. We stream row-group batches and output compressed Parquet directly to disk, keeping RAM footprint under 500 MB.
2. **UTF-8 Everywhere**: All file handlers explicitly use `encoding='utf-8'`.
3. **Column Pruning at Ingestion**: Ingest passes project only required columns; `friends` and raw user `name` are never materialized into Parquet.
4. **AST Defensive Parsing**: Attribute parsers wrap `ast.literal_eval` with robust try/except fallbacks, defaulting safely to null/empty dicts.
5. **Allow-List & Fuzzy Matching in Gate 1**: Before masking any detected PERSON entity, we verify against `food_vocab.txt`, business names, and rapidfuzz similarity >= 90 against the restaurant's name.
6. **BM25 Placeholder Stripping**: Placeholder strings (`[PERSON]`, `[PHONE]`, etc.) are stripped from text prior to FTS5 virtual table insertion and queries.
7. **Two-Layer Data Architecture**: Statistics layer processes all restaurant reviews; semantic layer embeds a stratified subset prioritizing negative complaints (70/30 ratio) balanced across restaurants.
8. **SQLite WAL Mode**: SQLite connections configure `PRAGMA journal_mode=WAL` and `PRAGMA synchronous=NORMAL`.
9. **Modular Architecture & Unit Testing**: Strict phase-by-phase gates with unit tests verifying each component before moving to the next phase.
