"""Phase 0 Smoke Test script.
Verifies all dependencies, imports, DuckDB JSON reads, and embedding model loading.
"""

import sys
import time
from pathlib import Path


def run_smoke_test():
    results = {}
    print("=" * 60)
    print("RESTAURANT OPERATIONS ASSISTANT - PHASE 0 SMOKE TEST")
    print("=" * 60)

    # 1. Imports check
    libs = [
        ("duckdb", "DuckDB SQL Engine"),
        ("pyarrow", "Apache PyArrow"),
        ("pandas", "Pandas"),
        ("numpy", "NumPy"),
        ("pydantic", "Pydantic v2"),
        ("pydantic_settings", "Pydantic Settings"),
        ("dotenv", "Python Dotenv"),
        ("tqdm", "TQDM Progress Bars"),
        ("rapidfuzz", "RapidFuzz"),
        ("spacy", "spaCy NLP"),
        ("vaderSentiment.vaderSentiment", "VADER Sentiment"),
        ("sentence_transformers", "Sentence Transformers"),
        ("sklearn", "Scikit-Learn"),
        ("statsmodels", "Statsmodels"),
        ("joblib", "Joblib"),
        ("fastapi", "FastAPI"),
        ("uvicorn", "Uvicorn Server"),
        ("streamlit", "Streamlit UI"),
        ("plotly", "Plotly Charts"),
        ("httpx", "HTTPX Client"),
        ("pytest", "Pytest Framework"),
    ]

    for mod_name, label in libs:
        try:
            __import__(mod_name)
            results[label] = "OK"
        except Exception as e:
            results[label] = f"FAIL ({e})"

    # 2. Check spaCy en_core_web_sm
    try:
        import spacy
        nlp = spacy.load("en_core_web_sm")
        doc = nlp("John ordered pizza at Joe's Diner in New York.")
        results["spaCy model (en_core_web_sm)"] = "OK"
    except Exception as e:
        results["spaCy model (en_core_web_sm)"] = f"FAIL ({e})"

    # 3. DuckDB 3-row JSON read from real dataset
    try:
        import duckdb
        con = duckdb.connect()
        bus_path = Path("dataset/yelp_academic_dataset_business.json").resolve()
        if bus_path.exists():
            df = con.execute(f"""
                SELECT business_id, name, city, state, stars
                FROM read_ndjson('{str(bus_path).replace(chr(92), "/")}', maximum_object_size=134217728, ignore_errors=true)
                LIMIT 3
            """).df()
            if len(df) == 3:
                results["DuckDB 3-row JSON read"] = f"OK ({len(df)} rows read)"
            else:
                results["DuckDB 3-row JSON read"] = f"FAIL (unexpected count: {len(df)})"
        else:
            results["DuckDB 3-row JSON read"] = "FAIL (dataset file not found)"
    except Exception as e:
        results["DuckDB 3-row JSON read"] = f"FAIL ({e})"

    # 4. SQLite FTS5 check
    try:
        import sqlite3
        con_sql = sqlite3.connect(":memory:")
        con_sql.execute("CREATE VIRTUAL TABLE test_fts USING fts5(content, tokenize='porter unicode61')")
        con_sql.execute("INSERT INTO test_fts(content) VALUES ('slow service waited forever')")
        res = con_sql.execute("SELECT * FROM test_fts WHERE test_fts MATCH 'wait*'").fetchall()
        if len(res) == 1:
            results["SQLite FTS5 Porter tokenize"] = "OK"
        else:
            results["SQLite FTS5 Porter tokenize"] = "FAIL (query did not match)"
    except Exception as e:
        results["SQLite FTS5 Porter tokenize"] = f"FAIL ({e})"

    # 5. Sentence transformer load check
    try:
        from sentence_transformers import SentenceTransformer
        t0 = time.time()
        # Use small model or verify import
        model = SentenceTransformer("BAAI/bge-small-en-v1.5")
        emb = model.encode(["Checking smoke test embedding latency"], normalize_embeddings=True)
        results["SentenceTransformer (BAAI/bge-small-en-v1.5)"] = f"OK (dim={emb.shape[1]}, load+enc={time.time()-t0:.2f}s)"
    except Exception as e:
        results["SentenceTransformer (BAAI/bge-small-en-v1.5)"] = f"FAIL ({e})"

    # Print summary table
    all_ok = True
    print("\n{:<45} | {:<20}".format("COMPONENT / MODULE", "STATUS"))
    print("-" * 68)
    for k, v in results.items():
        print("{:<45} | {:<20}".format(k, v))
        if not v.startswith("OK"):
            all_ok = False

    print("=" * 68)
    if all_ok:
        print("ALL SMOKE TESTS PASSED [OK]")
        return 0
    else:
        print("SOME SMOKE TESTS FAILED [FAIL]")
        return 1


if __name__ == "__main__":
    sys.exit(run_smoke_test())
