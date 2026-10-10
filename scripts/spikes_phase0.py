"""Phase 0 Spikes:
1. DuckDB JSON performance reading real business.json and 100k review.json slice.
2. Embedding model CPU speed (chunks/sec).
3. SQLite FTS5 BM25 functionality.
"""

import time
import sqlite3
import duckdb
from pathlib import Path
from sentence_transformers import SentenceTransformer


def spike_duckdb_json():
    print("--- Spike A: DuckDB JSON read performance ---")
    con = duckdb.connect()
    
    bus_path = Path("dataset/yelp_academic_dataset_business.json").resolve()
    t0 = time.time()
    bus_count = con.execute(f"""
        SELECT count(*) 
        FROM read_ndjson('{str(bus_path).replace(chr(92), "/")}', maximum_object_size=134217728, ignore_errors=true)
    """).fetchone()[0]
    bus_time = time.time() - t0
    print(f"Business count: {bus_count} rows in {bus_time:.2f}s ({bus_count/bus_time:.0f} rows/s)")

    rev_path = Path("dataset/yelp_academic_dataset_review.json").resolve()
    t0 = time.time()
    rev_slice_count = con.execute(f"""
        SELECT count(*) 
        FROM (
            SELECT review_id, business_id, stars, text 
            FROM read_ndjson('{str(rev_path).replace(chr(92), "/")}', 
                             maximum_object_size=134217728, 
                             ignore_errors=true,
                             columns={{'review_id': 'VARCHAR', 'business_id': 'VARCHAR', 'stars': 'DOUBLE', 'text': 'VARCHAR'}})
            LIMIT 100000
        )
    """).fetchone()[0]
    rev_time = time.time() - t0
    print(f"Review 100k slice: {rev_slice_count} rows in {rev_time:.2f}s ({rev_slice_count/rev_time:.0f} rows/s)")
    return {
        "bus_count": bus_count, "bus_time": bus_time,
        "rev_slice_count": rev_slice_count, "rev_time": rev_time
    }


def spike_embedding_speed():
    print("\n--- Spike D: Embedding model load & speed on CPU ---")
    t0 = time.time()
    model = SentenceTransformer("BAAI/bge-small-en-v1.5")
    load_time = time.time() - t0
    print(f"Model loaded in {load_time:.2f}s")

    sample_chunks = [
        "The wait time was over an hour and our food arrived cold and bland.",
        "Waitress was extremely rude when we asked for the check.",
        "Excellent crust and wonderful fresh basil on the margherita pizza.",
        "Terrible experience, undercooked chicken and dirty tables everywhere.",
        "Great atmosphere and craft beer selection, will definitely visit again."
    ] * 20  # 100 chunks

    t0 = time.time()
    embeddings = model.encode(
        sample_chunks,
        batch_size=32,
        normalize_embeddings=True,
        show_progress_bar=False
    )
    enc_time = time.time() - t0
    chunks_per_sec = len(sample_chunks) / enc_time
    print(f"Encoded {len(sample_chunks)} chunks in {enc_time:.2f}s ({chunks_per_sec:.1f} chunks/sec)")
    return {"load_time": load_time, "chunks_per_sec": chunks_per_sec}


def spike_fts5():
    print("\n--- Spike E: SQLite FTS5 & BM25 verification ---")
    con = sqlite3.connect(":memory:")
    con.execute("""
        CREATE VIRTUAL TABLE test_fts USING fts5(
            chunk_id, text, tokenize='porter unicode61'
        )
    """)
    samples = [
        ("c1", "waited an hour for food and the server was nowhere to be found"),
        ("c2", "best pizza in Philadelphia, wonderful crust"),
        ("c3", "cold food and slow service, terrible wait time"),
        ("c4", "the burger was juicy and flavorful, great fries")
    ]
    con.executemany("INSERT INTO test_fts(chunk_id, text) VALUES (?, ?)", samples)

    query = "waited hour food"
    cursor = con.execute("""
        SELECT chunk_id, bm25(test_fts) as rank, text 
        FROM test_fts 
        WHERE test_fts MATCH ? 
        ORDER BY rank
    """, (query,))
    rows = cursor.fetchall()
    for row in rows:
        print(f"Hit: {row[0]} | BM25: {row[1]:.4f} | Text: {row[2][:50]}")
    fts_ok = len(rows) > 0 and rows[0][0] == "c1"
    print(f"FTS5 test status: {'OK' if fts_ok else 'FAIL'}")
    return {"fts_ok": fts_ok}


if __name__ == "__main__":
    duck_res = spike_duckdb_json()
    emb_res = spike_embedding_speed()
    fts_res = spike_fts5()
    print("\nAll spikes completed successfully.")
