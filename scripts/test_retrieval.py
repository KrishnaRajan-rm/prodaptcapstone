import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import sqlite3
import numpy as np
from sentence_transformers import SentenceTransformer
from data_preprocessing.storage.vector_store import NumpyMemmapDenseStore


def test_bm25_search():
    print("=" * 60)
    print("1. SQLITE FTS5 BM25 SEARCH TEST")
    print("=" * 60)
    con = sqlite3.connect("data_store/sqlite/restaurant_ops.db")
    query = "waited hour food"
    cur = con.cursor()
    rows = cur.execute("""
        SELECT m.chunk_id, m.restaurant_name, m.city, m.stars, m.text_masked, bm25(chunks_fts) as rank
        FROM chunks_fts f
        JOIN chunks_meta m ON f.chunk_id = m.chunk_id
        WHERE chunks_fts MATCH ?
        ORDER BY rank
        LIMIT 3
    """, (query,)).fetchall()
    
    for r in rows:
        print(f"Hit: [{r[0]}] {r[1]} ({r[2]}, {r[3]} stars) | Score: {r[5]:.4f}")
        print(f"Text: {r[4][:120]}...\n")
    con.close()
    return len(rows) > 0


def test_dense_search():
    print("=" * 60)
    print("2. DENSE VECTOR COSINE SEARCH TEST")
    print("=" * 60)
    con = sqlite3.connect("data_store/sqlite/restaurant_ops.db")
    store = NumpyMemmapDenseStore("data_store/vectors/vectors.f16.npy", "data_store/vectors/vector_ids.parquet", dim=384)
    model = SentenceTransformer("BAAI/bge-small-en-v1.5")
    bge_prefix = "Represent this sentence for searching relevant passages: "
    q_emb = model.encode([bge_prefix + "waited an hour for food"], normalize_embeddings=True)[0]

    # Search over all populated chunks so far
    import json
    prog_path = "data_store/progress/vectors_progress.json"
    with open(prog_path, "r", encoding="utf-8") as f:
        p = json.load(f)
    n_done = p.get("completed_chunks", 3200)

    hits = store.search(q_emb, candidate_row_indexes=list(range(n_done)), top_k=3)
    for cid, score in hits:
        row = con.execute("SELECT restaurant_name, city, stars, text_masked FROM chunks_meta WHERE chunk_id = ?", (cid,)).fetchone()
        if row:
            print(f"Hit: [{cid}] {row[0]} ({row[1]}, {row[2]} stars) | Cosine: {score:.4f}")
            print(f"Text: {row[3][:120]}...\n")
    con.close()
    return len(hits) > 0


if __name__ == "__main__":
    bm25_ok = test_bm25_search()
    dense_ok = test_dense_search()
    print("=" * 60)
    print(f"BM25 Search:  {'[PASS]' if bm25_ok else '[FAIL]'}")
    print(f"Dense Search: {'[PASS]' if dense_ok else '[FAIL]'}")
    print("=" * 60)
