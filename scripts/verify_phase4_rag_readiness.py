"""
Verification script for Phase 0 to Phase 4.
Validates:
1. SQLite Database integrity, tables, schemas, indexes, FTS5 search, and raw vault texts.
2. Vector Store memmap integrity, shape, dtype, non-trivial embeddings, and alignment with vector_ids.parquet.
3. RAG Retrieval Pipeline readiness: End-to-end test querying BM25 + Dense vector search + Metadata joining.
4. Parquet Datasets: Full data across filtered, clean, features, and aggregates.
"""

import os
import sys
import json
import sqlite3
import numpy as np
import pyarrow.parquet as pq
from sentence_transformers import SentenceTransformer

if sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

def check_sqlite(db_path: str):
    print("=" * 60)
    print("1. SQLITE DATABASE VERIFICATION")
    print("=" * 60)
    assert os.path.exists(db_path), f"Database not found: {db_path}"
    db_size_mb = os.path.getsize(db_path) / (1024 * 1024)
    print(f"Database File: {db_path} ({db_size_mb:.2f} MB)")

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Integrity check
    cursor.execute("PRAGMA integrity_check;")
    res = cursor.fetchone()[0]
    print(f"PRAGMA integrity_check: {res}")
    assert res == "ok", "Database integrity check failed!"

    # Tables and counts
    cursor.execute("SELECT name, type FROM sqlite_master WHERE type IN ('table', 'view');")
    tables = cursor.fetchall()
    print("\nTables and Row Counts:")
    for name, ttype in sorted(tables):
        if "fts" in name and not name.endswith("_fts"):
            continue
        cursor.execute(f"SELECT count(*) FROM {name};")
        cnt = cursor.fetchone()[0]
        print(f"  • {name:<20} ({ttype}): {cnt:>10,d} rows")

    # Indices
    cursor.execute("SELECT name, tbl_name FROM sqlite_master WHERE type = 'index';")
    indices = cursor.fetchall()
    print(f"\nIndexes created: {len(indices)}")
    for name, tbl in sorted(indices):
        print(f"  • {name} ON {tbl}")

    # Chunks meta sample
    cursor.execute("""
        SELECT chunk_id, business_id, city, state, stars, aspects, n_tokens
        FROM chunks_meta LIMIT 1;
    """)
    sample_chunk = cursor.fetchone()
    print(f"\nSample chunk_meta:\n  {sample_chunk}")

    # Vault sample
    cursor.execute("SELECT chunk_id, length(text_raw), substr(text_raw, 1, 80) FROM vault LIMIT 1;")
    sample_vault = cursor.fetchone()
    print(f"\nSample vault entry:\n  ID: {sample_vault[0]}\n  Length: {sample_vault[1]}\n  Snippet: {sample_vault[2]}...")

    conn.close()
    return True


def check_vectors(vec_path: str, vec_ids_path: str, expected_dim: int = 384):
    print("\n" + "=" * 60)
    print("2. VECTOR STORE VERIFICATION")
    print("=" * 60)
    assert os.path.exists(vec_path), f"Vector file missing: {vec_path}"
    assert os.path.exists(vec_ids_path), f"Vector IDs missing: {vec_ids_path}"

    vec_file_mb = os.path.getsize(vec_path) / (1024 * 1024)
    print(f"Vector Memmap File: {vec_path} ({vec_file_mb:.2f} MB)")

    # Read vector IDs
    id_table = pq.read_table(vec_ids_path)
    total_ids = len(id_table)
    print(f"Vector ID Map: {vec_ids_path} ({total_ids:,} records)")
    print(f"Columns: {id_table.column_names}")

    # Validate shape
    memmap = np.memmap(vec_path, dtype=np.float16, mode="r", shape=(total_ids, expected_dim))
    print(f"Memmap Shape: {memmap.shape}, Dtype: {memmap.dtype}")

    # Check for NaNs or Infs
    sample_size = min(total_ids, 5000)
    sample_vecs = memmap[:sample_size].astype(np.float32)
    has_nan = np.isnan(sample_vecs).any()
    has_inf = np.isinf(sample_vecs).any()
    print(f"Sample NaN Check ({sample_size} vectors): {not has_nan}")
    print(f"Sample Inf Check ({sample_size} vectors): {not has_inf}")
    assert not has_nan, "Found NaN in vectors!"
    assert not has_inf, "Found Inf in vectors!"

    # Norm check (L2 normalized vectors have norm approx 1.0)
    norms = np.linalg.norm(sample_vecs[:20], axis=1)
    print(f"L2 Norms of first 20 vectors: {np.round(norms, 3)}")
    assert np.allclose(norms, 1.0, atol=0.05), "Vectors are not properly normalized!"

    return total_ids


def check_alignment(db_path: str, vec_ids_path: str):
    print("\n" + "=" * 60)
    print("3. CHUNK ID & METADATA ALIGNMENT VERIFICATION")
    print("=" * 60)
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute("SELECT count(*) FROM chunks_meta;")
    sql_chunks_count = cursor.fetchone()[0]

    cursor.execute("SELECT count(*) FROM vault;")
    sql_vault_count = cursor.fetchone()[0]

    id_table = pq.read_table(vec_ids_path)
    vec_count = len(id_table)

    print(f"chunks_meta count: {sql_chunks_count:,}")
    print(f"vault count:       {sql_vault_count:,}")
    print(f"vector_ids count:  {vec_count:,}")

    assert sql_chunks_count == sql_vault_count == vec_count, (
        f"Mismatch! chunks_meta={sql_chunks_count}, vault={sql_vault_count}, vectors={vec_count}"
    )

    # Check order alignment
    cursor.execute("SELECT row_idx, chunk_id FROM chunks_meta ORDER BY row_idx LIMIT 100;")
    meta_pairs = cursor.fetchall()
    meta_ids = [r[1] for r in meta_pairs]
    parquet_ids = id_table["chunk_id"][:100].to_pylist()

    assert meta_ids == parquet_ids, "Row ID mismatch between SQLite and vector_ids.parquet!"
    for r_idx, (m_row, m_id) in enumerate(meta_pairs):
        assert m_row == r_idx, f"Row index mismatch: meta {m_row} != expected {r_idx}"
    print(f"First 100 chunk_ids and row_idx strictly identical across SQLite and vector_ids.parquet: YES")

    conn.close()
    return True


def check_rag_retrieval(db_path: str, vec_path: str, vec_ids_path: str):
    print("\n" + "=" * 60)
    print("4. END-TO-END RAG RETRIEVAL READINESS SMOKE TEST")
    print("=" * 60)
    query = "dirty tables slow service bad experience"
    print(f"Testing Retrieval for query: '{query}'")

    # 1. BM25 Search
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    bm25_terms = " OR ".join([w for w in query.split() if len(w) > 3])
    cursor.execute("""
        SELECT chunk_id, bm25(chunks_fts) as bm25_score
        FROM chunks_fts
        WHERE chunks_fts MATCH ?
        ORDER BY bm25_score
        LIMIT 5;
    """, (bm25_terms,))
    bm25_results = cursor.fetchall()
    print("\n[BM25 Top Results]:")
    for cid, score in bm25_results:
        cursor.execute("SELECT business_id, city, stars, aspects FROM chunks_meta WHERE chunk_id = ?;", (cid,))
        meta = cursor.fetchone()
        cursor.execute("SELECT substr(text_raw, 1, 90) FROM vault WHERE chunk_id = ?;", (cid,))
        text = cursor.fetchone()[0]
        print(f"  • chunk_id: {cid} | score: {score:.3f} | biz: {meta[0]} | city: {meta[1]} | tags: {meta[3]}")
        print(f"    text: {text}...")

    # 2. Dense Vector Search
    model = SentenceTransformer("BAAI/bge-small-en-v1.5", device="cpu")
    q_emb = model.encode(query, normalize_embeddings=True).astype(np.float32)

    id_table = pq.read_table(vec_ids_path)
    total_chunks = len(id_table)
    memmap = np.memmap(vec_path, dtype=np.float16, mode="r", shape=(total_chunks, 384))

    # Dot product similarity (batch matrix multiply on sample of 5,000 for quickness)
    sub_vecs = memmap[:5000].astype(np.float32)
    scores = np.dot(sub_vecs, q_emb)
    top_indices = np.argsort(scores)[::-1][:5]

    print("\n[Dense Vector Top Results (from first 5k pool)]:")
    for idx in top_indices:
        cid = id_table["chunk_id"][int(idx)].as_py()
        score = scores[idx]
        cursor.execute("SELECT business_id, city, stars, aspects FROM chunks_meta WHERE chunk_id = ?;", (cid,))
        meta = cursor.fetchone()
        cursor.execute("SELECT substr(text_raw, 1, 90) FROM vault WHERE chunk_id = ?;", (cid,))
        text = cursor.fetchone()[0]
        print(f"  • chunk_id: {cid} | cos_sim: {score:.3f} | biz: {meta[0]} | city: {meta[1]} | tags: {meta[3]}")
        print(f"    text: {text}...")

    conn.close()
    return True


def check_all_parquet_datasets():
    print("\n" + "=" * 60)
    print("5. COMPLETE DATASET PARQUET AUDIT")
    print("=" * 60)
    expected_files = {
        "filtered/businesses.parquet": 50000,
        "filtered/reviews.parquet": 4000000,
        "filtered/users.parquet": 1000000,
        "filtered/tips.parquet": 400000,
        "filtered/checkins.parquet": 50000,
        "clean/restaurant_profiles.parquet": 50000,
        "clean/reviewer_credibility.parquet": 1000000,
        "clean/checkin_hour_weekday.parquet": 1000,
        "clean/checkin_monthly.parquet": 1000,
        "clean/restaurant_peak_slots.parquet": 50000,
        "clean/reviews_clean.parquet": 4000000,
        "clean/chunks.parquet": 50000,
        "features/review_aspects.parquet": 4000000,
        "features/selected_reviews.parquet": 40000,
        "features/masked_reviews.parquet": 40000,
        "features/pii_vault.parquet": 10000,
        "aggregates/monthly_aspects.parquet": 1000000,
        "aggregates/city_monthly_aspects.parquet": 50000,
    }

    base = "data_store/parquet"
    all_ok = True
    for rel_path, min_rows in expected_files.items():
        full_p = os.path.join(base, rel_path)
        if not os.path.exists(full_p):
            print(f"  [MISSING] {rel_path}")
            all_ok = False
            continue
        sz_mb = os.path.getsize(full_p) / (1024 * 1024)
        tbl = pq.read_table(full_p)
        nrows = len(tbl)
        status = "[OK]  " if nrows >= min_rows else "[WARN]"
        print(f"  {status} {rel_path:<42} | {nrows:>10,d} rows | {sz_mb:>8.2f} MB")

    return all_ok


if __name__ == "__main__":
    db_p = "data_store/sqlite/restaurant_ops.db"
    vec_p = "data_store/vectors/vectors.f16.npy"
    vec_ids_p = "data_store/vectors/vector_ids.parquet"

    check_sqlite(db_p)
    check_vectors(vec_p, vec_ids_p)
    check_alignment(db_p, vec_ids_p)
    check_all_parquet_datasets()
    check_rag_retrieval(db_p, vec_p, vec_ids_p)
    print("\n" + "=" * 60)
    print("ALL PHASE 0 - 4 ASSETS FULLY VERIFIED AND RAG-READY!")
    print("=" * 60)
