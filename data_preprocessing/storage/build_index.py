"""Phase 4: Storage and Index Builder.
Constructs SQLite FTS5 index, PII Vault, and Dense Vector Store.
Validates parity across Parquet, SQLite metadata, FTS5 virtual table, and vectors.
Tests both BM25 and dense retrieval for "waited an hour for food".
"""

import argparse
import sqlite3
import time
from pathlib import Path
import duckdb
import numpy as np
from sentence_transformers import SentenceTransformer

from app.core.config import settings
from app.core.logging import logger
from data_preprocessing.storage.sqlite_builder import build_sqlite_database
from data_preprocessing.storage.vector_store import build_embeddings, NumpyMemmapDenseStore


def run_build_index(resume: bool = True, batch_size: int = 64):
    logger.info("============================================================")
    logger.info("STARTING PHASE 4: STORAGE AND INDEX BUILDER")
    logger.info(f"Resume: {resume} | Batch Size: {batch_size}")
    logger.info("============================================================")
    t_start = time.time()

    clean_dir = settings.PARQUET_DIR / "clean"
    features_dir = settings.PARQUET_DIR / "features"
    chunks_path = clean_dir / "chunks.parquet"
    vault_path = features_dir / "pii_vault.parquet"
    db_path = settings.SQLITE_DIR / "restaurant_ops.db"
    vectors_dir = settings.VECTORS_DIR

    if not chunks_path.exists():
        logger.error(f"Missing chunks.parquet at {chunks_path}. Run Phase 3 first.")
        return 1

    # Step 1: SQLite Database Builder (Metadata + FTS5 + Vault)
    t0 = time.time()
    n_sqlite_chunks = build_sqlite_database(chunks_path, vault_path, db_path)
    logger.info(f"1. SQLite database and FTS5 index built in {time.time() - t0:.2f}s")

    # Step 2: Dense Vector Embeddings (NumPy Memmap)
    t0 = time.time()
    dense_store = build_embeddings(
        chunks_parquet_path=chunks_path,
        vectors_dir=vectors_dir,
        batch_size=batch_size,
        resume=resume
    )
    logger.info(f"2. Vector store built in {time.time() - t0:.2f}s")

    # Step 3: Cross-Store Parity Verification
    logger.info("Verifying parity across data stores...")
    duck_con = duckdb.connect()
    n_parquet = duck_con.execute(f"SELECT count(*) FROM read_parquet('{str(chunks_path).replace(chr(92), '/')}')").fetchone()[0]

    sql_con = sqlite3.connect(db_path)
    n_meta = sql_con.execute("SELECT count(*) FROM chunks_meta").fetchone()[0]
    n_fts = sql_con.execute("SELECT count(*) FROM chunks_fts").fetchone()[0]
    n_vectors = len(dense_store.idx_to_id)

    parity_ok = (n_parquet == n_meta == n_fts == n_vectors)

    logger.info("-" * 65)
    logger.info(f"PARQUET CHUNKS:     {n_parquet}")
    logger.info(f"SQLITE CHUNKS_META: {n_meta}")
    logger.info(f"SQLITE CHUNKS_FTS:  {n_fts}")
    logger.info(f"DENSE VECTORS:      {n_vectors}")
    logger.info(f"CROSS-STORE PARITY: {'[PASS]' if parity_ok else '[FAIL]'}")
    logger.info("-" * 65)

    # Step 4: Verification Queries for "waited an hour for food"
    test_query = "waited an hour for food"
    logger.info(f"Running verification retrieval tests for query: '{test_query}'...")

    # BM25 test query
    bm25_clean_query = "waited hour food"
    bm25_res = sql_con.execute("""
        SELECT m.chunk_id, m.restaurant_name, m.stars, m.text_masked, bm25(chunks_fts) as rank
        FROM chunks_fts f
        JOIN chunks_meta m ON f.chunk_id = m.chunk_id
        WHERE chunks_fts MATCH ?
        ORDER BY rank
        LIMIT 3
    """, (bm25_clean_query,)).fetchall()

    logger.info("\n--- BM25 Top Matches ---")
    for r in bm25_res:
        logger.info(f"[{r[0]}] {r[1]} ({r[2]} stars) | Score: {r[4]:.4f} | Text: {r[3][:70]}...")

    # Dense test query
    bge_prefix = "Represent this sentence for searching relevant passages: "
    embed_model = SentenceTransformer(settings.EMBEDDING_MODEL_NAME)
    query_emb = embed_model.encode([bge_prefix + test_query], normalize_embeddings=True)[0]
    dense_res = dense_store.search(query_emb, top_k=3)

    logger.info("\n--- Dense Top Matches ---")
    for cid, score in dense_res:
        row = sql_con.execute("SELECT restaurant_name, stars, text_masked FROM chunks_meta WHERE chunk_id = ?", (cid,)).fetchone()
        if row:
            logger.info(f"[{cid}] {row[0]} ({row[1]} stars) | Cosine: {score:.4f} | Text: {row[2][:70]}...")

    sql_con.close()
    logger.info("=" * 65)
    logger.info(f"PHASE 4 COMPLETED IN {time.time() - t_start:.2f}s")
    logger.info("=" * 65)
    return 0 if parity_ok and len(bm25_res) > 0 and len(dense_res) > 0 else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 4 Storage and Index Builder")
    parser.add_argument("--resume", action="store_true", default=True, help="Resume vector embeddings from checkpoint")
    parser.add_argument("--batch-size", type=int, default=64, help="Embedding batch size")
    args = parser.parse_args()

    run_build_index(resume=args.resume, batch_size=args.batch_size)
