"""Phase 4: SQLite Database Builder (Metadata, FTS5 BM25, and PII Vault).
Constructs:
- chunks_meta: indexed on city, state, business_id, stars, year_month, is_open, categories
- chunks_fts: FTS5 virtual table (porter tokenizer, PII placeholders stripped)
- vault: restricted raw review text table
- Empty tables: feedback, traces, llm_cache, request_log
"""

import json
import re
import sqlite3
import time
from pathlib import Path
import duckdb
import pandas as pd
from tqdm import tqdm

from app.core.config import settings
from app.core.logging import logger

PII_PLACEHOLDERS_REGEX = re.compile(r'\[(PERSON|EMAIL|PHONE|URL|CARD)\]', re.IGNORECASE)


def strip_pii_placeholders(text: str) -> str:
    """Removes PII placeholders so BM25 does not index them."""
    if not text:
        return ""
    cleaned = PII_PLACEHOLDERS_REGEX.sub(" ", text)
    return " ".join(cleaned.split())


def build_sqlite_database(
    chunks_parquet_path: Path,
    vault_parquet_path: Path,
    sqlite_db_path: Path,
    batch_size: int = 5000
):
    """Builds restaurant_ops.db with chunks_meta, FTS5 virtual table, vault, and system tables."""
    logger.info(f"Building SQLite database at {sqlite_db_path}...")
    sqlite_db_path.parent.mkdir(parents=True, exist_ok=True)

    con = sqlite3.connect(sqlite_db_path)
    cur = con.cursor()

    # Enable WAL mode and performance pragmas
    cur.execute("PRAGMA journal_mode = WAL;")
    cur.execute("PRAGMA synchronous = NORMAL;")
    cur.execute("PRAGMA temp_store = MEMORY;")
    cur.execute("PRAGMA cache_size = -64000;")  # 64MB cache

    # 1. chunks_meta table
    cur.execute("DROP TABLE IF EXISTS chunks_meta;")
    cur.execute("""
        CREATE TABLE chunks_meta (
            row_idx INTEGER PRIMARY KEY,
            chunk_id TEXT UNIQUE NOT NULL,
            review_id TEXT NOT NULL,
            business_id TEXT NOT NULL,
            restaurant_name TEXT NOT NULL,
            city TEXT NOT NULL,
            state TEXT NOT NULL,
            categories TEXT,
            price_range INTEGER,
            is_open INTEGER NOT NULL,
            stars REAL NOT NULL,
            business_stars REAL NOT NULL,
            date TEXT NOT NULL,
            year_month TEXT NOT NULL,
            credibility REAL NOT NULL,
            text_masked TEXT NOT NULL,
            aspects TEXT,
            aspect_sentiments TEXT,
            n_tokens INTEGER NOT NULL
        );
    """)

    # 2. FTS5 table
    cur.execute("DROP TABLE IF EXISTS chunks_fts;")
    cur.execute("""
        CREATE VIRTUAL TABLE chunks_fts USING fts5(
            chunk_id UNINDEXED,
            text,
            tokenize='porter unicode61'
        );
    """)

    # 3. Vault table
    cur.execute("DROP TABLE IF EXISTS vault;")
    cur.execute("""
        CREATE TABLE vault (
            chunk_id TEXT PRIMARY KEY,
            review_id TEXT NOT NULL,
            text_raw TEXT NOT NULL
        );
    """)

    # 4. Feedback table
    cur.execute("""
        CREATE TABLE IF NOT EXISTS feedback (
            feedback_id TEXT PRIMARY KEY,
            trace_id TEXT,
            rating INTEGER,
            flags TEXT,
            corrected_aspect TEXT,
            comment TEXT,
            role TEXT,
            created_at TEXT
        );
    """)

    # 5. Traces table
    cur.execute("""
        CREATE TABLE IF NOT EXISTS traces (
            trace_id TEXT PRIMARY KEY,
            request_id TEXT,
            timestamp TEXT,
            trace_json TEXT
        );
    """)

    # 6. LLM cache table
    cur.execute("""
        CREATE TABLE IF NOT EXISTS llm_cache (
            cache_key TEXT PRIMARY KEY,
            response_json TEXT,
            model TEXT,
            created_at TEXT
        );
    """)

    # 7. Request log table
    cur.execute("""
        CREATE TABLE IF NOT EXISTS request_log (
            request_id TEXT PRIMARY KEY,
            endpoint TEXT,
            timestamp TEXT,
            payload TEXT
        );
    """)
    con.commit()

    # Load and insert chunks
    logger.info("Reading chunks from Parquet and populating chunks_meta + chunks_fts...")
    duck_con = duckdb.connect()
    chunks_sql = f"read_parquet('{str(chunks_parquet_path).replace(chr(92), '/')}')"
    total_chunks = duck_con.execute(f"SELECT count(*) FROM {chunks_sql}").fetchone()[0]

    cursor = duck_con.execute(f"SELECT * FROM {chunks_sql}")
    row_idx = 0

    while True:
        chunk_df = cursor.fetch_df_chunk(batch_size)
        if chunk_df is None or len(chunk_df) == 0:
            break

        meta_rows = []
        fts_rows = []

        for _, row in chunk_df.iterrows():
            c_id = row["chunk_id"]
            aspects_json = json.dumps(row["aspects"]) if isinstance(row["aspects"], list) else str(row["aspects"])
            aspect_sentiments_json = str(row["aspect_sentiments"])

            # Clean text for FTS5
            fts_text = strip_pii_placeholders(row["text_masked"])

            meta_rows.append((
                row_idx,
                c_id,
                row["review_id"],
                row["business_id"],
                row["restaurant_name"],
                row["city"],
                row["state"],
                row["categories"],
                row["price_range"],
                int(row["is_open"]),
                float(row["stars"]),
                float(row["business_stars"]),
                row["date"],
                row["year_month"],
                float(row["credibility"]),
                row["text_masked"],
                aspects_json,
                aspect_sentiments_json,
                int(row["n_tokens"])
            ))

            fts_rows.append((c_id, fts_text))
            row_idx += 1

        cur.executemany("""
            INSERT INTO chunks_meta VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
        """, meta_rows)

        cur.executemany("""
            INSERT INTO chunks_fts (chunk_id, text) VALUES (?, ?);
        """, fts_rows)

        con.commit()
        logger.info(f"Populated {row_idx}/{total_chunks} chunks into SQLite...")

    # Build indices on chunks_meta
    logger.info("Creating indices on chunks_meta...")
    indices = [
        "CREATE INDEX idx_meta_city ON chunks_meta(city);",
        "CREATE INDEX idx_meta_state ON chunks_meta(state);",
        "CREATE INDEX idx_meta_business ON chunks_meta(business_id);",
        "CREATE INDEX idx_meta_stars ON chunks_meta(stars);",
        "CREATE INDEX idx_meta_year_month ON chunks_meta(year_month);",
        "CREATE INDEX idx_meta_is_open ON chunks_meta(is_open);"
    ]
    for idx_sql in indices:
        cur.execute(idx_sql)
    con.commit()

    # Populate Vault from pii_vault.parquet
    if vault_parquet_path.exists():
        logger.info("Populating vault table from pii_vault.parquet...")
        vault_df = duck_con.execute(f"""
            SELECT m.chunk_id, v.review_id, v.text_raw
            FROM read_parquet('{str(vault_parquet_path).replace(chr(92), '/')}') v
            JOIN {chunks_sql} m ON v.review_id = m.review_id
        """).df()

        vault_rows = [(r["chunk_id"], r["review_id"], r["text_raw"]) for _, r in vault_df.iterrows()]
        cur.executemany("INSERT OR REPLACE INTO vault (chunk_id, review_id, text_raw) VALUES (?, ?, ?);", vault_rows)
        con.commit()
        logger.info(f"Vault populated with {len(vault_rows)} raw records.")

    con.close()
    logger.info(f"SQLite database build complete: {row_idx} chunks indexed.")
    return row_idx
