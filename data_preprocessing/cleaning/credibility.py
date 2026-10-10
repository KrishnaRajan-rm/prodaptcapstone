"""Phase 2: Reviewer credibility calculation."""

import numpy as np
import pandas as pd
from pathlib import Path
import duckdb

from app.core.config import settings
from app.core.logging import logger


def calculate_elite_years(elite_val: str) -> int:
    """Parses comma-separated elite string (defensively handles duplicates like 20,20)."""
    if not elite_val or pd.isna(elite_val) or elite_val == "None":
        return 0
    parts = str(elite_val).split(",")
    valid_years = set()
    for p in parts:
        p_clean = p.strip()
        if p_clean.isdigit():
            valid_years.add(int(p_clean))
    return len(valid_years)


def compute_reviewer_credibility(users_parquet_path: Path, output_parquet_path: Path):
    """Calculates credibility for each reviewer using DuckDB and vectorization."""
    logger.info("Computing reviewer credibility scores...")
    con = duckdb.connect()
    
    # Register python function for elite years count
    con.create_function("count_elite", calculate_elite_years, ["VARCHAR"], "BIGINT")
    
    users_sql = f"read_parquet('{str(users_parquet_path).replace(chr(92), '/')}')"
    output_tmp = output_parquet_path.with_suffix(".parquet.tmp")

    query = f"""
        COPY (
            SELECT
                user_id,
                review_count,
                count_elite(elite) as elite_years,
                fans,
                average_stars,
                GREATEST(0.0, LEAST(1.0,
                    0.4 * (ln(review_count + 1.0) / ln(501.0)) +
                    0.3 * (CASE WHEN count_elite(elite) > 0 THEN 1.0 ELSE 0.0 END) +
                    0.2 * (ln(fans + 1.0) / ln(101.0)) +
                    0.1 * (CASE WHEN average_stars >= 2.0 AND average_stars <= 4.5 THEN 1.0 ELSE 0.0 END)
                )) AS credibility
            FROM {users_sql}
        ) TO '{str(output_tmp).replace(chr(92), "/")}' (FORMAT PARQUET, COMPRESSION ZSTD);
    """
    con.execute(query)
    output_tmp.replace(output_parquet_path)
    count = con.execute(f"SELECT count(*) FROM read_parquet('{str(output_parquet_path).replace(chr(92), "/")}')").fetchone()[0]
    logger.info(f"Reviewer credibility written for {count} users -> {output_parquet_path}")
