"""Phase 2: Review text cleaning and deduplication with batch processing."""

import shutil
import time
from pathlib import Path
import duckdb

from app.core.config import settings
from app.core.logging import logger


def clean_reviews(reviews_parquet_path: Path, output_parquet_path: Path):
    """Cleans review text using batched yearly streaming to ensure zero memory exhaustion."""
    logger.info("Cleaning and deduplicating reviews using annual batch processing...")
    con = duckdb.connect()
    temp_dir = settings.PROGRESS_DIR / "duckdb_temp"
    temp_dir.mkdir(parents=True, exist_ok=True)

    con.execute("PRAGMA memory_limit='3GB'")
    con.execute("PRAGMA threads=4")
    con.execute("PRAGMA preserve_insertion_order=false")
    con.execute(f"PRAGMA temp_directory='{str(temp_dir).replace(chr(92), '/')}'")

    rev_sql = f"read_parquet('{str(reviews_parquet_path).replace(chr(92), '/')}')"
    output_tmp = output_parquet_path.with_suffix(".parquet.tmp")

    # Get distinct years to process in memory-safe batches
    years = [row[0] for row in con.execute(f"SELECT DISTINCT substr(date, 1, 4) FROM {rev_sql} ORDER BY 1").fetchall() if row[0]]
    logger.info(f"Processing reviews across {len(years)} annual batches: {years}")

    # Temporary directory for yearly parquet batches
    batches_dir = settings.PROGRESS_DIR / "review_clean_batches"
    batches_dir.mkdir(parents=True, exist_ok=True)
    batch_files = []

    for yr in years:
        t_yr = time.time()
        yr_file = batches_dir / f"reviews_clean_{yr}.parquet"
        con.execute(f"""
            COPY (
                WITH normalized AS (
                    SELECT
                        review_id,
                        user_id,
                        business_id,
                        stars,
                        useful,
                        funny,
                        cool,
                        regexp_replace(
                            regexp_replace(trim(text), '[\\x00-\\x1f\\x7f-\\x9f]', '', 'g'),
                            '\\s+', ' ', 'g'
                        ) AS text_clean,
                        text AS text_original,
                        date,
                        substr(date, 1, 7) AS year_month
                    FROM {rev_sql}
                    WHERE substr(date, 1, 4) = '{yr}'
                )
                SELECT
                    review_id,
                    user_id,
                    business_id,
                    stars,
                    useful,
                    funny,
                    cool,
                    text_clean AS text,
                    text_original,
                    date,
                    year_month
                FROM normalized
                WHERE length(trim(text_clean)) > 0
                  AND (length(trim(text_clean)) - length(replace(trim(text_clean), ' ', '')) + 1) >= 3
                QUALIFY row_number() OVER (PARTITION BY user_id, business_id, text_clean ORDER BY date DESC) = 1
            ) TO '{str(yr_file).replace(chr(92), "/")}' (FORMAT PARQUET, COMPRESSION ZSTD);
        """)
        yr_count = con.execute(f"SELECT count(*) FROM read_parquet('{str(yr_file).replace(chr(92), "/")}')").fetchone()[0]
        batch_files.append(yr_file)
        logger.info(f"Batch year {yr}: {yr_count} reviews cleaned in {time.time() - t_yr:.2f}s")

    # Union all annual batches into the final clean reviews parquet
    logger.info(f"Combining {len(batch_files)} batches into final clean reviews parquet...")
    batch_glob = str(batches_dir / "reviews_clean_*.parquet").replace(chr(92), "/")
    con.execute(f"""
        COPY (
            SELECT * FROM read_parquet('{batch_glob}')
        ) TO '{str(output_tmp).replace(chr(92), "/")}' (FORMAT PARQUET, COMPRESSION ZSTD);
    """)
    output_tmp.replace(output_parquet_path)

    # Cleanup temporary batches
    shutil.rmtree(batches_dir, ignore_errors=True)

    total_count = con.execute(f"SELECT count(*) FROM read_parquet('{str(output_parquet_path).replace(chr(92), "/")}')").fetchone()[0]
    logger.info(f"Total clean reviews saved: {total_count} rows -> {output_parquet_path}")
