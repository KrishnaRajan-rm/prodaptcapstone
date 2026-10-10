"""Phase 3: Stratified Sampling for Semantic Layer.
Optimized column-pruned sampling:
- Samples IDs first using lightweight columns (review_id, business_id, stars)
- Stratifies 70% negative reviews (stars <= 3.0), 30% positive/neutral
- Balanced across restaurants and spread across years
- Joins back to fetch text columns only for the selected IDs
"""

from pathlib import Path
import duckdb

from app.core.config import settings
from app.core.logging import logger


def select_embedding_subset(
    reviews_clean_parquet: Path,
    output_selected_parquet: Path,
    max_chunks: int = 50000,
    random_seed: int = 42
):
    """Executes high-speed column-pruned stratified sampling."""
    logger.info(f"Selecting stratified embedding subset (cap={max_chunks}, seed={random_seed})...")
    con = duckdb.connect()
    temp_dir = settings.PROGRESS_DIR / "duckdb_temp"
    temp_dir.mkdir(parents=True, exist_ok=True)

    con.execute("PRAGMA memory_limit='3GB'")
    con.execute("PRAGMA threads=4")
    con.execute("PRAGMA preserve_insertion_order=false")
    con.execute(f"PRAGMA temp_directory='{str(temp_dir).replace(chr(92), '/')}'")

    rev_sql = f"read_parquet('{str(reviews_clean_parquet).replace(chr(92), '/')}')"
    output_tmp = output_selected_parquet.with_suffix(".parquet.tmp")
    output_tmp.parent.mkdir(parents=True, exist_ok=True)

    target_neg = int(max_chunks * 0.70)
    target_pos = max_chunks - target_neg

    query_sample = f"""
        COPY (
            WITH eligible_ids AS (
                SELECT
                    review_id,
                    business_id,
                    stars,
                    CASE WHEN stars <= 3.0 THEN 1 ELSE 0 END AS is_neg,
                    row_number() OVER (PARTITION BY business_id, CASE WHEN stars <= 3.0 THEN 1 ELSE 0 END ORDER BY hash(review_id)) AS rest_rn
                FROM {rev_sql}
            ),
            neg_ids AS (
                SELECT review_id
                FROM eligible_ids
                WHERE is_neg = 1
                ORDER BY rest_rn, hash(review_id)
                LIMIT {target_neg}
            ),
            pos_ids AS (
                SELECT review_id
                FROM eligible_ids
                WHERE is_neg = 0
                ORDER BY rest_rn, hash(review_id)
                LIMIT {target_pos}
            ),
            sampled_ids AS (
                SELECT review_id FROM neg_ids
                UNION ALL
                SELECT review_id FROM pos_ids
            )
            SELECT
                r.review_id,
                r.user_id,
                r.business_id,
                r.stars,
                r.useful,
                r.funny,
                r.cool,
                r.text,
                r.text_original,
                r.date,
                r.year_month
            FROM {rev_sql} r
            SEMI JOIN sampled_ids s ON r.review_id = s.review_id
        ) TO '{str(output_tmp).replace(chr(92), "/")}' (FORMAT PARQUET, COMPRESSION ZSTD);
    """
    con.execute(query_sample)
    output_tmp.replace(output_selected_parquet)

    count = con.execute(f"SELECT count(*) FROM read_parquet('{str(output_selected_parquet).replace(chr(92), "/")}')").fetchone()[0]
    neg_count = con.execute(f"SELECT count(*) FROM read_parquet('{str(output_selected_parquet).replace(chr(92), "/")}') WHERE stars <= 3.0").fetchone()[0]
    pos_count = count - neg_count
    logger.info(f"Selected {count} reviews for semantic layer: {neg_count} neg ({neg_count/count*100:.1f}%), {pos_count} pos ({pos_count/count*100:.1f}%) -> {output_selected_parquet}")
