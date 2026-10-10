"""Phase 3 Audit: Exports 150 random reviews with aspect tags to CSV for review."""

import csv
from pathlib import Path
import duckdb
from app.core.config import settings
from app.core.logging import logger


def run_aspect_audit(n_samples: int = 150):
    con = duckdb.connect()
    feat_path = settings.PARQUET_DIR / "features" / "review_aspects.parquet"
    rev_path = settings.PARQUET_DIR / "clean" / "reviews_clean.parquet"

    if not feat_path.exists() or not rev_path.exists():
        logger.error("Required parquet files missing for aspect audit.")
        return

    out_csv = settings.BASE_DIR / "aspect_audit_sample.csv"
    query = f"""
        SELECT
            r.review_id,
            r.business_id,
            r.stars,
            r.text,
            f.has_wait_time,
            f.has_food_quality,
            f.has_staff_behavior,
            f.has_order_accuracy,
            f.has_cleanliness,
            f.has_price_value,
            f.has_portion_size,
            f.has_ambience_noise,
            f.has_takeout_delivery,
            f.has_health_safety
        FROM read_parquet('{str(rev_path).replace(chr(92), '/')}') r
        JOIN read_parquet('{str(feat_path).replace(chr(92), '/')}') f
        ON r.review_id = f.review_id
        ORDER BY hash(r.review_id)
        LIMIT {n_samples}
    """
    df = con.execute(query).df()
    df.to_csv(out_csv, index=False, encoding="utf-8")
    logger.info(f"Aspect audit exported {len(df)} samples to {out_csv}")


if __name__ == "__main__":
    run_aspect_audit()
