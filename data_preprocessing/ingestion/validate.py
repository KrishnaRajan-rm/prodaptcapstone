"""Phase 1: Ingestion Validation.
Verifies data integrity across Parquet outputs:
- Row counts
- Orphan reviews (must be 0)
- Reviews without a user profile (report count)
- Restaurants with zero reviews (report count)
- Review ID duplicates (must be 0)
Saves validation_report.json in data_store/progress/.
"""

import json
from pathlib import Path
import duckdb

from app.core.config import settings
from app.core.logging import logger
from app.core.errors import DataIngestionError


def validate_ingestion() -> dict:
    filtered_dir = settings.PARQUET_DIR / "filtered"
    bus_path = filtered_dir / "businesses.parquet"
    rev_path = filtered_dir / "reviews.parquet"
    usr_path = filtered_dir / "users.parquet"
    tip_path = filtered_dir / "tips.parquet"
    chk_path = filtered_dir / "checkins.parquet"

    for p in [bus_path, rev_path, usr_path]:
        if not p.exists():
            raise DataIngestionError(f"Required table missing for validation: {p}")

    con = duckdb.connect()

    bus_sql = f"read_parquet('{str(bus_path).replace(chr(92), '/')}')"
    rev_sql = f"read_parquet('{str(rev_path).replace(chr(92), '/')}')"
    usr_sql = f"read_parquet('{str(usr_path).replace(chr(92), '/')}')"
    tip_sql = f"read_parquet('{str(tip_path).replace(chr(92), '/')}')" if tip_path.exists() else None
    chk_sql = f"read_parquet('{str(chk_path).replace(chr(92), '/')}')" if chk_path.exists() else None

    # Row counts
    n_businesses = con.execute(f"SELECT count(*) FROM {bus_sql}").fetchone()[0]
    n_reviews = con.execute(f"SELECT count(*) FROM {rev_sql}").fetchone()[0]
    n_users = con.execute(f"SELECT count(*) FROM {usr_sql}").fetchone()[0]
    n_tips = con.execute(f"SELECT count(*) FROM {tip_sql}").fetchone()[0] if tip_sql else 0
    n_checkins = con.execute(f"SELECT count(*) FROM {chk_sql}").fetchone()[0] if chk_sql else 0

    # 1. Orphan reviews (reviews pointing to unknown businesses) -> MUST BE 0
    orphan_reviews = con.execute(f"""
        SELECT count(*)
        FROM {rev_sql} r
        LEFT JOIN {bus_sql} b ON r.business_id = b.business_id
        WHERE b.business_id IS NULL
    """).fetchone()[0]

    # 2. Duplicate review_id -> MUST BE 0
    dup_reviews = con.execute(f"""
        SELECT count(*) - count(DISTINCT review_id)
        FROM {rev_sql}
    """).fetchone()[0]

    # 3. Reviews without user row
    reviews_without_user = con.execute(f"""
        SELECT count(*)
        FROM {rev_sql} r
        LEFT JOIN {usr_sql} u ON r.user_id = u.user_id
        WHERE u.user_id IS NULL
    """).fetchone()[0]

    # 4. Restaurants with zero reviews
    restaurants_with_zero_reviews = con.execute(f"""
        SELECT count(*)
        FROM {bus_sql} b
        LEFT JOIN {rev_sql} r ON b.business_id = r.business_id
        WHERE r.review_id IS NULL
    """).fetchone()[0]

    passed = (orphan_reviews == 0) and (dup_reviews == 0)

    report = {
        "status": "PASS" if passed else "FAIL",
        "row_counts": {
            "businesses": n_businesses,
            "reviews": n_reviews,
            "users": n_users,
            "tips": n_tips,
            "checkins": n_checkins
        },
        "integrity_checks": {
            "orphan_reviews": orphan_reviews,
            "duplicate_reviews": dup_reviews,
            "reviews_without_user_profile": reviews_without_user,
            "restaurants_with_zero_reviews": restaurants_with_zero_reviews
        }
    }

    report_path = settings.PROGRESS_DIR / "validation_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    logger.info("=" * 60)
    logger.info("INGESTION VALIDATION REPORT")
    logger.info("=" * 60)
    logger.info(f"Businesses: {n_businesses} | Reviews: {n_reviews} | Users: {n_users}")
    logger.info(f"Orphan Reviews: {orphan_reviews} (Gate: 0)")
    logger.info(f"Duplicate Reviews: {dup_reviews} (Gate: 0)")
    logger.info(f"Reviews missing user profile: {reviews_without_user}")
    logger.info(f"Restaurants with 0 reviews: {restaurants_with_zero_reviews}")
    logger.info(f"Validation Status: {'[PASS]' if passed else '[FAIL]'}")
    logger.info("=" * 60)

    if not passed:
        raise DataIngestionError(f"Ingestion validation failed: orphan_reviews={orphan_reviews}, dup_reviews={dup_reviews}")

    return report


if __name__ == "__main__":
    validate_ingestion()
