"""Phase 1: Streaming Ingestion Pipeline.
Converts raw Yelp JSON Lines to compressed Parquet with filtering, batching, and resumption.
"""

import argparse
import json
import os
import shutil
import time
from pathlib import Path
import duckdb

from app.core.config import settings
from app.core.logging import logger
from app.core.errors import DataIngestionError


def get_progress_file() -> Path:
    p = settings.PROGRESS_DIR / "ingestion.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def load_progress() -> dict:
    pf = get_progress_file()
    if pf.exists():
        try:
            with open(pf, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Could not read progress file: {e}")
    return {}


def save_progress(progress: dict):
    pf = get_progress_file()
    tmp = pf.with_suffix(".json.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(progress, f, indent=2)
    tmp.replace(pf)


def get_duckdb_con() -> duckdb.DuckDBPyConnection:
    temp_dir = settings.PROGRESS_DIR / "duckdb_temp"
    temp_dir.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    con.execute("PRAGMA memory_limit='3GB'")
    con.execute("PRAGMA threads=4")
    con.execute(f"PRAGMA temp_directory='{str(temp_dir).replace(chr(92), '/')}'")
    return con


def run_pass_0_businesses(mode: str, force: bool = False):
    """Pass 0: Read business.json, filter restaurants, save businesses.parquet & restaurant_ids.parquet."""
    progress = load_progress()
    if not force and progress.get(f"pass_0_{mode}", {}).get("completed"):
        logger.info(f"[Pass 0] Skipping (already completed for mode={mode}).")
        return

    logger.info(f"[Pass 0] Ingesting businesses (mode={mode})...")
    t0 = time.time()
    con = get_duckdb_con()
    bus_json = (settings.DATASET_DIR / "yelp_academic_dataset_business.json").resolve()
    if not bus_json.exists():
        raise DataIngestionError(f"Missing dataset file: {bus_json}")

    filtered_dir = settings.PARQUET_DIR / "filtered"
    filtered_dir.mkdir(parents=True, exist_ok=True)

    pos_clauses = [f"categories LIKE '%{cat}%'" for cat in settings.POSITIVE_CATEGORIES]
    neg_clauses = [f"(categories NOT LIKE '%{cat}%' OR categories IS NULL)" for cat in settings.NEGATIVE_CATEGORIES]
    cat_filter = f"({' OR '.join(pos_clauses)}) AND ({' AND '.join(neg_clauses)})"

    mode_filter = ""
    order_limit = ""
    if mode == "dev":
        mode_filter = f" AND city = '{settings.DEV_CITY}' AND review_count >= {settings.DEV_MIN_REVIEWS} AND review_count <= 1000"
        order_limit = f"ORDER BY review_count DESC LIMIT {settings.DEV_MAX_RESTAURANTS}"

    bus_parquet_tmp = filtered_dir / "businesses.parquet.tmp"
    bus_parquet = filtered_dir / "businesses.parquet"
    ids_parquet_tmp = filtered_dir / "restaurant_ids.parquet.tmp"
    ids_parquet = filtered_dir / "restaurant_ids.parquet"

    # Query to export businesses
    query_bus = f"""
        COPY (
            SELECT business_id, name, address, city, state, postal_code,
                   latitude, longitude, stars, review_count, is_open,
                   attributes, categories, hours
            FROM read_ndjson('{str(bus_json).replace(chr(92), "/")}',
                             maximum_object_size=134217728, ignore_errors=true)
            WHERE ({cat_filter}){mode_filter}
            {order_limit}
        ) TO '{str(bus_parquet_tmp).replace(chr(92), "/")}' (FORMAT PARQUET, COMPRESSION ZSTD);
    """
    con.execute(query_bus)
    bus_parquet_tmp.replace(bus_parquet)

    # Query to export restaurant_ids
    query_ids = f"""
        COPY (
            SELECT business_id
            FROM read_parquet('{str(bus_parquet).replace(chr(92), "/")}')
        ) TO '{str(ids_parquet_tmp).replace(chr(92), "/")}' (FORMAT PARQUET, COMPRESSION ZSTD);
    """
    con.execute(query_ids)
    ids_parquet_tmp.replace(ids_parquet)

    count = con.execute(f"SELECT count(*) FROM read_parquet('{str(bus_parquet).replace(chr(92), "/")}')").fetchone()[0]
    elapsed = time.time() - t0
    logger.info(f"[Pass 0] Done: {count} restaurants kept in {elapsed:.2f}s -> {bus_parquet}")

    progress[f"pass_0_{mode}"] = {
        "completed": True,
        "mode": mode,
        "restaurants_count": count,
        "elapsed_seconds": elapsed,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    save_progress(progress)


def run_pass_1_reviews(mode: str, force: bool = False):
    """Pass 1: Stream review.json, filter by restaurant IDs, save reviews.parquet & reviewer_user_ids.parquet."""
    progress = load_progress()
    if not force and progress.get(f"pass_1_{mode}", {}).get("completed"):
        logger.info(f"[Pass 1] Skipping (already completed for mode={mode}).")
        return

    logger.info(f"[Pass 1] Streaming reviews for {mode} mode...")
    t0 = time.time()
    con = get_duckdb_con()
    rev_json = (settings.DATASET_DIR / "yelp_academic_dataset_review.json").resolve()
    ids_parquet = (settings.PARQUET_DIR / "filtered" / "restaurant_ids.parquet").resolve()
    filtered_dir = settings.PARQUET_DIR / "filtered"

    if not rev_json.exists():
        raise DataIngestionError(f"Missing review dataset: {rev_json}")
    if not ids_parquet.exists():
        raise DataIngestionError(f"Missing restaurant_ids.parquet: Run Pass 0 first.")

    rev_parquet_tmp = filtered_dir / "reviews.parquet.tmp"
    rev_parquet = filtered_dir / "reviews.parquet"
    user_ids_parquet_tmp = filtered_dir / "reviewer_user_ids.parquet.tmp"
    user_ids_parquet = filtered_dir / "reviewer_user_ids.parquet"

    query_reviews = f"""
        COPY (
            SELECT r.review_id, r.user_id, r.business_id, r.stars,
                   r.useful, r.funny, r.cool, r.text, r.date
            FROM read_ndjson('{str(rev_json).replace(chr(92), "/")}',
                             maximum_object_size=134217728,
                             ignore_errors=true,
                             columns={{'review_id': 'VARCHAR', 'user_id': 'VARCHAR', 'business_id': 'VARCHAR',
                                      'stars': 'DOUBLE', 'useful': 'BIGINT', 'funny': 'BIGINT', 'cool': 'BIGINT',
                                      'text': 'VARCHAR', 'date': 'VARCHAR'}}) r
            SEMI JOIN read_parquet('{str(ids_parquet).replace(chr(92), "/")}') b
            ON r.business_id = b.business_id
        ) TO '{str(rev_parquet_tmp).replace(chr(92), "/")}' (FORMAT PARQUET, COMPRESSION ZSTD);
    """
    con.execute(query_reviews)
    rev_parquet_tmp.replace(rev_parquet)

    # Distinct user IDs
    query_user_ids = f"""
        COPY (
            SELECT DISTINCT user_id
            FROM read_parquet('{str(rev_parquet).replace(chr(92), "/")}')
        ) TO '{str(user_ids_parquet_tmp).replace(chr(92), "/")}' (FORMAT PARQUET, COMPRESSION ZSTD);
    """
    con.execute(query_user_ids)
    user_ids_parquet_tmp.replace(user_ids_parquet)

    rev_count = con.execute(f"SELECT count(*) FROM read_parquet('{str(rev_parquet).replace(chr(92), "/")}')").fetchone()[0]
    user_count = con.execute(f"SELECT count(*) FROM read_parquet('{str(user_ids_parquet).replace(chr(92), "/")}')").fetchone()[0]
    elapsed = time.time() - t0
    logger.info(f"[Pass 1] Done: {rev_count} reviews kept, {user_count} distinct reviewers in {elapsed:.2f}s -> {rev_parquet}")

    progress[f"pass_1_{mode}"] = {
        "completed": True,
        "mode": mode,
        "reviews_count": rev_count,
        "distinct_reviewers": user_count,
        "elapsed_seconds": elapsed,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    save_progress(progress)


def run_pass_2_users(mode: str, force: bool = False):
    """Pass 2: Stream user.json, filter by reviewer IDs, drop name/friends, save users.parquet."""
    progress = load_progress()
    if not force and progress.get(f"pass_2_{mode}", {}).get("completed"):
        logger.info(f"[Pass 2] Skipping (already completed for mode={mode}).")
        return

    logger.info(f"[Pass 2] Streaming users for {mode} mode...")
    t0 = time.time()
    con = get_duckdb_con()
    user_json = (settings.DATASET_DIR / "yelp_academic_dataset_user.json").resolve()
    reviewer_ids = (settings.PARQUET_DIR / "filtered" / "reviewer_user_ids.parquet").resolve()
    filtered_dir = settings.PARQUET_DIR / "filtered"

    if not user_json.exists():
        raise DataIngestionError(f"Missing user dataset: {user_json}")
    if not reviewer_ids.exists():
        raise DataIngestionError(f"Missing reviewer_user_ids.parquet: Run Pass 1 first.")

    users_parquet_tmp = filtered_dir / "users.parquet.tmp"
    users_parquet = filtered_dir / "users.parquet"

    # Project only needed columns; ignore friends and name
    query_users = f"""
        COPY (
            SELECT u.user_id, u.review_count, u.yelping_since, u.useful, u.funny, u.cool,
                   u.elite, u.fans, u.average_stars,
                   u.compliment_hot, u.compliment_more, u.compliment_profile, u.compliment_cute,
                   u.compliment_list, u.compliment_note, u.compliment_plain, u.compliment_cool,
                   u.compliment_funny, u.compliment_writer, u.compliment_photos
            FROM read_ndjson('{str(user_json).replace(chr(92), "/")}',
                             maximum_object_size=134217728,
                             ignore_errors=true,
                             columns={{'user_id': 'VARCHAR', 'review_count': 'BIGINT', 'yelping_since': 'VARCHAR',
                                      'useful': 'BIGINT', 'funny': 'BIGINT', 'cool': 'BIGINT', 'elite': 'VARCHAR',
                                      'fans': 'BIGINT', 'average_stars': 'DOUBLE',
                                      'compliment_hot': 'BIGINT', 'compliment_more': 'BIGINT', 'compliment_profile': 'BIGINT',
                                      'compliment_cute': 'BIGINT', 'compliment_list': 'BIGINT', 'compliment_note': 'BIGINT',
                                      'compliment_plain': 'BIGINT', 'compliment_cool': 'BIGINT', 'compliment_funny': 'BIGINT',
                                      'compliment_writer': 'BIGINT', 'compliment_photos': 'BIGINT'}}) u
            SEMI JOIN read_parquet('{str(reviewer_ids).replace(chr(92), "/")}') r
            ON u.user_id = r.user_id
        ) TO '{str(users_parquet_tmp).replace(chr(92), "/")}' (FORMAT PARQUET, COMPRESSION ZSTD);
    """
    con.execute(query_users)
    users_parquet_tmp.replace(users_parquet)

    count = con.execute(f"SELECT count(*) FROM read_parquet('{str(users_parquet).replace(chr(92), "/")}')").fetchone()[0]
    elapsed = time.time() - t0
    logger.info(f"[Pass 2] Done: {count} user profiles kept in {elapsed:.2f}s -> {users_parquet}")

    progress[f"pass_2_{mode}"] = {
        "completed": True,
        "mode": mode,
        "users_count": count,
        "elapsed_seconds": elapsed,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    save_progress(progress)


def run_pass_3_tips_checkins(mode: str, force: bool = False):
    """Pass 3: Stream tips (< 5 words dropped) and checkins for restaurants."""
    progress = load_progress()
    if not force and progress.get(f"pass_3_{mode}", {}).get("completed"):
        logger.info(f"[Pass 3] Skipping (already completed for mode={mode}).")
        return

    logger.info(f"[Pass 3] Processing tips and checkins for {mode} mode...")
    t0 = time.time()
    con = get_duckdb_con()
    ids_parquet = (settings.PARQUET_DIR / "filtered" / "restaurant_ids.parquet").resolve()
    tip_json = (settings.DATASET_DIR / "yelp_academic_dataset_tip.json").resolve()
    checkin_json = (settings.DATASET_DIR / "yelp_academic_dataset_checkin.json").resolve()
    filtered_dir = settings.PARQUET_DIR / "filtered"

    tips_parquet_tmp = filtered_dir / "tips.parquet.tmp"
    tips_parquet = filtered_dir / "tips.parquet"
    checkins_parquet_tmp = filtered_dir / "checkins.parquet.tmp"
    checkins_parquet = filtered_dir / "checkins.parquet"

    # Tips query (dropping tips < 5 words)
    query_tips = f"""
        COPY (
            SELECT t.user_id, t.business_id, t.text, t.date, t.compliment_count
            FROM read_ndjson('{str(tip_json).replace(chr(92), "/")}',
                             maximum_object_size=134217728,
                             ignore_errors=true,
                             columns={{'user_id': 'VARCHAR', 'business_id': 'VARCHAR', 'text': 'VARCHAR',
                                      'date': 'VARCHAR', 'compliment_count': 'BIGINT'}}) t
            SEMI JOIN read_parquet('{str(ids_parquet).replace(chr(92), "/")}') b
            ON t.business_id = b.business_id
            WHERE (length(trim(t.text)) - length(replace(trim(t.text), ' ', '')) + 1) >= 5
        ) TO '{str(tips_parquet_tmp).replace(chr(92), "/")}' (FORMAT PARQUET, COMPRESSION ZSTD);
    """
    con.execute(query_tips)
    tips_parquet_tmp.replace(tips_parquet)
    tips_count = con.execute(f"SELECT count(*) FROM read_parquet('{str(tips_parquet).replace(chr(92), "/")}')").fetchone()[0]

    # Checkins query
    query_checkins = f"""
        COPY (
            SELECT c.business_id, c.date
            FROM read_ndjson('{str(checkin_json).replace(chr(92), "/")}',
                             maximum_object_size=134217728,
                             ignore_errors=true,
                             columns={{'business_id': 'VARCHAR', 'date': 'VARCHAR'}}) c
            SEMI JOIN read_parquet('{str(ids_parquet).replace(chr(92), "/")}') b
            ON c.business_id = b.business_id
        ) TO '{str(checkins_parquet_tmp).replace(chr(92), "/")}' (FORMAT PARQUET, COMPRESSION ZSTD);
    """
    con.execute(query_checkins)
    checkins_parquet_tmp.replace(checkins_parquet)
    checkins_count = con.execute(f"SELECT count(*) FROM read_parquet('{str(checkins_parquet).replace(chr(92), "/")}')").fetchone()[0]

    elapsed = time.time() - t0
    logger.info(f"[Pass 3] Done: {tips_count} tips, {checkins_count} checkin records in {elapsed:.2f}s")

    progress[f"pass_3_{mode}"] = {
        "completed": True,
        "mode": mode,
        "tips_count": tips_count,
        "checkins_count": checkins_count,
        "elapsed_seconds": elapsed,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    save_progress(progress)


def run_ingestion_pipeline(mode: str = "dev", force: bool = False):
    """Executes all ingestion passes sequentially."""
    logger.info(f"Starting ingestion pipeline in MODE={mode.upper()} (force={force})...")
    run_pass_0_businesses(mode, force)
    run_pass_1_reviews(mode, force)
    run_pass_2_users(mode, force)
    run_pass_3_tips_checkins(mode, force)
    logger.info(f"Ingestion pipeline completed successfully for mode={mode}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Yelp Streaming Ingestion Pipeline")
    parser.add_argument("--mode", type=str, choices=["dev", "full"], default=settings.PIPELINE_MODE,
                        help="Ingestion mode: dev (focused city) or full (all restaurants)")
    parser.add_argument("--force", action="store_true", help="Force rerun all passes ignoring cached progress")
    args = parser.parse_args()

    run_ingestion_pipeline(mode=args.mode, force=args.force)
