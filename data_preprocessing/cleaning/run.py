"""Phase 2: Master Cleaning Pipeline Runner."""

import argparse
import time
from pathlib import Path

from app.core.config import settings
from app.core.logging import logger
from data_preprocessing.cleaning.checkins import clean_checkins
from data_preprocessing.cleaning.credibility import compute_reviewer_credibility
from data_preprocessing.cleaning.reviews import clean_reviews
from data_preprocessing.cleaning.restaurant_profiles import build_restaurant_profiles


def run_cleaning_pipeline(force: bool = False):
    logger.info("============================================================")
    logger.info("STARTING PHASE 2: DATA CLEANING PIPELINE")
    logger.info("============================================================")
    t_start = time.time()

    filtered_dir = settings.PARQUET_DIR / "filtered"
    clean_dir = settings.PARQUET_DIR / "clean"
    clean_dir.mkdir(parents=True, exist_ok=True)

    # 1. Reviewer credibility
    t0 = time.time()
    users_path = filtered_dir / "users.parquet"
    cred_path = clean_dir / "reviewer_credibility.parquet"
    if not force and cred_path.exists():
        logger.info(f"1. Reviewer credibility already exists at {cred_path}, skipping.")
    else:
        compute_reviewer_credibility(users_path, cred_path)
        logger.info(f"1. Reviewer credibility finished in {time.time() - t0:.2f}s")

    # 2. Checkins & Peak slots
    t0 = time.time()
    chk_path = filtered_dir / "checkins.parquet"
    peaks_path = clean_dir / "restaurant_peak_slots.parquet"
    if not force and peaks_path.exists():
        logger.info(f"2. Checkins & peak slots already exist, skipping.")
    else:
        clean_checkins(chk_path, clean_dir)
        logger.info(f"2. Checkins cleaning finished in {time.time() - t0:.2f}s")

    # 3. Restaurant Profiles
    t0 = time.time()
    bus_path = filtered_dir / "businesses.parquet"
    profiles_path = clean_dir / "restaurant_profiles.parquet"
    if not force and profiles_path.exists():
        logger.info(f"3. Restaurant profiles already exist at {profiles_path}, skipping.")
    else:
        build_restaurant_profiles(bus_path, peaks_path, profiles_path)
        logger.info(f"3. Restaurant profiles finished in {time.time() - t0:.2f}s")

    # 4. Review text cleaning & deduplication
    t0 = time.time()
    rev_path = filtered_dir / "reviews.parquet"
    clean_rev_path = clean_dir / "reviews_clean.parquet"
    if not force and clean_rev_path.exists():
        logger.info(f"4. Clean reviews already exist at {clean_rev_path}, skipping.")
    else:
        clean_reviews(rev_path, clean_rev_path)
        logger.info(f"4. Reviews cleaning finished in {time.time() - t0:.2f}s")

    logger.info("============================================================")
    logger.info(f"PHASE 2 COMPLETED SUCCESSFULLY IN {time.time() - t_start:.2f}s")
    logger.info("============================================================")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 2 Cleaning Pipeline")
    parser.add_argument("--force", action="store_true", help="Force rerun all cleaning steps")
    args = parser.parse_args()
    run_cleaning_pipeline(force=args.force)
