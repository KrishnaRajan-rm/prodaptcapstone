"""Phase 3: Master Preprocessing Pipeline Runner."""

import argparse
import time
from pathlib import Path

from app.core.config import settings
from app.core.logging import logger
from data_preprocessing.preprocessing.tagging import tag_review_aspects
from data_preprocessing.preprocessing.selection import select_embedding_subset
from data_preprocessing.preprocessing.pii_gate1 import run_pii_gate1
from data_preprocessing.preprocessing.chunking import build_chunks
from scripts.aspect_audit import run_aspect_audit
from scripts.pii_audit import run_pii_audit


def run_preprocessing_pipeline(max_chunks: int = 50000, force: bool = False):
    logger.info("============================================================")
    logger.info("STARTING PHASE 3: DATA PREPROCESSING PIPELINE")
    logger.info(f"Target semantic chunk sample: {max_chunks}")
    logger.info("============================================================")
    t_start = time.time()

    clean_dir = settings.PARQUET_DIR / "clean"
    features_dir = settings.PARQUET_DIR / "features"
    aggregates_dir = settings.PARQUET_DIR / "aggregates"
    features_dir.mkdir(parents=True, exist_ok=True)
    aggregates_dir.mkdir(parents=True, exist_ok=True)

    rev_clean_path = clean_dir / "reviews_clean.parquet"
    aspects_path = features_dir / "review_aspects.parquet"
    selected_path = features_dir / "selected_reviews.parquet"
    masked_path = features_dir / "masked_reviews.parquet"
    chunks_path = clean_dir / "chunks.parquet"

    # Step 1: Aspect tagging & aggregates
    if not force and aspects_path.exists():
        logger.info(f"1. Aspect features already exist at {aspects_path}, skipping.")
    else:
        t0 = time.time()
        tag_review_aspects(rev_clean_path, features_dir, aggregates_dir)
        logger.info(f"1. Review aspect tagging finished in {time.time() - t0:.2f}s")

    # Step 2: Stratified sampling for semantic layer
    if not force and selected_path.exists():
        logger.info(f"2. Selected reviews already exist at {selected_path}, skipping.")
    else:
        t0 = time.time()
        select_embedding_subset(rev_clean_path, selected_path, max_chunks=max_chunks)
        logger.info(f"2. Stratified selection finished in {time.time() - t0:.2f}s")

    # Step 3: PII Gate 1 Masking
    if not force and masked_path.exists():
        logger.info(f"3. Masked reviews already exist at {masked_path}, skipping.")
    else:
        t0 = time.time()
        run_pii_gate1(selected_path, features_dir)
        logger.info(f"3. PII Gate 1 finished in {time.time() - t0:.2f}s")

    # Step 4: Sentence Chunking + Sentiment
    if not force and chunks_path.exists():
        logger.info(f"4. Chunks already exist at {chunks_path}, skipping.")
    else:
        t0 = time.time()
        build_chunks(masked_path, chunks_path)
        logger.info(f"4. Chunking and sentiment finished in {time.time() - t0:.2f}s")

    # Step 5: Audits
    logger.info("Exporting audit samples for inspection...")
    run_aspect_audit(150)
    run_pii_audit(100)

    logger.info("============================================================")
    logger.info(f"PHASE 3 COMPLETED SUCCESSFULLY IN {time.time() - t_start:.2f}s")
    logger.info("============================================================")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 3 Preprocessing Pipeline")
    parser.add_argument("--chunks", type=int, default=50000, help="Max chunk sample for semantic layer")
    parser.add_argument("--force", action="store_true", help="Force rerun preprocessing")
    args = parser.parse_args()

    run_preprocessing_pipeline(max_chunks=args.chunks, force=args.force)
