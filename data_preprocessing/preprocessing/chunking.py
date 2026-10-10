"""Phase 3: Sentence chunking, sentence-level aspect tagging, and VADER sentiment analysis."""

import json
import re
from pathlib import Path
from typing import List, Dict, Tuple
import duckdb
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from app.core.config import settings
from app.core.logging import logger
from data_preprocessing.preprocessing.tagging import load_aspect_lexicon, build_aspect_regex_patterns

SENTENCE_SPLIT_REGEX = re.compile(r'(?<=[.!?])\s+')


def split_into_sentences(text: str) -> List[str]:
    """Splits text into sentences on punctuation boundaries."""
    if not text:
        return []
    sentences = [s.strip() for s in SENTENCE_SPLIT_REGEX.split(text) if s.strip()]
    return sentences if sentences else [text.strip()]


def estimate_tokens(text: str) -> int:
    """Estimates tokens as word count multiplied by 1.3."""
    words = len(text.split())
    return max(1, int(words * 1.3))


def chunk_text(text: str) -> List[Tuple[str, int]]:
    """Chunks text into <= 250 token units with 1-sentence overlap for longer texts."""
    sentences = split_into_sentences(text)
    total_tokens = estimate_tokens(text)

    # Review fits comfortably in 1 chunk
    if total_tokens <= 250 or len(sentences) <= 1:
        return [(text, total_tokens)]

    chunks = []
    current_sentences = []
    current_tokens = 0

    for sent in sentences:
        s_tokens = estimate_tokens(sent)
        if current_sentences and (current_tokens + s_tokens > 220):
            # Finalize current chunk
            chunk_str = " ".join(current_sentences)
            chunks.append((chunk_str, current_tokens))
            # Start next chunk with 1-sentence overlap
            overlap = [current_sentences[-1]] if len(current_sentences) > 0 else []
            current_sentences = overlap + [sent]
            current_tokens = sum(estimate_tokens(s) for s in current_sentences)
        else:
            current_sentences.append(sent)
            current_tokens += s_tokens

    if current_sentences:
        chunk_str = " ".join(current_sentences)
        chunks.append((chunk_str, current_tokens))

    return chunks


def build_chunks(
    masked_reviews_parquet: Path,
    output_chunks_parquet: Path,
    batch_size: int = 5000
):
    """Generates chunks with sentence-level aspects and VADER sentiment."""
    logger.info("Initializing Chunking and Sentiment analyzer (VADER)...")
    vader = SentimentIntensityAnalyzer()
    lexicon = load_aspect_lexicon()
    patterns = build_aspect_regex_patterns(lexicon, for_sql=False)
    aspect_names = list(lexicon["aspects"].keys())

    # Precompile regex for aspect detection in chunks
    aspect_compiled = {asp: re.compile(patterns[asp], re.IGNORECASE) for asp in aspect_names}

    con = duckdb.connect()
    con.execute("PRAGMA memory_limit='3GB'")
    con.execute("PRAGMA threads=4")

    # Join masked reviews with restaurant profiles and user credibility
    bus_sql = f"read_parquet('{str(settings.PARQUET_DIR / 'clean' / 'restaurant_profiles.parquet').replace(chr(92), '/')}')"
    cred_sql = f"read_parquet('{str(settings.PARQUET_DIR / 'clean' / 'reviewer_credibility.parquet').replace(chr(92), '/')}')"
    rev_sql = f"read_parquet('{str(settings.PARQUET_DIR / 'clean' / 'reviews_clean.parquet').replace(chr(92), '/')}')"
    mask_sql = f"read_parquet('{str(masked_reviews_parquet).replace(chr(92), '/')}')"

    query = f"""
        SELECT
            m.review_id,
            r.user_id,
            m.business_id,
            b.name AS restaurant_name,
            b.city,
            b.state,
            b.categories,
            b.price_range,
            b.is_open,
            m.stars,
            b.stars AS business_stars,
            m.date,
            m.year_month,
            COALESCE(c.credibility, 0.5) AS credibility,
            m.text_masked
        FROM {mask_sql} m
        JOIN {rev_sql} r ON m.review_id = r.review_id
        JOIN {bus_sql} b ON m.business_id = b.business_id
        LEFT JOIN {cred_sql} c ON r.user_id = c.user_id
    """

    cursor = con.execute(query)
    batches_dir = settings.PROGRESS_DIR / "chunk_batches"
    batches_dir.mkdir(parents=True, exist_ok=True)
    batch_idx = 0
    total_chunks = 0

    while True:
        df = cursor.fetch_df_chunk(batch_size)
        if df is None or len(df) == 0:
            break

        chunk_rows = []
        for _, row in df.iterrows():
            rev_id = row["review_id"]
            text_masked = row["text_masked"]
            chunks = chunk_text(text_masked)

            for idx, (c_text, n_tok) in enumerate(chunks):
                chunk_id = f"{rev_id}_{idx}"

                # Sentence-level aspects and VADER sentiment
                detected_aspects = []
                aspect_sentiments = {}
                sentences = split_into_sentences(c_text)

                for asp, regex in aspect_compiled.items():
                    matching_sents = [s for s in sentences if regex.search(s)]
                    if matching_sents:
                        detected_aspects.append(asp)
                        sent_scores = [vader.polarity_scores(s)["compound"] for s in matching_sents]
                        avg_sentiment = round(sum(sent_scores) / len(sent_scores), 3)
                        aspect_sentiments[asp] = avg_sentiment

                chunk_rows.append({
                    "chunk_id": chunk_id,
                    "review_id": rev_id,
                    "business_id": row["business_id"],
                    "restaurant_name": row["restaurant_name"],
                    "city": row["city"],
                    "state": row["state"],
                    "categories": row["categories"],
                    "price_range": row["price_range"],
                    "is_open": int(row["is_open"]),
                    "stars": float(row["stars"]),
                    "business_stars": float(row["business_stars"]),
                    "date": row["date"],
                    "year_month": row["year_month"],
                    "credibility": float(row["credibility"]),
                    "text_masked": c_text,
                    "aspects": detected_aspects,
                    "aspect_sentiments": json.dumps(aspect_sentiments),
                    "n_tokens": n_tok
                })

        batch_df = pd.DataFrame(chunk_rows)
        batch_out = batches_dir / f"chunk_batch_{batch_idx}.parquet"
        batch_df.to_parquet(batch_out, index=False, compression="zstd")
        total_chunks += len(batch_df)
        batch_idx += 1
        logger.info(f"Chunked batch {batch_idx}: generated {len(batch_df)} chunks (total so far: {total_chunks})")

    # Combine into final chunks.parquet
    logger.info("Combining chunk batches into final chunks.parquet...")
    output_tmp = output_chunks_parquet.with_suffix(".parquet.tmp")
    output_tmp.parent.mkdir(parents=True, exist_ok=True)
    batch_glob = str(batches_dir / "chunk_batch_*.parquet").replace(chr(92), "/")

    con.execute(f"""
        COPY (
            SELECT * FROM read_parquet('{batch_glob}')
        ) TO '{str(output_tmp).replace(chr(92), "/")}' (FORMAT PARQUET, COMPRESSION ZSTD);
    """)
    output_tmp.replace(output_chunks_parquet)

    # Cleanup temporary batches
    import shutil
    shutil.rmtree(batches_dir, ignore_errors=True)

    final_count = con.execute(f"SELECT count(*) FROM read_parquet('{str(output_chunks_parquet).replace(chr(92), "/")}')").fetchone()[0]
    logger.info(f"Final chunks.parquet generated with {final_count} chunks -> {output_chunks_parquet}")
