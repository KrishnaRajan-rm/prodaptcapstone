"""Phase 3: PII Gate 1 (Ingest Redaction).
Detects PERSON via spaCy en_core_web_sm (parser/lemmatizer disabled), plus regexes for email, phone, URL, card.
Strictly respects allow-list:
- food_vocab.txt (dish, culinary terms)
- business names and their tokens
- city names
- RapidFuzz ratio >= 90 against restaurant name
Outputs masked_reviews.parquet and pii_vault.parquet.
"""

import re
from pathlib import Path
from typing import Set, Dict, List
import spacy
from rapidfuzz import fuzz
import duckdb
import pandas as pd
from tqdm import tqdm

from app.core.config import settings
from app.core.logging import logger

EMAIL_REGEX = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,7}\b')
PHONE_REGEX = re.compile(r'\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b')
URL_REGEX = re.compile(r'https?://[^\s]+|www\.[^\s]+')
CARD_REGEX = re.compile(r'\b(?:\d{4}[-\s]?){3}\d{4}\b')


def load_food_vocab() -> Set[str]:
    vocab_path = Path(__file__).parent / "food_vocab.txt"
    if not vocab_path.exists():
        return set()
    with open(vocab_path, "r", encoding="utf-8") as f:
        return {line.strip().lower() for line in f if line.strip()}


def load_city_allowlist() -> Set[str]:
    con = duckdb.connect()
    bus_path = settings.PARQUET_DIR / "clean" / "restaurant_profiles.parquet"
    if not bus_path.exists():
        return set()
    cities = con.execute(f"SELECT DISTINCT lower(city) FROM read_parquet('{str(bus_path).replace(chr(92), '/')}')").fetchall()
    return {c[0] for c in cities if c[0]}


def mask_pii_texts(
    reviews_df: pd.DataFrame,
    nlp,
    food_vocab: Set[str],
    city_allowlist: Set[str]
) -> pd.DataFrame:
    """Masks PII across a DataFrame batch of reviews."""
    masked_records = []

    texts = reviews_df["text"].tolist()
    names = reviews_df["restaurant_name"].fillna("").tolist()

    # Pre-mask regexes
    regex_masked = []
    stats = {"person": 0, "email": 0, "phone": 0, "url": 0, "card": 0}

    for text in texts:
        t, n_e = EMAIL_REGEX.subn("[EMAIL]", text)
        t, n_p = PHONE_REGEX.subn("[PHONE]", t)
        t, n_u = URL_REGEX.subn("[URL]", t)
        t, n_c = CARD_REGEX.subn("[CARD]", t)
        stats["email"] += n_e
        stats["phone"] += n_p
        stats["url"] += n_u
        stats["card"] += n_c
        regex_masked.append(t)

    # spaCy NER for PERSON
    docs = list(nlp.pipe(regex_masked, batch_size=256))

    for doc, orig_text, rest_name in zip(docs, texts, names):
        rest_lower = rest_name.lower().strip()
        rest_tokens = set(re.findall(r'\b\w+\b', rest_lower))

        # Find PERSON spans to mask
        spans_to_mask = []
        for ent in doc.ents:
            if ent.label_ == "PERSON":
                ent_lower = ent.text.lower().strip()
                # Check allow-list
                if ent_lower in food_vocab:
                    continue
                if ent_lower in city_allowlist:
                    continue
                if ent_lower in rest_tokens or ent_lower == rest_lower:
                    continue
                if fuzz.ratio(ent_lower, rest_lower) >= 90:
                    continue
                # Also check single tokens against food vocab
                ent_words = ent_lower.split()
                if any(w in food_vocab for w in ent_words):
                    continue

                spans_to_mask.append((ent.start_char, ent.end_char))

        # Apply PERSON replacements from right to left
        t_clean = doc.text
        for start, end in sorted(spans_to_mask, key=lambda x: x[0], reverse=True):
            t_clean = t_clean[:start] + "[PERSON]" + t_clean[end:]
            stats["person"] += 1

        masked_records.append(t_clean)

    reviews_df["text_masked"] = masked_records
    return reviews_df, stats


def run_pii_gate1(selected_reviews_parquet: Path, output_features_dir: Path, batch_size: int = 5000):
    """Executes Gate 1 PII masking in memory-safe batches on selected reviews."""
    logger.info("Initializing PII Gate 1 (spaCy en_core_web_sm)...")
    nlp = spacy.load("en_core_web_sm", disable=["parser", "lemmatizer"])
    food_vocab = load_food_vocab()
    city_allowlist = load_city_allowlist()
    logger.info(f"Loaded {len(food_vocab)} food terms and {len(city_allowlist)} cities into allow-list.")

    con = duckdb.connect()
    bus_sql = f"read_parquet('{str(settings.PARQUET_DIR / 'clean' / 'restaurant_profiles.parquet').replace(chr(92), '/')}')"
    sel_sql = f"read_parquet('{str(selected_reviews_parquet).replace(chr(92), '/')}')"

    # Join selected reviews with restaurant names for fuzzy allowlist matching
    query = f"""
        SELECT s.review_id, s.user_id, s.business_id, b.name as restaurant_name,
               s.stars, s.useful, s.funny, s.cool, s.text, s.date, s.year_month
        FROM {sel_sql} s
        JOIN {bus_sql} b ON s.business_id = b.business_id
    """
    total_rows = con.execute(f"SELECT count(*) FROM {sel_sql}").fetchone()[0]
    logger.info(f"Processing {total_rows} selected reviews through PII Gate 1...")

    output_features_dir.mkdir(parents=True, exist_ok=True)
    masked_parquet = output_features_dir / "masked_reviews.parquet"
    vault_parquet = output_features_dir / "pii_vault.parquet"

    batches_dir = settings.PROGRESS_DIR / "pii_batches"
    batches_dir.mkdir(parents=True, exist_ok=True)

    df_cursor = con.execute(query)
    batch_idx = 0
    total_stats = {"person": 0, "email": 0, "phone": 0, "url": 0, "card": 0}

    while True:
        chunk_df = df_cursor.fetch_df_chunk(batch_size)
        if chunk_df is None or len(chunk_df) == 0:
            break

        chunk_df, b_stats = mask_pii_texts(chunk_df, nlp, food_vocab, city_allowlist)
        for k in total_stats:
            total_stats[k] += b_stats[k]

        batch_out = batches_dir / f"pii_batch_{batch_idx}.parquet"
        chunk_df.to_parquet(batch_out, index=False, compression="zstd")
        batch_idx += 1
        logger.info(f"PII Gate 1 processed batch {batch_idx} ({batch_idx * batch_size}/{total_rows} reviews)")

    # Combine into masked_reviews and pii_vault
    logger.info("Combining PII batches...")
    batch_glob = str(batches_dir / "pii_batch_*.parquet").replace(chr(92), "/")

    # Masked reviews
    masked_tmp = masked_parquet.with_suffix(".parquet.tmp")
    con.execute(f"""
        COPY (
            SELECT review_id, business_id, stars, text_masked, date, year_month
            FROM read_parquet('{batch_glob}')
        ) TO '{str(masked_tmp).replace(chr(92), "/")}' (FORMAT PARQUET, COMPRESSION ZSTD);
    """)
    masked_tmp.replace(masked_parquet)

    # Vault
    vault_tmp = vault_parquet.with_suffix(".parquet.tmp")
    con.execute(f"""
        COPY (
            SELECT review_id, text as text_raw
            FROM read_parquet('{batch_glob}')
        ) TO '{str(vault_tmp).replace(chr(92), "/")}' (FORMAT PARQUET, COMPRESSION ZSTD);
    """)
    vault_tmp.replace(vault_parquet)

    # Cleanup temporary batches
    import shutil
    shutil.rmtree(batches_dir, ignore_errors=True)

    logger.info("=" * 60)
    logger.info("PII GATE 1 MASKING STATISTICS:")
    logger.info(f"  [PERSON] Redactions: {total_stats['person']}")
    logger.info(f"  [EMAIL]  Redactions: {total_stats['email']}")
    logger.info(f"  [PHONE]  Redactions: {total_stats['phone']}")
    logger.info(f"  [URL]    Redactions: {total_stats['url']}")
    logger.info(f"  [CARD]   Redactions: {total_stats['card']}")
    logger.info("=" * 60)
