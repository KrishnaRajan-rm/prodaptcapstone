"""Phase 3: Aspect tagging and multi-level aggregation with annual batch processing."""

import re
import shutil
import time
import yaml
from pathlib import Path
import duckdb

from app.core.config import settings
from app.core.logging import logger


def load_aspect_lexicon() -> dict:
    lexicon_path = Path(__file__).parent / "aspect_lexicon.yaml"
    with open(lexicon_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_aspect_regex_patterns(lexicon: dict, for_sql: bool = True) -> dict:
    """Builds lowercase regex strings for each aspect and negative cues, escaping single quotes for SQL."""
    patterns = {}
    quote_rep = "''" if for_sql else "'"
    for aspect, data in lexicon["aspects"].items():
        escaped_pats = [re.escape(p.lower().strip()).replace("'", quote_rep) for p in data["patterns"]]
        patterns[aspect] = "|".join(escaped_pats)

    neg_pats = [re.escape(p.lower().strip()).replace("'", quote_rep) for p in lexicon.get("negative_cues", [])]
    patterns["_negative_cues"] = "|".join(neg_pats)
    return patterns


def tag_review_aspects(reviews_clean_parquet: Path, output_features_dir: Path, output_aggregates_dir: Path):
    """Tags all clean reviews with aspect and complaint flags using batched DuckDB regex."""
    logger.info("Starting review-level aspect tagging and feature generation...")
    con = duckdb.connect()
    temp_dir = settings.PROGRESS_DIR / "duckdb_temp"
    temp_dir.mkdir(parents=True, exist_ok=True)

    con.execute("PRAGMA memory_limit='3GB'")
    con.execute("PRAGMA threads=4")
    con.execute("PRAGMA preserve_insertion_order=false")
    con.execute(f"PRAGMA temp_directory='{str(temp_dir).replace(chr(92), '/')}'")

    lexicon = load_aspect_lexicon()
    patterns = build_aspect_regex_patterns(lexicon)
    neg_pattern = patterns["_negative_cues"]

    output_features_dir.mkdir(parents=True, exist_ok=True)
    output_aggregates_dir.mkdir(parents=True, exist_ok=True)

    review_aspects_path = output_features_dir / "review_aspects.parquet"
    monthly_aspects_path = output_aggregates_dir / "monthly_aspects.parquet"
    city_monthly_aspects_path = output_aggregates_dir / "city_monthly_aspects.parquet"

    # Build SQL projection expressions for each aspect
    aspect_cols = []
    agg_cols = []
    aspect_names = list(lexicon["aspects"].keys())

    for asp in aspect_names:
        pat = patterns[asp]
        asp_col = f"regexp_matches(lower(text), '{pat}') AS has_{asp}"
        cmp_col = f"(regexp_matches(lower(text), '{pat}') AND (stars <= 3.0 OR regexp_matches(lower(text), '{neg_pattern}'))) AS is_complaint_{asp}"
        aspect_cols.append(asp_col)
        aspect_cols.append(cmp_col)

        agg_cols.append(f"sum(CASE WHEN has_{asp} THEN 1 ELSE 0 END) AS n_{asp}_mentions")
        agg_cols.append(f"sum(CASE WHEN is_complaint_{asp} THEN 1 ELSE 0 END) AS n_{asp}_complaints")

    rev_sql = f"read_parquet('{str(reviews_clean_parquet).replace(chr(92), '/')}')"
    aspect_cols_sql = ",\n                    ".join(aspect_cols)

    # 1. Process in annual batches to ensure zero memory exhaustion
    years = [row[0] for row in con.execute(f"SELECT DISTINCT substr(date, 1, 4) FROM {rev_sql} ORDER BY 1").fetchall() if row[0]]
    logger.info(f"Tagging aspect flags across {len(years)} annual batches...")

    batches_dir = settings.PROGRESS_DIR / "aspect_tag_batches"
    batches_dir.mkdir(parents=True, exist_ok=True)
    batch_files = []

    for yr in years:
        t0 = time.time()
        yr_file = batches_dir / f"aspects_{yr}.parquet"
        con.execute(f"""
            COPY (
                SELECT
                    review_id,
                    business_id,
                    stars,
                    date,
                    year_month,
                    {aspect_cols_sql}
                FROM {rev_sql}
                WHERE substr(date, 1, 4) = '{yr}'
            ) TO '{str(yr_file).replace(chr(92), "/")}' (FORMAT PARQUET, COMPRESSION ZSTD);
        """)
        batch_files.append(yr_file)
        logger.info(f"Tagged aspects for year {yr} in {time.time() - t0:.2f}s")

    # Combine batches into final review_aspects.parquet
    logger.info("Combining aspect batches into final review_aspects.parquet...")
    batch_glob = str(batches_dir / "aspects_*.parquet").replace(chr(92), "/")
    feat_tmp = review_aspects_path.with_suffix(".parquet.tmp")
    con.execute(f"""
        COPY (
            SELECT * FROM read_parquet('{batch_glob}')
        ) TO '{str(feat_tmp).replace(chr(92), "/")}' (FORMAT PARQUET, COMPRESSION ZSTD);
    """)
    feat_tmp.replace(review_aspects_path)
    shutil.rmtree(batches_dir, ignore_errors=True)

    count_features = con.execute(f"SELECT count(*) FROM read_parquet('{str(review_aspects_path).replace(chr(92), "/")}')").fetchone()[0]
    logger.info(f"Review aspect features written for {count_features} reviews -> {review_aspects_path}")

    # 2. Monthly restaurant-level aggregates
    logger.info("Computing monthly restaurant-level aspect aggregates...")
    mon_tmp = monthly_aspects_path.with_suffix(".parquet.tmp")
    agg_cols_sql = ",\n                ".join(agg_cols)
    feat_sql = f"read_parquet('{str(review_aspects_path).replace(chr(92), '/')}')"

    query_monthly = f"""
        COPY (
            SELECT
                business_id,
                year_month,
                count(*) AS n_reviews,
                round(avg(stars), 2) AS avg_stars,
                {agg_cols_sql}
            FROM {feat_sql}
            GROUP BY 1, 2
        ) TO '{str(mon_tmp).replace(chr(92), "/")}' (FORMAT PARQUET, COMPRESSION ZSTD);
    """
    con.execute(query_monthly)
    mon_tmp.replace(monthly_aspects_path)
    count_monthly = con.execute(f"SELECT count(*) FROM read_parquet('{str(monthly_aspects_path).replace(chr(92), "/")}')").fetchone()[0]
    logger.info(f"Monthly aspects aggregate written: {count_monthly} rows -> {monthly_aspects_path}")

    # 3. Monthly city-level aggregates
    logger.info("Computing monthly city-level aspect aggregates...")
    city_tmp = city_monthly_aspects_path.with_suffix(".parquet.tmp")
    bus_sql = f"read_parquet('{str(settings.PARQUET_DIR / 'clean' / 'restaurant_profiles.parquet').replace(chr(92), '/')}')"

    query_city = f"""
        COPY (
            SELECT
                b.city,
                b.state,
                f.year_month,
                count(*) AS n_reviews,
                round(avg(f.stars), 2) AS avg_stars,
                {agg_cols_sql}
            FROM {feat_sql} f
            JOIN {bus_sql} b ON f.business_id = b.business_id
            GROUP BY 1, 2, 3
        ) TO '{str(city_tmp).replace(chr(92), "/")}' (FORMAT PARQUET, COMPRESSION ZSTD);
    """
    con.execute(query_city)
    city_tmp.replace(city_monthly_aspects_path)
    count_city = con.execute(f"SELECT count(*) FROM read_parquet('{str(city_monthly_aspects_path).replace(chr(92), "/")}')").fetchone()[0]
    logger.info(f"City monthly aspects aggregate written: {count_city} rows -> {city_monthly_aspects_path}")
