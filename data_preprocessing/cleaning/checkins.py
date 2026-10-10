"""Phase 2: Checkins transformation and peak slot analysis."""

from pathlib import Path
import duckdb

from app.core.config import settings
from app.core.logging import logger


def clean_checkins(checkins_parquet_path: Path, clean_dir: Path):
    """Processes checkin timestamps into hour-weekday distributions, monthly trends, and top 5 peak slots."""
    logger.info("Processing checkin timestamps and peak hours...")
    con = duckdb.connect()
    
    clean_dir.mkdir(parents=True, exist_ok=True)
    hw_parquet = clean_dir / "checkin_hour_weekday.parquet"
    monthly_parquet = clean_dir / "checkin_monthly.parquet"
    peaks_parquet = clean_dir / "restaurant_peak_slots.parquet"

    chk_sql = f"read_parquet('{str(checkins_parquet_path).replace(chr(92), '/')}')"

    # 1. Checkin hour-weekday breakdown
    logger.info("Computing checkin hour x weekday counts...")
    hw_tmp = hw_parquet.with_suffix(".parquet.tmp")
    query_hw = f"""
        COPY (
            WITH expanded AS (
                SELECT
                    business_id,
                    TRY_CAST(UNNEST(string_split(date, ', ')) AS TIMESTAMP) as ts
                FROM {chk_sql}
                WHERE date IS NOT NULL
            )
            SELECT
                business_id,
                dayname(ts) as weekday,
                hour(ts) as hour,
                count(*) as count
            FROM expanded
            WHERE ts IS NOT NULL
            GROUP BY 1, 2, 3
        ) TO '{str(hw_tmp).replace(chr(92), "/")}' (FORMAT PARQUET, COMPRESSION ZSTD);
    """
    con.execute(query_hw)
    hw_tmp.replace(hw_parquet)

    # 2. Checkin monthly counts
    logger.info("Computing checkin monthly counts...")
    mon_tmp = monthly_parquet.with_suffix(".parquet.tmp")
    query_monthly = f"""
        COPY (
            WITH expanded AS (
                SELECT
                    business_id,
                    TRY_CAST(UNNEST(string_split(date, ', ')) AS TIMESTAMP) as ts
                FROM {chk_sql}
                WHERE date IS NOT NULL
            )
            SELECT
                business_id,
                strftime(ts, '%Y-%m') as year_month,
                count(*) as count
            FROM expanded
            WHERE ts IS NOT NULL
            GROUP BY 1, 2
        ) TO '{str(mon_tmp).replace(chr(92), "/")}' (FORMAT PARQUET, COMPRESSION ZSTD);
    """
    con.execute(query_monthly)
    mon_tmp.replace(monthly_parquet)

    # 3. Peak slots per restaurant (top 5 slots)
    logger.info("Computing top 5 peak slots per restaurant...")
    peaks_tmp = peaks_parquet.with_suffix(".parquet.tmp")
    query_peaks = f"""
        COPY (
            WITH ranked AS (
                SELECT
                    business_id,
                    concat(substr(weekday, 1, 3), '-', lpad(CAST(hour AS VARCHAR), 2, '0')) as slot,
                    count,
                    row_number() OVER (PARTITION BY business_id ORDER BY count DESC) as rn
                FROM read_parquet('{str(hw_parquet).replace(chr(92), "/")}')
            )
            SELECT
                business_id,
                list(slot) as peak_slots
            FROM ranked
            WHERE rn <= 5
            GROUP BY business_id
        ) TO '{str(peaks_tmp).replace(chr(92), "/")}' (FORMAT PARQUET, COMPRESSION ZSTD);
    """
    con.execute(query_peaks)
    peaks_tmp.replace(peaks_parquet)

    logger.info("Checkin cleaning completed successfully.")
