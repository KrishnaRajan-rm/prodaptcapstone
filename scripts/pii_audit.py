"""Phase 3 Audit: Exports 100 random before/after PII masking pairs to CSV."""

from pathlib import Path
import duckdb
from app.core.config import settings
from app.core.logging import logger


def run_pii_audit(n_samples: int = 100):
    con = duckdb.connect()
    masked_path = settings.PARQUET_DIR / "features" / "masked_reviews.parquet"
    vault_path = settings.PARQUET_DIR / "features" / "pii_vault.parquet"

    if not masked_path.exists() or not vault_path.exists():
        logger.error("Required parquet files missing for PII audit.")
        return

    out_csv = settings.BASE_DIR / "pii_audit_sample.csv"
    query = f"""
        SELECT
            m.review_id,
            v.text_raw,
            m.text_masked
        FROM read_parquet('{str(masked_path).replace(chr(92), '/')}') m
        JOIN read_parquet('{str(vault_path).replace(chr(92), '/')}') v
        ON m.review_id = v.review_id
        WHERE m.text_masked LIKE '%[PERSON]%' 
           OR m.text_masked LIKE '%[EMAIL]%' 
           OR m.text_masked LIKE '%[PHONE]%'
        ORDER BY hash(m.review_id)
        LIMIT {n_samples}
    """
    df = con.execute(query).df()
    df.to_csv(out_csv, index=False, encoding="utf-8")
    logger.info(f"PII audit exported {len(df)} samples to {out_csv}")


if __name__ == "__main__":
    run_pii_audit()
