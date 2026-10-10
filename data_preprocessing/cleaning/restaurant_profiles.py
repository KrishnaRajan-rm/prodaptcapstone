"""Phase 2: Restaurant profiles assembly."""

import json
from pathlib import Path
import duckdb
import pandas as pd

from app.core.config import settings
from app.core.logging import logger
from data_preprocessing.cleaning.attributes import (
    parse_business_attributes,
    parse_business_hours,
    extract_primary_category
)


def build_restaurant_profiles(
    businesses_parquet_path: Path,
    peak_slots_parquet_path: Path,
    output_parquet_path: Path
):
    """Parses restaurant attributes, joins peak slots, and saves clean restaurant profiles."""
    logger.info("Building clean restaurant profiles...")
    con = duckdb.connect()

    bus_df = con.execute(f"SELECT * FROM read_parquet('{str(businesses_parquet_path).replace(chr(92), '/')}')").df()
    logger.info(f"Loaded {len(bus_df)} businesses for profile enrichment.")

    # Load peak slots
    peak_slots_map = {}
    if peak_slots_parquet_path.exists():
        peaks_df = con.execute(f"SELECT business_id, peak_slots FROM read_parquet('{str(peak_slots_parquet_path).replace(chr(92), '/')}')").df()
        for _, row in peaks_df.iterrows():
            peak_slots_map[row["business_id"]] = row["peak_slots"]

    # Parse attributes and hours row by row
    records = []
    for _, row in bus_df.iterrows():
        b_id = row["business_id"]
        raw_attrs = row["attributes"] if pd.notna(row.get("attributes")) else None
        raw_hours = row["hours"] if pd.notna(row.get("hours")) else None

        parsed_attrs = parse_business_attributes(raw_attrs)
        parsed_hours = parse_business_hours(raw_hours)
        primary_cat = extract_primary_category(row.get("categories"))
        peak_slots = peak_slots_map.get(b_id, [])

        rec = {
            "business_id": b_id,
            "name": row["name"],
            "address": row.get("address"),
            "city": row["city"],
            "state": row["state"],
            "postal_code": row.get("postal_code"),
            "latitude": row.get("latitude"),
            "longitude": row.get("longitude"),
            "stars": float(row["stars"]),
            "review_count": int(row["review_count"]),
            "is_open": int(row["is_open"]),
            "categories": row.get("categories") or "",
            "primary_category": primary_cat,
            "raw_attributes": json.dumps(raw_attrs) if raw_attrs else None,
            "raw_hours": json.dumps(raw_hours) if raw_hours else None,
            "price_range": parsed_attrs["price_range"],
            "noise_level": parsed_attrs["noise_level"],
            "alcohol": parsed_attrs["alcohol"],
            "wifi": parsed_attrs["wifi"],
            "outdoor_seating": parsed_attrs["outdoor_seating"],
            "good_for_kids": parsed_attrs["good_for_kids"],
            "has_tv": parsed_attrs["has_tv"],
            "reservations": parsed_attrs["reservations"],
            "delivery": parsed_attrs["delivery"],
            "takeout": parsed_attrs["takeout"],
            "parking_garage": parsed_attrs["parking_garage"],
            "parking_street": parsed_attrs["parking_street"],
            "parking_lot": parsed_attrs["parking_lot"],
            "parking_valet": parsed_attrs["parking_valet"],
            "peak_slots": peak_slots
        }
        rec.update(parsed_hours)
        records.append(rec)

    enriched_df = pd.DataFrame(records)
    output_tmp = output_parquet_path.with_suffix(".parquet.tmp")
    output_tmp.parent.mkdir(parents=True, exist_ok=True)
    enriched_df.to_parquet(output_tmp, index=False, compression="zstd")
    output_tmp.replace(output_parquet_path)

    logger.info(f"Restaurant profiles written: {len(enriched_df)} rows -> {output_parquet_path}")
