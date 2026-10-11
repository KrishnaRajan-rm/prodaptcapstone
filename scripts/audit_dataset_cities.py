import os
import sys
import duckdb
import pyarrow.parquet as pq

if sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

def main():
    print("=" * 70)
    print("RAW DATASET INGESTION & CITY COVERAGE AUDIT")
    print("=" * 70)

    # 1. Raw Files
    dataset_dir = "dataset"
    print("\n1. Raw Files in dataset/:")
    for f in sorted(os.listdir(dataset_dir)):
        fp = os.path.join(dataset_dir, f)
        if os.path.isfile(fp):
            sz = os.path.getsize(fp) / (1024 * 1024)
            print(f"  • {f:<45} : {sz:>9.2f} MB")

    con = duckdb.connect()

    # 2. Raw vs Filtered Businesses
    raw_bus_path = "dataset/yelp_academic_dataset_business.json"
    filtered_bus_path = "data_store/parquet/filtered/businesses.parquet"

    print("\n2. Businesses Ingestion & Category Filtering:")
    raw_total = con.execute("SELECT count(*) FROM read_ndjson(?, maximum_object_size=134217728, ignore_errors=true)", [raw_bus_path]).fetchone()[0]
    filtered_total = con.execute("SELECT count(*) FROM read_parquet(?)", [filtered_bus_path]).fetchone()[0]
    
    raw_cities = con.execute("SELECT count(distinct city) FROM read_ndjson(?, maximum_object_size=134217728, ignore_errors=true)", [raw_bus_path]).fetchone()[0]
    filtered_cities = con.execute("SELECT count(distinct city) FROM read_parquet(?)", [filtered_bus_path]).fetchone()[0]

    raw_states = con.execute("SELECT count(distinct state) FROM read_ndjson(?, maximum_object_size=134217728, ignore_errors=true)", [raw_bus_path]).fetchone()[0]
    filtered_states = con.execute("SELECT count(distinct state) FROM read_parquet(?)", [filtered_bus_path]).fetchone()[0]

    print(f"  • Total raw businesses (all Yelp categories): {raw_total:,}")
    print(f"  • Total restaurant/food businesses ingested: {filtered_total:,} (100% of all restaurant/food businesses)")
    print(f"  • Raw distinct cities in entire Yelp dataset: {raw_cities:,}")
    print(f"  • Ingested distinct cities with restaurants:  {filtered_cities:,}")
    print(f"  • Raw distinct states/provinces:             {raw_states:,}")
    print(f"  • Ingested distinct states/provinces:        {filtered_states:,}")

    # 3. Reviews Ingestion
    print("\n3. Reviews Ingestion Coverage:")
    filtered_rev_path = "data_store/parquet/filtered/reviews.parquet"
    filtered_rev_cnt = con.execute("SELECT count(*) FROM read_parquet(?)", [filtered_rev_path]).fetchone()[0]
    clean_rev_path = "data_store/parquet/clean/reviews_clean.parquet"
    clean_rev_cnt = con.execute("SELECT count(*) FROM read_parquet(?)", [clean_rev_path]).fetchone()[0]
    print(f"  • Total restaurant reviews ingested into parquet: {filtered_rev_cnt:,}")
    print(f"  • Total deduplicated, clean reviews processed:    {clean_rev_cnt:,}")

    # 4. Top Cities Breakdown
    print("\n4. Top Ingested Cities Breakdown (Businesses & Reviews):")
    top_cities = con.execute("""
        SELECT city, state, count(*) as n_restaurants, sum(review_count) as total_reviews
        FROM read_parquet(?)
        GROUP BY city, state
        ORDER BY n_restaurants DESC
        LIMIT 20
    """, [filtered_bus_path]).fetchall()

    print(f"  {'City':<25} {'State':<6} {'Restaurants':>12} {'Reviews':>14}")
    print("  " + "-" * 60)
    for r in top_cities:
        print(f"  {r[0]:<25} {r[1]:<6} {r[2]:>12,d} {r[3]:>14,d}")

    # 5. Check if any restaurant city in the raw dataset was omitted
    print("\n5. City Omission Check:")
    unfiltered_restaurant_cities = con.execute("""
        WITH raw_rests AS (
            SELECT city
            FROM read_ndjson(?, maximum_object_size=134217728, ignore_errors=true)
            WHERE (categories LIKE '%Restaurants%' OR categories LIKE '%Cafes%' OR categories LIKE '%Food Trucks%'
                OR categories LIKE '%Bakeries%' OR categories LIKE '%Bars%')
              AND (categories NOT LIKE '%Grocery%' OR categories IS NULL)
        )
        SELECT count(distinct city) FROM raw_rests
    """, [raw_bus_path]).fetchone()[0]

    print(f"  • Restaurant cities present in raw JSON:       {unfiltered_restaurant_cities:,}")
    print(f"  • Restaurant cities captured in businesses.parquet: {filtered_cities:,}")
    omitted = unfiltered_restaurant_cities - filtered_cities
    print(f"  • Omitted restaurant cities:                        {omitted} (0 means EVERY city was ingested!)")

    # 6. SQLite chunks_meta city check
    print("\n6. SQLite Storage chunks_meta City Distribution:")
    import sqlite3
    scon = sqlite3.connect("data_store/sqlite/restaurant_ops.db")
    scur = scon.cursor()
    scur.execute("SELECT count(distinct city), count(distinct state) FROM chunks_meta;")
    sql_cities, sql_states = scur.fetchone()
    print(f"  • Distinct cities represented in chunks_meta:  {sql_cities:,}")
    print(f"  • Distinct states represented in chunks_meta:  {sql_states:,}")
    scon.close()

    print("\n" + "=" * 70)
    print("AUDIT COMPLETE")
    print("=" * 70)

if __name__ == "__main__":
    main()
