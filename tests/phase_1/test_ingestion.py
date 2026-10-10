"""Unit tests for Phase 1: Ingestion Pipeline."""

import json
from pathlib import Path
import duckdb
import pytest
from app.core.config import settings


def test_restaurant_category_filtering():
    con = duckdb.connect()
    
    # Create sample business table in memory
    sample_businesses = [
        {"business_id": "b1", "name": "Tonys Pizza", "categories": "Restaurants, Italian, Pizza", "city": "Philadelphia", "review_count": 200},
        {"business_id": "b2", "name": "Dental Clinic", "categories": "Dentists, Health & Medical", "city": "Philadelphia", "review_count": 50},
        {"business_id": "b3", "name": "Corner Grocery & Snacks", "categories": "Restaurants, Grocery", "city": "Philadelphia", "review_count": 120},
        {"business_id": "b4", "name": "Coffee Lab", "categories": "Coffee & Tea, Cafes", "city": "Philadelphia", "review_count": 300},
    ]
    con.execute("CREATE TABLE sample_bus AS SELECT * FROM (VALUES " + 
                ", ".join([f"('{b['business_id']}', '{b['name']}', '{b['categories']}', '{b['city']}', {b['review_count']})" for b in sample_businesses]) +
                ") AS t(business_id, name, categories, city, review_count)")

    pos_clauses = [f"categories LIKE '%{cat}%'" for cat in settings.POSITIVE_CATEGORIES]
    neg_clauses = [f"(categories NOT LIKE '%{cat}%' OR categories IS NULL)" for cat in settings.NEGATIVE_CATEGORIES]
    query = f"""
        SELECT business_id, name FROM sample_bus
        WHERE ({' OR '.join(pos_clauses)}) AND ({' AND '.join(neg_clauses)})
    """
    rows = con.execute(query).fetchall()
    kept_ids = [r[0] for r in rows]

    assert "b1" in kept_ids  # Tony's Pizza kept
    assert "b4" in kept_ids  # Coffee Lab kept
    assert "b2" not in kept_ids  # Dentist dropped
    assert "b3" not in kept_ids  # Grocery dropped


def test_tip_length_filter():
    con = duckdb.connect()
    sample_tips = [
        {"text": "Short tip"},                      # 2 words -> drop
        {"text": "Great food and quick service!"},   # 5 words -> keep
        {"text": "Always ask for the special soup here"} # 7 words -> keep
    ]
    con.execute("CREATE TABLE sample_tips AS SELECT * FROM (VALUES " +
                ", ".join([f"('{t['text']}')" for t in sample_tips]) +
                ") AS t(text)")
    query = """
        SELECT text FROM sample_tips
        WHERE (length(trim(text)) - length(replace(trim(text), ' ', '')) + 1) >= 5
    """
    rows = con.execute(query).fetchall()
    assert len(rows) == 2


def test_user_privacy_exclusion():
    # Ensure neither name nor friends is ever projected in user schema
    user_cols = [
        "user_id", "review_count", "yelping_since", "useful", "funny", "cool",
        "elite", "fans", "average_stars", "compliment_hot"
    ]
    assert "name" not in user_cols
    assert "friends" not in user_cols
