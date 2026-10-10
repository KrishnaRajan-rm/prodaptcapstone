"""Tests for Phase 0: Bootstrap, Environment, Configuration, and Storage Engines."""

import sqlite3
import duckdb
import pytest
from app.core.config import settings
from app.core.schemas import BusinessSchema, ReviewSchema, ChunkSchema
from app.core.logging import setup_logger
from app.core.errors import RestaurantOpsError


def test_config_loads():
    assert settings.BASE_DIR.exists()
    assert settings.DEV_CITY is not None
    assert settings.EMBEDDING_DIM == 384
    assert len(settings.POSITIVE_CATEGORIES) > 0


def test_logger_setup(tmp_path):
    log_file = tmp_path / "test.log"
    logger = setup_logger("test_logger", log_file=log_file)
    logger.info("Test log entry")
    assert log_file.exists()
    content = log_file.read_text(encoding="utf-8")
    assert "Test log entry" in content


def test_schemas_validation():
    bus = BusinessSchema(
        business_id="b123",
        name="Pasta Palace",
        city="Philadelphia",
        state="PA",
        stars=4.5,
        review_count=150,
        is_open=1,
        categories="Restaurants, Italian"
    )
    assert bus.business_id == "b123"
    assert bus.stars == 4.5

    rev = ReviewSchema(
        review_id="r123",
        user_id="u123",
        business_id="b123",
        stars=4.0,
        text="Great food and service!",
        date="2022-01-01"
    )
    assert rev.stars == 4.0

    chunk = ChunkSchema(
        chunk_id="r123_0",
        review_id="r123",
        business_id="b123",
        restaurant_name="Pasta Palace",
        city="Philadelphia",
        state="PA",
        categories="Restaurants, Italian",
        stars=4.0,
        business_stars=4.5,
        date="2022-01-01",
        year_month="2022-01",
        credibility=0.85,
        text_masked="Great food and service!",
        aspects=["food_quality"],
        aspect_sentiments={"food_quality": 0.8},
        n_tokens=25
    )
    assert chunk.chunk_id == "r123_0"


def test_duckdb_engine():
    con = duckdb.connect()
    res = con.execute("SELECT 42 as answer").fetchone()
    assert res[0] == 42


def test_sqlite_fts5_support():
    con = sqlite3.connect(":memory:")
    con.execute("CREATE VIRTUAL TABLE t USING fts5(content, tokenize='porter unicode61')")
    con.execute("INSERT INTO t(content) VALUES ('waiter was slow and food was cold')")
    hits = con.execute("SELECT content FROM t WHERE t MATCH 'slow'").fetchall()
    assert len(hits) == 1
