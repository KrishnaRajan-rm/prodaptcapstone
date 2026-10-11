"""Durable SQLite persistence layer for requests, traces, and manager feedback."""

import json
import sqlite3
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.core.config import settings
from app.core.logging import logger
from app.api.schemas import FeedbackCreate, FeedbackSummaryResponse


DB_PATH = settings.SQLITE_DB_PATH if hasattr(settings, "SQLITE_DB_PATH") else (
    settings.BASE_DIR / "data_store" / "sqlite" / "restaurant_ops.db"
)


def get_db_connection() -> sqlite3.Connection:
    """Creates a connection to SQLite database with foreign keys and WAL mode enabled."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(str(DB_PATH))
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA journal_mode=WAL;")
    return con


def init_db():
    """Ensures required tables exist."""
    con = get_db_connection()
    try:
        con.execute("""
            CREATE TABLE IF NOT EXISTS feedback (
                feedback_id TEXT PRIMARY KEY,
                trace_id TEXT,
                rating INTEGER,
                flags TEXT,
                corrected_aspect TEXT,
                comment TEXT,
                role TEXT,
                created_at TEXT
            );
        """)
        con.execute("""
            CREATE TABLE IF NOT EXISTS traces (
                trace_id TEXT PRIMARY KEY,
                request_id TEXT,
                timestamp TEXT,
                trace_json TEXT
            );
        """)
        con.execute("""
            CREATE TABLE IF NOT EXISTS request_log (
                request_id TEXT PRIMARY KEY,
                endpoint TEXT,
                timestamp TEXT,
                payload TEXT
            );
        """)
        con.commit()
    finally:
        con.close()


def save_analysis_result(request_id: str, endpoint: str, payload: Dict[str, Any], trace_data: Optional[Dict[str, Any]] = None):
    """Persists analysis request payload and execution trace."""
    init_db()
    con = get_db_connection()
    try:
        now_ts = time.strftime("%Y-%m-%d %H:%M:%S")
        # Save request log
        con.execute(
            "INSERT OR REPLACE INTO request_log (request_id, endpoint, timestamp, payload) VALUES (?, ?, ?, ?)",
            (request_id, endpoint, now_ts, json.dumps(payload, default=str))
        )
        # Save trace
        if trace_data:
            trace_id = str(uuid.uuid4())
            con.execute(
                "INSERT OR REPLACE INTO traces (trace_id, request_id, timestamp, trace_json) VALUES (?, ?, ?, ?)",
                (trace_id, request_id, now_ts, json.dumps(trace_data, default=str))
            )
        con.commit()
    except Exception as e:
        logger.warning(f"Could not persist analysis record {request_id}: {e}")
    finally:
        con.close()


def get_analysis_by_request_id(request_id: str) -> Optional[Dict[str, Any]]:
    """Retrieves persisted analysis result and trace from SQLite."""
    init_db()
    con = get_db_connection()
    try:
        row = con.execute("SELECT payload FROM request_log WHERE request_id = ?", (request_id,)).fetchone()
        if not row:
            return None
        return json.loads(row["payload"])
    except Exception as e:
        logger.warning(f"Error retrieving analysis {request_id}: {e}")
        return None
    finally:
        con.close()


def save_feedback(feedback: FeedbackCreate) -> str:
    """Inserts a new manager feedback record."""
    init_db()
    con = get_db_connection()
    try:
        feedback_id = f"fb_{uuid.uuid4().hex[:10]}"
        now_ts = time.strftime("%Y-%m-%d %H:%M:%S")
        con.execute(
            """
            INSERT INTO feedback (feedback_id, trace_id, rating, flags, corrected_aspect, comment, role, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                feedback_id,
                feedback.request_id,
                feedback.rating,
                feedback.flags or "",
                "",
                feedback.comment or "",
                feedback.role or "manager",
                now_ts,
            )
        )
        con.commit()
        return feedback_id
    finally:
        con.close()


def get_feedback_summary() -> FeedbackSummaryResponse:
    """Calculates aggregated feedback statistics from the database."""
    init_db()
    con = get_db_connection()
    try:
        rows = con.execute("SELECT rating, flags FROM feedback").fetchall()
        total = len(rows)
        if total == 0:
            return FeedbackSummaryResponse(
                total_feedback=0,
                average_rating=5.0,
                useful_count=0,
                not_useful_count=0,
                common_flags={},
            )

        ratings = [r["rating"] for r in rows if r["rating"] is not None]
        avg_rating = sum(ratings) / len(ratings) if ratings else 0.0
        useful = sum(1 for r in ratings if r >= 4)
        not_useful = sum(1 for r in ratings if r < 4)

        flag_counts: Dict[str, int] = {}
        for r in rows:
            flags = (r["flags"] or "").split(",")
            for f in flags:
                f = f.strip().lower()
                if f:
                    flag_counts[f] = flag_counts.get(f, 0) + 1

        return FeedbackSummaryResponse(
            total_feedback=total,
            average_rating=round(avg_rating, 2),
            useful_count=useful,
            not_useful_count=not_useful,
            common_flags=flag_counts,
        )
    finally:
        con.close()
