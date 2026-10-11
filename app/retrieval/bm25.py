"""BM25 Keyword Retrieval Module using SQLite FTS5.
Searches chunks_fts virtual table with Porter stemming, safe parameterization,
PII placeholder stripping, and metadata filtering on chunks_meta.
"""

import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.core.config import settings
from app.core.logging import logger

PII_PLACEHOLDERS_REGEX = re.compile(r'\[(PERSON|EMAIL|PHONE|URL|CARD)\]', re.IGNORECASE)
SAFE_TOKEN_REGEX = re.compile(r'[a-zA-Z0-9_\-]+')


@dataclass
class BM25Candidate:
    chunk_id: str
    bm25_score: float
    bm25_rank: int
    review_id: str
    business_id: str
    restaurant_name: str
    city: str
    state: str
    categories: Optional[str]
    price_range: Optional[int]
    is_open: int
    stars: float
    business_stars: float
    date: str
    year_month: str
    credibility: float
    text_masked: str
    aspects: str
    n_tokens: int


def clean_query_text(query: str) -> str:
    """Removes PII placeholders and non-alphanumeric noise from search queries."""
    if not query:
        return ""
    text = PII_PLACEHOLDERS_REGEX.sub(" ", query)
    tokens = SAFE_TOKEN_REGEX.findall(text)
    # Filter very short noise tokens
    tokens = [t.lower() for t in tokens if len(t) > 1 and t.lower() not in {"and", "or", "not", "the", "for", "with"}]
    return " OR ".join(tokens)


class BM25Retriever:
    """Retrieves review chunks via SQLite FTS5 BM25 keyword matching."""

    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = Path(db_path or (settings.SQLITE_DIR / "restaurant_ops.db"))
        if not self.db_path.exists():
            raise FileNotFoundError(f"SQLite database not found at {self.db_path}")

    def _get_connection(self) -> sqlite3.Connection:
        con = sqlite3.connect(str(self.db_path), timeout=15.0)
        con.row_factory = sqlite3.Row
        return con

    def search(
        self,
        query: str,
        top_k: int = 100,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[BM25Candidate]:
        """Searches chunks_fts using BM25 ranking with optional metadata filtering."""
        if not query or not query.strip():
            return []

        match_expr = clean_query_text(query)
        if not match_expr:
            return []

        where_clauses = ["chunks_fts MATCH ?"]
        params: List[Any] = [match_expr]

        if filters:
            if "city" in filters and filters["city"]:
                where_clauses.append("m.city = ?")
                params.append(filters["city"])
            if "state" in filters and filters["state"]:
                where_clauses.append("m.state = ?")
                params.append(filters["state"])
            if "business_id" in filters and filters["business_id"]:
                where_clauses.append("m.business_id = ?")
                params.append(filters["business_id"])
            if "is_open" in filters and filters["is_open"] is not None:
                where_clauses.append("m.is_open = ?")
                params.append(int(filters["is_open"]))
            if "min_stars" in filters and filters["min_stars"] is not None:
                where_clauses.append("m.stars >= ?")
                params.append(float(filters["min_stars"]))
            if "max_stars" in filters and filters["max_stars"] is not None:
                where_clauses.append("m.stars <= ?")
                params.append(float(filters["max_stars"]))
            if "aspect" in filters and filters["aspect"]:
                where_clauses.append("m.aspects LIKE ?")
                params.append(f"%{filters['aspect']}%")

        sql = f"""
            SELECT 
                f.chunk_id,
                bm25(chunks_fts) as score,
                m.review_id,
                m.business_id,
                m.restaurant_name,
                m.city,
                m.state,
                m.categories,
                m.price_range,
                m.is_open,
                m.stars,
                m.business_stars,
                m.date,
                m.year_month,
                m.credibility,
                m.text_masked,
                m.aspects,
                m.n_tokens
            FROM chunks_fts f
            JOIN chunks_meta m ON f.chunk_id = m.chunk_id
            WHERE {' AND '.join(where_clauses)}
            ORDER BY score ASC
            LIMIT ?;
        """
        params.append(top_k)

        candidates: List[BM25Candidate] = []
        try:
            with self._get_connection() as con:
                cursor = con.cursor()
                cursor.execute(sql, params)
                rows = cursor.fetchall()

                for rank, row in enumerate(rows, start=1):
                    # Filter out chunks that are purely PII placeholders
                    text_no_pii = PII_PLACEHOLDERS_REGEX.sub("", row["text_masked"]).strip()
                    if not text_no_pii:
                        continue

                    candidates.append(BM25Candidate(
                        chunk_id=row["chunk_id"],
                        bm25_score=float(row["score"]),
                        bm25_rank=rank,
                        review_id=row["review_id"],
                        business_id=row["business_id"],
                        restaurant_name=row["restaurant_name"],
                        city=row["city"],
                        state=row["state"],
                        categories=row["categories"],
                        price_range=row["price_range"],
                        is_open=int(row["is_open"]),
                        stars=float(row["stars"]),
                        business_stars=float(row["business_stars"]),
                        date=row["date"],
                        year_month=row["year_month"],
                        credibility=float(row["credibility"]),
                        text_masked=row["text_masked"],
                        aspects=row["aspects"] or "[]",
                        n_tokens=int(row["n_tokens"])
                    ))
        except sqlite3.OperationalError as e:
            logger.warning(f"FTS5 query operational error: {e}. Query: {match_expr}")
            return []

        return candidates
