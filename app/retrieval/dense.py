"""Dense Semantic Retrieval Module using NumPy Memmap and BAAI/bge-small-en-v1.5.
Searches stored 384-d float16 document embeddings with cosine similarity,
candidate row index pre-filtering, and metadata enrichment from SQLite.
"""

import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np
from sentence_transformers import SentenceTransformer

from app.core.config import settings
from app.core.logging import logger
from data_preprocessing.storage.vector_store import NumpyMemmapDenseStore

BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


@dataclass
class DenseCandidate:
    chunk_id: str
    dense_similarity: float
    dense_rank: int
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


class DenseRetriever:
    """Retrieves review chunks via dense vector cosine similarity."""

    _model_instance: Optional[SentenceTransformer] = None

    def __init__(
        self,
        vectors_path: Optional[Path] = None,
        ids_path: Optional[Path] = None,
        db_path: Optional[Path] = None,
        model_name: Optional[str] = None
    ):
        self.vectors_path = Path(vectors_path or (settings.VECTORS_DIR / "vectors.f16.npy"))
        self.ids_path = Path(ids_path or (settings.VECTORS_DIR / "vector_ids.parquet"))
        self.db_path = Path(db_path or (settings.SQLITE_DIR / "restaurant_ops.db"))
        self.model_name = model_name or settings.EMBEDDING_MODEL_NAME

        if not self.vectors_path.exists():
            raise FileNotFoundError(f"Vector store file not found at {self.vectors_path}")
        if not self.ids_path.exists():
            raise FileNotFoundError(f"Vector IDs mapping not found at {self.ids_path}")
        if not self.db_path.exists():
            raise FileNotFoundError(f"SQLite database not found at {self.db_path}")

        # Initialize memmap store
        self.dense_store = NumpyMemmapDenseStore(
            vectors_path=self.vectors_path,
            ids_path=self.ids_path,
            dim=settings.EMBEDDING_DIM
        )

    @classmethod
    def _get_model(cls, model_name: str) -> SentenceTransformer:
        if cls._model_instance is None:
            logger.info(f"Loading embedding model '{model_name}' on CPU...")
            cls._model_instance = SentenceTransformer(model_name, device="cpu")
        return cls._model_instance

    def _get_candidate_row_indexes(self, filters: Dict[str, Any]) -> Optional[List[int]]:
        """Queries SQLite chunks_meta to retrieve candidate row_idx values for pre-filtering."""
        where_clauses = []
        params: List[Any] = []

        if "city" in filters and filters["city"]:
            where_clauses.append("city = ?")
            params.append(filters["city"])
        if "state" in filters and filters["state"]:
            where_clauses.append("state = ?")
            params.append(filters["state"])
        if "business_id" in filters and filters["business_id"]:
            where_clauses.append("business_id = ?")
            params.append(filters["business_id"])
        if "is_open" in filters and filters["is_open"] is not None:
            where_clauses.append("is_open = ?")
            params.append(int(filters["is_open"]))
        if "min_stars" in filters and filters["min_stars"] is not None:
            where_clauses.append("stars >= ?")
            params.append(float(filters["min_stars"]))
        if "max_stars" in filters and filters["max_stars"] is not None:
            where_clauses.append("stars <= ?")
            params.append(float(filters["max_stars"]))
        if "aspect" in filters and filters["aspect"]:
            where_clauses.append("aspects LIKE ?")
            params.append(f"%{filters['aspect']}%")

        if not where_clauses:
            return None

        sql = f"SELECT row_idx FROM chunks_meta WHERE {' AND '.join(where_clauses)}"
        with sqlite3.connect(str(self.db_path)) as con:
            cur = con.cursor()
            cur.execute(sql, params)
            return [r[0] for r in cur.fetchall()]

    def search(
        self,
        query: str,
        top_k: int = 100,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[DenseCandidate]:
        """Encodes query, executes cosine similarity search, and enriches candidates with metadata."""
        if not query or not query.strip():
            return []

        # Candidate row index pre-filtering
        candidate_row_indexes = self._get_candidate_row_indexes(filters) if filters else None
        if candidate_row_indexes is not None and len(candidate_row_indexes) == 0:
            return []

        # Encode query
        model = self._get_model(self.model_name)
        prefixed_query = BGE_QUERY_PREFIX + query.strip()
        query_emb = model.encode(prefixed_query, normalize_embeddings=True).astype(np.float32)

        # Vector search
        raw_matches = self.dense_store.search(
            query_vector=query_emb,
            candidate_row_indexes=candidate_row_indexes,
            top_k=top_k
        )
        if not raw_matches:
            return []

        # Fetch metadata in a single query
        chunk_ids = [m[0] for m in raw_matches]
        scores_map = {m[0]: m[1] for m in raw_matches}
        placeholders = ",".join("?" for _ in chunk_ids)

        sql = f"""
            SELECT 
                chunk_id, review_id, business_id, restaurant_name, city, state,
                categories, price_range, is_open, stars, business_stars, date,
                year_month, credibility, text_masked, aspects, n_tokens
            FROM chunks_meta
            WHERE chunk_id IN ({placeholders});
        """

        meta_rows_map = {}
        with sqlite3.connect(str(self.db_path)) as con:
            con.row_factory = sqlite3.Row
            cur = con.cursor()
            cur.execute(sql, chunk_ids)
            for r in cur.fetchall():
                meta_rows_map[r["chunk_id"]] = r

        candidates: List[DenseCandidate] = []
        for rank, cid in enumerate(chunk_ids, start=1):
            if cid not in meta_rows_map:
                continue
            row = meta_rows_map[cid]
            candidates.append(DenseCandidate(
                chunk_id=cid,
                dense_similarity=float(scores_map[cid]),
                dense_rank=rank,
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

        return candidates
