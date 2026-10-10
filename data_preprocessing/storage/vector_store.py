"""Phase 4: Dense Vector Store (NumPy Memmap float16 + BAAI/bge-small-en-v1.5).
Implements the DenseStore interface:
- Stores normalized 384-d float16 embeddings in vectors.f16.npy
- Maps row_idx to chunk_id in vector_ids.parquet
- Supports checkpointing and resumption
- Candidate row index filtered matrix-vector search
"""

import json
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Tuple, Optional
import numpy as np
import duckdb
import pandas as pd
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from app.core.config import settings
from app.core.logging import logger


class DenseStore(ABC):
    """Abstract interface for dense vector storage."""

    @abstractmethod
    def search(self, query_vector: np.ndarray, candidate_row_indexes: Optional[List[int]] = None, top_k: int = 10) -> List[Tuple[str, float]]:
        pass


class NumpyMemmapDenseStore(DenseStore):
    """NumPy memmap dense store with normalized float16 vectors on disk."""

    def __init__(self, vectors_path: Path, ids_path: Path, dim: int = 384):
        self.vectors_path = Path(vectors_path)
        self.ids_path = Path(ids_path)
        self.dim = dim
        self.memmap: Optional[np.memmap] = None
        self.id_to_idx: dict = {}
        self.idx_to_id: List[str] = []
        self._load()

    def _load(self):
        if self.vectors_path.exists() and self.ids_path.exists():
            con = duckdb.connect()
            df = con.execute(f"SELECT row_idx, chunk_id FROM read_parquet('{str(self.ids_path).replace(chr(92), '/')}') ORDER BY row_idx").df()
            n_rows = len(df)
            self.idx_to_id = df["chunk_id"].tolist()
            self.id_to_idx = {cid: idx for idx, cid in enumerate(self.idx_to_id)}
            self.memmap = np.memmap(
                str(self.vectors_path),
                dtype=np.float16,
                mode="r",
                shape=(n_rows, self.dim)
            )
            logger.info(f"Loaded NumPy Memmap vector store with {n_rows} vectors (dim={self.dim}).")

    def search(self, query_vector: np.ndarray, candidate_row_indexes: Optional[List[int]] = None, top_k: int = 10) -> List[Tuple[str, float]]:
        """Calculates cosine similarity dot product over candidate rows and returns top-k."""
        if self.memmap is None or len(self.idx_to_id) == 0:
            return []

        q_vec = query_vector.astype(np.float32)
        norm = np.linalg.norm(q_vec)
        if norm > 0:
            q_vec = q_vec / norm

        if candidate_row_indexes is not None and len(candidate_row_indexes) > 0:
            valid_indices = [idx for idx in candidate_row_indexes if 0 <= idx < len(self.idx_to_id)]
            if not valid_indices:
                return []
            candidate_vecs = np.asarray(self.memmap[valid_indices], dtype=np.float32)
            scores = np.dot(candidate_vecs, q_vec)
            top_local_idx = np.argsort(scores)[::-1][:top_k]
            return [(self.idx_to_id[valid_indices[i]], float(scores[i])) for i in top_local_idx]
        else:
            all_vecs = np.asarray(self.memmap, dtype=np.float32)
            scores = np.dot(all_vecs, q_vec)
            top_idx = np.argsort(scores)[::-1][:top_k]
            return [(self.idx_to_id[i], float(scores[i])) for i in top_idx]


def build_embeddings(
    chunks_parquet_path: Path,
    vectors_dir: Path,
    batch_size: int = 64,
    checkpoint_every: int = 50,
    resume: bool = True
):
    """Encodes chunk texts in batches using BAAI/bge-small-en-v1.5 and writes float16 memmap."""
    logger.info("Initializing vector embeddings pipeline...")
    vectors_dir.mkdir(parents=True, exist_ok=True)
    vectors_path = vectors_dir / "vectors.f16.npy"
    ids_path = vectors_dir / "vector_ids.parquet"
    progress_file = settings.PROGRESS_DIR / "vectors_progress.json"

    con = duckdb.connect()
    chunks_sql = f"read_parquet('{str(chunks_parquet_path).replace(chr(92), '/')}')"
    chunks_df = con.execute(f"SELECT chunk_id, text_masked FROM {chunks_sql} ORDER BY chunk_id").df()
    total_chunks = len(chunks_df)
    dim = settings.EMBEDDING_DIM

    logger.info(f"Loaded {total_chunks} chunks to embed (dim={dim}).")

    start_idx = 0
    if resume and progress_file.exists() and vectors_path.exists():
        try:
            with open(progress_file, "r", encoding="utf-8") as f:
                p_data = json.load(f)
                start_idx = p_data.get("completed_chunks", 0)
                logger.info(f"Resuming embeddings from chunk index {start_idx}/{total_chunks}...")
        except Exception as e:
            logger.warning(f"Could not read vectors progress: {e}")

    # Initialize memmap
    mode = "r+" if vectors_path.exists() and start_idx > 0 else "w+"
    mmap = np.memmap(str(vectors_path), dtype=np.float16, mode=mode, shape=(total_chunks, dim))

    # Save vector IDs mapping
    if not ids_path.exists() or start_idx == 0:
        ids_df = pd.DataFrame({
            "row_idx": np.arange(total_chunks),
            "chunk_id": chunks_df["chunk_id"].values
        })
        ids_df.to_parquet(ids_path, index=False, compression="zstd")

    if start_idx >= total_chunks:
        logger.info("All embeddings already generated.")
        return NumpyMemmapDenseStore(vectors_path, ids_path, dim)

    logger.info(f"Loading embedding model: {settings.EMBEDDING_MODEL_NAME} on CPU...")
    model = SentenceTransformer(settings.EMBEDDING_MODEL_NAME)

    texts = chunks_df["text_masked"].tolist()
    t_start = time.time()
    n_batches = (total_chunks - start_idx + batch_size - 1) // batch_size

    for b in range(n_batches):
        cur_start = start_idx + b * batch_size
        cur_end = min(total_chunks, cur_start + batch_size)
        batch_texts = texts[cur_start:cur_end]

        embs = model.encode(
            batch_texts,
            batch_size=len(batch_texts),
            normalize_embeddings=True,
            show_progress_bar=False
        )

        mmap[cur_start:cur_end] = embs.astype(np.float16)

        # Checkpoint
        if (b + 1) % checkpoint_every == 0 or cur_end == total_chunks:
            mmap.flush()
            elapsed = time.time() - t_start
            processed = cur_end - start_idx
            speed = processed / max(elapsed, 0.001)
            eta_sec = (total_chunks - cur_end) / max(speed, 0.001)
            logger.info(f"Embedded {cur_end}/{total_chunks} chunks ({speed:.1f} chunks/sec, ETA: {eta_sec:.1f}s)")
            with open(progress_file, "w", encoding="utf-8") as f:
                json.dump({"completed_chunks": cur_end, "total_chunks": total_chunks}, f)

    mmap.flush()
    logger.info(f"Vector embeddings complete: {total_chunks} vectors written to {vectors_path}")
    return NumpyMemmapDenseStore(vectors_path, ids_path, dim)
