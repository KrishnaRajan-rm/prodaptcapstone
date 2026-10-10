"""Unit tests for Phase 4: Storage, SQLite FTS5, and Dense Vector Store."""

import sqlite3
import numpy as np
import pytest
from pathlib import Path
from data_preprocessing.storage.sqlite_builder import strip_pii_placeholders
from data_preprocessing.storage.vector_store import NumpyMemmapDenseStore


def test_strip_pii_placeholders():
    raw_text = "Hello [PERSON], call me at [PHONE] or email [EMAIL] with card [CARD]."
    cleaned = strip_pii_placeholders(raw_text)
    assert "[PERSON]" not in cleaned
    assert "[PHONE]" not in cleaned
    assert "[EMAIL]" not in cleaned
    assert "[CARD]" not in cleaned
    assert "Hello" in cleaned


def test_numpy_memmap_dense_store(tmp_path):
    vec_path = tmp_path / "vectors.f16.npy"
    ids_path = tmp_path / "vector_ids.parquet"
    
    # Create 4 synthetic 4-dimensional normalized vectors
    dim = 4
    vecs = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.7071, 0.7071, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0]
    ], dtype=np.float32)

    # Save memmap
    mmap = np.memmap(str(vec_path), dtype=np.float16, mode="w+", shape=(4, dim))
    mmap[:] = vecs.astype(np.float16)
    mmap.flush()

    import pandas as pd
    ids_df = pd.DataFrame({
        "row_idx": [0, 1, 2, 3],
        "chunk_id": ["c0", "c1", "c2", "c3"]
    })
    ids_df.to_parquet(ids_path, index=False)

    store = NumpyMemmapDenseStore(vec_path, ids_path, dim=dim)
    
    # Query aligned with vector 0
    query = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    hits = store.search(query, top_k=2)
    assert len(hits) == 2
    assert hits[0][0] == "c0"
    assert hits[0][1] > 0.99
    assert hits[1][0] == "c2"

    # Query with candidate filtering
    filtered_hits = store.search(query, candidate_row_indexes=[1, 3], top_k=2)
    assert len(filtered_hits) == 2
    assert "c0" not in [h[0] for h in filtered_hits]
