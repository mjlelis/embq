"""Tests for the dataset loaders in `embq.io`."""

import numpy as np
import pandas as pd
import pytest

from embq.io import load_embeddings


def test_load_npy(tmp_path):
    path = tmp_path / "vecs.npy"
    data = np.random.default_rng(0).standard_normal((10, 8)).astype(np.float32)
    np.save(path, data)

    loaded = load_embeddings(str(path))

    assert loaded.shape == (10, 8)
    assert loaded.dtype == np.float32
    np.testing.assert_allclose(loaded, data)


def test_load_npy_casts_to_float32(tmp_path):
    path = tmp_path / "vecs64.npy"
    np.save(path, np.ones((4, 4), dtype=np.float64))

    loaded = load_embeddings(str(path))

    assert loaded.dtype == np.float32


def test_load_parquet_embedding_column(tmp_path):
    path = tmp_path / "vecs.parquet"
    vectors = [list(np.arange(4, dtype=np.float32)) for _ in range(5)]
    pd.DataFrame({"embedding": vectors}).to_parquet(path)

    loaded = load_embeddings(str(path))

    assert loaded.shape == (5, 4)
    assert loaded.dtype == np.float32


def test_unsupported_extension_raises(tmp_path):
    path = tmp_path / "vecs.csv"
    path.write_text("not embeddings")

    with pytest.raises(ValueError, match="Unsupported file format"):
        load_embeddings(str(path))
