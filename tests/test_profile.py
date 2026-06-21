"""End-to-end tests for the `embq.profile` entry point (PyO3 kernel)."""

import numpy as np
import pytest

from embq import MethodResult, Report, profile


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(42)


def _result_by_method(report: Report) -> dict[str, MethodResult]:
    return {r.method: r for r in report.results}


def test_profile_returns_all_default_methods(rng):
    emb = rng.standard_normal((1000, 128)).astype(np.float32)

    report = profile(emb, k=10)
    methods = {r.method for r in report.results}

    assert methods == {"int8", "binary", "binary_rescore"}


def test_int8_recall_is_high(rng):
    # int8 is a near-lossless quantization, so recall@10 should stay high.
    emb = rng.standard_normal((2000, 128)).astype(np.float32)

    report = _result_by_method(profile(emb, k=10))

    assert report["int8"].recall_at_k > 0.9


def test_rescore_recovers_recall_lost_by_binary(rng):
    # Binary quantization is lossy; rescoring with fp32 must never be worse.
    emb = rng.standard_normal((2000, 128)).astype(np.float32)

    report = _result_by_method(profile(emb, k=10, oversample=20))

    assert report["binary_rescore"].recall_at_k >= report["binary"].recall_at_k


def test_compression_ratios(rng):
    # 128 fp32 dims = 512 B. int8 = 128 B (4x). binary = 16 B (32x).
    emb = rng.standard_normal((500, 128)).astype(np.float32)

    report = _result_by_method(profile(emb, k=10))

    assert report["int8"].bytes_per_vec == 128
    assert report["int8"].compression_ratio == pytest.approx(4.0)
    assert report["binary"].bytes_per_vec == 16
    assert report["binary"].compression_ratio == pytest.approx(32.0)


def test_perfect_recall_when_queries_are_database_rows(rng):
    # Querying with exact database rows: the nearest neighbour of each query
    # is itself, so even lossy int8 should recover it with high recall.
    emb = rng.standard_normal((1000, 64)).astype(np.float32)
    queries = emb[:50].copy()

    report = _result_by_method(profile(emb, queries=queries, k=1))

    assert report["int8"].recall_at_k > 0.95
    assert report["binary_rescore"].recall_at_k > 0.95


def test_explicit_method_subset(rng):
    emb = rng.standard_normal((400, 32)).astype(np.float32)

    report = profile(emb, k=5, methods=["int8"])

    assert [r.method for r in report.results] == ["int8"]


def test_recall_within_valid_range(rng):
    emb = rng.standard_normal((800, 96)).astype(np.float32)

    for r in profile(emb, k=10).results:
        assert 0.0 <= r.recall_at_k <= 1.0
        assert r.latency_ms >= 0.0


def test_oversample_increases_or_holds_rescore_recall(rng):
    emb = rng.standard_normal((2000, 128)).astype(np.float32)

    low = _result_by_method(profile(emb, k=10, oversample=2))["binary_rescore"]
    high = _result_by_method(profile(emb, k=10, oversample=50))["binary_rescore"]

    # More candidates can only help (or tie) the rescore stage.
    assert high.recall_at_k >= low.recall_at_k - 1e-6
