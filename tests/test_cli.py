"""Tests for the `embq` command-line interface."""

import subprocess
import sys

import numpy as np


def _make_npy(tmp_path):
    path = tmp_path / "data.npy"
    data = np.random.default_rng(7).standard_normal((500, 64)).astype(np.float32)
    np.save(path, data)
    return path


def test_cli_run_prints_report(tmp_path):
    path = _make_npy(tmp_path)

    proc = subprocess.run(
        [sys.executable, "-m", "embq.cli", "run", "--embeddings", str(path), "--k", "10"],
        capture_output=True,
        text=True,
    )

    assert proc.returncode == 0, proc.stderr
    assert "embq Quantization Profile" in proc.stdout
    assert "int8" in proc.stdout
    assert "binary_rescore" in proc.stdout


def test_cli_missing_file_exits_nonzero(tmp_path):
    proc = subprocess.run(
        [sys.executable, "-m", "embq.cli", "run", "--embeddings", str(tmp_path / "missing.npy")],
        capture_output=True,
        text=True,
    )

    assert proc.returncode == 1
    assert "Error:" in proc.stdout
