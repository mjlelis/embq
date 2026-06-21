"""Reproducible demo for embq.

Generates a synthetic, clustered embedding set (a rough stand-in for real
embeddings, which live near a low-dimensional manifold) and prints the
quantization trade-off table. The numbers shown in the README come straight
from this script with the seed below, so anyone can reproduce them:

    python examples/leaderboard/run.py
"""

import numpy as np

from embq import profile
from embq.report import print_report


def make_dataset(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    dim, n_clusters, per_cluster = 256, 50, 40
    centers = rng.standard_normal((n_clusters, dim)).astype(np.float32)
    noise = 0.15 * rng.standard_normal((n_clusters * per_cluster, dim)).astype(np.float32)
    return (np.repeat(centers, per_cluster, axis=0) + noise).astype(np.float32)


def main() -> None:
    emb = make_dataset()
    print(f"Dataset: n={emb.shape[0]}, dim={emb.shape[1]} (50 clusters x 40 points)")
    report = profile(emb, k=10, oversample=20)
    print_report(report, k=10)


if __name__ == "__main__":
    main()
