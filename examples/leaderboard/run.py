"""Reproducible leaderboard demo for embq.

This script measures the quantization trade-off (recall vs. compression) on
**real** sentence embeddings, at a scale where the binary+rescore shortlist is
a small fraction of the database — which is exactly what keeps the numbers
honest. A tiny synthetic database can make binary+rescore look perfect
(recall = 1.0) simply because the shortlist covers most of the corpus; at real
scale that artifact disappears.

By default it:

  1. Loads a public text corpus via `datasets` (AG News, ~120k news snippets).
  2. Embeds it with a sentence-transformers model (BAAI/bge-small-en-v1.5,
     384-dim, L2-normalized).
  3. Holds out a disjoint set of texts (AG News *test* split) as queries.
  4. Computes exact fp32 ground truth and reports recall@k for int8, binary,
     and binary+rescore across a sweep of oversample factors.

Run it:

    pip install -r examples/leaderboard/requirements.txt   # extra deps
    python examples/leaderboard/run.py                      # real data (default)
    python examples/leaderboard/run.py --synthetic          # offline / CI fallback

NOTE ON METHODOLOGY: queries are held-out embeddings, not human relevance
judgements. We therefore measure how faithfully each *quantization* reproduces
the fp32 nearest neighbours — not end-to-end retrieval relevance.
"""

from __future__ import annotations

import argparse
import sys
import time

import numpy as np
from tabulate import tabulate

from embq import profile

# Print Unicode (em dashes, arrows) reliably even on Windows code pages.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"
DEFAULT_DATASET = "fancyzhx/ag_news"  # public, stable; disjoint train/test splits
DEFAULT_N = 50_000
DEFAULT_QUERIES = 1_000
DEFAULT_K = 10
OVERSAMPLE_SWEEP = (4, 20, 100)
SEED = 0


def die(msg: str) -> "None":
    """Fail loudly with a clear message — never fabricate a silent fallback."""
    print(f"\nERROR: {msg}", file=sys.stderr)
    sys.exit(1)


# --------------------------------------------------------------------------- #
# Corpus loading
# --------------------------------------------------------------------------- #
def load_corpus_dataset(n: int, n_queries: int) -> "tuple[list[str], list[str], str]":
    try:
        from datasets import load_dataset
    except ImportError:
        die(
            "`datasets` is not installed. Install the demo extras:\n"
            "    pip install -r examples/leaderboard/requirements.txt"
        )
    try:
        train = load_dataset(DEFAULT_DATASET, split="train")
        test = load_dataset(DEFAULT_DATASET, split="test")
    except Exception as exc:  # noqa: BLE001 - surface the real cause to the user
        die(
            f"Could not download dataset '{DEFAULT_DATASET}': {exc}\n"
            "Check your network/HF access, or run offline with --synthetic."
        )

    if n > len(train):
        die(f"Requested n={n} but '{DEFAULT_DATASET}' train split has only {len(train)} rows.")

    docs = list(train["text"][:n])
    # Queries come from the *test* split: guaranteed disjoint from the index.
    n_queries = min(n_queries, len(test))
    queries = list(test["text"][:n_queries])
    source = f"{DEFAULT_DATASET} (train[:{n}] indexed, test[:{n_queries}] as queries)"
    return docs, queries, source


def load_corpus_file(path: str, n: int, n_queries: int) -> "tuple[list[str], list[str], str]":
    try:
        with open(path, encoding="utf-8") as fh:
            lines = [ln.strip() for ln in fh if ln.strip()]
    except OSError as exc:
        die(f"Could not read corpus file '{path}': {exc}")

    if len(lines) < n_queries + 1:
        die(f"Corpus '{path}' has only {len(lines)} non-empty lines; need > {n_queries}.")

    n = min(n, len(lines) - n_queries)
    # Last n_queries lines are held out as queries (disjoint from the index).
    docs = lines[:n]
    queries = lines[-n_queries:]
    source = f"{path} ({n} indexed, {n_queries} held out as queries)"
    return docs, queries, source


def load_queries_file(path: str) -> "list[str]":
    try:
        with open(path, encoding="utf-8") as fh:
            queries = [ln.strip() for ln in fh if ln.strip()]
    except OSError as exc:
        die(f"Could not read queries file '{path}': {exc}")
    if not queries:
        die(f"Queries file '{path}' is empty.")
    return queries


# --------------------------------------------------------------------------- #
# Embedding
# --------------------------------------------------------------------------- #
def embed(texts: "list[str]", model_name: str) -> np.ndarray:
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        die(
            "`sentence-transformers` is not installed. Install the demo extras:\n"
            "    pip install -r examples/leaderboard/requirements.txt"
        )
    try:
        model = SentenceTransformer(model_name)
    except Exception as exc:  # noqa: BLE001
        die(f"Could not load model '{model_name}': {exc}\nCheck network/HF access.")

    vecs = model.encode(
        texts,
        normalize_embeddings=True,
        batch_size=256,
        convert_to_numpy=True,
        show_progress_bar=True,
    )
    return np.ascontiguousarray(vecs, dtype=np.float32)


# --------------------------------------------------------------------------- #
# Synthetic fallback (offline / CI only)
# --------------------------------------------------------------------------- #
def make_synthetic(n: int, n_queries: int, dim: int = 384) -> "tuple[np.ndarray, np.ndarray, str]":
    rng = np.random.default_rng(SEED)
    n_clusters = max(2, n // 200)
    centers = rng.standard_normal((n_clusters, dim)).astype(np.float32)

    def sample(count: int) -> np.ndarray:
        idx = rng.integers(0, n_clusters, size=count)
        noise = 0.35 * rng.standard_normal((count, dim)).astype(np.float32)
        v = centers[idx] + noise
        v /= np.linalg.norm(v, axis=1, keepdims=True)
        return v.astype(np.float32)

    db = sample(n)
    queries = sample(n_queries)
    source = f"synthetic ({n_clusters} clusters, dim={dim}) — OFFLINE FALLBACK, not representative"
    return db, queries, source


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #
def run_method(db: np.ndarray, queries: np.ndarray, method: str, k: int, oversample: int):
    """Run a single method via the embq kernel and return its MethodResult."""
    report = profile(db, queries=queries, k=k, methods=[method], oversample=oversample)
    return report.results[0]


def main() -> None:
    parser = argparse.ArgumentParser(description="embq quantization leaderboard (real embeddings)")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="sentence-transformers model")
    parser.add_argument("--n", type=int, default=DEFAULT_N, help="number of documents to index")
    parser.add_argument("--queries", help="optional path to a queries file (one per line)")
    parser.add_argument("--n-queries", type=int, default=DEFAULT_QUERIES, help="held-out query count")
    parser.add_argument("--corpus", help="path to a custom corpus (one document per line)")
    parser.add_argument("--k", type=int, default=DEFAULT_K, help="top-k for recall")
    parser.add_argument("--synthetic", action="store_true", help="offline fallback, do not download")
    args = parser.parse_args()

    np.random.seed(SEED)

    # ---- assemble texts --------------------------------------------------- #
    if args.synthetic:
        db, q, source = make_synthetic(args.n, args.n_queries)
        model_label = "—  (synthetic)"
    else:
        if args.corpus:
            docs, queries, source = load_corpus_file(args.corpus, args.n, args.n_queries)
        else:
            docs, queries, source = load_corpus_dataset(args.n, args.n_queries)
        if args.queries:
            queries = load_queries_file(args.queries)
            source += f" | queries overridden by {args.queries}"

        print(f"Embedding {len(docs)} docs + {len(queries)} queries with {args.model} ...")
        t0 = time.perf_counter()
        db = embed(docs, args.model)
        q = embed(queries, args.model)
        print(f"  embedding took {time.perf_counter() - t0:.1f}s")
        model_label = args.model

    dim = db.shape[1]

    # ---- reproducible header --------------------------------------------- #
    print("\n" + "=" * 72)
    print("embq quantization leaderboard")
    print("=" * 72)
    print(f"model       : {model_label}")
    print(f"corpus      : {source}")
    print(f"N (indexed) : {db.shape[0]}")
    print(f"queries     : {q.shape[0]} (held out)")
    print(f"dims        : {dim}")
    print(f"k           : {args.k}")
    print(f"seed        : {SEED}")
    print("=" * 72)

    fp32_bytes = dim * 4
    rows = []
    latencies = []

    # int8 and binary: oversample is not applicable.
    for method in ("int8", "binary"):
        r = run_method(db, q, method, args.k, oversample=1)
        latencies.append((r.method, r.latency_ms))
        note = "" if method == "int8" else "lossy; recover with rescore"
        rows.append([
            method,
            f"{r.recall_at_k:.4f}",
            f"{r.bytes_per_vec} B",
            f"{r.compression_ratio:.1f}x",
            "—",
            note,
        ])

    # binary+rescore: sweep oversample so the recovery curve is explicit.
    for ov in OVERSAMPLE_SWEEP:
        r = run_method(db, q, "binary_rescore", args.k, oversample=ov)
        latencies.append((f"binary+rescore (os={ov})", r.latency_ms))
        rows.append([
            "binary+rescore",
            f"{r.recall_at_k:.4f}",
            f"{r.bytes_per_vec} B",
            f"{r.compression_ratio:.1f}x",
            str(ov),
            "requires fp32 retained for re-rank → RAM-only win",
        ])

    headers = ["Method", f"Recall@{args.k}", "Bytes/vec (RAM)", "Compression", "Oversample", "Notes"]
    print("\n" + tabulate(rows, headers=headers, tablefmt="github", disable_numparse=True))

    print(f"\nfp32 baseline: {fp32_bytes} B/vec, recall 1.0000 by construction (exact oracle).")
    print("Latency/query (machine-dependent, indicative only):")
    for name, ms in latencies:
        print(f"  {name:28s} {ms:.4f} ms")


if __name__ == "__main__":
    main()
