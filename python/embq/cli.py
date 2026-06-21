import argparse
import sys
from .io import load_embeddings
from .report import print_report
from . import profile

def main():
    parser = argparse.ArgumentParser(prog="embq", description="Quantization Tolerance Profiler")
    subparsers = parser.add_subparsers(dest="command")
    
    run_parser = subparsers.add_parser("run", help="Run profile")
    run_parser.add_argument("--embeddings", required=True, help="Path to .npy or .parquet embeddings")
    run_parser.add_argument("--queries", help="Path to .npy or .parquet queries (optional)")
    run_parser.add_argument("--k", type=int, default=10, help="Top-k for recall calculation")
    run_parser.add_argument("--sample-queries", type=int, default=100, help="Number of queries to sample if not provided")
    run_parser.add_argument("--oversample", type=int, default=20, help="Oversampling for binary_rescore (k * oversample)")
    
    args = parser.parse_args()
    
    if args.command == "run":
        try:
            print(f"Loading embeddings from {args.embeddings}...")
            emb = load_embeddings(args.embeddings)
            
            queries = None
            if args.queries:
                print(f"Loading queries from {args.queries}...")
                queries = load_embeddings(args.queries)
            
            print(f"Profiling (k={args.k}, n={len(emb)}, dim={emb.shape[1]})...")
            report = profile(emb, queries=queries, k=args.k, oversample=args.oversample)
            
            print_report(report, args.k)
            
        except Exception as e:
            print(f"Error: {e}")
            sys.exit(1)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
