from tabulate import tabulate

def print_report(report, k):
    data = []
    for res in report.results:
        data.append([
            res.method,
            f"{res.recall_at_k:.4f}",
            f"{res.latency_ms:.4f}ms",
            f"{res.bytes_per_vec} B",
            f"{res.compression_ratio:.1f}x"
        ])
    
    headers = ["Method", f"Recall@{k}", "Latency/query", "Bytes/vec", "Compression"]
    print("\n### embq Quantization Profile ###")
    print(tabulate(data, headers=headers, tablefmt="grid"))
    
    # Simple recommendation
    best_int8 = next((r for r in report.results if r.method == "int8"), None)
    if best_int8 and best_int8.recall_at_k > 0.95:
        print(f"\nRecommendation: int8 looks great! {best_int8.compression_ratio:.1f}x compression with >95% recall.")
    
    best_rescore = next((r for r in report.results if r.method == "binary_rescore"), None)
    if best_rescore:
        print(f"Binary+Rescore: {best_rescore.compression_ratio:.1f}x smaller, {best_rescore.recall_at_k:.4f} recall.")
