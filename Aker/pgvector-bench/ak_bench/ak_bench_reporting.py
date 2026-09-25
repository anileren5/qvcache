"""Reporting helpers.

Output formats are kept compatible with the legacy scripts:
- report.csv (TSV)
- search-results.pkl
- trace-extract-info.csv (TSV)
"""

from __future__ import annotations

import pickle
from typing import List


def writeReport(
    report_path: str,
    workload_name: str,
    workload_type: str,
    search_params: str,
    qps: float,
    avg_search_latency: float,
    p50_search_latency: float,
    p99_search_latency: float,
    avg_delete_latency: float,
    p50_delete_latency: float,
    p99_delete_latency: float,
    avg_insert_latency: float,
    p50_insert_latency: float,
    p99_insert_latency: float,
    avg_recall: float,
) -> None:
    """Write legacy-compatible report.csv."""

    with open(report_path, "w") as f:
        f.write("Name\tWorkload Type\tSearch Params\tQPS\t")
        f.write("Avg Search Latency (s)\t50%ile Search Latency (s)\t99%ile Search Latency (s)\t")
        f.write("Avg Delete Latency (s)\t50%ile Delete Latency (s)\t99%ile Delete Latency (s)\t")
        f.write("Avg Insert Latency (s)\t50%ile Insert Latency (s)\t99%ile Insert Latency (s)\t")
        f.write("Avg Recall\n")

        f.write(f"{workload_name}\t{workload_type}\t{search_params}\t{qps:.2f}\t")
        f.write(f"{avg_search_latency:.4f}\t{p50_search_latency:.4f}\t{p99_search_latency:.4f}\t")
        f.write(f"{avg_delete_latency:.4f}\t{p50_delete_latency:.4f}\t{p99_delete_latency:.4f}\t")
        f.write(f"{avg_insert_latency:.4f}\t{p50_insert_latency:.4f}\t{p99_insert_latency:.4f}\t")
        f.write(f"{avg_recall:.4f}\n")


def writeSearchResultsPkl(path: str, results: List[dict]) -> None:
    """Write legacy-compatible search-results.pkl."""

    with open(path, "wb") as f:
        pickle.dump(results, f)


def writeTraceExtractInfo(path: str, results: List[dict], recalls: List[float]) -> None:
    """Write legacy-compatible trace-extract-info.csv."""

    recalls_copy = list(recalls)

    with open(path, "w") as f:
        for i, result in enumerate(results):
            operation_type = result.get("operation", "")
            operation_letter = operation_type[0].upper() if operation_type else "U"

            latency = float(result.get("latency", 0.0))
            qps_moment = float(result.get("qps_moment", 0.0))

            if operation_letter == "S" and recalls_copy:
                rec = recalls_copy.pop(0)
                f.write(f"{i}\t{operation_letter}\t{rec:.4f}\t{latency:.6f}\t{qps_moment:.6f}\n")
            else:
                f.write(f"{i}\t{operation_letter}\t0.0000\t{latency:.6f}\t{qps_moment:.6f}\n")
