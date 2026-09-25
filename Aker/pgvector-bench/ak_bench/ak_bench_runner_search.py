"""Search-workload runner.

This runner is compatible with the legacy Workload A trace format:
- trace: list[dict]
- each dict has keys: operation, payload, gt_ids, gt_scores

It writes legacy-compatible outputs:
- report.csv
- search-results.pkl
- trace-extract-info.csv
"""

from __future__ import annotations

import logging
import os
import time
from typing import List, Tuple

import numpy as np

from ak_bench.ak_bench_config import BenchConfig, DISTANCE_OPERATOR, TABLE_NAME
from ak_bench.ak_bench_dscheck import checkIndexExists
from ak_bench.ak_bench_pg import PostgresClient
from ak_bench.ak_bench_reporting import writeReport, writeSearchResultsPkl, writeTraceExtractInfo
from ak_bench.ak_bench_trace_schema import ensureSearchGtPresent, loadTrace, validateSearchTrace
from ak_bench.ak_bench_util import ensureDir


class SearchWorkloadRunner:
    """Execute a Search-workload trace against PostgreSQL."""

    def __init__(self, config: BenchConfig):
        self._config = config

    def run(self, output_dir: str) -> None:
        """Run the benchmark and write outputs into output_dir."""

        ensureDir(output_dir)

        conn = PostgresClient(self._config).connect()
        cursor = conn.cursor()

        ready = checkIndexExists(conn, self._config)
        if not ready:
            logging.error("Index check failed. The run may be invalid.")

        trace_path = self._config.dataset.gt_trace_path
        if not os.path.exists(trace_path):
            raise FileNotFoundError(f"Trace file {trace_path} does not exist")

        trace = loadTrace(trace_path)
        validateSearchTrace(trace)
        ensureSearchGtPresent(trace)

        limit = self._config.workload.limit

        #
        # Set index search parameters (kept compatible with legacy scripts).
        #
        index_type = self._config.pgvector.index_type
        if index_type == "hnsw" and self._config.pgvector.ef_search is not None:
            cursor.execute(f"SET hnsw.ef_search = {self._config.pgvector.ef_search};")
        elif index_type == "ivfflat" and self._config.pgvector.nprobe is not None:
            cursor.execute(f"SET ivfflat.probes = {self._config.pgvector.nprobe};")

        search_sql = (
            f"SELECT id, embedding {DISTANCE_OPERATOR} %s AS _score "
            f"FROM {TABLE_NAME} ORDER BY _score LIMIT %s;"
        )

        results: List[dict] = []
        recalls: List[float] = []

        workload_start = time.perf_counter()

        show_progress_percentage = 1
        for i, request in enumerate(trace):
            operation = request["operation"]
            if operation != "search":
                raise ValueError(f"Search-workload trace contains non-search operation: {operation}")

            search_vector = request["payload"]
            if not isinstance(search_vector, np.ndarray):
                raise TypeError(f"Expected search_vector to be np.ndarray, got {type(search_vector)}")

            request_start = time.perf_counter()
            cursor.execute(search_sql, (search_vector, limit), binary=True, prepare=True)
            rows = cursor.fetchall()
            request_end = time.perf_counter()

            result_ids = [row[0] for row in rows]
            result_scores = [row[1] for row in rows]

            # Recall@K: intersection of GT and result ids.
            gt_ids = set(request["gt_ids"][:limit])
            res_ids = set(result_ids)
            recall = len(gt_ids.intersection(res_ids)) / float(limit)
            recalls.append(recall)

            results.append(
                {
                    "operation": "search",
                    "search_vector": search_vector,
                    "result_ids": result_ids,
                    "result_scores": result_scores,
                    "gt_ids": request["gt_ids"],
                    "gt_scores": request["gt_scores"],
                    "latency": request_end - request_start,
                    "latency_accumulated": request_end - workload_start,
                    "qps_moment": (i + 1) / (request_end - workload_start)
                    if (request_end - workload_start) > 0
                    else 0,
                }
            )

            percentage = (i + 1) / len(trace) * 100
            if percentage >= show_progress_percentage:
                logging.info("Progress: %.2f%% completed.", percentage)
                show_progress_percentage += 1

        workload_end = time.perf_counter()

        #
        # Aggregate stats.
        #
        search_latencies = [r["latency"] for r in results]
        avg_search_latency, p50_search_latency, p99_search_latency = self._summarizeLatencies(search_latencies)

        # Search-workload has no insert/delete ops.
        avg_delete_latency = 0.0
        p50_delete_latency = 0.0
        p99_delete_latency = 0.0
        avg_insert_latency = 0.0
        p50_insert_latency = 0.0
        p99_insert_latency = 0.0

        avg_recall = float(np.mean(recalls)) if recalls else 0.0
        #
        # QPS (end-to-end): keep the legacy definition (total wall time for the workload).
        #
        total_time = workload_end - workload_start
        qps = len(results) / total_time if total_time > 0 else 0.0

        search_params = ""
        if index_type == "hnsw":
            search_params = f"ef_search={self._config.pgvector.ef_search}"
        elif index_type == "ivfflat":
            search_params = f"nprobe={self._config.pgvector.nprobe}"
        else:
            search_params = "unknown"

        #
        # Write outputs.
        #
        writeReport(
            os.path.join(output_dir, "report.csv"),
            workload_name=self._config.workload.name,
            workload_type=self._config.workload.wtype,
            search_params=search_params,
            qps=qps,
            avg_search_latency=avg_search_latency,
            p50_search_latency=p50_search_latency,
            p99_search_latency=p99_search_latency,
            avg_delete_latency=avg_delete_latency,
            p50_delete_latency=p50_delete_latency,
            p99_delete_latency=p99_delete_latency,
            avg_insert_latency=avg_insert_latency,
            p50_insert_latency=p50_insert_latency,
            p99_insert_latency=p99_insert_latency,
            avg_recall=avg_recall,
        )

        writeSearchResultsPkl(os.path.join(output_dir, "search-results.pkl"), results)
        writeTraceExtractInfo(os.path.join(output_dir, "trace-extract-info.csv"), results, recalls)

        conn.close()

    def _summarizeLatencies(self, latencies: List[float]) -> Tuple[float, float, float]:
        """Return (avg, p50, p99)."""

        if not latencies:
            return 0.0, 0.0, 0.0

        arr = np.array(latencies, dtype=np.float64)
        avg = float(np.mean(arr))
        p50 = float(np.percentile(arr, 50))
        p99 = float(np.percentile(arr, 99))
        return avg, p50, p99
