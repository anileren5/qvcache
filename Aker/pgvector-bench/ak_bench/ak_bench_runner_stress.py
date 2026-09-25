"""Stress-workload runner.

This runner is compatible with the legacy stress trace format:
- trace: dict{search:list[dict], insert:list[dict]}

The stress workload is interpreted as:
- Optional prefill phase: execute the search list once (legacy behavior)
- Optional insert phase: insert vectors from trace['insert']
- Invalidate phase: call aker_invalidate_random(fraction)
- Search phase: execute the search list again and record latency/recall

Outputs are written in the same filenames as legacy scripts.
"""

from __future__ import annotations

import copy
import logging
import os
import time
from typing import List, Tuple

import numpy as np

from ak_bench.ak_bench_config import BenchConfig, DISTANCE_OPERATOR, TABLE_NAME
from ak_bench.ak_bench_dscheck import checkIndexExists
from ak_bench.ak_bench_pg import PostgresClient
from ak_bench.ak_bench_reporting import writeReport, writeSearchResultsPkl, writeTraceExtractInfo
from ak_bench.ak_bench_trace_schema import ensureStressGtPresent, loadTrace, validateStressTrace
from ak_bench.ak_bench_util import ensureDir


class StressWorkloadRunner:
    """Execute a Stress-workload trace against PostgreSQL."""

    def __init__(self, config: BenchConfig):
        self._config = config

    def run(self, output_dir: str, invalidate_fraction: float) -> None:
        """Run stress workload and write outputs."""

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
        validateStressTrace(trace)
        ensureStressGtPresent(trace)

        search_trace = trace["search"]
        insert_trace = trace["insert"]

        # Legacy runner truncates insert list to search length.
        insert_trace = insert_trace[: len(search_trace)]

        stress_mode = self._config.getStressMode()
        limit = self._config.workload.limit

        #
        # Set index search parameters.
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
        insert_sql = f"INSERT INTO {TABLE_NAME} (id, embedding) VALUES (%s, %s);"

        #
        # Current row count (debug logging).
        #
        cursor.execute(f"SELECT COUNT(*) FROM {TABLE_NAME};")
        row_count = cursor.fetchone()[0]
        logging.info("Rows in '%s': %d", TABLE_NAME, row_count)

        cursor.execute(f"SELECT MAX(id) FROM {TABLE_NAME};")
        max_vector_id = cursor.fetchone()[0] or 0

        #
        # Phase 1: Prefill (legacy behavior: run the search list once).
        #
        self._runPrefillSearch(cursor, search_sql, search_trace, limit)

        # Remaining search requests (legacy uses deepcopy).
        remaining_search_requests = copy.deepcopy(search_trace)

        #
        # Phase 2: Optional inserts.
        #
        if stress_mode in ("stress-insert", "stress-mixed"):
            max_vector_id = self._runInsertPhase(cursor, insert_sql, insert_trace, max_vector_id)

        #
        # Phase 3: Invalidate random cache entries (implemented inside pgvector/Aker).
        #
        cursor.execute(f"SELECT aker_invalidate_random({invalidate_fraction});")
        logging.info(">> Aker invalidation requested: %.2f%%", invalidate_fraction * 100.0)

        #
        # Phase 4: Timed search workload.
        #
        results, recalls = self._runTimedSearchPhase(
            cursor,
            search_sql,
            remaining_search_requests,
            limit,
            with_recall=(stress_mode != "stress-mixed"),
        )

        #
        # Aggregate stats.
        #
        search_latencies = [r["latency"] for r in results if r.get("operation") == "search"]
        avg_search_latency, p50_search_latency, p99_search_latency = self._summarizeLatencies(search_latencies)

        avg_recall = float(np.mean(recalls)) if recalls else 0.0

        # Keep the same QPS definition as Search-workload: end-to-end wall time.
        total_time = results[-1]["latency_accumulated"] if results else 0.0
        qps = len(results) / total_time if total_time > 0 else 0.0

        # Stress runner only performs searches in the timed phase.
        avg_delete_latency = 0.0
        p50_delete_latency = 0.0
        p99_delete_latency = 0.0
        avg_insert_latency = 0.0
        p50_insert_latency = 0.0
        p99_insert_latency = 0.0

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

    def _runPrefillSearch(self, cursor, search_sql: str, search_trace: List[dict], limit: int) -> None:
        """Warm up by executing the search list once (legacy behavior)."""

        if not search_trace:
            return

        logging.info("Prefilling by running %d searches...", len(search_trace))

        show_progress_percentage = 1
        for i, request in enumerate(search_trace):
            search_vector = request["payload"]
            cursor.execute(search_sql, (search_vector, limit), binary=True, prepare=True)
            cursor.fetchall()

            percentage = (i + 1) / len(search_trace) * 100
            if percentage >= show_progress_percentage:
                logging.info("Prefill progress: %.2f%%", percentage)
                show_progress_percentage += 1

    def _runInsertPhase(self, cursor, insert_sql: str, insert_trace: List[dict], max_vector_id: int) -> int:
        """Insert vectors sequentially."""

        logging.info("Insert phase: inserting %d vectors...", len(insert_trace))

        show_progress_percentage = 1
        for i, request in enumerate(insert_trace):
            insert_vector = request["payload"]
            max_vector_id += 1
            cursor.execute(insert_sql, (max_vector_id, insert_vector), binary=True, prepare=True)

            percentage = (i + 1) / len(insert_trace) * 100
            if percentage >= show_progress_percentage:
                logging.info("Insert progress: %.2f%%", percentage)
                show_progress_percentage += 1

        return max_vector_id

    def _runTimedSearchPhase(
        self,
        cursor,
        search_sql: str,
        search_requests: List[dict],
        limit: int,
        with_recall: bool,
    ) -> Tuple[List[dict], List[float]]:
        """Execute timed search phase and return (results, recalls)."""

        results: List[dict] = []
        recalls: List[float] = []

        show_progress_percentage = 1
        workload_start = time.perf_counter()

        for i, request in enumerate(search_requests):
            search_vector = request["payload"]

            request_start = time.perf_counter()
            cursor.execute(search_sql, (search_vector, limit), binary=True, prepare=True)
            rows = cursor.fetchall()
            request_end = time.perf_counter()

            result_ids = [row[0] for row in rows]
            result_scores = [row[1] for row in rows]

            entry = {
                "operation": "search",
                "latency": request_end - request_start,
                "latency_accumulated": request_end - workload_start,
                "qps_moment": (i + 1) / (request_end - workload_start)
                if (request_end - workload_start) > 0
                else 0,
                "result_ids": result_ids,
                "result_scores": result_scores,
            }

            if with_recall:
                entry["gt_ids"] = request["gt_ids"]
                entry["gt_scores"] = request["gt_scores"]

                gt_ids = set(request["gt_ids"][:limit])
                res_ids = set(result_ids)
                recalls.append(len(gt_ids.intersection(res_ids)) / float(limit))

            results.append(entry)

            percentage = (i + 1) / len(search_requests) * 100
            if percentage >= show_progress_percentage:
                elapsed = time.perf_counter() - workload_start
                qps_now = (i + 1) / elapsed if elapsed > 0 else 0.0
                logging.info("Progress: %.2f%% (QPS: %.2f)", percentage, qps_now)
                show_progress_percentage += 1

        return results, recalls

    def _summarizeLatencies(self, latencies: List[float]) -> Tuple[float, float, float]:
        """Return (avg, p50, p99)."""

        if not latencies:
            return 0.0, 0.0, 0.0

        arr = np.array(latencies, dtype=np.float64)
        avg = float(np.mean(arr))
        p50 = float(np.percentile(arr, 50))
        p99 = float(np.percentile(arr, 99))
        return avg, p50, p99
