"""PostgreSQL-backed exact GT generator.

This matches the legacy approach:
- For each query vector, run a linear-scan exact search with `enable_indexscan=off`.
- Use L2 (`<->`) and Top-100.

The output arrays are stored as:
- gt_ids: np.int32
- gt_scores: np.float32
"""

from __future__ import annotations

import logging
import multiprocessing
from concurrent.futures import ThreadPoolExecutor
from typing import List, Sequence, Tuple

import numpy as np

from ak_bench.ak_bench_config import BenchConfig, DISTANCE_OPERATOR, GT_TOPK, TABLE_NAME
from ak_bench.ak_bench_pg import PostgresClient


class PostgresExactGtProvider:
    """Compute ground truth using PostgreSQL exact scans."""

    def __init__(self, config: BenchConfig):
        self._config = config

    def computeGtForSearchTrace(self, trace: List[dict]) -> None:
        """Fill gt_ids/gt_scores for each search entry in-place."""

        segments = self._findSearchSegments(trace)
        for start, end in segments:
            logging.info("Running exact GT for segment (%d, %d)", start, end)
            self._runExactSearch((start, end), trace)

    def _findSearchSegments(self, op_list: List[dict]) -> List[Tuple[int, int]]:
        """Return ranges of consecutive search operations."""

        segments: List[Tuple[int, int]] = []
        start = None

        for idx, item in enumerate(op_list):
            if item.get("operation") == "search":
                if start is None:
                    start = idx
            else:
                if start is not None:
                    segments.append((start, idx - 1))
                    start = None

        if start is not None:
            segments.append((start, len(op_list) - 1))

        return segments

    def _runExactSearch(self, search_range: Tuple[int, int], trace: List[dict]) -> None:
        """Run multi-threaded GT computation for a contiguous range."""

        start_idx, end_idx = search_range
        query_count = end_idx - start_idx + 1

        if query_count <= 0:
            return

        thread_num = 1
        try:
            thread_num = max(1, multiprocessing.cpu_count() // 2)
        except Exception as e:
            logging.warning("Could not detect CPU cores. Falling back to 1 thread: %s", e)
            thread_num = 1

        if query_count < (thread_num * 100):
            thread_num = 1

        index_ranges: List[Tuple[int, int]] = []
        chunk = query_count // thread_num
        remainder = query_count % thread_num
        offset = 0
        for i in range(thread_num):
            sub_len = chunk + (1 if i < remainder else 0)
            sub_start = start_idx + offset
            sub_end = sub_start + sub_len - 1
            offset += sub_len
            if sub_len > 0:
                index_ranges.append((sub_start, sub_end))

        with ThreadPoolExecutor(max_workers=thread_num) as executor:
            futures = []
            for tid, (sub_start, sub_end) in enumerate(index_ranges):
                futures.append(
                    executor.submit(self._runExactSearchRange, tid, sub_start, sub_end, trace)
                )

            for future in futures:
                future.result()

    def _runExactSearchRange(self, tid: int, start: int, end: int, trace: List[dict]) -> None:
        """Compute GT for a subrange using a dedicated DB connection."""

        conn = PostgresClient(self._config).connect()
        cursor = conn.cursor()

        completed = 0
        for i in range(start, end + 1):
            query_vector = trace[i]["payload"]

            query_string = (
                f"SELECT id, embedding {DISTANCE_OPERATOR} '{query_vector.tolist()}' AS _score "
                f"FROM {TABLE_NAME} ORDER BY _score LIMIT {GT_TOPK};"
            )

            with conn.transaction():
                cursor.execute("SET LOCAL enable_indexscan = off")
                cursor.execute(query_string)

            searched = cursor.fetchall()

            ids = np.array([x[0] for x in searched], dtype=np.int32)
            scores = np.array([x[1] for x in searched], dtype=np.float32)

            trace[i]["gt_ids"] = ids
            trace[i]["gt_scores"] = scores

            completed += 1
            if completed % 100 == 0:
                denom = max(1, (end - start + 1))
                logging.info("GT thread %d progress: %.2f%%", tid, (completed / denom) * 100.0)

        conn.close()
