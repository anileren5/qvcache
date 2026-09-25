"""Trace (workload) GT hole fixer.

Legacy scripts provided a `fix.py` utility that fills missing GT for search operations.
This module provides the same feature while keeping the pickle formats unchanged.

This fixer does not change existing GT entries; it only fills missing ones.
"""

from __future__ import annotations

import logging
from typing import Any, List

import numpy as np

from ak_bench.ak_bench_config import BenchConfig, DISTANCE_OPERATOR, GT_TOPK, TABLE_NAME
from ak_bench.ak_bench_pg import PostgresClient
from ak_bench.ak_bench_trace_schema import SearchTrace, StressTrace, loadTrace, saveTrace, validateSearchTrace, validateStressTrace


class TraceGtFixer:
    """Fill missing GT entries in legacy-compatible traces."""

    def __init__(self, config: BenchConfig):
        self._config = config

    def fixSearchTraceInPlace(self, trace: SearchTrace) -> int:
        """Fill missing GT in a Search-workload trace (list[dict]).

        Returns:
            Number of entries fixed.
        """

        fixed = 0

        conn = PostgresClient(self._config).connect()
        cursor = conn.cursor()

        for i, entry in enumerate(trace):
            if entry.get("operation") != "search":
                continue

            gt_ids = entry.get("gt_ids")
            gt_scores = entry.get("gt_scores")

            if gt_ids is not None and gt_scores is not None and len(gt_ids) == GT_TOPK:
                continue

            query_vector = entry["payload"]

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

            entry["gt_ids"] = ids
            entry["gt_scores"] = scores

            fixed += 1
            if fixed % 100 == 0:
                logging.info("Fixed %d GT holes so far...", fixed)

        conn.close()
        return fixed

    def fixTraceFile(self, trace_path: str) -> int:
        """Fix a trace file in-place.

        Supports both:
        - Search-workload trace: list[dict]
        - Stress-workload trace: dict{search:list[dict], insert:list[dict]}

        Returns:
            Number of entries fixed.
        """

        trace_obj: Any = loadTrace(trace_path)

        if isinstance(trace_obj, list):
            validateSearchTrace(trace_obj)
            fixed = self.fixSearchTraceInPlace(trace_obj)
            saveTrace(trace_path, trace_obj)
            return fixed

        if isinstance(trace_obj, dict):
            validateStressTrace(trace_obj)
            fixed = self.fixSearchTraceInPlace(trace_obj["search"])
            saveTrace(trace_path, trace_obj)
            return fixed

        raise TypeError(f"Unsupported trace object type: {type(trace_obj)}")
