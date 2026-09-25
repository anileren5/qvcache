"""Workload (trace) generators.

This module generates pickle traces compatible with the legacy scripts.
Only Search-workload and Stress-workload are supported.
"""

from __future__ import annotations

import glob
import logging
import os
from pathlib import Path
from typing import List, Optional

import numpy as np

from ak_bench.ak_bench_config import BenchConfig
from ak_bench.ak_bench_gt_postgres import PostgresExactGtProvider
from ak_bench.ak_bench_gt_numpy import NumpyExactGtProvider
from ak_bench.ak_bench_pg import PostgresClient
from ak_bench.ak_bench_trace_schema import makeInsertEntry, makeSearchEntry, saveTrace


def _inferBaseVectorCountFromPath(base_path: str) -> int:
    """Infer total base vector count from one or more .npy files.

    This is used to avoid PostgreSQL dependency when gt_backend=="numpy".
    Assumption: DB ids are contiguous starting at 0 and match the base file order.
    """

    file_paths: List[str]
    if "*" in base_path:
        file_paths = sorted(glob.glob(base_path))
    else:
        file_paths = [base_path]

    if not file_paths:
        raise FileNotFoundError(f"No base files matched: {base_path}")

    total = 0
    for file_path in file_paths:
        arr = np.load(file_path, mmap_mode="r")
        total += int(arr.shape[0])

    if total <= 0:
        raise RuntimeError(f"Invalid base vector count inferred from {base_path}: {total}")

    return total


class SearchWorkloadGenerator:
    """Generate a Search-workload trace (legacy workload A format)."""

    def __init__(self, config: BenchConfig):
        self._config = config

    def generate(
        self,
        force: bool = False,
        gt_backend: str = "postgres",
        gt_numpy_workers: int = 8,
        gt_numpy_base_chunk_rows: int = 100000,
        gt_numpy_query_batch_size: int = 16,
    ) -> None:
        """Generate the trace file if missing (or if force==True)."""

        trace_path = self._config.dataset.gt_trace_path

        if os.path.exists(trace_path) and not force:
            logging.info("Trace file %s already exists. Skipping generation.", trace_path)
            return

        query_vectors = np.load(self._config.dataset.search_path)

        trace = [makeSearchEntry(vector) for vector in query_vectors]

        # Fill GT using the selected backend.
        if gt_backend == "postgres":
            PostgresExactGtProvider(self._config).computeGtForSearchTrace(trace)
        elif gt_backend == "numpy":
            if self._config.dataset.base_path is None:
                raise ValueError("dataset.base must be set when --gt-backend=numpy")

            # Infer the base row count from the base vector files.
            # Assumption: DB ids are contiguous starting at 0 and match base file order.
            base_count = _inferBaseVectorCountFromPath(self._config.dataset.base_path)
            logging.info("Inferred base vector count from dataset.base: %d", base_count)

            NumpyExactGtProvider(
                self._config,
                base_path=self._config.dataset.base_path,
                base_vector_count=base_count,
                workers=gt_numpy_workers,
                base_chunk_rows=gt_numpy_base_chunk_rows,
                query_batch_size=gt_numpy_query_batch_size,
            ).computeGtForSearchTrace(trace)
        else:
            raise ValueError(f"Unknown gt_backend: {gt_backend}")

        saveTrace(trace_path, trace)
        logging.info("Wrote Search-workload trace: %s", trace_path)

    def _getDbMaxIdPlusOne(self) -> int:
        """Return MAX(id)+1 from DB, assuming ids are contiguous from 0."""

        conn = PostgresClient(self._config).connect()
        cursor = conn.cursor()

        cursor.execute("SELECT MAX(id) FROM items;")
        result = cursor.fetchone()

        conn.close()

        if result is None or result[0] is None:
            raise RuntimeError("DB returned NULL for MAX(id); dataset may be empty")

        return int(result[0]) + 1


class StressWorkloadGenerator:
    """Generate a Stress-workload trace (legacy stress trace format)."""

    def __init__(self, config: BenchConfig):
        self._config = config

    def generate(
        self,
        force: bool = False,
        gt_backend: str = "postgres",
        gt_numpy_workers: int = 8,
        gt_numpy_base_chunk_rows: int = 100000,
        gt_numpy_query_batch_size: int = 16,
    ) -> None:
        """Generate the stress trace file if missing (or if force==True)."""

        trace_path = self._config.dataset.gt_trace_path

        if os.path.exists(trace_path) and not force:
            logging.info("Trace file %s already exists. Skipping generation.", trace_path)
            return

        if self._config.dataset.base_path is None:
            raise ValueError("dataset.base must be set for Stress-workload")
        if self._config.dataset.split_num is None:
            raise ValueError("dataset.split_num must be set for Stress-workload")

        base_vectors = self._loadBaseVectors(self._config.dataset.base_path, self._config.dataset.split_num)
        search_vectors = np.load(self._config.dataset.search_path)

        split_num = self._config.dataset.split_num

        if split_num > len(base_vectors) // 2:
            raise ValueError(
                f"split_num {split_num} is larger than half of base vectors length {len(base_vectors) // 2}"
            )

        first_split = base_vectors[:split_num]
        second_split = base_vectors[split_num: split_num * 2]

        #
        # Match legacy semantics: limit inserts to 5% of the first split.
        #
        first_split_len = len(first_split)
        second_split_len = int(0.05 * first_split_len)
        second_split = second_split[:second_split_len]

        insert_vectors = second_split

        search_trace = [makeSearchEntry(v) for v in search_vectors]
        insert_trace = [makeInsertEntry(v) for v in insert_vectors]

        #
        # Match legacy semantics: compute GT after inserts.
        # - postgres backend: physically insert into DB, then compute GT via exact scan.
        # - numpy backend: do NOT depend on DB; infer base_count from base files and
        #   include inserted vectors as an extra segment during GT computation.
        #
        base_count_before_inserts: int
        if gt_backend == "postgres":
            base_count_before_inserts = self._getDbMaxIdPlusOne()
            self._bulkInsertIntoDb(insert_trace, start_id=base_count_before_inserts)
        elif gt_backend == "numpy":
            base_count_before_inserts = _inferBaseVectorCountFromPath(self._config.dataset.base_path)
            logging.info("Inferred base vector count from dataset.base: %d", base_count_before_inserts)
        else:
            raise ValueError(f"Unknown gt_backend: {gt_backend}")

        # Compute GT after inserts.
        if gt_backend == "postgres":
            PostgresExactGtProvider(self._config).computeGtForSearchTrace(search_trace)
        elif gt_backend == "numpy":
            if self._config.dataset.base_path is None:
                raise ValueError("dataset.base must be set when --gt-backend=numpy")

            # Match DB id semantics by using max id before inserts.
            base_count = base_count_before_inserts

            NumpyExactGtProvider(
                self._config,
                base_path=self._config.dataset.base_path,
                base_vector_count=base_count,
                extra_vectors=np.asarray(insert_vectors),
                extra_start_id=base_count,
                workers=gt_numpy_workers,
                base_chunk_rows=gt_numpy_base_chunk_rows,
                query_batch_size=gt_numpy_query_batch_size,
            ).computeGtForSearchTrace(search_trace)
        else:
            raise ValueError(f"Unknown gt_backend: {gt_backend}")

        trace = {
            "search": search_trace,
            "insert": insert_trace,
        }

        saveTrace(trace_path, trace)
        logging.info("Wrote Stress-workload trace: %s", trace_path)

    def _loadBaseVectors(self, base_path: str, split_num: int) -> np.ndarray:
        """Load base vectors (supporting glob patterns) up to 2*split_num."""

        if "*" not in base_path:
            return np.load(base_path)

        matching_files = sorted(glob.glob(base_path))
        if not matching_files:
            raise FileNotFoundError(f"No files found matching pattern: {base_path}")

        loaded_vectors = 0
        base_vectors: Optional[np.ndarray] = None

        for file_idx, file_path in enumerate(matching_files):
            logging.info(
                "Loading base vectors from file %d/%d: %s",
                file_idx + 1,
                len(matching_files),
                Path(file_path).name,
            )

            vectors = np.load(file_path)
            if base_vectors is None:
                base_vectors = vectors
            else:
                base_vectors = np.concatenate((base_vectors, vectors), axis=0)

            loaded_vectors += len(vectors)
            if loaded_vectors >= split_num * 2:
                break

        if base_vectors is None:
            raise RuntimeError("Failed to load any base vectors")

        return base_vectors

    def _bulkInsertIntoDb(self, insert_trace: List[dict], start_id: int) -> None:
        """Insert vectors into DB sequentially, matching legacy behavior.

        Args:
            insert_trace: Trace entries with operation=="insert".
            start_id: The first id to assign for the insertion.
        """

        conn = PostgresClient(self._config).connect()
        cursor = conn.cursor()

        # Match legacy semantics: insert ids are contiguous and start at MAX(id)+1.
        # We take start_id explicitly to keep DB ids consistent with numpy GT.
        max_id = start_id - 1

        insert_sql = "INSERT INTO items (id, embedding) VALUES (%s, %s);"

        for i, req in enumerate(insert_trace):
            vector = req["payload"]
            with conn.transaction():
                cursor.execute(insert_sql, (max_id + 1, vector))
                max_id += 1

            if (i + 1) % 100 == 0:
                logging.info("Inserted %d/%d vectors...", i + 1, len(insert_trace))

        conn.close()

    def _getDbMaxIdPlusOne(self) -> int:
        """Return MAX(id)+1 from DB, assuming ids are contiguous from 0."""

        conn = PostgresClient(self._config).connect()
        cursor = conn.cursor()

        cursor.execute("SELECT MAX(id) FROM items;")
        result = cursor.fetchone()

        conn.close()

        if result is None or result[0] is None:
            raise RuntimeError("DB returned NULL for MAX(id); dataset may be empty")

        return int(result[0]) + 1
