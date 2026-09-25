"""NumPy-backed exact GT generator.

This module provides an optional ground-truth backend that computes exact
Top-K (K=100) neighbors with L2 distance using NumPy.

Design goals:
1) Keep the legacy workload file format unchanged.
2) Keep id semantics compatible with the legacy pipeline: base ids are
   contiguous starting from 0.
3) Compute scores using Euclidean distance (sqrt of squared distance),
   matching PostgreSQL `<->` (L2).

Performance notes:
- The core computation is based on batched dot products (SIMD/BLAS).
- Multi-threading is implemented at the Python level by splitting query
  batches across a ThreadPoolExecutor. For best results, set BLAS threads
  (e.g., OMP_NUM_THREADS) appropriately to avoid oversubscription.
"""

from __future__ import annotations

import glob
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import numpy as np

from ak_bench.ak_bench_config import BenchConfig, GT_TOPK


@dataclass(frozen=True)
class _NumpyGtParams:
    """Execution parameters for NumPy GT."""

    workers: int
    base_chunk_rows: int
    query_batch_size: int


class NumpyExactGtProvider:
    """Compute exact GT using NumPy (SIMD + optional threading)."""

    def __init__(
        self,
        config: BenchConfig,
        base_path: str,
        base_vector_count: int,
        *,
        workers: int = 8,
        base_chunk_rows: int = 100000,
        query_batch_size: int = 16,
        extra_vectors: Optional[np.ndarray] = None,
        extra_start_id: Optional[int] = None,
    ):
        if base_vector_count <= 0:
            raise ValueError(f"base_vector_count must be positive, got {base_vector_count}")
        if workers <= 0:
            raise ValueError(f"workers must be positive, got {workers}")
        if base_chunk_rows <= 0:
            raise ValueError(f"base_chunk_rows must be positive, got {base_chunk_rows}")
        if query_batch_size <= 0:
            raise ValueError(f"query_batch_size must be positive, got {query_batch_size}")

        if (extra_vectors is None) != (extra_start_id is None):
            raise ValueError("extra_vectors and extra_start_id must be set together")

        self._config = config
        self._base_path = base_path
        self._base_vector_count = base_vector_count
        self._params = _NumpyGtParams(
            workers=workers,
            base_chunk_rows=base_chunk_rows,
            query_batch_size=query_batch_size,
        )

        self._extra_vectors = extra_vectors
        self._extra_start_id = extra_start_id

    def computeGtForSearchTrace(self, trace: List[dict]) -> None:
        """Fill gt_ids/gt_scores for each search entry in-place."""

        if not trace:
            return

        segments = self._findSearchSegments(trace)
        for start, end in segments:
            logging.info(
                "Running numpy GT for segment (%d, %d) with base_count=%d",
                start,
                end,
                self._base_vector_count,
            )
            self._computeSegment(trace, start, end)

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

    def _computeSegment(self, trace: List[dict], start: int, end: int) -> None:
        """Compute GT for a contiguous [start, end] range."""

        query_vectors = np.stack([trace[i]["payload"] for i in range(start, end + 1)], axis=0)
        query_vectors = np.asarray(query_vectors, dtype=np.float32)

        # Maintain best K per query as squared distances to avoid sqrt until the end.
        q_count = query_vectors.shape[0]
        best_ids = np.full((q_count, GT_TOPK), -1, dtype=np.int32)
        best_dist_sq = np.full((q_count, GT_TOPK), np.inf, dtype=np.float32)

        for base_chunk, base_start_id in self._iterAllBaseChunks():
            base_chunk_f32 = np.asarray(base_chunk, dtype=np.float32)
            base_norm = np.sum(base_chunk_f32 * base_chunk_f32, axis=1, dtype=np.float32)

            self._updateTopK(
                query_vectors=query_vectors,
                best_ids=best_ids,
                best_dist_sq=best_dist_sq,
                base_chunk=base_chunk_f32,
                base_norm=base_norm,
                base_start_id=base_start_id,
            )

        # Finalize: sort and write back to trace.
        for local_i in range(q_count):
            order = np.argsort(best_dist_sq[local_i], kind="stable")
            final_ids = best_ids[local_i][order]
            final_scores = np.sqrt(best_dist_sq[local_i][order]).astype(np.float32, copy=False)

            trace[start + local_i]["gt_ids"] = final_ids.astype(np.int32, copy=False)
            trace[start + local_i]["gt_scores"] = final_scores

    def _iterAllBaseChunks(self) -> Iterable[Tuple[np.ndarray, int]]:
        """Yield all base chunks, including optional extra vectors."""

        yield from self._iterBaseChunksFromFiles()

        if self._extra_vectors is not None and self._extra_start_id is not None:
            extra = np.asarray(self._extra_vectors)
            extra_total = extra.shape[0]

            for offset in range(0, extra_total, self._params.base_chunk_rows):
                end = min(extra_total, offset + self._params.base_chunk_rows)
                yield extra[offset:end], int(self._extra_start_id + offset)

    def _iterBaseChunksFromFiles(self) -> Iterable[Tuple[np.ndarray, int]]:
        """Iterate over .npy base vectors up to base_vector_count."""

        base_path = self._base_path
        file_paths: List[str]

        if "*" in base_path:
            file_paths = sorted(glob.glob(base_path))
        else:
            file_paths = [base_path]

        if not file_paths:
            raise FileNotFoundError(f"No base files matched: {base_path}")

        loaded_total = 0
        for file_idx, file_path in enumerate(file_paths):
            if loaded_total >= self._base_vector_count:
                break

            arr = np.load(file_path, mmap_mode="r")
            file_rows = int(arr.shape[0])
            remaining = self._base_vector_count - loaded_total
            take_rows = min(file_rows, remaining)

            logging.info(
                "Loading base vectors file %d/%d: %s (take_rows=%d)",
                file_idx + 1,
                len(file_paths),
                Path(file_path).name,
                take_rows,
            )

            for offset in range(0, take_rows, self._params.base_chunk_rows):
                end = min(take_rows, offset + self._params.base_chunk_rows)
                chunk = arr[offset:end]
                yield chunk, int(loaded_total + offset)

            loaded_total += take_rows

        if loaded_total < self._base_vector_count:
            raise RuntimeError(
                f"Base vectors exhausted early: loaded {loaded_total}, expected {self._base_vector_count}" 
            )

    def _updateTopK(
        self,
        *,
        query_vectors: np.ndarray,
        best_ids: np.ndarray,
        best_dist_sq: np.ndarray,
        base_chunk: np.ndarray,
        base_norm: np.ndarray,
        base_start_id: int,
    ) -> None:
        """Update best top-K results for all queries using one base chunk."""

        from concurrent.futures import ThreadPoolExecutor

        q_count = int(query_vectors.shape[0])
        batch_size = self._params.query_batch_size

        batches: List[Tuple[int, int]] = []
        for q_start in range(0, q_count, batch_size):
            q_end = min(q_count, q_start + batch_size)
            batches.append((q_start, q_end))

        # If the workload is small, avoid thread overhead.
        use_workers = self._params.workers
        if len(batches) <= 1:
            use_workers = 1

        with ThreadPoolExecutor(max_workers=use_workers) as executor:
            futures = []
            for q_start, q_end in batches:
                futures.append(
                    executor.submit(
                        self._processQueryBatch,
                        query_vectors,
                        best_ids,
                        best_dist_sq,
                        base_chunk,
                        base_norm,
                        base_start_id,
                        q_start,
                        q_end,
                    )
                )

            for f in futures:
                f.result()

    def _processQueryBatch(
        self,
        query_vectors: np.ndarray,
        best_ids: np.ndarray,
        best_dist_sq: np.ndarray,
        base_chunk: np.ndarray,
        base_norm: np.ndarray,
        base_start_id: int,
        q_start: int,
        q_end: int,
    ) -> None:
        """Compute top-K for a query batch against a base chunk."""

        Q = query_vectors[q_start:q_end]
        q_norm = np.sum(Q * Q, axis=1, dtype=np.float32)

        # dist_sq = ||q||^2 + ||x||^2 - 2 q·x
        dot = Q @ base_chunk.T
        dist_sq = q_norm[:, None] + base_norm[None, :] - (2.0 * dot)

        # Numerical guard: avoid tiny negatives.
        np.maximum(dist_sq, 0.0, out=dist_sq)

        local_k = min(GT_TOPK, int(base_chunk.shape[0]))
        if local_k <= 0:
            return

        for i in range(q_end - q_start):
            row = dist_sq[i]
            if local_k == row.shape[0]:
                local_idx = np.arange(row.shape[0])
            else:
                local_idx = np.argpartition(row, local_k - 1)[:local_k]

            local_d = row[local_idx].astype(np.float32, copy=False)
            local_ids = (base_start_id + local_idx).astype(np.int32, copy=False)

            cur_d = best_dist_sq[q_start + i]
            cur_ids = best_ids[q_start + i]

            merged_d = np.concatenate((cur_d, local_d), axis=0)
            merged_ids = np.concatenate((cur_ids, local_ids), axis=0)

            select_k = min(GT_TOPK, int(merged_d.shape[0]))
            sel = np.argpartition(merged_d, select_k - 1)[:select_k]

            new_d = merged_d[sel]
            new_ids = merged_ids[sel]

            order = np.argsort(new_d, kind="stable")
            new_d = new_d[order]
            new_ids = new_ids[order]

            # Pad if needed (rare): keep arrays fixed-size.
            if select_k < GT_TOPK:
                pad = GT_TOPK - select_k
                new_d = np.concatenate((new_d, np.full((pad,), np.inf, dtype=np.float32)), axis=0)
                new_ids = np.concatenate((new_ids, np.full((pad,), -1, dtype=np.int32)), axis=0)

            best_dist_sq[q_start + i] = new_d[:GT_TOPK]
            best_ids[q_start + i] = new_ids[:GT_TOPK]
