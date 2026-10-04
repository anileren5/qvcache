"""pgvector datastore management helpers.

These helpers are intended to match the legacy pipeline semantics:
- Create extension + table (items)
- COPY vectors with explicit contiguous IDs (0..N-1)
- Build HNSW/IVFFlat index with CREATE INDEX CONCURRENTLY

This module is used by the refactor-sample end-to-end scripts.
"""

from __future__ import annotations

import glob
import logging
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import psycopg

from ak_bench.ak_bench_config import BenchConfig, TABLE_NAME
from ak_bench.ak_bench_pg import PostgresClient


class PgvectorAdmin:
    """Administrative operations for the pgvector benchmark datastore."""

    def __init__(self, config: BenchConfig):
        self._config = config

    def createTable(self) -> None:
        """Create the pgvector extension and recreate the items table."""

        if self._config.dataset.dim is None:
            raise ValueError("dataset.dim must be set to create the table")

        conn = PostgresClient(self._config).connect()

        # Keep SQL identical to legacy create.py.
        conn.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        conn.commit()

        conn.execute(f"DROP TABLE IF EXISTS {TABLE_NAME} CASCADE;")

        conn.execute(
            f"""
            CREATE TABLE {TABLE_NAME} (
                id SERIAL PRIMARY KEY,
                embedding vector({self._config.dataset.dim}) NOT NULL
            );"""
        )
        conn.execute(f"ALTER TABLE {TABLE_NAME} ALTER COLUMN embedding SET STORAGE PLAIN;")

        conn.commit()

        # Legacy create.py performs a table check (SELECT *). The table is empty here.
        cursor = conn.cursor()
        cursor.execute(f"SELECT * FROM {TABLE_NAME};")
        _ = cursor.fetchall()

        conn.close()

    def copyBaseVectors(self, max_rows: Optional[int] = None) -> int:
        """COPY base vectors into the items table with contiguous IDs.

        Args:
            max_rows: If set, only copy up to this many vectors.

        Returns:
            Number of rows inserted.
        """

        base_path = self._config.dataset.base_path
        if base_path is None or base_path.strip() == "":
            raise ValueError("dataset.base must be set for COPY")

        paths: list[str]
        if "*" in base_path:
            paths = sorted(glob.glob(base_path))
            if not paths:
                raise FileNotFoundError(f"No files matched dataset.base pattern: {base_path}")
        else:
            paths = [base_path]

        conn = PostgresClient(self._config).connect()
        cursor = conn.cursor()

        inserted = 0
        next_id = 0

        with cursor.copy(f"COPY {TABLE_NAME} (id, embedding) FROM STDIN WITH (FORMAT BINARY)") as copy:
            copy.set_types(["integer", "vector"])

            for file_path in paths:
                vecs = np.load(file_path, mmap_mode="r")
                logging.info("Loaded base vectors: %s (rows=%d)", Path(file_path).name, len(vecs))

                for row in vecs:
                    if max_rows is not None and inserted >= max_rows:
                        break

                    # Keep the same representation as legacy upload.py: np.array(list)
                    if isinstance(row, np.ndarray):
                        payload = np.array(row.tolist())
                    else:
                        payload = np.array(row)

                    copy.write_row((next_id, payload))
                    next_id += 1
                    inserted += 1

                if max_rows is not None and inserted >= max_rows:
                    break

        cursor.execute(f"SELECT COUNT(*) FROM {TABLE_NAME};")
        row_count = cursor.fetchone()[0]
        logging.info("Number of rows in '%s' table: %d", TABLE_NAME, row_count)

        conn.close()

        return inserted

    def alterSystemForBuild(self, max_worker_processes: int = 100) -> None:
        """Apply ALTER SYSTEM parameters used by the legacy scripts."""

        conn = PostgresClient(self._config).connect()
        cursor = conn.cursor()
        cursor.execute(f"ALTER SYSTEM SET max_worker_processes = {max_worker_processes};")
        conn.close()

    def buildIndex(self) -> None:
        """Build a pgvector index using legacy parameters."""

        cfg = self._config.pgvector

        conn = PostgresClient(self._config).connect()
        cursor = conn.cursor()

        # Keep the legacy build.py behavior.
        cursor.execute("SET max_parallel_workers = 96;")
        cursor.execute("SET max_parallel_maintenance_workers = 96;")

        if cfg.index_type not in ("hnsw", "ivfflat"):
            raise ValueError(f"Unsupported index type: {cfg.index_type}")

        if cfg.distance_type not in (
            "vector_l2_ops",
            "vector_ip_ops",
            "vector_cosine_ops",
            "vector_l1_ops",
        ):
            raise ValueError(f"Unsupported distance type: {cfg.distance_type}")

        if cfg.index_type == "hnsw":
            cursor.execute(
                f"""
                CREATE INDEX CONCURRENTLY {cfg.index_name} ON {TABLE_NAME} USING hnsw (embedding {cfg.distance_type})
                WITH (m = {cfg.m}, ef_construction = {cfg.ef_construction});
                """
            )
        else:
            cursor.execute(
                f"""
                CREATE INDEX CONCURRENTLY {cfg.index_name} ON {TABLE_NAME} USING ivfflat (embedding {cfg.distance_type})
                WITH (lists = {cfg.nlist});
                """
            )

        logging.info(
            "Index %s created successfully (type=%s, distance=%s)",
            cfg.index_name,
            cfg.index_type,
            cfg.distance_type,
        )

        conn.close()
