"""Index existence checks.

This matches the legacy `dscheck.py` behavior, including its fallback index name.

In addition to checking index existence, this module logs useful datastore
metadata right after index build:
- min/max vector IDs
- table and index sizes
- current public tables and indexes
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import psycopg

from ak_bench.ak_bench_config import BenchConfig, TABLE_NAME


def _as_text(value: Any) -> str:
    """Normalize a value for logging/comparison.

    psycopg may return `bytes` for text fields under certain server/client
    encoding combinations (e.g., SQL_ASCII). This helper makes comparisons
    robust by decoding bytes as UTF-8 with replacement.
    """

    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def _bytes_to_mb(size_bytes: Optional[int]) -> Optional[float]:
    if size_bytes is None:
        return None
    return float(size_bytes) / 1024.0 / 1024.0


def checkIndexExists(conn: psycopg.Connection, config: BenchConfig) -> bool:
    """Return True if the configured index exists.

    Also prints datastore diagnostics (IDs, sizes, table/index lists) to make
    "right after build" failures easier to debug.
    """

    cursor = conn.cursor()

    index_name = config.pgvector.index_name
    qualified_table = f"public.{TABLE_NAME}"

    #
    # Diagnostics: tables and indexes.
    #
    cursor.execute(
        """
        SELECT tablename
        FROM pg_tables
        WHERE schemaname = 'public'
        ORDER BY tablename;
        """
    )
    tables = [_as_text(r[0]) for r in cursor.fetchall()]
    logging.info("Public tables: %s", tables)

    cursor.execute(
        """
        SELECT indexname, tablename, indexdef
        FROM pg_indexes
        WHERE schemaname = 'public'
        ORDER BY tablename, indexname;
        """
    )
    indexes = [(_as_text(r[0]), _as_text(r[1]), _as_text(r[2])) for r in cursor.fetchall()]
    logging.info("Public indexes (%d):", len(indexes))
    for name, tbl, idef in indexes:
        logging.info("  - %s on %s: %s", name, tbl, idef)

    #
    # Diagnostics: ID range.
    #
    cursor.execute(f"SELECT MIN(id), MAX(id) FROM {qualified_table};")
    min_id, max_id = cursor.fetchone()
    logging.info("Table '%s' id range: min_id=%s max_id=%s", qualified_table, min_id, max_id)

    #
    # Diagnostics: sizes.
    #
    cursor.execute(
        """
        SELECT
            pg_relation_size(%s::regclass) AS heap_bytes,
            pg_size_pretty(pg_relation_size(%s::regclass)) AS heap_pretty,
            pg_total_relation_size(%s::regclass) AS total_bytes,
            pg_size_pretty(pg_total_relation_size(%s::regclass)) AS total_pretty;
        """,
        (qualified_table, qualified_table, qualified_table, qualified_table),
    )
    heap_bytes, heap_pretty, total_bytes, total_pretty = cursor.fetchone()
    heap_mb = _bytes_to_mb(heap_bytes)
    total_mb = _bytes_to_mb(total_bytes)
    logging.info(
        "Table '%s' size (vectors+ids heap): %s (%.2f MB), total=%s (%.2f MB)",
        qualified_table,
        _as_text(heap_pretty),
        heap_mb if heap_mb is not None else -1.0,
        _as_text(total_pretty),
        total_mb if total_mb is not None else -1.0,
    )

    cursor.execute(
        """
        SELECT
            to_regclass(%s) AS idx,
            pg_relation_size(to_regclass(%s)) AS idx_bytes,
            pg_size_pretty(pg_relation_size(to_regclass(%s))) AS idx_pretty;
        """,
        (index_name, index_name, index_name),
    )
    idx_reg, idx_bytes, idx_pretty = cursor.fetchone()
    if idx_reg is None:
        logging.warning("Index '%s' size: N/A (to_regclass returned NULL)", index_name)
    else:
        idx_mb = _bytes_to_mb(idx_bytes)
        logging.info(
            "Index '%s' size: %s (%.2f MB)",
            index_name,
            _as_text(idx_pretty),
            idx_mb if idx_mb is not None else -1.0,
        )

    #
    # Existence check.
    #
    cursor.execute(
        """
        SELECT 1
        FROM pg_indexes
        WHERE schemaname = 'public' AND indexname = %s
        LIMIT 1;
        """,
        (index_name,),
    )

    exists = cursor.fetchone() is not None

    # Keep the legacy-style log line, but normalize types for readability.
    cursor.execute(
        """
        SELECT indexname
        FROM pg_indexes
        WHERE schemaname = 'public' AND indexname = %s;
        """,
        (index_name,),
    )
    index_lists = cursor.fetchall()
    flat = [_as_text(item[0]) for item in index_lists]
    logging.info("Checking for index '%s' in the database: %s", index_name, flat)

    if exists:
        logging.info("Index '%s' exists in the database.", index_name)
        return True

    logging.error("Index '%s' does not exist in the database.", index_name)
    return False
