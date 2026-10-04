"""PostgreSQL connection helpers.

This keeps the same connection semantics as the legacy scripts:
- autocommit enabled
- ensure pgvector extension exists
- register pgvector type adapters
"""

from __future__ import annotations

from dataclasses import dataclass

import pgvector.psycopg
import psycopg

from ak_bench.ak_bench_config import BenchConfig


@dataclass
class PostgresClient:
    """A small wrapper around psycopg connection."""

    config: BenchConfig

    def connect(self) -> psycopg.Connection:
        """Create a new PostgreSQL connection."""

        db_params = {
            "host": self.config.postgres.host,
            "dbname": self.config.postgres.database,
            "user": self.config.postgres.user,
            "password": self.config.postgres.password,
            "autocommit": True,
            "port": self.config.postgres.port,
        }

        conn = psycopg.connect(**db_params)

        #
        # Ensure pgvector is available and register vector adapters.
        #
        conn.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        pgvector.psycopg.register_vector(conn)

        return conn
