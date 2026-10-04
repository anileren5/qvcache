"""Configuration loader for the refactor sample.

This module intentionally keeps compatibility with the legacy INI configs.
Only a subset of keys are used for Search-workload and Stress-workload.

In this sample scope, we hard-code:
- L2 distance operator (<->)
- GT Top-100
- Table name: items

We still parse pgvector index build parameters to support an end-to-end pipeline.
"""

from __future__ import annotations

import configparser
from dataclasses import dataclass
from typing import Optional

#
# Hard-coded semantics for the current experiment scope.
#
TABLE_NAME = "items"
DISTANCE_OPERATOR = "<->"  # L2
GT_TOPK = 100


@dataclass(frozen=True)
class PostgresConfig:
    """PostgreSQL connection parameters."""

    host: str
    port: int
    user: str
    password: str
    database: str


@dataclass(frozen=True)
class DatasetConfig:
    """Dataset and trace paths."""

    base_path: Optional[str]
    search_path: str
    gt_trace_path: str
    split_num: Optional[int]
    dim: Optional[int]


@dataclass(frozen=True)
class PgvectorConfig:
    """pgvector index parameters (build + runtime)."""

    index_type: str
    index_name: str
    distance_type: str

    # Build parameters.
    m: int
    ef_construction: int
    nlist: int

    # Runtime parameters.
    ef_search: Optional[int]
    nprobe: Optional[int]


@dataclass(frozen=True)
class WorkloadConfig:
    """Workload settings."""

    wtype: str
    name: str
    limit: int
    insert_ratio: Optional[float]


@dataclass(frozen=True)
class BenchConfig:
    """Unified configuration object."""

    config_path: str
    postgres: PostgresConfig
    dataset: DatasetConfig
    pgvector: PgvectorConfig
    workload: WorkloadConfig

    def getWorkloadKind(self) -> str:
        """Return the normalized workload kind: 'search' or 'stress'."""

        wtype = self.workload.wtype.strip().lower()

        if wtype in ("workloada", "search-workload", "search"):
            return "search"

        if wtype.startswith("stress-") or wtype in ("stress-workload", "stress"):
            return "stress"

        # Keep legacy names observable to the caller.
        raise ValueError(f"Unsupported workload type: {self.workload.wtype}")

    def getStressMode(self) -> str:
        """Return the stress mode string compatible with legacy runners."""

        wtype = self.workload.wtype.strip().lower()

        if wtype in ("stress-workload", "stress"):
            # Default to the most common legacy mode.
            return "stress-insert"

        if wtype.startswith("stress-"):
            return wtype

        raise ValueError(f"Not a stress workload: {self.workload.wtype}")


def loadFromIni(config_path: str) -> BenchConfig:
    """Load legacy INI config into a typed BenchConfig."""

    parser = configparser.ConfigParser()
    parser.read(config_path)

    def _get_optional_str(section: str, key: str) -> Optional[str]:
        value = parser.get(section, key, fallback="").strip()
        if value == "":
            return None
        return value

    def _get_optional_int(section: str, key: str) -> Optional[int]:
        value = parser.get(section, key, fallback="").strip()
        if value == "":
            return None
        try:
            return int(value)
        except ValueError as e:
            raise ValueError(f"Invalid integer for [{section}] {key}: {value!r}") from e

    def _get_int_with_default(section: str, key: str, default: int) -> int:
        value = parser.get(section, key, fallback="").strip()
        if value == "":
            return default
        try:
            return int(value)
        except ValueError as e:
            raise ValueError(f"Invalid integer for [{section}] {key}: {value!r}") from e

    def _get_optional_float(section: str, key: str) -> Optional[float]:
        value = parser.get(section, key, fallback="").strip()
        if value == "":
            return None
        try:
            return float(value)
        except ValueError as e:
            raise ValueError(f"Invalid float for [{section}] {key}: {value!r}") from e

    #
    # PostgreSQL section.
    #
    postgres = PostgresConfig(
        host=parser.get("postgres", "host"),
        port=_get_int_with_default("postgres", "port", 5432),
        user=parser.get("postgres", "user"),
        password=parser.get("postgres", "password"),
        database=parser.get("postgres", "database"),
    )

    #
    # Dataset section.
    #
    dataset = DatasetConfig(
        base_path=_get_optional_str("dataset", "base"),
        search_path=parser.get("dataset", "search"),
        gt_trace_path=parser.get("dataset", "gt_trace"),
        split_num=_get_optional_int("dataset", "split_num"),
        dim=_get_optional_int("dataset", "dim"),
    )

    #
    # Workload section.
    #
    workload = WorkloadConfig(
        wtype=parser.get("workload", "wtype"),
        name=parser.get("workload", "name", fallback="unknown"),
        limit=_get_int_with_default("workload", "limit", 10),
        insert_ratio=_get_optional_float("workload", "insert_ratio"),
    )

    #
    # pgvector section.
    #
    index_type = parser.get("pgvector", "type")
    index_name = parser.get("pgvector", "index_name", fallback="")
    if index_name.strip() == "":
        # Keep a deterministic fallback index name when configs omit it.
        index_name = f"index_{index_type}_idx"

    distance_type = parser.get("pgvector", "distance", fallback="vector_l2_ops")

    pgvector = PgvectorConfig(
        index_type=index_type,
        index_name=index_name,
        distance_type=distance_type,
        m=_get_int_with_default("pgvector", "m", 16),
        ef_construction=_get_int_with_default("pgvector", "ef_construction", 64),
        nlist=_get_int_with_default("pgvector", "nlist", 100),
        ef_search=_get_optional_int("pgvector", "ef_search"),
        nprobe=_get_optional_int("pgvector", "nprobe"),
    )

    config = BenchConfig(
        config_path=config_path,
        postgres=postgres,
        dataset=dataset,
        pgvector=pgvector,
        workload=workload,
    )

    if config.getWorkloadKind() == "stress" and config.dataset.split_num is None:
        raise ValueError("dataset.split_num must be set for Stress-workload")

    return config
