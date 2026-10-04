"""CLI entrypoint for the refactor sample.

This CLI focuses on two workloads:
- Search-workload (legacy workloada trace format)
- Stress-workload (legacy stress trace format)

All outputs are kept compatible with legacy scripts.

This sample also provides a minimal end-to-end pgvector pipeline:
- create table
- COPY vectors
- build index
- check index
- fix trace GT holes
"""

from __future__ import annotations

import argparse
import logging
import os

from ak_bench.ak_bench_config import loadFromIni
from ak_bench.ak_bench_pgvector_admin import PgvectorAdmin
from ak_bench.ak_bench_runner_search import SearchWorkloadRunner
from ak_bench.ak_bench_runner_stress import StressWorkloadRunner
from ak_bench.ak_bench_trace_fix import TraceGtFixer
from ak_bench.ak_bench_util import setupLogging
from ak_bench.ak_bench_workload_gen import SearchWorkloadGenerator, StressWorkloadGenerator


def _addGtBackendArgs(parser: argparse.ArgumentParser) -> None:
    def _env_int(key: str, default: int) -> int:
        raw = os.environ.get(key, "").strip()
        if raw == "":
            return default
        try:
            return int(raw)
        except ValueError:
            logging.warning("Ignoring invalid env %s=%r; using default %d", key, raw, default)
            return default

    def _env_str(key: str, default: str, allowed: tuple[str, ...]) -> str:
        raw = os.environ.get(key, "").strip()
        if raw == "":
            return default
        if raw not in allowed:
            logging.warning("Ignoring invalid env %s=%r; using default %s", key, raw, default)
            return default
        return raw

    parser.add_argument(
        "--gt-backend",
        type=str,
        default=_env_str("GT_BACKEND", "postgres", ("postgres", "numpy")),
        choices=("postgres", "numpy"),
        help="GT backend: postgres (legacy exact scan) or numpy (SIMD + threaded)",
    )
    parser.add_argument(
        "--gt-numpy-workers",
        type=int,
        default=_env_int("GT_NUMPY_WORKERS", 8),
        help="Number of worker threads for numpy GT (only if --gt-backend=numpy)",
    )
    parser.add_argument(
        "--gt-numpy-base-chunk-rows",
        type=int,
        default=_env_int("GT_NUMPY_BASE_CHUNK_ROWS", 100000),
        help="Base chunk rows for numpy GT (only if --gt-backend=numpy)",
    )
    parser.add_argument(
        "--gt-numpy-query-batch-size",
        type=int,
        default=_env_int("GT_NUMPY_QUERY_BATCH_SIZE", 16),
        help="Query batch size for numpy GT (only if --gt-backend=numpy)",
    )


def _buildParser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ak_bench", description="Refactor-sample benchmark CLI")

    subparsers = parser.add_subparsers(dest="command", required=True)

    #
    # Workload generation.
    #
    gen_search = subparsers.add_parser("generate-search-workload", help="Generate Search-workload trace (.pkl)")
    gen_search.add_argument("--config", type=str, required=True)
    gen_search.add_argument("--force", action="store_true", help="Overwrite existing trace")
    _addGtBackendArgs(gen_search)

    gen_stress = subparsers.add_parser("generate-stress-workload", help="Generate Stress-workload trace (.pkl)")
    gen_stress.add_argument("--config", type=str, required=True)
    gen_stress.add_argument("--force", action="store_true", help="Overwrite existing trace")
    _addGtBackendArgs(gen_stress)

    #
    # Benchmark runners.
    #
    run_search = subparsers.add_parser("run-search-workload", help="Run Search-workload benchmark")
    run_search.add_argument("--config", type=str, required=True)
    run_search.add_argument("--output-dir", type=str, default=".")

    run_stress = subparsers.add_parser("run-stress-workload", help="Run Stress-workload benchmark")
    run_stress.add_argument("--config", type=str, required=True)
    run_stress.add_argument("--invalidate", type=float, default=0.0)
    run_stress.add_argument("--output-dir", type=str, default=".")

    #
    # End-to-end pgvector pipeline (table/COPY/index).
    #
    create_tbl = subparsers.add_parser("create-table", help="Create extension + items table")
    create_tbl.add_argument("--config", type=str, required=True)

    alter_sys = subparsers.add_parser("alter-system-for-build", help="ALTER SYSTEM for faster index build")
    alter_sys.add_argument("--config", type=str, required=True)
    alter_sys.add_argument("--max-worker-processes", type=int, default=100)

    copy_vec = subparsers.add_parser("copy-vectors", help="COPY base vectors into items with contiguous IDs")
    copy_vec.add_argument("--config", type=str, required=True)
    copy_vec.add_argument(
        "--max-rows",
        type=int,
        default=0,
        help="If >0, only copy this many vectors (useful for quick tests)",
    )

    build_idx = subparsers.add_parser("build-index", help="Build pgvector index (hnsw/ivfflat)")
    build_idx.add_argument("--config", type=str, required=True)

    dscheck = subparsers.add_parser("dscheck-index", help="Check if the configured index exists")
    dscheck.add_argument("--config", type=str, required=True)

    #
    # Legacy GT hole fixer.
    #
    fix_trace = subparsers.add_parser("fix-trace", help="Fill missing GT entries in an existing trace file")
    fix_trace.add_argument("--config", type=str, required=True)
    fix_trace.add_argument(
        "--trace-path",
        type=str,
        default="",
        help="Optional explicit trace path. Default: dataset.gt_trace from config.",
    )

    return parser


def runMain() -> None:
    setupLogging("INFO")

    parser = _buildParser()
    args = parser.parse_args()

    config = loadFromIni(args.config)

    if args.command == "generate-search-workload":
        if config.getWorkloadKind() not in ("search",):
            logging.warning("Config wtype is '%s' (not search). Generating search trace anyway.", config.workload.wtype)
        SearchWorkloadGenerator(config).generate(
            force=args.force,
            gt_backend=args.gt_backend,
            gt_numpy_workers=args.gt_numpy_workers,
            gt_numpy_base_chunk_rows=args.gt_numpy_base_chunk_rows,
            gt_numpy_query_batch_size=args.gt_numpy_query_batch_size,
        )
        return

    if args.command == "generate-stress-workload":
        if config.getWorkloadKind() not in ("stress",):
            logging.warning("Config wtype is '%s' (not stress). Generating stress trace anyway.", config.workload.wtype)
        StressWorkloadGenerator(config).generate(
            force=args.force,
            gt_backend=args.gt_backend,
            gt_numpy_workers=args.gt_numpy_workers,
            gt_numpy_base_chunk_rows=args.gt_numpy_base_chunk_rows,
            gt_numpy_query_batch_size=args.gt_numpy_query_batch_size,
        )
        return

    if args.command == "run-search-workload":
        SearchWorkloadRunner(config).run(output_dir=args.output_dir)
        return

    if args.command == "run-stress-workload":
        StressWorkloadRunner(config).run(output_dir=args.output_dir, invalidate_fraction=args.invalidate)
        return

    if args.command == "create-table":
        PgvectorAdmin(config).createTable()
        return

    if args.command == "alter-system-for-build":
        PgvectorAdmin(config).alterSystemForBuild(max_worker_processes=args.max_worker_processes)
        return


    if args.command == "copy-vectors":
        max_rows = args.max_rows if args.max_rows > 0 else None
        inserted = PgvectorAdmin(config).copyBaseVectors(max_rows=max_rows)
        logging.info("COPY completed. Inserted rows: %d", inserted)
        return

    if args.command == "build-index":
        PgvectorAdmin(config).buildIndex()
        return

    if args.command == "dscheck-index":
        from ak_bench.ak_bench_dscheck import checkIndexExists
        from ak_bench.ak_bench_pg import PostgresClient

        conn = PostgresClient(config).connect()
        exists = checkIndexExists(conn, config)
        conn.close()

        if not exists:
            raise SystemExit(2)
        return

    if args.command == "fix-trace":
        trace_path = args.trace_path.strip() if args.trace_path.strip() != "" else config.dataset.gt_trace_path
        fixed = TraceGtFixer(config).fixTraceFile(trace_path)
        logging.info("Trace fixed. Updated entries: %d", fixed)
        return

    raise RuntimeError(f"Unknown command: {args.command}")


# Backwards-compatible entrypoint name.
main = runMain
