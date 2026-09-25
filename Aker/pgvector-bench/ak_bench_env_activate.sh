#!/bin/bash

set -euo pipefail

#
# This script sets up the runtime environment for pgvector-bench.
# Assumption: user launches scripts from the pgvector-bench working directory.
#

BENCH_ROOT="$(pwd)"
VENV_DIR="${BENCH_ROOT}/.venv"

#
# Python package location.
# Layout assumption: user launches scripts from the pgvector-bench/ directory.
#
PYTHON_ROOT="${BENCH_ROOT}"

#
# Ensure PostgreSQL binaries are discoverable (same default as legacy scripts).
#
export PATH=/usr/local/pgsql/bin:"${PATH}"
export LD_LIBRARY_PATH=/usr/local/pgsql/lib:"${LD_LIBRARY_PATH:-}"
export MANPATH=/usr/local/pgsql/share/man:"${MANPATH:-}"

#
# Ensure Python can import the bench package.
#
if [[ -z "${PYTHONPATH:-}" ]]; then
    export PYTHONPATH="${PYTHON_ROOT}"
else
    export PYTHONPATH="${PYTHON_ROOT}:${PYTHONPATH}"
fi

ensure_working_dir() {
    #
    # Validate the expected directory structure under the current working directory.
    #
    if [[ ! -d "${BENCH_ROOT}/configs" ]]; then
        printf "[ERROR] configs/ directory not found. Please run from pgvector-bench/\n" >&2
        exit 1
    fi
    if [[ ! -d "${BENCH_ROOT}/ak_bench" ]]; then
        printf "[ERROR] ak_bench package not found. Please run from pgvector-bench/\n" >&2
        exit 1
    fi
}

ensure_venv() {
    #
    # Create and initialize a venv if missing.
    #
    ensure_working_dir

    if [[ ! -d "${VENV_DIR}" ]]; then
        python3 -m venv "${VENV_DIR}"
        # shellcheck disable=SC1091
        source "${VENV_DIR}/bin/activate"
        pip install --upgrade pip >/dev/null
        pip install numpy psycopg pgvector psycopg-binary >/dev/null
    else
        # shellcheck disable=SC1091
        source "${VENV_DIR}/bin/activate"
    fi
}

run_bench_cli() {
    #
    # Run the pgvector-bench CLI.
    #
    ensure_venv
    python3 -m ak_bench "$@"
}

run_bench_cli_numactl() {
    #
    # Run the pgvector-bench CLI under a fixed NUMA binding.
    # This matches the legacy experiment scripts for benchmarking.
    #
    ensure_venv

    if ! command -v numactl >/dev/null 2>&1; then
        printf "[ERROR] numactl is required for benchmarking but was not found in PATH\n" >&2
        exit 1
    fi

    numactl --cpunodebind=0 --membind=0 python3 -m ak_bench "$@"
}

export -f ensure_working_dir
export -f ensure_venv
export -f run_bench_cli
export -f run_bench_cli_numactl
