#!/bin/bash

set -euo pipefail

#
# Prepare pgvector storage (legacy-style), without running benchmarks.
#
# This script performs:
#   1) initdb (PGDATA)
#   2) start postgres (pg_ctl)
#   3) create extension + items table
#   4) COPY base vectors (explicit contiguous IDs)
#   5) ALTER SYSTEM for faster index build
#   6) restart postgres
#   7) build index (hnsw / ivfflat)
#   8) dscheck index
#   9) stop postgres
#  10) create a clean snapshot (PGDATA-clean)
#
# Notes:
# - This is the "base pgvector" pipeline that you can use even if the Aker-integrated
#   pgvector build cannot build the index.
# - No workload files are generated in this script.
#

BENCH_ROOT="$(pwd)"
# shellcheck disable=SC1091
source "${BENCH_ROOT}/ak_bench_env_activate.sh"
# shellcheck disable=SC1091
source "${BENCH_ROOT}/ak_bench_pg_safeguard.sh"

CONFIG_PATH=""
FORCE_REBUILD=false
MAX_WORKER_PROCESSES=100

while [[ $# -gt 0 ]]; do
    case "$1" in
        --config)
            CONFIG_PATH="$2"
            shift 2
            ;;
        --force)
            FORCE_REBUILD=true
            shift 1
            ;;
        --max-worker-processes)
            MAX_WORKER_PROCESSES="$2"
            shift 2
            ;;
        *)
            printf "Unknown argument: %s\n" "$1" >&2
            exit 1
            ;;
    esac
 done

if [[ -z "${CONFIG_PATH}" ]]; then
    printf "--config is required\n" >&2
    exit 1
fi

CONFIG_PATH="$(resolve_config_path "${CONFIG_PATH}")"

# Current refactor requires Docker mode by default.
prepare_docker_environment "${CONFIG_PATH}"

if [[ -z "${AK_BENCH_DOCKER_CONTAINER_INTERNAL:-}" ]]; then
    require_cmd initdb
    require_cmd pg_ctl
fi

DATASTORE_VALUE="$(ini_get_value "${CONFIG_PATH}" "postgres" "datastore")"
DATASTORE_PATH="$(resolve_datastore_path "${DATASTORE_VALUE}")"
if [[ -z "${DATASTORE_PATH}" ]]; then
    log_err "postgres.datastore must be set for storage preparation"
    exit 1
fi

DATASTORE_CLEAN_VALUE="$(ini_get_value "${CONFIG_PATH}" "postgres" "datastore_clean")"
if [[ -z "${DATASTORE_CLEAN_VALUE}" ]]; then
    DATASTORE_CLEAN_PATH="${DATASTORE_PATH}-clean"
else
    DATASTORE_CLEAN_PATH="$(resolve_datastore_path "${DATASTORE_CLEAN_VALUE}")"
fi

PSQL_CONFIG_VALUE="$(ini_get_value "${CONFIG_PATH}" "postgres" "psql_config")"
if [[ -z "${PSQL_CONFIG_VALUE}" && -f "${BENCH_ROOT}/configs/postgresql.conf" ]]; then
    PSQL_CONFIG_VALUE="configs/postgresql.conf"
fi
PSQL_CONFIG_PATH="$(resolve_path_under_root "${PSQL_CONFIG_VALUE}")"

PG_LOG_VALUE="$(ini_get_value "${CONFIG_PATH}" "postgres" "pg_log")"
if [[ -z "${PG_LOG_VALUE}" ]]; then
    PG_LOG_VALUE="output/postgres.log"
fi
PG_LOG_PATH="$(resolve_path_under_root "${PG_LOG_VALUE}")"

HOST="$(ini_get_value "${CONFIG_PATH}" "postgres" "host")"
PORT="$(ini_get_value "${CONFIG_PATH}" "postgres" "port")"
PGUSER="$(ini_get_value "${CONFIG_PATH}" "postgres" "user")"

if [[ -d "${DATASTORE_CLEAN_PATH}" && "${FORCE_REBUILD}" != "true" ]]; then
    log_info "Clean snapshot already exists; skip storage preparation: ${DATASTORE_CLEAN_PATH}"
    log_info "Use --force to rebuild from scratch."
    exit 0
fi

log_info "Preparing pgvector storage from scratch"
log_info "PGDATA: ${DATASTORE_PATH}"
log_info "Clean : ${DATASTORE_CLEAN_PATH}"

# Stop and reset PGDATA.
pgctl_stop_force_all "${DATASTORE_PATH}"
rm -rf "${DATASTORE_PATH}" "${DATASTORE_CLEAN_PATH}"
mkdir -p "$(dirname -- "${DATASTORE_PATH}")"

pg_initdb "${DATASTORE_PATH}" "${PGUSER}"

# Start postgres.
export_env_from_ini "${CONFIG_PATH}"
pgctl_start "${DATASTORE_PATH}" "${PSQL_CONFIG_PATH}" "${PG_LOG_PATH}"
pg_wait_ready "${HOST}" "${PORT}" "${PGUSER}"

log_info "Postgres started, preparing storage..."

# Create table.
run_bench_cli create-table --config "${CONFIG_PATH}"

# COPY vectors.
run_bench_cli copy-vectors --config "${CONFIG_PATH}"

# Legacy: increase max_worker_processes and restart before index build.
run_bench_cli alter-system-for-build --config "${CONFIG_PATH}" --max-worker-processes "${MAX_WORKER_PROCESSES}"
pgctl_stop "${DATASTORE_PATH}"
export_env_from_ini "${CONFIG_PATH}"
pgctl_start "${DATASTORE_PATH}" "${PSQL_CONFIG_PATH}" "${PG_LOG_PATH}"
pg_wait_ready "${HOST}" "${PORT}" "${PGUSER}"

# Build index.
run_bench_cli build-index --config "${CONFIG_PATH}"

# dscheck.
run_bench_cli dscheck-index --config "${CONFIG_PATH}"

# Stop and create clean snapshot.
pgctl_stop "${DATASTORE_PATH}"
cp -r "${DATASTORE_PATH}" "${DATASTORE_CLEAN_PATH}"

log_info "Storage preparation completed"
log_info "Clean snapshot created: ${DATASTORE_CLEAN_PATH}"

docker_cleanup_tmp_traces

maybe_shutdown_docker_container "${CONFIG_PATH}"
