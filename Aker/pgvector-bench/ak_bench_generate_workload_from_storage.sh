#!/bin/bash

set -euo pipefail

#
# Generate workload trace (.pkl) when storage/index already exists.
#
# This script:
#   1) restores the clean snapshot (PGDATA-clean -> PGDATA) if configured
#      (without dropping OS page cache, since this is not a benchmark phase)
#   2) starts postgres (pg_ctl) and waits for readiness
#   3) generates the workload trace (Search-workload or Stress-workload)
#   4) runs the legacy GT-hole fixer (fix-trace)
#   5) collects newly created /tmp traces into output dir
#   6) stops postgres (if managed via PGDATA)
#
# Notes:
# - Workload format is kept compatible with legacy scripts.
# - GT is generated using the selected backend (default: postgres exact scan).
# - Unlike benchmark runs, this script does NOT enforce numactl.
#

BENCH_ROOT="$(pwd)"
# shellcheck disable=SC1091
source "${BENCH_ROOT}/ak_bench_env_activate.sh"
# shellcheck disable=SC1091
source "${BENCH_ROOT}/ak_bench_pg_safeguard.sh"

CONFIG_PATH=""
OUTPUT_ROOT="output"
FORCE_GENERATE=false
AKER_CONFIG_OVERRIDE=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --config)
            CONFIG_PATH="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_ROOT="$2"
            shift 2
            ;;
        --force)
            FORCE_GENERATE=true
            shift 1
            ;;
        --aker-config)
            AKER_CONFIG_OVERRIDE="$2"
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

# Resolve GT backend selection.
#
# Policy:
# - postgres (default): use Docker+PostgreSQL to generate GT and run fix-trace.
# - numpy: do NOT start Docker/PostgreSQL for this script; generate GT using numpy only.
#
# We export [env] from INI so users can specify GT_BACKEND/GT_NUMPY_* in configs.
# If the user already exported GT_* env vars before running this script, keep them.
GT_BACKEND_PRE="${GT_BACKEND:-}"
GT_NUMPY_WORKERS_PRE="${GT_NUMPY_WORKERS:-}"
GT_NUMPY_BASE_CHUNK_ROWS_PRE="${GT_NUMPY_BASE_CHUNK_ROWS:-}"
GT_NUMPY_QUERY_BATCH_SIZE_PRE="${GT_NUMPY_QUERY_BATCH_SIZE:-}"

export_env_from_ini "${CONFIG_PATH}" "${AKER_CONFIG_OVERRIDE}"

if [[ -n "${GT_BACKEND_PRE}" ]]; then
    export GT_BACKEND="${GT_BACKEND_PRE}"
fi
if [[ -n "${GT_NUMPY_WORKERS_PRE}" ]]; then
    export GT_NUMPY_WORKERS="${GT_NUMPY_WORKERS_PRE}"
fi
if [[ -n "${GT_NUMPY_BASE_CHUNK_ROWS_PRE}" ]]; then
    export GT_NUMPY_BASE_CHUNK_ROWS="${GT_NUMPY_BASE_CHUNK_ROWS_PRE}"
fi
if [[ -n "${GT_NUMPY_QUERY_BATCH_SIZE_PRE}" ]]; then
    export GT_NUMPY_QUERY_BATCH_SIZE="${GT_NUMPY_QUERY_BATCH_SIZE_PRE}"
fi

GT_BACKEND_SELECTED="${GT_BACKEND:-postgres}"
GT_BACKEND_SELECTED="$(printf '%s' "${GT_BACKEND_SELECTED}" | tr '[:upper:]' '[:lower:]')"
if [[ "${GT_BACKEND_SELECTED}" != "postgres" ]]; then
    GT_BACKEND_SELECTED="numpy"
fi
log_info "GT backend selected for workload generation: ${GT_BACKEND_SELECTED}"

mkdir -p "${OUTPUT_ROOT}"

RUN_ID="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="${OUTPUT_ROOT}/workloads/workload_${RUN_ID}"
MERGED_TMP_DIR="${OUTPUT_ROOT}/merged_tmp_traces"
mkdir -p "${RUN_DIR}" "${MERGED_TMP_DIR}"

TMP_BEFORE_LIST="${RUN_DIR}/tmp_trace_before.txt"
: > "${TMP_BEFORE_LIST}"

if [[ "${GT_BACKEND_SELECTED}" == "postgres" ]]; then
    restore_clean_snapshot_and_start_no_cache_drop "${CONFIG_PATH}" "workload-generate" "${AKER_CONFIG_OVERRIDE}"
    capture_tmp_trace_list "${TMP_BEFORE_LIST}"
else
    log_info "Skipping Docker/PostgreSQL startup because gt-backend=numpy"
fi

WORKLOAD_TYPE="$(ini_get_value "${CONFIG_PATH}" "workload" "wtype")"

case "${WORKLOAD_TYPE}" in
    search-workload)
        if [[ "${FORCE_GENERATE}" == "true" ]]; then
            run_bench_cli generate-search-workload --config "${CONFIG_PATH}" --force --gt-backend "${GT_BACKEND_SELECTED}"
        else
            run_bench_cli generate-search-workload --config "${CONFIG_PATH}" --gt-backend "${GT_BACKEND_SELECTED}"
        fi
        ;;
    stress-workload)
        if [[ "${FORCE_GENERATE}" == "true" ]]; then
            run_bench_cli generate-stress-workload --config "${CONFIG_PATH}" --force --gt-backend "${GT_BACKEND_SELECTED}"
        else
            run_bench_cli generate-stress-workload --config "${CONFIG_PATH}" --gt-backend "${GT_BACKEND_SELECTED}"
        fi
        ;;
    *)
        printf "[ERROR] Unsupported workload.wtype for this script: %s\n" "${WORKLOAD_TYPE}" >&2
        exit 1
        ;;
esac

if [[ "${GT_BACKEND_SELECTED}" == "postgres" ]]; then
    run_bench_cli fix-trace --config "${CONFIG_PATH}"
else
    log_info "Skipping fix-trace because gt-backend=numpy (no PostgreSQL dependency)"
fi

# Copy the generated pkl into the run directory for convenience.
GT_TRACE_PATH="$(ini_get_value "${CONFIG_PATH}" "dataset" "gt_trace")"
if [[ -n "${GT_TRACE_PATH}" && -f "${GT_TRACE_PATH}" ]]; then
    cp -a "${GT_TRACE_PATH}" "${RUN_DIR}/"
fi

if [[ "${GT_BACKEND_SELECTED}" == "postgres" ]]; then
    collect_new_tmp_traces "${TMP_BEFORE_LIST}" "${RUN_DIR}/tmp_traces" "${MERGED_TMP_DIR}"

    maybe_stop_postgres "${CONFIG_PATH}"

    docker_cleanup_tmp_traces

    maybe_shutdown_docker_container "${CONFIG_PATH}"
fi

printf "[OK] Workload generation finished: %s\n" "${RUN_DIR}"
