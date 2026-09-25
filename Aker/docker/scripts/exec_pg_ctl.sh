#!/bin/bash
set -euo pipefail

# Start/stop PostgreSQL inside a running container using pg_ctl.
# This script parses the same INI config used by pgvector-bench and executes pg_ctl via docker exec.
#
# Design note:
# - To avoid brittle quoting, config-derived paths are passed through docker exec -e
#   and expanded inside the container shell.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

usage() {
    cat <<EOF
Usage: $0 <container_name> <start|stop|status> <bench_ini_config> [numa_node]

Examples:
  $0 aker_std start pgvector-bench/configs/search_workload_hnsw.ini 0
  $0 aker_std stop  pgvector-bench/configs/search_workload_hnsw.ini

Notes:
- The benchmark INI is parsed to locate:
  - postgres.datastore (PGDATA)
  - postgres.psql_config (postgresql.conf)
  - postgres.pg_log (log file)
  - postgres.port (optional)
- The project root is expected to be bind-mounted into the container at the same path.
EOF
}

if [[ $# -lt 3 ]]; then
    usage
    exit 1
fi

container_name="$1"
action="$2"
config_arg="$3"
numa_node="${4:-}"
os_user="${AK_BENCH_DOCKER_OS_USER:-akerbench}"

# Reuse the INI parsing helpers from the benchmark safeguard script.
# shellcheck disable=SC1090
source "${PROJECT_ROOT}/pgvector-bench/ak_bench_pg_safeguard.sh"

config_path="$(resolve_config_path "${config_arg}")"
if [[ -z "${config_path}" || ! -f "${config_path}" ]]; then
    printf "[exec_pg_ctl] ERROR: config not found: %s\n" "${config_arg}" >&2
    exit 1
fi

datastore_path="$(ini_get_value "${config_path}" "postgres" "datastore")"
datastore_path="$(resolve_datastore_path "${datastore_path}")"
if [[ -z "${datastore_path}" ]]; then
    printf "[exec_pg_ctl] ERROR: postgres.datastore is empty; cannot manage pg_ctl.\n" >&2
    exit 1
fi

psql_config_path="$(ini_get_value "${config_path}" "postgres" "psql_config")"
if [[ -z "${psql_config_path}" && -f "${PROJECT_ROOT}/pgvector-bench/configs/postgresql.conf" ]]; then
    psql_config_path="pgvector-bench/configs/postgresql.conf"
fi
psql_config_path="$(resolve_path_under_root "${psql_config_path}")"

pg_log_path="$(ini_get_value "${config_path}" "postgres" "pg_log")"
if [[ -z "${pg_log_path}" ]]; then
    pg_log_path="pgvector-bench/output/postgres.log"
fi
pg_log_path="$(resolve_path_under_root "${pg_log_path}")"

port="$(ini_get_value "${config_path}" "postgres" "port")"

docker_exec_pg() {
    local inner_cmd="$1"

    docker exec -u "${os_user}" \
        -e AKER_PGDATA="${datastore_path}" \
        -e AKER_PSQL_CONFIG="${psql_config_path}" \
        -e AKER_PG_LOG="${pg_log_path}" \
        -e AKER_PG_PORT="${port}" \
        -e AKER_NUMA_NODE="${numa_node}" \
        "${container_name}" \
        bash -lc "${inner_cmd}"
}

case "${action}" in
    start)
        docker_exec_pg '
            set -euo pipefail

            numa_prefix=()
            if [[ -n "${AKER_NUMA_NODE}" ]]; then
                numa_prefix=(numactl --cpunodebind="${AKER_NUMA_NODE}" --membind="${AKER_NUMA_NODE}")
            fi

            mkdir -p "$(dirname -- "${AKER_PG_LOG}")"

            extra_opts=()
            if [[ -n "${AKER_PSQL_CONFIG}" ]]; then
                extra_opts+=( -o "--config-file=${AKER_PSQL_CONFIG}" )
            fi
            if [[ -n "${AKER_PG_PORT}" ]]; then
                extra_opts+=( -o "-p ${AKER_PG_PORT}" )
            fi

            "${numa_prefix[@]}" pg_ctl -D "${AKER_PGDATA}" -l "${AKER_PG_LOG}" "${extra_opts[@]}" start
        '
        ;;
    stop)
        docker_exec_pg '
            set -euo pipefail

            numa_prefix=()
            if [[ -n "${AKER_NUMA_NODE}" ]]; then
                numa_prefix=(numactl --cpunodebind="${AKER_NUMA_NODE}" --membind="${AKER_NUMA_NODE}")
            fi

            "${numa_prefix[@]}" pg_ctl -D "${AKER_PGDATA}" -m fast stop || true
        '
        ;;
    status)
        docker_exec_pg '
            set -euo pipefail
            pg_ctl -D "${AKER_PGDATA}" status
        '
        ;;
    *)
        usage
        exit 1
        ;;
esac
