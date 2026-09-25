#!/bin/bash
set -euo pipefail

# Run a benchmark container in "idle" mode.
# The benchmark harness can start/stop PostgreSQL via docker exec + pg_ctl.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

usage() {
    cat <<EOF
Usage: $0 <image_tag> <container_name> [network_mode] [host_port]

Examples:
  # Host networking (recommended on Linux for lowest overhead):
  $0 aker_pgvector_vanilla:latest aker_vanilla host

  # Bridge networking with explicit port mapping:
  $0 aker_pgvector_standard:latest aker_std bridge 5433

Notes:
- This script is for *execution*. The benchmark harness starts/stops PostgreSQL via docker exec + pg_ctl.
- The project root is bind-mounted into the container at the SAME absolute path.
  This keeps benchmark config paths consistent between host and container.
- Containers are NOT removed automatically. The harness can stop containers for reuse.
EOF
}

if [[ $# -lt 2 ]]; then
    usage
    exit 1
fi

image_tag="$1"
container_name="$2"
network_mode="${3:-host}"
host_port="${4:-}"

require_cmd() {
    local cmd="$1"
    if ! command -v "${cmd}" >/dev/null 2>&1; then
        printf "[run_container] ERROR: required command not found: %s\n" "${cmd}" >&2
        exit 1
    fi
}

require_cmd docker

container_exists() {
    docker ps -a --format '{{.Names}}' | grep -Fxq "${container_name}" >/dev/null 2>&1
}

container_running() {
    docker ps --format '{{.Names}}' | grep -Fxq "${container_name}" >/dev/null 2>&1
}

force_recreate="${AK_BENCH_DOCKER_FORCE_RECREATE:-0}"
docker_seccomp="${AK_BENCH_DOCKER_SECCOMP:-}"
docker_privileged="${AK_BENCH_DOCKER_PRIVILEGED:-0}"

if [[ "${force_recreate}" == "1" ]] && container_exists; then
    if container_running; then
        docker stop "${container_name}" >/dev/null 2>&1 || true
    fi
    docker rm -f "${container_name}" >/dev/null 2>&1 || true
fi

if container_exists; then
    if container_running; then
        printf "[run_container] reuse running container: %s (%s)\n" "${container_name}" "${image_tag}"
        exit 0
    fi
    docker start "${container_name}" >/dev/null
    printf "[run_container] started existing container: %s\n" "${container_name}"
    exit 0
fi

docker_args=(
    run -d
    --name "${container_name}"
    --cap-add SYS_NICE
    -e "HOST_UID=$(id -u)"
    -e "HOST_GID=$(id -g)"
    -e "AK_BENCH_USER=akerbench"
    -v "${PROJECT_ROOT}:${PROJECT_ROOT}"
    -w "${PROJECT_ROOT}"
)


if [[ "${docker_privileged}" == "1" ]]; then
    docker_args+=(--privileged)
elif [[ -n "${docker_seccomp}" ]]; then
    if [[ "${docker_seccomp}" == "unconfined" ]]; then
        docker_args+=(--security-opt seccomp=unconfined)
    else
        docker_args+=(--security-opt "seccomp=${docker_seccomp}")
    fi
fi


if [[ "${network_mode}" == "host" ]]; then
    docker_args+=(--network host)
else
    if [[ -z "${host_port}" ]]; then
        printf "[run_container] ERROR: host_port is required when network_mode != host\n" >&2
        exit 1
    fi
    docker_args+=( -p "${host_port}:5432" )
fi

docker "${docker_args[@]}" "${image_tag}" >/dev/null

if [[ "${network_mode}" == "host" ]]; then
    printf "[run_container] started %s (%s) network=%s\n" "${container_name}" "${image_tag}" "${network_mode}"
else
    printf "[run_container] started %s (%s) network=%s host_port=%s->5432\n" "${container_name}" "${image_tag}" "${network_mode}" "${host_port}"
fi
