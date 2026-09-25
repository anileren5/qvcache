#!/bin/bash
set -euo pipefail

# Build all container images needed for the benchmark:
# - Vanilla pgvector
# - Aker-integrated pgvector (standard / potluck / proximity)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# Persist all build output to a host-side log file for post-mortem debugging.
LOG_DIR="${PROJECT_ROOT}/temp"
mkdir -p "${LOG_DIR}"
LOG_TIMESTAMP="$(date +"%Y%m%d_%H%M%S")"
LOG_FILE="${LOG_DIR}/docker_build_images_${LOG_TIMESTAMP}.log"
exec > >(tee -a "${LOG_FILE}") 2>&1

IMAGE_PREFIX="${IMAGE_PREFIX:-aker_pgvector}"
POSTGRESQL_VERSION="${POSTGRESQL_VERSION:-16.2}"
BASE_IMAGE="${BASE_IMAGE:-${IMAGE_PREFIX}_pg16_base:${POSTGRESQL_VERSION}}"

log() {
    printf "[build_images] %s\n" "$*"
}

build_base() {
    log "Building PostgreSQL ${POSTGRESQL_VERSION} base image: ${BASE_IMAGE}"
    docker build \
        -f "${PROJECT_ROOT}/docker/images/Dockerfile.pg16_base" \
        --build-arg "POSTGRESQL_VERSION=${POSTGRESQL_VERSION}" \
        -t "${BASE_IMAGE}" \
        "${PROJECT_ROOT}"
}

build_vanilla() {
    local tag="${IMAGE_PREFIX}_vanilla:latest"
    log "Building vanilla image: ${tag}"
    docker build \
        -f "${PROJECT_ROOT}/docker/images/Dockerfile.pgvector_vanilla" \
        --build-arg "BASE_IMAGE=${BASE_IMAGE}" \
        -t "${tag}" \
        "${PROJECT_ROOT}"
}

build_aker_variant() {
    local aker_mode="$1"
    local tag="${IMAGE_PREFIX}_${aker_mode}:latest"

    log "Building Aker image (${aker_mode}): ${tag}"
    docker build \
        -f "${PROJECT_ROOT}/docker/images/Dockerfile.pgvector_aker" \
        --build-arg "BASE_IMAGE=${BASE_IMAGE}" \
        --build-arg "AKER_MODE=${aker_mode}" \
        -t "${tag}" \
        "${PROJECT_ROOT}"
}

main() {
    log "Logging to: ${LOG_FILE}"
    build_base
    build_vanilla
    build_aker_variant standard
    build_aker_variant potluck
    build_aker_variant proximity

    log "Done."
    log "Built images:"
    log "  ${IMAGE_PREFIX}_vanilla:latest"
    log "  ${IMAGE_PREFIX}_standard:latest"
    log "  ${IMAGE_PREFIX}_potluck:latest"
    log "  ${IMAGE_PREFIX}_proximity:latest"
}

main "$@"
