#!/usr/bin/env bash

set -euo pipefail

# Quick end-to-end smoke test runner for the Aker test image.
#
# This script:
#  1) Builds the Aker test Docker image if missing.
#  2) Launches a container, executes the test binary, and collects traces.
#  3) Copies /tmp/aker_trace_* from the container into <project root>/temp.
#  4) Stops and removes the container.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${SCRIPT_DIR}"

IMAGE_TAG="${AKER_TEST_IMAGE_TAG:-aker_test:latest}"
CONTAINER_NAME="${AKER_QUICK_TEST_CONTAINER_NAME:-aker_quick_test}"
TEST_BIN_PATH="${AKER_TEST_BIN_PATH:-/usr/local/bin/aker-random-cache-test}"
TEMP_DIR="${PROJECT_ROOT}/temp"
BUILD_SCRIPT="${PROJECT_ROOT}/docker/scripts/build_aker_test_image.sh"

require_cmd() {
    local cmd="$1"
    if ! command -v "${cmd}" >/dev/null 2>&1; then
        printf "[ak_quick_test] ERROR: required command not found: %s\n" "${cmd}" >&2
        exit 1
    fi
}

docker_image_exists() {
    docker image inspect "${IMAGE_TAG}" >/dev/null 2>&1
}

container_exists() {
    docker ps -a --format '{{.Names}}' | grep -Fxq "${CONTAINER_NAME}" >/dev/null 2>&1
}

cleanup() {
    if container_exists; then
        docker rm -f "${CONTAINER_NAME}" >/dev/null 2>&1 || true
    fi
}

trap cleanup EXIT

require_cmd docker

mkdir -p "${TEMP_DIR}"

if ! docker_image_exists; then
    printf "[ak_quick_test] building docker image: %s\n" "${IMAGE_TAG}"
    if [[ ! -f "${BUILD_SCRIPT}" ]]; then
        printf "[ak_quick_test] ERROR: build script not found: %s\n" "${BUILD_SCRIPT}" >&2
        exit 1
    fi
    AKER_TEST_IMAGE_TAG="${IMAGE_TAG}" bash "${BUILD_SCRIPT}"
fi

if container_exists; then
    docker rm -f "${CONTAINER_NAME}" >/dev/null 2>&1 || true
fi

printf "[ak_quick_test] starting container: %s (%s)\n" "${CONTAINER_NAME}" "${IMAGE_TAG}"
docker run -d --name "${CONTAINER_NAME}" "${IMAGE_TAG}" bash -lc "sleep infinity" >/dev/null

printf "[ak_quick_test] running test: %s\n" "${TEST_BIN_PATH}"

set +e
docker exec "${CONTAINER_NAME}" "${TEST_BIN_PATH}" "$@"
test_rc=$?
set -e

mapfile -t trace_paths < <(
    docker exec "${CONTAINER_NAME}" bash -lc 'ls -1d /tmp/aker_trace_* 2>/dev/null || true'
)

if [[ ${#trace_paths[@]} -eq 0 ]]; then
    printf "[ak_quick_test] no traces found under /tmp/aker_trace_*\n"
else
    printf "[ak_quick_test] copying traces to: %s\n" "${TEMP_DIR}"
    for trace_path in "${trace_paths[@]}"; do
        base_name="$(basename "${trace_path}")"
        dest_name="${base_name}"
        suffix=0
        while [[ -e "${TEMP_DIR}/${dest_name}" ]]; do
            suffix=$((suffix + 1))
            dest_name="${base_name}_copy${suffix}"
        done

        docker cp "${CONTAINER_NAME}:${trace_path}" "${TEMP_DIR}/${dest_name}" >/dev/null
        printf "[ak_quick_test]  - %s\n" "${TEMP_DIR}/${dest_name}"
    done
fi

printf "[ak_quick_test] stopping container: %s\n" "${CONTAINER_NAME}"
docker stop "${CONTAINER_NAME}" >/dev/null || true
docker rm "${CONTAINER_NAME}" >/dev/null || true

trap - EXIT
exit "${test_rc}"
