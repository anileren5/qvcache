#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# Persist all build output to a host-side log file for post-mortem debugging.
LOG_DIR="${PROJECT_ROOT}/temp"
mkdir -p "${LOG_DIR}"
LOG_TIMESTAMP="$(date +"%Y%m%d_%H%M%S")"
LOG_FILE="${LOG_DIR}/docker_build_aker_test_${LOG_TIMESTAMP}.log"
exec > >(tee -a "${LOG_FILE}") 2>&1

IMAGE_TAG="${AKER_TEST_IMAGE_TAG:-aker_test:latest}"
DOCKERFILE_PATH="${PROJECT_ROOT}/docker/images/Dockerfile.aker_test"

FAISS_REPO_URL="${FAISS_REPO_URL:-https://github.com/facebookresearch/faiss.git}"
FAISS_REF="${FAISS_REF:-}"
AKER_MODE="${AKER_MODE:-standard}"
AKER_REPO_URL="${AKER_REPO_URL:-https://github.com/sjoon-oh/Aker.git}"
AKER_REF="${AKER_REF:-release}"

docker build \
    -f "${DOCKERFILE_PATH}" \
    -t "${IMAGE_TAG}" \
    --build-arg "FAISS_REPO_URL=${FAISS_REPO_URL}" \
    --build-arg "FAISS_REF=${FAISS_REF}" \
    --build-arg "AKER_MODE=${AKER_MODE}" \
    --build-arg "AKER_REPO_URL=${AKER_REPO_URL}" \
    --build-arg "AKER_REF=${AKER_REF}" \
    "${PROJECT_ROOT}"

echo "Logging to: ${LOG_FILE}"
echo "Built image: ${IMAGE_TAG}"
echo "Run: docker run --rm ${IMAGE_TAG} --help"
