#!/usr/bin/env bash
# Aker: download official simZipf query streams (Hugging Face) and convert them
# to DiskANN bins against the existing SPACEV-1M index. Does not edit Aker sources.
#
# Official files: spacev-sim-100k-{0.3,0.6,0.99}.npy
# https://huggingface.co/datasets/sjoon-oh/aker
#
# Default is SPACEV-1M. For SPACEV-10M use ./scripts/aker/prepare_paper_scale.sh.
set -euo pipefail

cd "$(dirname "$0")/../.." || exit 1

DATASET="${DATASET:-spacev-small-test}"
DATA_TYPE="${DATA_TYPE:-int8}"
SKEW="${SKEW:-0.3}"
HF_REPO="${HF_REPO:-https://huggingface.co/datasets/sjoon-oh/aker/resolve/main}"
AKER_DATASET_DIR="${AKER_DATASET_DIR:-Aker/pgvector-bench/dataset/spacev-small-test}"
OUT_DIR="${OUT_DIR:-data/${DATASET}}"
K="${K:-100}"
METRIC="${METRIC:-l2}"
SKIP_GT="${SKIP_GT:-0}"
NPY_PREFIX="${NPY_PREFIX:-spacev}"

STREAM_TAG="sim-100k-${SKEW}"
NPY_NAME="${NPY_PREFIX}-sim-100k-${SKEW}.npy"
NPY_PATH="${AKER_DATASET_DIR}/${NPY_NAME}"
BASE_BIN="${OUT_DIR}/${DATASET}_base.bin"
QUERY_BIN="${OUT_DIR}/${DATASET}_query_${STREAM_TAG}.bin"
GT_BIN="${OUT_DIR}/${DATASET}_groundtruth_${STREAM_TAG}.bin"

mkdir -p "${AKER_DATASET_DIR}" "${OUT_DIR}"

if [[ ! -f "${BASE_BIN}" ]]; then
  echo "Error: missing ${BASE_BIN}. Run ./scripts/aker/prepare_search_workload.sh first."
  exit 1
fi

echo "=========================================="
echo "Aker simZipf stream -> DiskANN bins"
echo "=========================================="
echo "Skew:   ${SKEW}"
echo "Query:  ${QUERY_BIN}"
echo "GT:     ${GT_BIN}"
echo ""

if [[ ! -f "${NPY_PATH}" ]]; then
  echo "Downloading ${NPY_NAME} ..."
  python3 - "${HF_REPO}/${NPY_NAME}" "${NPY_PATH}" <<'PY'
import sys, urllib.request
url, dest = sys.argv[1], sys.argv[2]
urllib.request.urlretrieve(url, dest)
print(f"Wrote {dest}")
PY
else
  echo "npy exists: ${NPY_PATH}"
fi

if [[ ! -f "${QUERY_BIN}" ]]; then
  python3 scripts/aker/npy_to_diskann_bin.py \
    --input "${NPY_PATH}" \
    --output "${QUERY_BIN}" \
    --dtype "${DATA_TYPE}"
else
  echo "Query bin exists: ${QUERY_BIN}"
fi

if [[ "${SKIP_GT}" != "1" ]]; then
  COMPUTE_GT_BIN="./build/benchmarks/compute_groundtruth"
  if [[ ! -x "${COMPUTE_GT_BIN}" ]]; then
    echo "Error: ${COMPUTE_GT_BIN} not found. Build the project first."
    exit 1
  fi
  if [[ ! -f "${GT_BIN}" ]]; then
    echo ""
    echo "Computing exact groundtruth vs SPACEV-1M (100k queries; can take a while)..."
    "${COMPUTE_GT_BIN}" "${BASE_BIN}" "${QUERY_BIN}" "${GT_BIN}" "${DATA_TYPE}" "${K}" "${METRIC}"
  else
    echo "Groundtruth exists: ${GT_BIN}"
  fi
fi

echo ""
echo "Ready simZipf ${SKEW}:"
echo "  ${QUERY_BIN}"
echo "  ${GT_BIN}"
echo ""
echo "Run:"
echo "  DATASET=${DATASET} DATA_TYPE=${DATA_TYPE} QUERY_STREAM=${STREAM_TAG} \\"
echo "    ./scripts/aker/aker_benchmark_search_workload.sh"
echo "  DATASET=${DATASET} DATA_TYPE=${DATA_TYPE} QUERY_STREAM=${STREAM_TAG} \\"
echo "    ./scripts/qvcache/qvcache_benchmark_search_workload.sh"
