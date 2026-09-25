#!/usr/bin/env bash
# Aker: prepare SPACEV-10M search-workload for DiskANN.
# Does not edit Aker sources.
#
# SPACEV (web search): 10M x dim-100 int8 + spacev-sim-100k-{skew}.npy
# Files: https://huggingface.co/datasets/sjoon-oh/aker
set -euo pipefail

cd "$(dirname "$0")/../.." || exit 1

SKEW="${SKEW:-0.3}"
HF_REPO="${HF_REPO:-https://huggingface.co/datasets/sjoon-oh/aker/resolve/main}"
K="${K:-100}"
METRIC="${METRIC:-l2}"
SKIP_GT="${SKIP_GT:-0}"
SKIP_INDEX="${SKIP_INDEX:-0}"
BUILD_INDEX="${BUILD_INDEX:-1}"
BUILD_THREADS="${BUILD_THREADS:-8}"

DATASET="${DATASET:-spacev-10m}"
DATA_TYPE="${DATA_TYPE:-int8}"
AKER_DATASET_DIR="${AKER_DATASET_DIR:-Aker/pgvector-bench/dataset/spacev-10m}"
BASE_NPY_NAME="spacev-10m.npy"
QUERY_NPY_NAME="spacev-sim-100k-${SKEW}.npy"
R="${R:-64}"
L="${L:-128}"
B="${B:-8}"
M="${M:-8}"

STREAM_TAG="sim-100k-${SKEW}"
OUT_DIR="${OUT_DIR:-data/${DATASET}}"
BASE_NPY="${AKER_DATASET_DIR}/${BASE_NPY_NAME}"
QUERY_NPY="${AKER_DATASET_DIR}/${QUERY_NPY_NAME}"
BASE_BIN="${OUT_DIR}/${DATASET}_base.bin"
QUERY_BIN="${OUT_DIR}/${DATASET}_query_${STREAM_TAG}.bin"
GT_BIN="${OUT_DIR}/${DATASET}_groundtruth_${STREAM_TAG}.bin"
INDEX_PREFIX="./index/${DATASET}/${DATASET}"

mkdir -p "${AKER_DATASET_DIR}" "${OUT_DIR}" "./index/${DATASET}"

download_hf() {
  local name="$1"
  local dest="$2"
  if [[ -f "${dest}" ]]; then
    echo "npy exists: ${dest}"
    return 0
  fi
  echo "Downloading ${name} from Hugging Face (~1 GB for spacev-10m.npy)..."
  python3 - "${HF_REPO}/${name}" "${dest}" <<'PY'
import sys, urllib.request
url, dest = sys.argv[1], sys.argv[2]
print(f"GET {url}")
urllib.request.urlretrieve(url, dest)
print(f"Wrote {dest}")
PY
}

echo "=========================================="
echo "Aker SPACEV-10M -> DiskANN"
echo "=========================================="
echo "Dataset: ${DATASET} (${DATA_TYPE})"
echo "Skew:   ${SKEW}"
echo "Base:   ${BASE_BIN}"
echo "Query:  ${QUERY_BIN}"
echo "GT:     ${GT_BIN}"
echo ""
echo "Exact GT vs 10M x 100k is expensive. Set SKIP_GT=1 to convert/index first."
echo ""

download_hf "${BASE_NPY_NAME}" "${BASE_NPY}"
download_hf "${QUERY_NPY_NAME}" "${QUERY_NPY}"

if [[ ! -f "${BASE_BIN}" ]]; then
  python3 scripts/aker/npy_to_diskann_bin.py \
    --input "${BASE_NPY}" \
    --output "${BASE_BIN}" \
    --dtype "${DATA_TYPE}"
else
  echo "Base bin exists: ${BASE_BIN}"
fi

if [[ ! -f "${QUERY_BIN}" ]]; then
  python3 scripts/aker/npy_to_diskann_bin.py \
    --input "${QUERY_NPY}" \
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
    echo "Computing exact groundtruth vs SPACEV-10M (100k queries; this can take many hours)..."
    "${COMPUTE_GT_BIN}" "${BASE_BIN}" "${QUERY_BIN}" "${GT_BIN}" "${DATA_TYPE}" "${K}" "${METRIC}"
  else
    echo "Groundtruth exists: ${GT_BIN}"
  fi
fi

if [[ "${SKIP_INDEX}" != "1" && "${BUILD_INDEX}" == "1" ]]; then
  if [[ ! -f "${INDEX_PREFIX}_disk.index" && ! -f "./index/${DATASET}/${DATASET}_disk.index" ]]; then
    echo ""
    echo "Building DiskANN index at ${INDEX_PREFIX} ..."
    ./build/benchmarks/build_index "${DATA_TYPE}" \
      --data_file "${BASE_BIN}" \
      --index_prefix_path "${INDEX_PREFIX}" \
      --R "${R}" \
      --L "${L}" \
      --B "${B}" \
      --M "${M}" \
      --T "${BUILD_THREADS}" \
      --dist_metric "${METRIC}" \
      --single_file_index 0 \
      --sector_len 4096
  else
    echo "DiskANN index already present under ./index/${DATASET}/"
  fi
fi

echo ""
echo "Ready SPACEV-10M ${STREAM_TAG}:"
echo "  ${BASE_BIN}"
echo "  ${QUERY_BIN}"
echo "  ${GT_BIN}"
echo ""
echo "Run:"
echo "  QUERY_STREAM=${STREAM_TAG} ./scripts/aker/aker_benchmark_search_workload.sh"
echo "  QUERY_STREAM=${STREAM_TAG} ./scripts/qvcache/qvcache_benchmark_search_workload.sh"
echo ""
echo "Paper Pool 1%/5% of 10M vectors: AKER_POOL_SIZE=100000 or 500000"
