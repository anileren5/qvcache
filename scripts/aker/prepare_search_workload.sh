#!/usr/bin/env bash
# Aker: convert Aker pgvector-bench SPACEV search-workload files to DiskANN bins,
# then optionally compute exact GT and build a DiskANN index.
# Does not modify Aker sources.
set -euo pipefail

cd "$(dirname "$0")/../.." || exit 1

DATASET="${DATASET:-spacev-small-test}"
DATA_TYPE="${DATA_TYPE:-int8}"
AKER_DATASET_DIR="${AKER_DATASET_DIR:-external/Aker/pgvector-bench/dataset/spacev-small-test}"
OUT_DIR="${OUT_DIR:-data/${DATASET}}"
K="${K:-100}"
METRIC="${METRIC:-l2}"
SKIP_GT="${SKIP_GT:-0}"
SKIP_INDEX="${SKIP_INDEX:-0}"
BUILD_INDEX="${BUILD_INDEX:-1}"

BASE_NPY_GLOB="${AKER_DATASET_DIR}/spacev-1m-split.part_*.npy"
QUERY_NPY="${AKER_DATASET_DIR}/example-queryset.npy"
BASE_BIN="${OUT_DIR}/${DATASET}_base.bin"
QUERY_BIN="${OUT_DIR}/${DATASET}_query.bin"
GT_BIN="${OUT_DIR}/${DATASET}_groundtruth.bin"

mkdir -p "${OUT_DIR}"

echo "=========================================="
echo "Aker search-workload -> DiskANN bins"
echo "=========================================="
echo "Dataset: ${DATASET} (${DATA_TYPE})"
echo "Source:  ${AKER_DATASET_DIR}"
echo "Output:  ${OUT_DIR}"
echo ""

if [[ ! -f "${BASE_BIN}" ]]; then
  python3 scripts/aker/npy_to_diskann_bin.py \
    --input "${BASE_NPY_GLOB}" \
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
    echo "Computing exact groundtruth (can take a while for 1M x 30k)..."
    "${COMPUTE_GT_BIN}" "${BASE_BIN}" "${QUERY_BIN}" "${GT_BIN}" "${DATA_TYPE}" "${K}" "${METRIC}"
  else
    echo "Groundtruth exists: ${GT_BIN}"
  fi
fi

if [[ "${SKIP_INDEX}" != "1" && "${BUILD_INDEX}" == "1" ]]; then
  INDEX_PREFIX="./index/${DATASET}/${DATASET}"
  if [[ ! -f "${INDEX_PREFIX}_disk.index" && ! -f "./index/${DATASET}/${DATASET}_disk.index" ]]; then
    echo ""
    echo "Building DiskANN index at ${INDEX_PREFIX} ..."
    mkdir -p "./index/${DATASET}"
    ./build/benchmarks/build_index "${DATA_TYPE}" \
      --data_file "${BASE_BIN}" \
      --index_prefix_path "${INDEX_PREFIX}" \
      --R 64 \
      --L 128 \
      --B 8 \
      --M 8 \
      --T "${BUILD_THREADS:-8}" \
      --dist_metric "${METRIC}" \
      --single_file_index 0 \
      --sector_len 4096
  else
    echo "DiskANN index already present under ./index/${DATASET}/"
  fi
fi

echo ""
echo "Ready:"
echo "  ${BASE_BIN}"
echo "  ${QUERY_BIN}"
echo "  ${GT_BIN}"
echo ""
echo "Run both caches on this stream:"
echo "  ./scripts/aker/aker_benchmark_search_workload.sh"
echo "  ./scripts/qvcache/qvcache_benchmark_search_workload.sh"
