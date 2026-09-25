#!/usr/bin/env bash
# Build a simZipf+ stream (Zipf over unique queryset, fresh ε-interpolation)
# and exact GT vs the DiskANN base. Does not overwrite official sim-100k-*.bin.
set -euo pipefail

cd "$(dirname "$0")/../.." || exit 1

DATASET="${DATASET:-spacev-small-test}"
DATA_TYPE="${DATA_TYPE:-int8}"
DISTRIBUTION="${DISTRIBUTION:-zipfian}"
SKEW="${SKEW:-0.99}"
EPSILON="${EPSILON:-0.01}"
N_STREAM="${N_STREAM:-100000}"
SEED="${SEED:-42}"
K="${K:-100}"
METRIC="${METRIC:-l2}"
SKIP_GT="${SKIP_GT:-1}"

OUT_DIR="${OUT_DIR:-data/${DATASET}}"
QUERYSET_BIN="${QUERYSET_BIN:-${OUT_DIR}/${DATASET}_query.bin}"
BASE_BIN="${BASE_BIN:-${OUT_DIR}/${DATASET}_base.bin}"
if [[ "${DISTRIBUTION}" == "uniform" ]]; then
  STREAM_TAG="simplus-100k-uniform"
else
  STREAM_TAG="simplus-100k-${SKEW}"
fi
QUERY_BIN="${OUT_DIR}/${DATASET}_query_${STREAM_TAG}.bin"
GT_BIN="${OUT_DIR}/${DATASET}_groundtruth_${STREAM_TAG}.bin"
STATS_JSON="${OUT_DIR}/${DATASET}_${STREAM_TAG}.stats.json"

if [[ ! -f "${QUERYSET_BIN}" ]]; then
  echo "Error: missing unique queryset ${QUERYSET_BIN}"
  echo "Run ./scripts/aker/prepare_search_workload.sh first."
  exit 1
fi
if [[ ! -f "${BASE_BIN}" ]]; then
  echo "Error: missing base ${BASE_BIN}"
  exit 1
fi

echo "=========================================="
echo "simZipf+ -> DiskANN"
echo "=========================================="
echo "Dataset: ${DATASET} (${DATA_TYPE})"
echo "Distribution: ${DISTRIBUTION}"
echo "Skew:    ${SKEW}"
echo "Epsilon: ${EPSILON}"
echo "N:       ${N_STREAM}"
echo "Query:   ${QUERY_BIN}"
echo "GT:      ${GT_BIN}"
echo ""

python3 scripts/aker/generate_simzipf_plus.py \
  --queryset "${QUERYSET_BIN}" \
  --output "${QUERY_BIN}" \
  --dtype "${DATA_TYPE}" \
  --distribution "${DISTRIBUTION}" \
  --skew "${SKEW}" \
  --epsilon "${EPSILON}" \
  --n-stream "${N_STREAM}" \
  --seed "${SEED}" \
  --stats-json "${STATS_JSON}"

if [[ "${SKIP_GT}" != "1" ]]; then
  COMPUTE_GT_BIN="./build/benchmarks/compute_groundtruth"
  if [[ ! -x "${COMPUTE_GT_BIN}" ]]; then
    echo "Error: ${COMPUTE_GT_BIN} not found. Build the project first."
    exit 1
  fi
  echo ""
  echo "Computing exact groundtruth vs ${BASE_BIN} ..."
  "${COMPUTE_GT_BIN}" "${BASE_BIN}" "${QUERY_BIN}" "${GT_BIN}" "${DATA_TYPE}" "${K}" "${METRIC}"
fi

echo ""
echo "Ready simZipf+ ${STREAM_TAG}:"
echo "  ${QUERY_BIN}"
echo "  ${GT_BIN}"
echo ""
echo "Run:"
echo "  QUERY_STREAM=${STREAM_TAG} ./scripts/qvcache/qvcache_benchmark_search_workload.sh"
echo "  QUERY_STREAM=${STREAM_TAG} ./scripts/aker/aker_benchmark_search_workload.sh"
