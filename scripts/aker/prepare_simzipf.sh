#!/usr/bin/env bash
# Reconstruct Aker simZipf (frozen 50-groups) with Zipf or uniform over groups.
# Does not overwrite Hugging Face sim-100k-{0.3,0.6,0.99}.bin.
set -euo pipefail

cd "$(dirname "$0")/../.." || exit 1

DATASET="${DATASET:-spacev-small-test}"
DATA_TYPE="${DATA_TYPE:-int8}"
DISTRIBUTION="${DISTRIBUTION:-uniform}"
SKEW="${SKEW:-0.99}"
N_STREAM="${N_STREAM:-100000}"
GROUP_SIZE="${GROUP_SIZE:-50}"
LAMBDA_MAX="${LAMBDA_MAX:-0.5}"
SEED="${SEED:-42}"
K="${K:-100}"
METRIC="${METRIC:-l2}"
SKIP_GT="${SKIP_GT:-1}"

OUT_DIR="${OUT_DIR:-data/${DATASET}}"
QUERYSET_BIN="${QUERYSET_BIN:-${OUT_DIR}/${DATASET}_query.bin}"
BASE_BIN="${BASE_BIN:-${OUT_DIR}/${DATASET}_base.bin}"
if [[ "${DISTRIBUTION}" == "uniform" ]]; then
  STREAM_TAG="sim-100k-uniform"
else
  STREAM_TAG="sim-100k-${SKEW}"
  if [[ -f "${OUT_DIR}/${DATASET}_query_${STREAM_TAG}.bin" ]]; then
    echo "Refusing to overwrite official/existing ${STREAM_TAG}. Use a new tag."
    exit 1
  fi
fi

QUERY_BIN="${OUT_DIR}/${DATASET}_query_${STREAM_TAG}.bin"
GT_BIN="${OUT_DIR}/${DATASET}_groundtruth_${STREAM_TAG}.bin"
STATS_JSON="${OUT_DIR}/${DATASET}_${STREAM_TAG}.stats.json"

if [[ ! -f "${QUERYSET_BIN}" ]]; then
  echo "Error: missing unique queryset ${QUERYSET_BIN}"
  exit 1
fi

echo "=========================================="
echo "simZipf (frozen groups) -> DiskANN"
echo "=========================================="
echo "Distribution: ${DISTRIBUTION}"
echo "Query: ${QUERY_BIN}"
echo ""

python3 scripts/aker/generate_simzipf.py \
  --queryset "${QUERYSET_BIN}" \
  --output "${QUERY_BIN}" \
  --dtype "${DATA_TYPE}" \
  --distribution "${DISTRIBUTION}" \
  --skew "${SKEW}" \
  --n-stream "${N_STREAM}" \
  --group-size "${GROUP_SIZE}" \
  --lambda-max "${LAMBDA_MAX}" \
  --seed "${SEED}" \
  --stats-json "${STATS_JSON}"

if [[ "${SKIP_GT}" != "1" ]]; then
  COMPUTE_GT_BIN="./build/benchmarks/compute_groundtruth"
  if [[ ! -x "${COMPUTE_GT_BIN}" ]]; then
    echo "Error: ${COMPUTE_GT_BIN} not found."
    exit 1
  fi
  "${COMPUTE_GT_BIN}" "${BASE_BIN}" "${QUERY_BIN}" "${GT_BIN}" "${DATA_TYPE}" "${K}" "${METRIC}"
fi

echo ""
echo "Ready ${STREAM_TAG}:"
echo "  ${QUERY_BIN}"
echo "  QUERY_STREAM=${STREAM_TAG} ./scripts/qvcache/qvcache_benchmark_search_workload.sh"
