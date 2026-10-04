#!/usr/bin/env bash
# Aker: insert one SPACEV query, shrink interpolation noise until an approx hit.
set -euo pipefail
cd "$(dirname "$0")/../.." || exit 1

DATASET="${DATASET:-spacev-10m}"
DATA_TYPE="${DATA_TYPE:-int8}"
QUERY_STREAM="${QUERY_STREAM:-sim-100k-0.3}"
QUERY_INDEX="${QUERY_INDEX:-0}"
MU0="${MU0:-1.0}"
MU_DECAY="${MU_DECAY:-0.5}"
MAX_ITERS="${MAX_ITERS:-24}"
AKER_CONFIG="${AKER_CONFIG:-external/Aker/bootstrap/aker-standard.ini}"

DATA_PATH="data/${DATASET}/${DATASET}_base.bin"
if [[ "${QUERY_STREAM}" = "query" ]]; then
  QUERY_PATH="data/${DATASET}/${DATASET}_query.bin"
else
  QUERY_PATH="data/${DATASET}/${DATASET}_query_${QUERY_STREAM}.bin"
fi
DISK_INDEX_PREFIX="./index/${DATASET}/${DATASET}"

if [[ ! -x ./build/benchmarks/aker_approx_hit_probe ]]; then
  echo "Build first: cmake --build build --target aker_approx_hit_probe -j\"\$(nproc)\""
  exit 1
fi
if [[ ! -f "${QUERY_PATH}" ]]; then
  echo "Missing ${QUERY_PATH}"
  exit 1
fi
if [[ ! -f "${DISK_INDEX_PREFIX}_disk.index" ]]; then
  echo "Missing DiskANN index at ${DISK_INDEX_PREFIX}_disk.index"
  exit 1
fi

export LD_LIBRARY_PATH="/app/external/Aker/build/lib:/usr/local/lib:/opt/boost_1_86/lib:${LD_LIBRARY_PATH:-}"

./build/benchmarks/aker_approx_hit_probe \
  --data_type "${DATA_TYPE}" \
  --data_path "${DATA_PATH}" \
  --query_path "${QUERY_PATH}" \
  --disk_index_prefix "${DISK_INDEX_PREFIX}" \
  --aker_config "${AKER_CONFIG}" \
  --R 64 --disk_L 32 --K 10 --B 8 --M 8 \
  --build_threads 8 --beamwidth 2 \
  --disk_index_already_built 1 \
  --query_index "${QUERY_INDEX}" \
  --mu0 "${MU0}" \
  --mu_decay "${MU_DECAY}" \
  --max_iters "${MAX_ITERS}"
