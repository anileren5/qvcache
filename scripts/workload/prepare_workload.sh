#!/usr/bin/env bash
# Prepare DiskANN query/base bins for Aker and QVCache search workloads.
#
# Usage:
#   ./scripts/workload/prepare_workload.sh base
#   ./scripts/workload/prepare_workload.sh simzipf
#   ./scripts/workload/prepare_workload.sh simzipf+
#   ./scripts/workload/prepare_workload.sh simzipf+2
#
# Optional: SKEW=0.99 DISTRIBUTION=zipfian DATASET=spacev-1m SKIP_GT=1
# SPACEV-10M official simZipf: DATASET=spacev-10m ./scripts/workload/prepare_workload.sh simzipf
set -euo pipefail

cd "$(dirname "$0")/../.." || exit 1

WORKLOAD="${1:-${WORKLOAD:-}}"
DATASET="${DATASET:-spacev-1m}"
if [[ "${DATASET}" == "sphere-10m" ]]; then
  DATA_TYPE="${DATA_TYPE:-float}"
else
  DATA_TYPE="${DATA_TYPE:-int8}"
fi
SKEW="${SKEW:-0.99}"
DISTRIBUTION="${DISTRIBUTION:-zipfian}"
N_STREAM="${N_STREAM:-100000}"
SEED="${SEED:-42}"
K="${K:-100}"
METRIC="${METRIC:-l2}"
EPSILON="${EPSILON:-0.01}"
LAMBDA_MAX="${LAMBDA_MAX:-0.5}"
GROUP_SIZE="${GROUP_SIZE:-50}"
HF_REPO="${HF_REPO:-https://huggingface.co/datasets/sjoon-oh/aker/resolve/main}"
OUT_DIR="${OUT_DIR:-data/${DATASET}}"
BUILD_THREADS="${BUILD_THREADS:-8}"
SKIP_INDEX="${SKIP_INDEX:-1}"
BUILD_INDEX="${BUILD_INDEX:-0}"

normalize_workload() {
  case "$1" in
    base|queryset|search) echo "base" ;;
    simZipf|simzipf|sim) echo "simzipf" ;;
    simZipf+|simzipf+|simplus) echo "simzipf+" ;;
    simZipf+2|simzipf+2|simplus2) echo "simzipf+2" ;;
    "") echo "" ;;
    *) echo "$1" ;;
  esac
}

WORKLOAD="$(normalize_workload "${WORKLOAD}")"
if [[ -z "${WORKLOAD}" ]]; then
  echo "Usage: $0 base|simzipf|simzipf+|simzipf+2" >&2
  exit 1
fi

if [[ "${SKEW}" == "uniform" ]]; then
  DISTRIBUTION="uniform"
  SKEW_TAG="uniform"
else
  SKEW_TAG="${SKEW}"
fi

# 100000 -> 100k, 1000000 -> 1m
stream_size_tag() {
  local n="${1}"
  if (( n % 1000000 == 0 )); then
    echo "$((n / 1000000))m"
  elif (( n % 1000 == 0 )); then
    echo "$((n / 1000))k"
  else
    echo "${n}"
  fi
}
STREAM_SIZE="$(stream_size_tag "${N_STREAM}")"

if [[ "${DATASET}" == "spacev-1m" ]]; then
  AKER_DATASET_DIR="${AKER_DATASET_DIR:-external/Aker/pgvector-bench/dataset/spacev-small-test}"
else
  AKER_DATASET_DIR="${AKER_DATASET_DIR:-external/Aker/pgvector-bench/dataset/${DATASET}}"
fi
BASE_BIN="${OUT_DIR}/base.bin"
QUERYSET_BIN="${OUT_DIR}/query.bin"
QUERYSET_GT="${OUT_DIR}/groundtruth.bin"
INDEX_PREFIX="./index/${DATASET}/${DATASET}"
COMPUTE_GT_BIN="./build/benchmarks/compute_groundtruth"

stream_dir() { echo "${OUT_DIR}/queries/${1}"; }
stream_query() { echo "${OUT_DIR}/queries/${1}/query.bin"; }
stream_gt() { echo "${OUT_DIR}/queries/${1}/groundtruth.bin"; }
stream_stats() { echo "${OUT_DIR}/queries/${1}/stats.json"; }

mkdir -p "${AKER_DATASET_DIR}" "${OUT_DIR}" "./index/${DATASET}"

download_hf() {
  local name="$1"
  local dest="$2"
  if [[ -f "${dest}" ]]; then
    echo "npy exists: ${dest}"
    return 0
  fi
  echo "Downloading ${name} ..."
  python3 - "${HF_REPO}/${name}" "${dest}" <<'PY'
import sys, urllib.request
url, dest = sys.argv[1], sys.argv[2]
print(f"GET {url}")
urllib.request.urlretrieve(url, dest)
print(f"Wrote {dest}")
PY
}

npy_to_bin() {
  local src="$1"
  local dest="$2"
  if [[ -f "${dest}" ]]; then
    echo "bin exists: ${dest}"
    return 0
  fi
  python3 scripts/workload/npy_to_diskann_bin.py --input "${src}" --output "${dest}" --dtype "${DATA_TYPE}"
}

maybe_gt() {
  local query_bin="$1"
  local gt_bin="$2"
  local skip="${SKIP_GT:-0}"
  if [[ "${skip}" == "1" ]]; then
    return 0
  fi
  if [[ ! -x "${COMPUTE_GT_BIN}" ]]; then
    echo "Error: ${COMPUTE_GT_BIN} not found. Build the project first." >&2
    exit 1
  fi
  if [[ -f "${gt_bin}" ]]; then
    echo "Groundtruth exists: ${gt_bin}"
    return 0
  fi
  echo "Computing exact groundtruth -> ${gt_bin}"
  "${COMPUTE_GT_BIN}" "${BASE_BIN}" "${query_bin}" "${gt_bin}" "${DATA_TYPE}" "${K}" "${METRIC}"
}

maybe_index() {
  if [[ "${SKIP_INDEX}" == "1" || "${BUILD_INDEX}" != "1" ]]; then
    return 0
  fi
  if [[ -f "${INDEX_PREFIX}_disk.index" || -f "./index/${DATASET}/${DATASET}_disk.index" ]]; then
    echo "DiskANN index already present under ./index/${DATASET}/"
    return 0
  fi
  echo "Building DiskANN index at ${INDEX_PREFIX} ..."
  ./build/benchmarks/diskann_build_index "${DATA_TYPE}" \
    --data_file "${BASE_BIN}" \
    --index_prefix_path "${INDEX_PREFIX}" \
    --R "${R:-64}" \
    --L "${L:-128}" \
    --B "${B:-8}" \
    --M "${M:-8}" \
    --T "${BUILD_THREADS}" \
    --dist_metric "${METRIC}" \
    --single_file_index 0 \
    --sector_len 4096
}

prepare_base_1m() {
  local base_glob="${AKER_DATASET_DIR}/spacev-1m-split.part_*.npy"
  local query_npy="${AKER_DATASET_DIR}/example-queryset.npy"
  if [[ ! -f "${BASE_BIN}" ]]; then
    python3 scripts/workload/npy_to_diskann_bin.py --input "${base_glob}" --output "${BASE_BIN}" --dtype "${DATA_TYPE}"
  else
    echo "Base bin exists: ${BASE_BIN}"
  fi
  if [[ ! -f "${QUERYSET_BIN}" ]]; then
    python3 scripts/workload/npy_to_diskann_bin.py --input "${query_npy}" --output "${QUERYSET_BIN}" --dtype "${DATA_TYPE}"
  else
    echo "Queryset bin exists: ${QUERYSET_BIN}"
  fi
  maybe_gt "${QUERYSET_BIN}" "${QUERYSET_GT}"
  maybe_index
}

prepare_simzipf_10m() {
  SKIP_GT="${SKIP_GT:-1}"
  local stream_tag="sim-100k-${SKEW_TAG}"
  if [[ "${DISTRIBUTION}" == "uniform" ]]; then
    echo "SPACEV-10M official files are Zipf streams. Use SKEW=0.3|0.6|0.99." >&2
    exit 1
  fi
  download_hf "spacev-10m.npy" "${AKER_DATASET_DIR}/spacev-10m.npy"
  download_hf "spacev-sim-100k-${SKEW_TAG}.npy" "${AKER_DATASET_DIR}/spacev-sim-100k-${SKEW_TAG}.npy"
  npy_to_bin "${AKER_DATASET_DIR}/spacev-10m.npy" "${BASE_BIN}"
  mkdir -p "$(stream_dir "${stream_tag}")"
  npy_to_bin "${AKER_DATASET_DIR}/spacev-sim-100k-${SKEW_TAG}.npy" "$(stream_query "${stream_tag}")"
  maybe_gt "$(stream_query "${stream_tag}")" "$(stream_gt "${stream_tag}")"
  echo ""
  echo "Ready ${stream_tag}:"
  echo "  QUERY_STREAM=${stream_tag} ./scripts/aker/diskann_search.sh"
  echo "  QUERY_STREAM=${stream_tag} ./scripts/qvcache/diskann_search.sh"
}

prepare_simzipf_official() {
  local stream_tag="sim-100k-${SKEW_TAG}"
  local npy_name="spacev-sim-100k-${SKEW_TAG}.npy"
  if [[ ! -f "${BASE_BIN}" ]]; then
    echo "Error: missing ${BASE_BIN}. Run: ./scripts/workload/prepare_workload.sh base" >&2
    exit 1
  fi
  if [[ "${DISTRIBUTION}" == "uniform" ]]; then
    echo "Official Hugging Face simZipf files are Zipf streams. Use SKEW=0.3|0.6|0.99," >&2
    echo "or DISTRIBUTION=zipfian. For a generated uniform stream, use generate mode." >&2
    exit 1
  fi
  download_hf "${npy_name}" "${AKER_DATASET_DIR}/${npy_name}"
  mkdir -p "$(stream_dir "${stream_tag}")"
  npy_to_bin "${AKER_DATASET_DIR}/${npy_name}" "$(stream_query "${stream_tag}")"
  maybe_gt "$(stream_query "${stream_tag}")" "$(stream_gt "${stream_tag}")"
  echo ""
  echo "Ready ${stream_tag}:"
  echo "  QUERY_STREAM=${stream_tag} ./scripts/aker/diskann_search.sh"
  echo "  QUERY_STREAM=${stream_tag} ./scripts/qvcache/diskann_search.sh"
}

ensure_queryset() {
  if [[ ! -f "${QUERYSET_BIN}" || ! -f "${BASE_BIN}" ]]; then
    echo "Error: missing ${QUERYSET_BIN} or ${BASE_BIN}." >&2
    echo "Run: ./scripts/workload/prepare_workload.sh base" >&2
    exit 1
  fi
}

prepare_generated() {
  local kind="$1"
  local stream_tag=""
  local skip_default=""
  ensure_queryset
  case "${kind}" in
    simzipf)
      if [[ "${DISTRIBUTION}" == "uniform" ]]; then
        stream_tag="sim-${STREAM_SIZE}-uniform"
      else
        stream_tag="sim-${STREAM_SIZE}-${SKEW_TAG}-gen"
      fi
      SKIP_GT="${SKIP_GT:-1}"
      mkdir -p "$(stream_dir "${stream_tag}")"
      python3 scripts/workload/generate_simzipf.py \
        --queryset "${QUERYSET_BIN}" \
        --output "$(stream_query "${stream_tag}")" \
        --dtype "${DATA_TYPE}" \
        --distribution "${DISTRIBUTION}" \
        --skew "${SKEW}" \
        --n-stream "${N_STREAM}" \
        --group-size "${GROUP_SIZE}" \
        --lambda-max "${LAMBDA_MAX}" \
        --seed "${SEED}" \
        --stats-json "$(stream_stats "${stream_tag}")"
      ;;
    simzipf+)
      # Fixed interpolation weight ε (default 0.01).
      local lam_tag="lam${EPSILON}"
      if [[ "${DISTRIBUTION}" == "uniform" ]]; then
        stream_tag="simplus-${STREAM_SIZE}-uniform-${lam_tag}"
      else
        stream_tag="simplus-${STREAM_SIZE}-${SKEW_TAG}-${lam_tag}"
      fi
      SKIP_GT="${SKIP_GT:-1}"
      mkdir -p "$(stream_dir "${stream_tag}")"
      python3 scripts/workload/generate_simzipf_plus.py \
        --queryset "${QUERYSET_BIN}" \
        --output "$(stream_query "${stream_tag}")" \
        --dtype "${DATA_TYPE}" \
        --distribution "${DISTRIBUTION}" \
        --skew "${SKEW}" \
        --epsilon "${EPSILON}" \
        --n-stream "${N_STREAM}" \
        --seed "${SEED}" \
        --stats-json "$(stream_stats "${stream_tag}")"
      ;;
    simzipf+2)
      # Per-access ε ~ Unif(0, lambda_max) (default 0.5). Keep the simplus prefix.
      local lam_tag="lamU0-${LAMBDA_MAX}"
      if [[ "${DISTRIBUTION}" == "uniform" ]]; then
        stream_tag="simplus-${STREAM_SIZE}-uniform-${lam_tag}"
      else
        stream_tag="simplus-${STREAM_SIZE}-${SKEW_TAG}-${lam_tag}"
      fi
      SKIP_GT="${SKIP_GT:-0}"
      mkdir -p "$(stream_dir "${stream_tag}")"
      python3 scripts/workload/generate_simzipf_plus.py \
        --queryset "${QUERYSET_BIN}" \
        --output "$(stream_query "${stream_tag}")" \
        --dtype "${DATA_TYPE}" \
        --distribution "${DISTRIBUTION}" \
        --skew "${SKEW}" \
        --lambda-max "${LAMBDA_MAX}" \
        --n-stream "${N_STREAM}" \
        --seed "${SEED}" \
        --stats-json "$(stream_stats "${stream_tag}")"
      ;;
  esac
  maybe_gt "$(stream_query "${stream_tag}")" "$(stream_gt "${stream_tag}")"
  echo ""
  echo "Ready ${stream_tag}:"
  echo "  QUERY_STREAM=${stream_tag} ./scripts/aker/diskann_search.sh"
  echo "  QUERY_STREAM=${stream_tag} ./scripts/qvcache/diskann_search.sh"
}

prepare_sphere_simzipf() {
  local skew dest stream_tag
  for skew in 0.3 0.6 0.99; do
    stream_tag="sim-100k-${skew}"
    dest="$(stream_dir "${stream_tag}")"
    mkdir -p "${OUT_DIR}/raw" "${dest}"
    download_hf "triviaqa-sim-100k-${skew}.npy" "${OUT_DIR}/raw/triviaqa-sim-100k-${skew}.npy"
    download_hf "triviaqa-sim-100k-${skew}.pkl" "${OUT_DIR}/raw/triviaqa-sim-100k-${skew}.pkl"
    if [[ -f "$(stream_query "${stream_tag}")" && -f "$(stream_gt "${stream_tag}")" ]]; then
      echo "bins exist: ${dest}"
      continue
    fi
    python3 scripts/workload/import_official_triviaqa.py \
      --npy "${OUT_DIR}/raw/triviaqa-sim-100k-${skew}.npy" \
      --pkl "${OUT_DIR}/raw/triviaqa-sim-100k-${skew}.pkl" \
      --out-dir "${dest}" \
      --skew "${skew}"
  done
  echo ""
  echo "Official TriviaQA simZipf streams are under ${OUT_DIR}/queries/."
  echo "Search needs SPHERE-10M base.bin and DATA_TYPE=float:"
  echo "  DATA_TYPE=float DATASET=sphere-10m QUERY_STREAM=sim-100k-0.99 ./scripts/aker/diskann_search.sh"
}

prepare_sphere_simplus() {
  local skew dest src_bin src_npy stream_tag found=0
  for skew in 0.3 0.6 0.99; do
    stream_tag="simplus-100k-${skew}"
    dest="$(stream_dir "${stream_tag}")"
    src_bin="${OUT_DIR}/raw/${stream_tag}/query.bin"
    src_npy="${OUT_DIR}/raw/${stream_tag}/query.npy"
    mkdir -p "${dest}"
    if [[ -f "$(stream_query "${stream_tag}")" ]]; then
      echo "bin exists: $(stream_query "${stream_tag}")"
      found=1
      continue
    fi
    if [[ -f "${src_bin}" ]]; then
      cp "${src_bin}" "$(stream_query "${stream_tag}")"
      if [[ -f "${OUT_DIR}/raw/${stream_tag}/stats.json" ]]; then
        cp "${OUT_DIR}/raw/${stream_tag}/stats.json" "$(stream_stats "${stream_tag}")"
      fi
      found=1
      continue
    fi
    if [[ -f "${src_npy}" ]]; then
      python3 scripts/workload/npy_to_diskann_bin.py \
        --input "${src_npy}" --output "$(stream_query "${stream_tag}")" --dtype float
      found=1
      continue
    fi
    echo "Missing ${stream_tag}. Build it on Clariden with scripts/workload/run_triviaqa_simzipf_plus.sbatch" >&2
  done
  if [[ "${found}" -eq 0 ]]; then
    exit 1
  fi
  echo ""
  echo "TriviaQA simZipf+ uses a different paraphrase each time a Zipf group is chosen."
  echo "  DATA_TYPE=float DATASET=sphere-10m QUERY_STREAM=simplus-100k-0.99 ./scripts/aker/diskann_search.sh"
}

echo "=========================================="
echo "Prepare workload: ${WORKLOAD}"
echo "=========================================="
echo "Dataset: ${DATASET} (${DATA_TYPE})"
echo ""

case "${WORKLOAD}" in
  base)
    if [[ "${DATASET}" == "spacev-10m" ]]; then
      echo "SPACEV-10M base is prepared with: DATASET=spacev-10m $0 simzipf" >&2
      exit 1
    fi
    prepare_base_1m
    echo ""
    echo "Ready base bins under ${OUT_DIR}"
    ;;
  simzipf)
    if [[ "${DATASET}" == "sphere-10m" ]]; then
      prepare_sphere_simzipf
    elif [[ "${DATASET}" == "spacev-10m" ]]; then
      prepare_simzipf_10m
    elif [[ "${GENERATE:-0}" == "1" ]]; then
      prepare_generated simzipf
    else
      prepare_simzipf_official
    fi
    ;;
  simzipf+)
    if [[ "${DATASET}" == "sphere-10m" ]]; then
      prepare_sphere_simplus
    else
      prepare_generated simzipf+
    fi
    ;;
  simzipf+2)
    if [[ "${DATASET}" == "sphere-10m" ]]; then
      echo "TriviaQA simZipf+ has no interpolation weight. Use: DATASET=sphere-10m $0 simzipf+" >&2
      exit 1
    fi
    prepare_generated simzipf+2
    ;;
  *)
    echo "Unknown workload '${WORKLOAD}'. Use base, simzipf, simzipf+, or simzipf+2." >&2
    exit 1
    ;;
esac
