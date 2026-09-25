#!/usr/bin/env bash
# Sequential overnight job: SPACEV-1M simZipf (0.3/0.6/0.99) + windowed SIFT
# for QVCache and Aker, then SPACEV-10M exact GT for 0.3 and 0.6.
# Survives SSH drop when launched with: docker exec -d qvcache ...
set -u
cd /app || exit 1

LOGDIR="${LOGDIR:-/app/logs/overnight}"
STATUS="${LOGDIR}/status.log"
mkdir -p "${LOGDIR}"

ts() { date -u +"%Y-%m-%dT%H:%M:%SZ"; }

status() {
  echo "$(ts) $*" | tee -a "${STATUS}"
}

run_step() {
  local name="$1"
  shift
  local logfile="${LOGDIR}/${name}.log"
  status "START ${name}"
  status "CURRENT ${name} log=${logfile}"
  set +e
  bash -lc "$*" >"${logfile}" 2>&1
  local rc=$?
  status "END ${name} exit=${rc}"
  if [[ ${rc} -ne 0 ]]; then
    status "FAIL ${name} (continuing)"
  fi
  return 0
}

{
  echo "=========================================="
  echo "Overnight cluster job"
  echo "started_utc=$(ts)"
  echo "pid=$$ hostname=$(hostname)"
  echo "logdir=${LOGDIR}"
  echo "=========================================="
} | tee "${STATUS}"

export LD_LIBRARY_PATH="/app/Aker/build/lib:/usr/local/lib:/opt/boost_1_86/lib:${LD_LIBRARY_PATH:-}"

# SPACEV-10M sim-100k-0.6 query bin (npy may still need a download).
if [[ ! -f data/spacev-10m/spacev-10m_query_sim-100k-0.6.bin ]]; then
  status "START prepare_spacev10m_query_0.6"
  set +e
  SKEW=0.6 SKIP_GT=1 SKIP_INDEX=1 ./scripts/aker/prepare_paper_scale.sh \
    >"${LOGDIR}/prepare_spacev10m_query_0.6.log" 2>&1
  rc=$?
  status "END prepare_spacev10m_query_0.6 exit=${rc}"
fi

# --- SPACEV-1M search-workload (existing GT) ---
for skew in 0.3 0.6 0.99; do
  stream="sim-100k-${skew}"
  tag="${skew}"
  run_step "qvcache_sim_${tag}" \
    "QUERY_STREAM=${stream} ./scripts/qvcache/qvcache_benchmark_search_workload.sh"
  run_step "aker_sim_${tag}" \
    "QUERY_STREAM=${stream} ./scripts/aker/aker_benchmark_search_workload.sh"
done

# --- Windowed SIFT ---
run_step "qvcache_windowed_sift" "./scripts/qvcache/qvcache_benchmark_diskann.sh"
run_step "aker_windowed_sift" "./scripts/aker/aker_benchmark_diskann.sh"

# --- SPACEV-10M exact GT (long) ---
for skew in 0.3 0.6; do
  stream="sim-100k-${skew}"
  q="data/spacev-10m/spacev-10m_query_${stream}.bin"
  gt="data/spacev-10m/spacev-10m_groundtruth_${stream}.bin"
  if [[ -f "${gt}" ]]; then
    status "SKIP spacev10m_gt_${skew} already exists ${gt}"
    continue
  fi
  if [[ ! -f "${q}" ]]; then
    status "SKIP spacev10m_gt_${skew} missing query bin ${q}"
    continue
  fi
  run_step "spacev10m_gt_${skew}" \
    "./build/benchmarks/compute_groundtruth data/spacev-10m/spacev-10m_base.bin ${q} ${gt} int8 100 l2"
done

status "CURRENT none"
status "ALL_DONE utc=$(ts)"
echo "$(ts) ALL_DONE" >>"${LOGDIR}/supervisor.out"
