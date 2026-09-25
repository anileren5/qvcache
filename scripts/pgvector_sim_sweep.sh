#!/usr/bin/env bash
# Sequential pgvector search-workload sweep: simZipf + simZipf+ for QVCache and Aker.
# Run inside the qvcache container. Survives SSH drop when launched with:
#   docker exec -d qvcache bash /app/scripts/pgvector_sim_sweep.sh
set -u
cd /app || exit 1

LOGDIR="${LOGDIR:-/app/logs/pgvector_sweep}"
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
  echo "pgvector simZipf / simZipf+ sweep"
  echo "started_utc=$(ts)"
  echo "pid=$$ hostname=$(hostname)"
  echo "logdir=${LOGDIR}"
  echo "=========================================="
} | tee "${STATUS}"

export LD_LIBRARY_PATH="/app/Aker/build/lib:/usr/local/lib:/opt/boost_1_86/lib:${LD_LIBRARY_PATH:-}"

streams=(
  sim-100k-0.3
  sim-100k-0.6
  sim-100k-0.99
  simplus-100k-0.3
  simplus-100k-0.6
  simplus-100k-0.99
)

for stream in "${streams[@]}"; do
  q="data/spacev-small-test/spacev-small-test_query_${stream}.bin"
  gt="data/spacev-small-test/spacev-small-test_groundtruth_${stream}.bin"
  if [[ ! -f "${q}" || ! -f "${gt}" ]]; then
    status "SKIP ${stream} missing ${q} or ${gt}"
    continue
  fi
  run_step "qvcache_${stream}" \
    "QUERY_STREAM=${stream} ./scripts/qvcache/qvcache_benchmark_pgvector.sh"
  run_step "aker_${stream}" \
    "QUERY_STREAM=${stream} ./scripts/aker/aker_benchmark_pgvector.sh"
done

status "ALL_DONE"
