#!/usr/bin/env bash
# SPACEV-1M sweep: simZipf / simZipf+ / simZipf+2 at 0.99, 0.6, 0.3, uniform.
# QVCache and Aker run concurrently per (workload, skew). Order is skew-major:
# all three generators at 0.99, then 0.6, then 0.3, then uniform.
# Launch with: docker exec -d qvcache ./scripts/sim_workload_sweep.sh
set -u
cd "$(dirname "$0")/.." || exit 1
if [[ -d /app ]]; then
  cd /app || exit 1
fi

ROOT="${LOGDIR:-logs}"
STATUS="${ROOT}/sim_sweep.status"
mkdir -p "${ROOT}/simZipf" "${ROOT}/simZipf+" "${ROOT}/simZipf+2"

ts() { date -u +"%Y-%m-%dT%H:%M:%SZ"; }
status() { echo "$(ts) $*" | tee -a "${STATUS}"; }

{
  echo "=========================================="
  echo "simZipf / simZipf+ / simZipf+2 sweep"
  echo "started_utc=$(ts)"
  echo "pid=$$ hostname=$(hostname)"
  echo "logdir=${ROOT}"
  echo "=========================================="
} | tee "${STATUS}"

export LD_LIBRARY_PATH="/app/external/Aker/build/lib:/usr/local/lib:/opt/boost_1_86/lib:${LD_LIBRARY_PATH:-}"

run_pair() {
  local group="$1"
  local tag="$2"
  local stream="$3"
  local qlog="${ROOT}/${group}/qvcache_${tag}.log"
  local alog="${ROOT}/${group}/aker_${tag}.log"

  local qbin="data/spacev-small-test/spacev-small-test_query_${stream}.bin"
  local gtbin="data/spacev-small-test/spacev-small-test_groundtruth_${stream}.bin"
  if [[ ! -f "${qbin}" || ! -f "${gtbin}" ]]; then
    status "SKIP ${group} ${tag} missing ${qbin} or ${gtbin}"
    return 0
  fi

  status "START ${group} ${tag} stream=${stream}"
  status "CURRENT ${group} ${tag} qv=${qlog} aker=${alog}"
  set +e
  QUERY_STREAM="${stream}" ./scripts/qvcache/qvcache_benchmark_search_workload.sh >"${qlog}" 2>&1 &
  local qpid=$!
  QUERY_STREAM="${stream}" ./scripts/aker/aker_benchmark_search_workload.sh >"${alog}" 2>&1 &
  local apid=$!
  wait "${qpid}"
  local qrc=$?
  wait "${apid}"
  local arc=$?
  set -u
  status "END ${group} ${tag} qvcache=${qrc} aker=${arc}"
  if [[ ${qrc} -ne 0 || ${arc} -ne 0 ]]; then
    status "FAIL ${group} ${tag} (continuing)"
  fi
  return 0
}

for tag in 0.99 0.6 0.3 uniform; do
  run_pair "simZipf" "${tag}" "sim-100k-${tag}"
  run_pair "simZipf+" "${tag}" "simplus-100k-${tag}"
  run_pair "simZipf+2" "${tag}" "simplus2-100k-${tag}"
done

status "CURRENT none"
status "ALL_DONE utc=$(ts)"
