#!/bin/bash
# Mixed search/insert/delete workload: Aker + pgvector.
# Same knobs as scripts/qvcache/qvcache_benchmark_update_pgvector.sh.
# Mutates the PostgreSQL table. Recreate the index afterwards if you need a clean search run.
set -euo pipefail

cd "$(dirname "$0")/../.." || exit 1

DATASET="${DATASET:-spacev-small-test}"
DATA_TYPE="${DATA_TYPE:-int8}"
QUERY_STREAM="${QUERY_STREAM:-sim-100k-0.99}"
DATA_PATH="data/$DATASET/${DATASET}_base.bin"
if [ "$QUERY_STREAM" = "query" ]; then
  QUERY_PATH="data/$DATASET/${DATASET}_query.bin"
else
  QUERY_PATH="data/$DATASET/${DATASET}_query_${QUERY_STREAM}.bin"
fi

K=10
SEARCH_THREADS=1
AKER_CONFIG="Aker/bootstrap/aker-standard.ini"
AKER_TOP_DELTA=5
AKER_POOL_SIZE=200000
METRIC="l2"

N_OPS="${N_OPS:-10000}"
INSERT_RATIO="${INSERT_RATIO:-0.05}"
DELETE_RATIO="${DELETE_RATIO:-0.05}"
REPORT_INTERVAL="${REPORT_INTERVAL:-100}"
SEED="${SEED:-1}"
CHECK_VISIBILITY="${CHECK_VISIBILITY:-1}"
AKER_PROCESS_LOG="${AKER_PROCESS_LOG:-1}"

TABLE_NAME="${TABLE_NAME:-spacev_1m}"
EF_SEARCH="${EF_SEARCH:-200}"
DB_HOST="${DB_HOST:-localhost}"
DB_PORT="${DB_PORT:-7000}"
DB_NAME="${DB_NAME:-postgres}"
DB_USER="${DB_USER:-postgres}"
DB_PASSWORD="${DB_PASSWORD:-postgres}"
if [ -f /.dockerenv ] || [ -n "${DOCKER_CONTAINER:-}" ]; then
    DB_HOST="postgres"
    if [ "$DB_PORT" = "7000" ]; then
        DB_PORT="5432"
    fi
fi

if [ ! -x "./build/benchmarks/aker_update_pgvector" ]; then
    echo "Error: ./build/benchmarks/aker_update_pgvector not found."
    echo "Build Aker (./scripts/aker/build.sh) then: cmake --build build --target aker_update_pgvector"
    exit 1
fi

echo "=========================================="
echo "Aker update-workload - pgvector"
echo "=========================================="
echo "Dataset: $DATASET ($DATA_TYPE)"
echo "n_ops=$N_OPS insert_ratio=$INSERT_RATIO delete_ratio=$DELETE_RATIO"
echo "PostgreSQL: $DB_HOST:$DB_PORT/$DB_NAME table=$TABLE_NAME"
echo "NOTE: this run mutates the table. Rebuild the pgvector index afterwards."
echo "=========================================="

export LD_LIBRARY_PATH="/app/Aker/build/lib:/usr/local/lib:/opt/boost_1_86/lib:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS="${SEARCH_THREADS}"
export OMP_MAX_ACTIVE_LEVELS=1

filter_aker_debug() {
    grep -v --line-buffered '\[debug\]' | grep -v --line-buffered '\[info\].*\[ANNSCache\]'
}

./build/benchmarks/aker_update_pgvector \
  --data_type "$DATA_TYPE" \
  --data_path "$DATA_PATH" \
  --query_path "$QUERY_PATH" \
  --aker_config "$AKER_CONFIG" \
  --table_name "$TABLE_NAME" \
  --db_host "$DB_HOST" \
  --db_port "$DB_PORT" \
  --db_name "$DB_NAME" \
  --db_user "$DB_USER" \
  --db_password "$DB_PASSWORD" \
  --hnsw_ef_search "$EF_SEARCH" \
  --K "$K" \
  --aker_top_delta "$AKER_TOP_DELTA" \
  --aker_pool_size "$AKER_POOL_SIZE" \
  --search_threads "$SEARCH_THREADS" \
  --metric "$METRIC" \
  --n_ops "$N_OPS" \
  --insert_ratio "$INSERT_RATIO" \
  --delete_ratio "$DELETE_RATIO" \
  --report_interval "$REPORT_INTERVAL" \
  --seed "$SEED" \
  --check_visibility "$CHECK_VISIBILITY" \
  --aker_process_log "$AKER_PROCESS_LOG" \
  > >(filter_aker_debug) 2> >(filter_aker_debug >&2)
