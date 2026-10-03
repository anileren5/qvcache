#!/bin/bash
# Aker §5.4 refresh stress-test: Aker + pgvector on SPACEV-1M.
# Warmup is search-only on simZipf 0.99. After optional inserts and deletes,
# exact top-k is loaded from LIVE_GT_PATH if that sidecar already exists
# (same insert/delete/k/seed/query stream), otherwise computed and
# written. DELETE_ONLY=1 skips the 5% insert batch (INSERT_FRAC=0).
# Search-only GT files are never overwritten. Mutates the
# PostgreSQL table.
set -euo pipefail

cd "$(dirname "$0")/../.." || exit 1

DATASET="${DATASET:-spacev-small-test}"
DATA_TYPE="${DATA_TYPE:-int8}"
# STREAM=simZipf | simZipf+ | simZipf+2  (optional SKEW, default 0.99)
# or QUERY_STREAM=sim-100k-0.99 / simplus-100k-0.99 / ...
# shellcheck source=../update/resolve_query_stream.sh
source "$(dirname "$0")/../update/resolve_query_stream.sh"
DATA_PATH="data/$DATASET/${DATASET}_base.bin"
if [ "$QUERY_STREAM" = "query" ]; then
  QUERY_PATH="data/$DATASET/${DATASET}_query.bin"
else
  QUERY_PATH="data/$DATASET/${DATASET}_query_${QUERY_STREAM}.bin"
fi
if [ ! -f "$QUERY_PATH" ]; then
  echo "Error: missing query file $QUERY_PATH"
  echo "Set STREAM=simZipf|simZipf+|simZipf+2 (optional SKEW=0.99) or QUERY_STREAM=<file tag>."
  exit 1
fi

K=10
SEARCH_THREADS=1
AKER_CONFIG="Aker/bootstrap/aker-standard.ini"
# Paper AK-D10: Δ=10. Pool matches QVCache MEMORY_INDEX_MAX_POINTS.
AKER_TOP_DELTA="${AKER_TOP_DELTA:-10}"
AKER_POOL_SIZE="${AKER_POOL_SIZE:-200000}"
METRIC="l2"

INSERT_FRAC="${INSERT_FRAC:-0.05}"
DELETE_RATE="${DELETE_RATE:-0.05}"
DELETE_ONLY="${DELETE_ONLY:-0}"
if [ "$DELETE_ONLY" != "0" ]; then
  INSERT_FRAC=0
fi
N_WARMUP="${N_WARMUP:-0}"
N_EVAL="${N_EVAL:-0}"
REPORT_INTERVAL="${REPORT_INTERVAL:-100}"
GT_THREADS="${GT_THREADS:-8}"
SEED="${SEED:-1}"
AKER_PROCESS_LOG="${AKER_PROCESS_LOG:-1}"
REBUILD_INDEX="${REBUILD_INDEX:-1}"
LIVE_GT_PATH="${LIVE_GT_PATH:-data/$DATASET/${DATASET}_groundtruth_${QUERY_STREAM}.refresh_ins${INSERT_FRAC}_del${DELETE_RATE}_k${K}_seed${SEED}.bin}"
WARMUP_GT_PATH="${WARMUP_GT_PATH:-data/$DATASET/${DATASET}_groundtruth_${QUERY_STREAM}.bin}"
if [ ! -f "$WARMUP_GT_PATH" ]; then
  echo "Error: missing search-only groundtruth $WARMUP_GT_PATH"
  echo "Warmup recall loads this file; it is never computed or overwritten."
  exit 1
fi

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

if [ ! -x "./build/benchmarks/aker_refresh_pgvector" ]; then
    echo "Error: ./build/benchmarks/aker_refresh_pgvector not found."
    echo "Build Aker (./scripts/aker/build.sh) then: cmake --build build --target aker_refresh_pgvector"
    exit 1
fi
echo "=========================================="
echo "Aker refresh stress-test - pgvector"
echo "=========================================="
echo "Dataset: $DATASET ($DATA_TYPE)  STREAM=${STREAM:-} SKEW=$SKEW  stream=$QUERY_STREAM"
echo "insert_frac=$INSERT_FRAC delete_rate=$DELETE_RATE delete_only=$DELETE_ONLY"
echo "n_warmup=$N_WARMUP n_eval=$N_EVAL pool=$AKER_POOL_SIZE delta=$AKER_TOP_DELTA"
echo "PostgreSQL: $DB_HOST:$DB_PORT/$DB_NAME table=$TABLE_NAME"
echo "rebuild_index=$REBUILD_INDEX (required: each run inserts/deletes in the table)"
echo "=========================================="

if [ "$REBUILD_INDEX" != "0" ]; then
  echo "Rebuilding $TABLE_NAME from $DATA_PATH ..."
  DATASET="$DATASET" TABLE_NAME="$TABLE_NAME" DATA_PATH="$DATA_PATH" \
    ./python/scripts/build_index/build_pgvector_index.sh
fi

export LD_LIBRARY_PATH="/app/Aker/build/lib:/usr/local/lib:/opt/boost_1_86/lib:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS="${SEARCH_THREADS}"
export OMP_MAX_ACTIVE_LEVELS=1

filter_aker_debug() {
    grep -v --line-buffered '\[debug\]' | grep -v --line-buffered '\[info\].*\[ANNSCache\]'
}

./build/benchmarks/aker_refresh_pgvector \
  --data_type "$DATA_TYPE" \
  --data_path "$DATA_PATH" \
  --query_path "$QUERY_PATH" \
  --live_gt_path "$LIVE_GT_PATH" \
  --warmup_gt_path "$WARMUP_GT_PATH" \
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
  --insert_frac "$INSERT_FRAC" \
  --delete_rate "$DELETE_RATE" \
  --n_warmup "$N_WARMUP" \
  --n_eval "$N_EVAL" \
  --report_interval "$REPORT_INTERVAL" \
  --gt_threads "$GT_THREADS" \
  --seed "$SEED" \
  --aker_process_log "$AKER_PROCESS_LOG" \
  > >(filter_aker_debug) 2> >(filter_aker_debug >&2)
