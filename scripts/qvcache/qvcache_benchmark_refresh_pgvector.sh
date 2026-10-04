#!/bin/bash
# Aker §5.4 refresh stress-test: QVCache + pgvector on SPACEV-1M.
# Warmup is search-only on simZipf 0.99. After optional inserts and deletes,
# exact top-k is loaded from LIVE_GT_PATH if that sidecar already exists
# (same insert/delete/k/seed/query stream), otherwise computed and
# written. DELETE_ONLY=1 skips the 5% insert batch (INSERT_FRAC=0).
# Search-only GT files are never overwritten. Client is one
# thread. Mutates the PostgreSQL table.
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

R=64
MEMORY_L="${MEMORY_L:-32}"
K=10
B=8
M=8
ALPHA=1.2
SEARCH_THREADS=1
BUILD_THREADS=8
DISK_INDEX_PREFIX="./index/${DATASET}/${DATASET}"
BEAMWIDTH=2
P=0.90
DEVIATION_FACTOR="${DEVIATION_FACTOR:-0.0}"
USE_REGIONAL_THETA=1
PCA_DIM=16
BUCKETS_PER_DIM=8
# θ ← α · θ on v's 16-D cell and L1-nearby cells. 1.0 = no penalty.
WRITE_THETA_DISCOUNT="${WRITE_THETA_DISCOUNT:-0.80}"
WRITE_L1_RADIUS="${WRITE_L1_RADIUS:-1}"
MEMORY_INDEX_MAX_POINTS="${MEMORY_INDEX_MAX_POINTS:-200000}"
MAX_REGIONS=1000000
N_ASYNC_INSERT_THREADS=4
LAZY_THETA_UPDATES=1
NUMBER_OF_MINI_INDEXES="${NUMBER_OF_MINI_INDEXES:-4}"
SEARCH_STRATEGY="${SEARCH_STRATEGY:-SEQUENTIAL}"
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

if [ ! -x "./build/benchmarks/qvcache_refresh_pgvector" ]; then
    echo "Error: ./build/benchmarks/qvcache_refresh_pgvector not found."
    echo "cmake --build build --target qvcache_refresh_pgvector"
    exit 1
fi
echo "=========================================="
echo "QVCache refresh stress-test - pgvector"
echo "=========================================="
echo "Dataset: $DATASET ($DATA_TYPE)  STREAM=${STREAM:-} SKEW=$SKEW  stream=$QUERY_STREAM"
echo "insert_frac=$INSERT_FRAC delete_rate=$DELETE_RATE delete_only=$DELETE_ONLY"
echo "n_warmup=$N_WARMUP n_eval=$N_EVAL pool=$MEMORY_INDEX_MAX_POINTS"
echo "memory_L=$MEMORY_L search_strategy=$SEARCH_STRATEGY"
echo "write_theta_discount=$WRITE_THETA_DISCOUNT write_l1_radius=$WRITE_L1_RADIUS"
echo "PostgreSQL: $DB_HOST:$DB_PORT/$DB_NAME table=$TABLE_NAME"
echo "rebuild_index=$REBUILD_INDEX (required: each run inserts/deletes in the table)"
echo "=========================================="

if [ "$REBUILD_INDEX" != "0" ]; then
  echo "Rebuilding $TABLE_NAME from $DATA_PATH ..."
  DATASET="$DATASET" TABLE_NAME="$TABLE_NAME" DATA_PATH="$DATA_PATH" \
    ./python/scripts/build_index/build_pgvector_index.sh
fi

export OMP_NUM_THREADS="${SEARCH_THREADS}"
export OMP_MAX_ACTIVE_LEVELS=1

./build/benchmarks/qvcache_refresh_pgvector \
  --data_type "$DATA_TYPE" \
  --data_path "$DATA_PATH" \
  --query_path "$QUERY_PATH" \
  --live_gt_path "$LIVE_GT_PATH" \
  --warmup_gt_path "$WARMUP_GT_PATH" \
  --disk_index_prefix "$DISK_INDEX_PREFIX" \
  --table_name "$TABLE_NAME" \
  --db_host "$DB_HOST" \
  --db_port "$DB_PORT" \
  --db_name "$DB_NAME" \
  --db_user "$DB_USER" \
  --db_password "$DB_PASSWORD" \
  --hnsw_ef_search "$EF_SEARCH" \
  --R "$R" \
  --memory_L "$MEMORY_L" \
  --K "$K" \
  --B "$B" \
  --M "$M" \
  --alpha "$ALPHA" \
  --build_threads "$BUILD_THREADS" \
  --search_threads "$SEARCH_THREADS" \
  --beamwidth "$BEAMWIDTH" \
  --p "$P" \
  --deviation_factor "$DEVIATION_FACTOR" \
  --memory_index_max_points "$MEMORY_INDEX_MAX_POINTS" \
  --use_regional_theta "$USE_REGIONAL_THETA" \
  --pca_dim "$PCA_DIM" \
  --buckets_per_dim "$BUCKETS_PER_DIM" \
  --max_regions "$MAX_REGIONS" \
  --n_async_insert_threads "$N_ASYNC_INSERT_THREADS" \
  --lazy_theta_updates "$LAZY_THETA_UPDATES" \
  --number_of_mini_indexes "$NUMBER_OF_MINI_INDEXES" \
  --search_strategy "$SEARCH_STRATEGY" \
  --metric "$METRIC" \
  --insert_frac "$INSERT_FRAC" \
  --delete_rate "$DELETE_RATE" \
  --n_warmup "$N_WARMUP" \
  --n_eval "$N_EVAL" \
  --report_interval "$REPORT_INTERVAL" \
  --gt_threads "$GT_THREADS" \
  --seed "$SEED" \
  --write_theta_discount "$WRITE_THETA_DISCOUNT" \
  --write_l1_radius "$WRITE_L1_RADIUS"
