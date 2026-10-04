#!/bin/bash
# Mixed search/insert/delete workload: QVCache + pgvector.
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

R=64
MEMORY_L=16
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
MEMORY_INDEX_MAX_POINTS=200000
MAX_REGIONS=1000000
N_ASYNC_INSERT_THREADS=4
LAZY_THETA_UPDATES=1
NUMBER_OF_MINI_INDEXES="${NUMBER_OF_MINI_INDEXES:-4}"
SEARCH_STRATEGY="${SEARCH_STRATEGY:-SEQUENTIAL}"
METRIC="l2"

N_OPS="${N_OPS:-10000}"
INSERT_RATIO="${INSERT_RATIO:-0.05}"
DELETE_RATIO="${DELETE_RATIO:-0.05}"
REPORT_INTERVAL="${REPORT_INTERVAL:-100}"
SEED="${SEED:-1}"
CHECK_VISIBILITY="${CHECK_VISIBILITY:-1}"

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

if [ ! -x "./build/benchmarks/qvcache_update_pgvector" ]; then
    echo "Error: ./build/benchmarks/qvcache_update_pgvector not found."
    echo "cmake --build build --target qvcache_update_pgvector"
    exit 1
fi

echo "=========================================="
echo "QVCache update-workload - pgvector"
echo "=========================================="
echo "Dataset: $DATASET ($DATA_TYPE)"
echo "n_ops=$N_OPS insert_ratio=$INSERT_RATIO delete_ratio=$DELETE_RATIO"
echo "PostgreSQL: $DB_HOST:$DB_PORT/$DB_NAME table=$TABLE_NAME"
echo "NOTE: this run mutates the table. Rebuild the pgvector index afterwards."
echo "=========================================="

export OMP_NUM_THREADS="${SEARCH_THREADS}"
export OMP_MAX_ACTIVE_LEVELS=1

./build/benchmarks/qvcache_update_pgvector \
  --data_type "$DATA_TYPE" \
  --data_path "$DATA_PATH" \
  --query_path "$QUERY_PATH" \
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
  --n_ops "$N_OPS" \
  --insert_ratio "$INSERT_RATIO" \
  --delete_ratio "$DELETE_RATIO" \
  --report_interval "$REPORT_INTERVAL" \
  --seed "$SEED" \
  --check_visibility "$CHECK_VISIBILITY"
