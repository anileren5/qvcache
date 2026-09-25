#!/bin/bash
# Sequential search-workload on C++ QVCache with a native pgvector backend.
# Same query/base/gt files as scripts/qvcache/qvcache_benchmark_search_workload.sh.
set -euo pipefail

cd "$(dirname "$0")/../.." || exit 1

DATASET="${DATASET:-spacev-small-test}"
DATA_TYPE="${DATA_TYPE:-int8}"
QUERY_STREAM="${QUERY_STREAM:-sim-100k-0.99}"
DATA_PATH="data/$DATASET/${DATASET}_base.bin"
if [ "$QUERY_STREAM" = "query" ]; then
  QUERY_PATH="data/$DATASET/${DATASET}_query.bin"
  GROUNDTRUTH_PATH="data/$DATASET/${DATASET}_groundtruth.bin"
else
  QUERY_PATH="data/$DATASET/${DATASET}_query_${QUERY_STREAM}.bin"
  GROUNDTRUTH_PATH="data/$DATASET/${DATASET}_groundtruth_${QUERY_STREAM}.bin"
fi

N_SPLIT=1
N_SPLIT_REPEAT=1
WINDOW_SIZE=1
N_REPEAT=1
STRIDE=1
N_ROUND=1
REPORT_INTERVAL=100

R=64
MEMORY_L=16
DISK_L=32
K=10
B=8
M=8
ALPHA=1.2
SEARCH_THREADS=1
BUILD_THREADS=8
DISK_INDEX_PREFIX="./index/${DATASET}/${DATASET}"
DISK_INDEX_ALREADY_BUILT=1
BEAMWIDTH=2
USE_RECONSTRUCTED_VECTORS=0
P=0.90
DEVIATION_FACTOR="${DEVIATION_FACTOR:-0.0}"
SECTOR_LEN=4096
USE_REGIONAL_THETA=1
LEARN_PCA_FROM_QUERIES="${LEARN_PCA_FROM_QUERIES:-0}"
PCA_DIM=16
BUCKETS_PER_DIM=8
MEMORY_INDEX_MAX_POINTS=200000
MAX_REGIONS=1000000
N_ASYNC_INSERT_THREADS=4
LAZY_THETA_UPDATES=1
NUMBER_OF_MINI_INDEXES=8
SEARCH_MINI_INDEXES_IN_PARALLEL=false
MAX_SEARCH_THREADS=32
SEARCH_STRATEGY="SEQUENTIAL_LRU_STOP_FIRST_HIT"
METRIC="l2"

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

if [ ! -f "$QUERY_PATH" ] || [ ! -f "$DATA_PATH" ]; then
    echo "Error: missing DiskANN bins ($QUERY_PATH)."
    echo "1M simZipf: DATASET=spacev-small-test ./scripts/aker/prepare_sim_workload.sh"
    echo "1M simZipf+: DATASET=spacev-small-test ./scripts/aker/prepare_simzipf_plus.sh"
    exit 1
fi

if [ ! -f "$GROUNDTRUTH_PATH" ]; then
    echo "Error: Groundtruth file not found: $GROUNDTRUTH_PATH"
    exit 1
fi

if [ ! -x ./build/benchmarks/qvcache_benchmark_pgvector ]; then
    echo "Error: ./build/benchmarks/qvcache_benchmark_pgvector not found."
    echo "Rebuild with libpq-dev installed, then: cmake --build build --target qvcache_benchmark_pgvector"
    exit 1
fi

echo "=========================================="
echo "QVCache search-workload - pgvector (C++)"
echo "=========================================="
echo "Dataset: $DATASET ($DATA_TYPE)"
echo "Query stream: $QUERY_STREAM"
echo "Query file: $QUERY_PATH"
echo "Groundtruth file: $GROUNDTRUTH_PATH"
echo "PostgreSQL: $DB_HOST:$DB_PORT/$DB_NAME table=$TABLE_NAME"
echo "hnsw.ef_search: $EF_SEARCH"
echo "deviation_factor: $DEVIATION_FACTOR"
echo "learn_pca_from_queries: $LEARN_PCA_FROM_QUERIES"
echo "Report interval: $REPORT_INTERVAL queries"
echo "=========================================="
echo ""

./build/benchmarks/qvcache_benchmark_pgvector \
  --data_type "$DATA_TYPE" \
  --data_path "$DATA_PATH" \
  --query_path "$QUERY_PATH" \
  --groundtruth_path "$GROUNDTRUTH_PATH" \
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
  --disk_L "$DISK_L" \
  --K "$K" \
  --B "$B" \
  --M "$M" \
  --alpha "$ALPHA" \
  --search_threads "$SEARCH_THREADS" \
  --build_threads "$BUILD_THREADS" \
  --disk_index_already_built "$DISK_INDEX_ALREADY_BUILT" \
  --beamwidth "$BEAMWIDTH" \
  --use_reconstructed_vectors "$USE_RECONSTRUCTED_VECTORS" \
  --p "$P" \
  --deviation_factor "$DEVIATION_FACTOR" \
  --sector_len "$SECTOR_LEN" \
  --use_regional_theta "$USE_REGIONAL_THETA" \
  --learn_pca_from_queries "$LEARN_PCA_FROM_QUERIES" \
  --pca_dim "$PCA_DIM" \
  --buckets_per_dim "$BUCKETS_PER_DIM" \
  --memory_index_max_points "$MEMORY_INDEX_MAX_POINTS" \
  --max_regions "$MAX_REGIONS" \
  --n_splits "$N_SPLIT" \
  --n_split_repeat "$N_SPLIT_REPEAT" \
  --n_async_insert_threads "$N_ASYNC_INSERT_THREADS" \
  --lazy_theta_updates "$LAZY_THETA_UPDATES" \
  --number_of_mini_indexes "$NUMBER_OF_MINI_INDEXES" \
  --search_mini_indexes_in_parallel "$SEARCH_MINI_INDEXES_IN_PARALLEL" \
  --max_search_threads "$MAX_SEARCH_THREADS" \
  --search_strategy "$SEARCH_STRATEGY" \
  --metric "$METRIC" \
  --window_size "$WINDOW_SIZE" \
  --n_repeat "$N_REPEAT" \
  --stride "$STRIDE" \
  --n_round "$N_ROUND" \
  --report_interval "$REPORT_INTERVAL" &

BENCHMARK_PID=$!

MAX_RSS=0
while kill -0 "$BENCHMARK_PID" 2>/dev/null; do
    if [ -f "/proc/$BENCHMARK_PID/status" ]; then
        current_rss=$(grep "^VmHWM:" "/proc/$BENCHMARK_PID/status" 2>/dev/null | awk '{print $2}')
        if [ -n "$current_rss" ] && [ "$current_rss" -gt "$MAX_RSS" ]; then
            MAX_RSS=$current_rss
        fi
    fi
    sleep 0.01
done

wait $BENCHMARK_PID
BENCHMARK_EXIT_CODE=$?

if [ -f "/proc/$BENCHMARK_PID/status" ]; then
    final_rss=$(grep "^VmHWM:" "/proc/$BENCHMARK_PID/status" 2>/dev/null | awk '{print $2}')
    if [ -n "$final_rss" ] && [ "$final_rss" -gt "$MAX_RSS" ]; then
        MAX_RSS=$final_rss
    fi
fi

if [ $BENCHMARK_EXIT_CODE -ne 0 ]; then
    exit $BENCHMARK_EXIT_CODE
fi

echo ""
echo "=========================================="
echo "Memory Usage Statistics"
echo "=========================================="
if [ -n "$MAX_RSS" ] && [ "$MAX_RSS" -gt 0 ]; then
    MAX_RSS_MB=$((MAX_RSS / 1024))
    echo "Maximum resident set size (RSS): ${MAX_RSS} KB (${MAX_RSS_MB} MB)"
else
    echo "Warning: Could not determine maximum memory usage"
fi
