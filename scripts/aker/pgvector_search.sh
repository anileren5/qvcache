#!/bin/bash
# Sequential search-workload on Aker with a native pgvector backend.
# Same query/base/gt files as scripts/aker/diskann_search.sh.
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
DISK_L=32
K=10
B=8
M=8
SEARCH_THREADS=1
BUILD_THREADS=8
DISK_INDEX_PREFIX="./index/${DATASET}/${DATASET}"
DISK_INDEX_ALREADY_BUILT=1
BEAMWIDTH=2
SECTOR_LEN=4096
METRIC="l2"

AKER_CONFIG="external/Aker/bootstrap/aker-standard.ini"
AKER_TOP_DELTA=5
# Same result-vector budget as QVCache MEMORY_INDEX_MAX_POINTS.
AKER_POOL_SIZE=200000

TABLE_NAME="${TABLE_NAME:-spacev_1m}"
EF_SEARCH="${EF_SEARCH:-400}"
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
    echo "Prepare: ./scripts/workload/prepare_workload.sh simzipf   # or simzipf+ / simzipf+2 / base"
    exit 1
fi

if [ ! -f "$GROUNDTRUTH_PATH" ]; then
    echo "Error: Groundtruth file not found: $GROUNDTRUTH_PATH"
    exit 1
fi

if [ ! -x "./build/benchmarks/aker_pgvector_search" ]; then
    echo "Error: ./build/benchmarks/aker_pgvector_search not found."
    echo "Build Aker (./scripts/aker/build.sh) then: cmake --build build --target aker_pgvector_search"
    exit 1
fi

echo "=========================================="
echo "Aker search-workload - pgvector (C++)"
echo "=========================================="
echo "Dataset: $DATASET ($DATA_TYPE)"
echo "Query stream: $QUERY_STREAM"
echo "Query file: $QUERY_PATH"
echo "Groundtruth file: $GROUNDTRUTH_PATH"
echo "PostgreSQL: $DB_HOST:$DB_PORT/$DB_NAME table=$TABLE_NAME"
echo "hnsw.ef_search: $EF_SEARCH"
echo "Aker config: $AKER_CONFIG"
echo "Report interval: $REPORT_INTERVAL queries"
echo "=========================================="
echo ""

export LD_LIBRARY_PATH="/app/external/Aker/build/lib:/usr/local/lib:/opt/boost_1_86/lib:${LD_LIBRARY_PATH:-}"
# FAISS nested OpenMP: pin to SEARCH_THREADS so pgvector approx-hits match DiskANN.
export OMP_NUM_THREADS="${SEARCH_THREADS}"
export OMP_MAX_ACTIVE_LEVELS=1

filter_aker_debug() {
    grep -v --line-buffered '\[debug\]' | grep -v --line-buffered '\[info\].*\[ANNSCache\]'
}

./build/benchmarks/aker_pgvector_search \
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
  --disk_L "$DISK_L" \
  --K "$K" \
  --B "$B" \
  --M "$M" \
  --search_threads "$SEARCH_THREADS" \
  --build_threads "$BUILD_THREADS" \
  --disk_index_already_built "$DISK_INDEX_ALREADY_BUILT" \
  --beamwidth "$BEAMWIDTH" \
  --sector_len "$SECTOR_LEN" \
  --metric "$METRIC" \
  --aker_config "$AKER_CONFIG" \
  --aker_pool_size "$AKER_POOL_SIZE" \
  --aker_top_delta "$AKER_TOP_DELTA" \
  --n_splits "$N_SPLIT" \
  --n_split_repeat "$N_SPLIT_REPEAT" \
  --window_size "$WINDOW_SIZE" \
  --n_repeat "$N_REPEAT" \
  --stride "$STRIDE" \
  --n_round "$N_ROUND" \
  --report_interval "$REPORT_INTERVAL" \
  > >(filter_aker_debug) 2> >(filter_aker_debug >&2) &

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
