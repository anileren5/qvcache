#!/bin/bash
# Aker: sequential search-workload (Aker pgvector-bench workloada) on DiskANN + QVCache.
# Same query/base/gt files as scripts/aker/aker_benchmark_search_workload.sh.
set -e

cd "$(dirname "$0")/../.." || exit 1

# Aker: SPACEV-1M for now. Override DATASET=spacev-10m for paper-scale.
DATASET="${DATASET:-spacev-small-test}"
DATA_TYPE="${DATA_TYPE:-int8}"
# Aker: official simZipf stream (default). Use QUERY_STREAM=query for the unique example-queryset.
QUERY_STREAM="${QUERY_STREAM:-sim-100k-0.3}"
DATA_PATH="data/$DATASET/${DATASET}_base.bin"
if [ "$QUERY_STREAM" = "query" ]; then
  QUERY_PATH="data/$DATASET/${DATASET}_query.bin"
  GROUNDTRUTH_PATH="data/$DATASET/${DATASET}_groundtruth.bin"
else
  QUERY_PATH="data/$DATASET/${DATASET}_query_${QUERY_STREAM}.bin"
  GROUNDTRUTH_PATH="data/$DATASET/${DATASET}_groundtruth_${QUERY_STREAM}.bin"
fi

# Aker search-workload: sequential file order. Dummy window args are unused when
# REPORT_INTERVAL > 0; metrics are logged every N queries.
N_SPLIT=1
N_SPLIT_REPEAT=1
WINDOW_SIZE=1
N_REPEAT=1
STRIDE=1
N_ROUND=1
REPORT_INTERVAL=100

R=64
MEMORY_L=32
DISK_L=64
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
DEVIATION_FACTOR=0.00
SECTOR_LEN=4096
USE_REGIONAL_THETA=1
LEARN_PCA_FROM_QUERIES="${LEARN_PCA_FROM_QUERIES:-0}"
PCA_DIM=16
BUCKETS_PER_DIM=8
MEMORY_INDEX_MAX_POINTS=200000
MAX_REGIONS=1000000
N_ASYNC_INSERT_THREADS=4
LAZY_THETA_UPDATES=1
NUMBER_OF_MINI_INDEXES=4
MAX_SEARCH_THREADS="${MAX_SEARCH_THREADS:-32}"
SEARCH_STRATEGY="${SEARCH_STRATEGY:-SEQUENTIAL}"
METRIC="l2"

if [ ! -f "$QUERY_PATH" ] || [ ! -f "$DATA_PATH" ]; then
    echo "Error: missing DiskANN bins ($QUERY_PATH)."
    echo "SPACEV-10M: ./scripts/aker/prepare_paper_scale.sh"
    echo "1M simZipf: DATASET=spacev-small-test ./scripts/aker/prepare_sim_workload.sh"
    echo "1M simZipf+: DATASET=spacev-small-test ./scripts/aker/prepare_simzipf_plus.sh"
    echo "1M example queryset: ./scripts/aker/prepare_search_workload.sh"
    exit 1
fi

if [ ! -f "$GROUNDTRUTH_PATH" ]; then
    echo "Error: Groundtruth file not found: $GROUNDTRUTH_PATH"
    echo "Run ./scripts/aker/prepare_sim_workload.sh (or prepare_search_workload.sh)."
    exit 1
fi

echo "=========================================="
echo "QVCache search-workload - DiskANN"
echo "=========================================="
echo "Dataset: $DATASET ($DATA_TYPE)"
echo "Query stream: $QUERY_STREAM"
echo "Search strategy: $SEARCH_STRATEGY"
echo "Query file: $QUERY_PATH"
echo "Groundtruth file: $GROUNDTRUTH_PATH"
echo "Report interval: $REPORT_INTERVAL queries"
echo "=========================================="
echo ""

./build/benchmarks/qvcache_benchmark_diskann \
  --data_type "$DATA_TYPE" \
  --data_path "$DATA_PATH" \
  --query_path "$QUERY_PATH" \
  --groundtruth_path "$GROUNDTRUTH_PATH" \
  --disk_index_prefix "$DISK_INDEX_PREFIX" \
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
