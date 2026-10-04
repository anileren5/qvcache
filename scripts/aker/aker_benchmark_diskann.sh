#!/bin/bash

# Aker: windowed DiskANN benchmark using Aker as the result cache.
# Workload files and windowing match scripts/qvcache/qvcache_benchmark_diskann.sh.

set -e

# Change to project root
cd "$(dirname "$0")/../.." || exit 1

# Define variables
DATASET="sift"
DATA_TYPE="float"
DATA_PATH="data/$DATASET/${DATASET}_base.bin"

# Noisy query parameters
N_SPLIT=10
N_SPLIT_REPEAT=10
NOISE_RATIO=0.01

# Window parameters
WINDOW_SIZE=4
N_REPEAT=2
STRIDE=1
N_ROUND=1

# Construct query and groundtruth paths based on noisy query parameters
# Format noise_ratio to match Python script (remove trailing zeros)
NOISE_STR=$(echo "$NOISE_RATIO" | sed 's/\.0*$//;s/\.$//')
QUERY_PATH="data/$DATASET/${DATASET}_query_nsplit-${N_SPLIT}_nrepeat-${N_SPLIT_REPEAT}_noise-${NOISE_STR}.bin"
GROUNDTRUTH_PATH="data/$DATASET/${DATASET}_groundtruth_nsplit-${N_SPLIT}_nrepeat-${N_SPLIT_REPEAT}_noise-${NOISE_STR}.bin"

# DiskANN backend parameters (shared with the QVCache DiskANN script)
R=64
DISK_L=32
K=10
B=8
M=8
# Aker: match QVCache SEARCH_THREADS. Aker holds one spinlock for lookup (and
# FAISS on miss), so 24 threads make hits wait tens of ms even when insert is off-path.
SEARCH_THREADS=1
BUILD_THREADS=8
DISK_INDEX_PREFIX="./index/${DATASET}/${DATASET}"
DISK_INDEX_ALREADY_BUILT=1
BEAMWIDTH=2
SECTOR_LEN=4096
METRIC="l2"

# Aker cache parameters
AKER_CONFIG="external/Aker/bootstrap/aker-standard.ini"
AKER_TOP_DELTA=5
# Same result-vector budget as QVCache MEMORY_INDEX_MAX_POINTS.
AKER_POOL_SIZE=200000

# Validate window parameters
MIN_SPLIT_REPEAT=$(( 1 + (WINDOW_SIZE / STRIDE) * N_REPEAT * N_ROUND ))
if [ "$N_SPLIT_REPEAT" -lt "$MIN_SPLIT_REPEAT" ]; then
    echo "Error: n_split_repeat ($N_SPLIT_REPEAT) must be >= 1 + (window_size / stride) * n_repeat * n_round = $MIN_SPLIT_REPEAT"
    exit 1
fi

if [ ! -f "$QUERY_PATH" ]; then
    echo "Error: Query file not found: $QUERY_PATH"
    echo "Please run generate_noisy_queries.sh first to generate the query file."
    exit 1
fi

if [ ! -f "$GROUNDTRUTH_PATH" ]; then
    echo "Error: Groundtruth file not found: $GROUNDTRUTH_PATH"
    echo "Please run generate_noisy_queries.sh first to generate the groundtruth file."
    exit 1
fi

if [ ! -x "./build/benchmarks/aker_benchmark_diskann" ]; then
    echo "Error: ./build/benchmarks/aker_benchmark_diskann not found."
    echo "Build Aker (./scripts/aker/build.sh) then rebuild QVCache (./build.sh)."
    exit 1
fi

echo "=========================================="
echo "Windowed Aker Benchmark - DiskANN"
echo "=========================================="
echo "Dataset: $DATASET"
echo "Query file: $QUERY_PATH"
echo "Groundtruth file: $GROUNDTRUTH_PATH"
echo "Aker config: $AKER_CONFIG"
echo "Noise parameters: n_split=$N_SPLIT, n_repeat=$N_SPLIT_REPEAT, noise_ratio=$NOISE_RATIO"
echo "Window parameters: window_size=$WINDOW_SIZE, n_repeat=$N_REPEAT, stride=$STRIDE, n_round=$N_ROUND"
echo "=========================================="
echo ""

export LD_LIBRARY_PATH="/app/external/Aker/build/lib:/usr/local/lib:/opt/boost_1_86/lib:${LD_LIBRARY_PATH:-}"
# FAISS nested OpenMP: pin to SEARCH_THREADS so pgvector approx-hits match DiskANN.
export OMP_NUM_THREADS="${SEARCH_THREADS}"
export OMP_MAX_ACTIVE_LEVELS=1

# Aker: Boost.Log debug goes to stdout with the JSON; strip it live.
filter_aker_debug() {
    grep -v --line-buffered '\[debug\]' | grep -v --line-buffered '\[info\].*\[ANNSCache\]'
}

./build/benchmarks/aker_benchmark_diskann \
  --data_type "$DATA_TYPE" \
  --data_path "$DATA_PATH" \
  --query_path "$QUERY_PATH" \
  --groundtruth_path "$GROUNDTRUTH_PATH" \
  --disk_index_prefix "$DISK_INDEX_PREFIX" \
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
