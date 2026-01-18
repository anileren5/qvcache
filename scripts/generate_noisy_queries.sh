#!/bin/bash

# Script to generate noisy queries using generate_noisy_queries.py
# This script wraps the Python script for easier command-line usage

set -e

# Change to project root
cd "$(dirname "$0")/.." || exit 1

# Default values (can be overridden by command-line arguments)
DATASET="siftsmall"
N_SPLIT="10"
N_SPLIT_REPEAT="20"
NOISE_RATIO="0.01"
RANDOM_SEED="42"
DATA_DIR="data"
DATA_TYPE="float"
K="100"
METRIC="l2"

# Parse arguments (optional - if provided, override defaults)
if [ $# -ge 1 ]; then
    DATASET="$1"
fi
if [ $# -ge 2 ]; then
    N_SPLIT="$2"
fi
if [ $# -ge 3 ]; then
    N_SPLIT_REPEAT="$3"
fi
if [ $# -ge 4 ]; then
    NOISE_RATIO="$4"
fi
if [ $# -ge 5 ]; then
    RANDOM_SEED="$5"
fi
if [ $# -ge 6 ]; then
    DATA_DIR="$6"
fi
if [ $# -ge 7 ]; then
    DATA_TYPE="$7"
fi
if [ $# -ge 8 ]; then
    K="$8"
fi
if [ $# -ge 9 ]; then
    METRIC="$9"
fi

# Check if Python script exists
PYTHON_SCRIPT="scripts/generate_noisy_queries.py"
if [ ! -f "$PYTHON_SCRIPT" ]; then
    echo "Error: Python script not found: $PYTHON_SCRIPT"
    exit 1
fi

# Check if query file exists
QUERY_FILE="$DATA_DIR/$DATASET/${DATASET}_query.bin"
if [ ! -f "$QUERY_FILE" ]; then
    echo "Error: Query file not found: $QUERY_FILE"
    exit 1
fi

echo "=========================================="
echo "Generating Noisy Queries"
echo "=========================================="
echo "Dataset: $DATASET"
echo "Query file: $QUERY_FILE"
echo "n_split: $N_SPLIT"
echo "n_split_repeat: $N_SPLIT_REPEAT"
echo "noise_ratio: $NOISE_RATIO"
echo "random_seed: $RANDOM_SEED"
echo "=========================================="
echo ""

# Run the Python script
python3 "$PYTHON_SCRIPT" \
  --dataset "$DATASET" \
  --n_split "$N_SPLIT" \
  --n_split_repeat "$N_SPLIT_REPEAT" \
  --noise_ratio "$NOISE_RATIO" \
  --random_seed "$RANDOM_SEED" \
  --data_dir "$DATA_DIR" \
  --dtype "$DATA_TYPE"

if [ $? -ne 0 ]; then
    echo ""
    echo "✗ Error: Failed to generate noisy queries"
    exit 1
fi

# Construct paths for compute_groundtruth
# Format noise_ratio to match Python script (remove trailing zeros)
NOISE_STR=$(echo "$NOISE_RATIO" | sed 's/\.0*$//;s/\.$//')
DATASET_DIR="$DATA_DIR/$DATASET"
BASE_FILE="$DATASET_DIR/${DATASET}_base.bin"
# Use the same naming convention as the query file
GENERATED_QUERY_FILE="$DATASET_DIR/${DATASET}_query_nsplit-${N_SPLIT}_nrepeat-${N_SPLIT_REPEAT}_noise-${NOISE_STR}.bin"
# Derive groundtruth filename from query filename by replacing "query" with "groundtruth"
OUTPUT_GROUNDTRUTH_FILE=$(echo "$GENERATED_QUERY_FILE" | sed 's/_query_/_groundtruth_/')

# Check if compute_groundtruth binary exists
COMPUTE_GT_BIN="./build/benchmarks/compute_groundtruth"
if [ ! -f "$COMPUTE_GT_BIN" ]; then
    echo ""
    echo "Warning: compute_groundtruth binary not found: $COMPUTE_GT_BIN"
    echo "Skipping groundtruth computation. Please build the project first."
    exit 0
fi

# Check if base file exists
if [ ! -f "$BASE_FILE" ]; then
    echo ""
    echo "Warning: Base file not found: $BASE_FILE"
    echo "Skipping groundtruth computation."
    exit 0
fi

# Check if generated query file exists
if [ ! -f "$GENERATED_QUERY_FILE" ]; then
    echo ""
    echo "Warning: Generated query file not found: $GENERATED_QUERY_FILE"
    echo "Skipping groundtruth computation."
    exit 0
fi

echo ""
echo "=========================================="
echo "Computing Groundtruth for Noisy Queries"
echo "=========================================="
echo "Base file: $BASE_FILE"
echo "Query file: $GENERATED_QUERY_FILE"
echo "Output file: $OUTPUT_GROUNDTRUTH_FILE"
echo "Data type: $DATA_TYPE"
echo "K: $K"
echo "Metric: $METRIC"
echo "=========================================="
echo ""

# Run compute_groundtruth
"$COMPUTE_GT_BIN" "$BASE_FILE" "$GENERATED_QUERY_FILE" "$OUTPUT_GROUNDTRUTH_FILE" "$DATA_TYPE" "$K" "$METRIC"

if [ $? -eq 0 ]; then
    echo ""
    echo "✓ Success! Noisy queries generated and groundtruth computed."
    echo "  Query file: $GENERATED_QUERY_FILE"
    echo "  Groundtruth file: $OUTPUT_GROUNDTRUTH_FILE"
else
    echo ""
    echo "✗ Error: Failed to compute groundtruth"
    exit 1
fi