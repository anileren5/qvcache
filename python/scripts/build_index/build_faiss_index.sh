#!/bin/bash

# Script to build FAISS index from binary data files

# Change to project root (go up 3 levels from scripts/build_index/)
cd "$(dirname "$0")/../../.." || exit 1

# Configuration
DATASET="siftsmall"
DATA_PATH="data/$DATASET/${DATASET}_base.bin"
INDEX_PATH="./faiss_index.bin"  # Path to save FAISS index
INDEX_TYPE="HNSW"  # HNSW (approximate) or Flat (exact)

# HNSW parameters (only used if INDEX_TYPE="HNSW")
HNSW_M=32  # Number of bi-directional links
HNSW_EF_CONSTRUCTION=200  # ef_construction parameter

# Options
RECREATE=true  # Set to true to recreate index even if it exists

# Check if data file exists
if [ ! -f "$DATA_PATH" ]; then
    echo "Error: Data file not found: $DATA_PATH"
    exit 1
fi

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# Add python directory to PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)/python"

# Build FAISS index
echo "Building FAISS index..."
echo "Data file: $DATA_PATH"
echo "Index path: $INDEX_PATH"
echo "Index type: $INDEX_TYPE"

RECREATE_FLAG=""
if [ "$RECREATE" = true ]; then
    RECREATE_FLAG="--recreate"
    echo "Recreating index..."
fi

python3 python/scripts/build_index/build_faiss_index.py \
    --data_path "$DATA_PATH" \
    --index_path "$INDEX_PATH" \
    --index_type "$INDEX_TYPE" \
    --hnsw_m "$HNSW_M" \
    --hnsw_ef_construction "$HNSW_EF_CONSTRUCTION" \
    $RECREATE_FLAG

echo ""
echo "FAISS index build completed!"

