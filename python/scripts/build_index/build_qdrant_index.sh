#!/bin/bash

# Script to build Qdrant index from binary data files

# Change to project root (go up 3 levels from scripts/build_index/)
cd "$(dirname "$0")/../../.." || exit 1

# Configuration
DATASET="siftsmall"
DATA_PATH="data/$DATASET/${DATASET}_base.bin"
COLLECTION_NAME="vectors"  # Qdrant collection name

# Detect if running inside Docker and set Qdrant URL accordingly
if [ -f /.dockerenv ] || [ -n "$DOCKER_CONTAINER" ]; then
    QDRANT_URL="http://qdrant:6333"  # Docker internal network
else
    QDRANT_URL="http://localhost:6333"  # Local machine
fi

# Allow override via environment variable
if [ -n "$QDRANT_URL_ENV" ]; then
    QDRANT_URL="$QDRANT_URL_ENV"
fi

# Options
RECREATE=true  # Set to true to recreate collection even if it exists

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

# Build Qdrant index
echo "Building Qdrant index..."
echo "Data file: $DATA_PATH"
echo "Collection name: $COLLECTION_NAME"
echo "Qdrant URL: $QDRANT_URL"

RECREATE_FLAG=""
if [ "$RECREATE" = true ]; then
    RECREATE_FLAG="--recreate"
    echo "Recreating collection..."
fi

python3 python/scripts/build_index/build_qdrant_index.py \
    --data_path "$DATA_PATH" \
    --collection_name "$COLLECTION_NAME" \
    --qdrant_url "$QDRANT_URL" \
    $RECREATE_FLAG

echo ""
echo "Qdrant index build completed!"

