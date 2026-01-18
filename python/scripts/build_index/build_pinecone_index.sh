#!/bin/bash

# Script to build Pinecone index from binary data files

# Change to project root (go up 3 levels from scripts/build_index/)
cd "$(dirname "$0")/../../.." || exit 1

# ============================================================================
# DATASET CONFIGURATION
# ============================================================================
DATASET="siftsmall"
DATA_PATH="data/$DATASET/${DATASET}_base.bin"

# ============================================================================
# CLOUD PINECONE CONFIGURATION - UPDATE THESE VALUES FOR YOUR SETUP
# ============================================================================
# Your Pinecone API Key (starts with "pcsk_")
# Get this from your Pinecone dashboard: https://app.pinecone.io/
# You can also set this via PINECONE_API_KEY environment variable
PINECONE_API_KEY="${PINECONE_API_KEY:-YOUR_PINECONE_API_KEY_HERE}"

# Your Pinecone region/environment (e.g., "us-east-1", "us-west-2", "eu-west-1")
# Check your Pinecone dashboard for the correct region
PINECONE_ENVIRONMENT="us-east-1"

# Your Pinecone index name (check your Pinecone dashboard)
INDEX_NAME=$DATASET 

# Not needed for cloud Pinecone (only used for local Docker setup)
PINECONE_HOST=""

# Validate API key is set
if [ "$PINECONE_API_KEY" = "YOUR_PINECONE_API_KEY_HERE" ] || [ -z "$PINECONE_API_KEY" ]; then
    echo "ERROR: Pinecone API key is not set!"
    echo "Please update PINECONE_API_KEY in this script with your actual API key from https://app.pinecone.io/"
    exit 1
fi

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

# Build Pinecone index
echo "Building Pinecone index..."
echo "Data file: $DATA_PATH"
echo "Index name: $INDEX_NAME"
echo "API key: ${PINECONE_API_KEY:0:10}..."  # Show first 10 chars only

RECREATE_FLAG=""
if [ "$RECREATE" = true ]; then
    RECREATE_FLAG="--recreate"
    echo "Recreating index..."
fi

ENV_FLAG=""
if [ -n "$PINECONE_ENVIRONMENT" ]; then
    ENV_FLAG="--environment $PINECONE_ENVIRONMENT"
fi

HOST_FLAG=""
if [ -n "$PINECONE_HOST" ]; then
    HOST_FLAG="--host $PINECONE_HOST"
fi

python3 python/scripts/build_index/build_pinecone_index.py \
    --data_path "$DATA_PATH" \
    --index_name "$INDEX_NAME" \
    --api_key "$PINECONE_API_KEY" \
    $ENV_FLAG \
    $HOST_FLAG \
    $RECREATE_FLAG

echo ""
echo "Pinecone index build completed!"


