#!/bin/bash
set -euo pipefail

# Script to build pgvector index from binary data files

# Change to project root (go up 3 levels from scripts/build_index/)
cd "$(dirname "$0")/../../.." || exit 1

# ============================================================================
# DATASET CONFIGURATION
# ============================================================================
# SPACEV-10M DiskANN bin (int8). Table name cannot contain '-'.
DATASET="${DATASET:-spacev-10m}"
DATA_PATH="${DATA_PATH:-data/$DATASET/${DATASET}_base.bin}"

# ============================================================================
# POSTGRESQL CONFIGURATION
# ============================================================================
# Host: from the qvcache container use service name "postgres" (auto-detected).
# From the host machine: DB_HOST=localhost DB_PORT=7000 (compose maps 7000->5432).
DB_HOST="${DB_HOST:-localhost}"
DB_PORT="${DB_PORT:-7000}"
DB_NAME="${DB_NAME:-postgres}"
DB_USER="${DB_USER:-postgres}"
DB_PASSWORD="${DB_PASSWORD:-postgres}"

TABLE_NAME="${TABLE_NAME:-spacev_10m}"

# Distance metric for the index (l2 or cosine)
METRIC="l2"  # Default: l2 

# Detect if running inside Docker and set PostgreSQL host accordingly
if [ -f /.dockerenv ] || [ -n "$DOCKER_CONTAINER" ]; then
    DB_HOST="postgres"
    # Compose maps host 7000 -> container 5432; inside the network use 5432.
    if [ "$DB_PORT" = "7000" ]; then
        DB_PORT="5432"
    fi
fi

# Options
RECREATE=true  # Set to true to recreate table even if it exists

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
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)/python"

# Build pgvector index
echo "Building pgvector index..."
echo "Data file: $DATA_PATH"
echo "Table name: $TABLE_NAME"
echo "Metric: $METRIC"
echo "PostgreSQL host: $DB_HOST"
echo "PostgreSQL port: $DB_PORT"
echo "Database: $DB_NAME"
echo "User: $DB_USER"

RECREATE_FLAG=""
if [ "$RECREATE" = true ]; then
    RECREATE_FLAG="--recreate"
    echo "Recreating table..."
fi

python3 python/scripts/build_index/build_pgvector_index.py \
    --data_path "$DATA_PATH" \
    --table_name "$TABLE_NAME" \
    --metric "$METRIC" \
    --db_host "$DB_HOST" \
    --db_port "$DB_PORT" \
    --db_name "$DB_NAME" \
    --db_user "$DB_USER" \
    --db_password "$DB_PASSWORD" \
    $RECREATE_FLAG

echo ""
echo "pgvector index build completed!"



