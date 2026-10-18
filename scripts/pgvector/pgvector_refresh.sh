#!/bin/bash
# Refresh insert/delete latency on pgvector alone.
# No QVCache and no Aker. Inserts are the same interpolated batch as the
# cache refresh benches (INSERT_FRAC, SEED), then the same random delete
# sample (DELETE_RATE, SEED+1). The query stream does not affect either batch.
# Mutates the PostgreSQL table.
set -euo pipefail

cd "$(dirname "$0")/../.." || exit 1

DATASET="${DATASET:-spacev-1m}"
DATA_TYPE="${DATA_TYPE:-int8}"
DATA_PATH="${DATA_PATH:-data/${DATASET}/base.bin}"
if [ ! -f "$DATA_PATH" ]; then
  echo "Error: missing base file $DATA_PATH"
  exit 1
fi

INSERT_FRAC="${INSERT_FRAC:-0.05}"
DELETE_RATE="${DELETE_RATE:-0.05}"
DELETE_ONLY="${DELETE_ONLY:-0}"
if [ "$DELETE_ONLY" != "0" ]; then
  INSERT_FRAC=0
fi
SEED="${SEED:-1}"
REPORT_INTERVAL="${REPORT_INTERVAL:-1000}"
REBUILD_INDEX="${REBUILD_INDEX:-1}"

TABLE_NAME="${TABLE_NAME:-spacev_1m}"
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

if [ ! -x "./build/benchmarks/pgvector_refresh" ]; then
    echo "Error: ./build/benchmarks/pgvector_refresh not found."
    echo "cmake --build build --target pgvector_refresh"
    exit 1
fi
echo "=========================================="
echo "pgvector refresh insert/delete - no cache"
echo "=========================================="
echo "Dataset: $DATASET ($DATA_TYPE)"
echo "insert_frac=$INSERT_FRAC delete_rate=$DELETE_RATE delete_only=$DELETE_ONLY seed=$SEED"
echo "PostgreSQL: $DB_HOST:$DB_PORT/$DB_NAME table=$TABLE_NAME"
echo "rebuild_index=$REBUILD_INDEX (required: each run inserts/deletes in the table)"
echo "=========================================="

if [ "$REBUILD_INDEX" != "0" ]; then
  echo "Rebuilding $TABLE_NAME from $DATA_PATH ..."
  DATASET="$DATASET" TABLE_NAME="$TABLE_NAME" DATA_PATH="$DATA_PATH" \
    ./scripts/pgvector/build_index.sh
fi

./build/benchmarks/pgvector_refresh \
  --data_type "$DATA_TYPE" \
  --data_path "$DATA_PATH" \
  --table_name "$TABLE_NAME" \
  --db_host "$DB_HOST" \
  --db_port "$DB_PORT" \
  --db_name "$DB_NAME" \
  --db_user "$DB_USER" \
  --db_password "$DB_PASSWORD" \
  --insert_frac "$INSERT_FRAC" \
  --delete_rate "$DELETE_RATE" \
  --seed "$SEED" \
  --report_interval "$REPORT_INTERVAL"
