#!/bin/bash
# Script to start SPTAG server

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR" || exit 1

DATASET="siftsmall"
INDEX_NAME="MyIndex"
INDEX_FOLDER="data/${DATASET}/index"
CONFIG_FILE="sptag_service.ini"

echo "=========================================="
echo "Starting SPTAG Server"
echo "=========================================="

# Check if server executable exists
if [ ! -f "./Release/server" ]; then
    echo "Error: SPTAG server not found at ./Release/server"
    echo "Please build SPTAG first by running the build process"
    exit 1
fi

# Check if index folder exists
if [ ! -d "${INDEX_FOLDER}" ]; then
    echo "Error: Index folder not found: ${INDEX_FOLDER}"
    echo "Please build the index first by running: ./build_index_siftsmall.sh"
    exit 1
fi

# Create service config file if it doesn't exist
if [ ! -f "${CONFIG_FILE}" ]; then
    echo "Creating service config file: ${CONFIG_FILE}"
    cat > "${CONFIG_FILE}" << EOF
[Service]
ListenAddr=0.0.0.0
ListenPort=8000
ThreadNumber=8
SocketThreadNumber=8

[QueryConfig]
DefaultMaxResultNumber=10
DefaultSeparator=|

[Index]
List=${INDEX_NAME}

[Index_${INDEX_NAME}]
IndexFolder=${INDEX_FOLDER}
EOF
    echo "Config file created at ${CONFIG_FILE}"
fi

echo "Server configuration:"
echo "  Config file: ${CONFIG_FILE}"
echo "  Index folder: ${INDEX_FOLDER}"
echo "  Listening on: 0.0.0.0:8000"
echo "=========================================="
echo ""

# Start the server
exec ./Release/server -m socket -c "${CONFIG_FILE}"

