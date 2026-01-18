#!/bin/bash
# Script to build SPTAG index for siftsmall dataset

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR" || exit 1

DATASET="siftsmall"
DATA_PATH="data/${DATASET}/${DATASET}_base.bin"
INDEX_PATH="data/${DATASET}/index"
ALGORITHM="BKT"  # BKT or KDT
DIST_CALC_METHOD="L2"  # L2 or Cosine
DIMENSION=128

# Build parameters
NUM_THREADS=4
BKT_NEIGHBORHOOD_SIZE=32
BKT_LEAF_SIZE=8
NEIGHBORHOOD_SIZE=32
CEF=1000
MAX_CHECK=8192

echo "=========================================="
echo "Building SPTAG Index for ${DATASET}"
echo "=========================================="
echo "Data file: ${DATA_PATH}"
echo "Index path: ${INDEX_PATH}"
echo "Algorithm: ${ALGORITHM}"
echo "Distance method: ${DIST_CALC_METHOD}"
echo "Dimension: ${DIMENSION}"
echo "=========================================="

# Check if data file exists
if [ ! -f "${DATA_PATH}" ]; then
    echo "Error: Data file not found: ${DATA_PATH}"
    exit 1
fi

# Create index directory if it doesn't exist
mkdir -p "${INDEX_PATH}"

# Check if index already exists
if [ -f "${INDEX_PATH}/indexloader.ini" ]; then
    echo "Warning: Index already exists at ${INDEX_PATH}"
    read -p "Do you want to rebuild it? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Index build cancelled."
        exit 0
    fi
    rm -rf "${INDEX_PATH}"/*
fi

# Build the index using indexbuilder
echo "Building index (this may take a while)..."
./Release/indexbuilder \
    -d ${DIMENSION} \
    -v Float \
    -f DEFAULT \
    -i ${DATA_PATH} \
    -o ${INDEX_PATH} \
    -a ${ALGORITHM} \
    -t ${NUM_THREADS} \
    Index.DistCalcMethod=${DIST_CALC_METHOD} \
    Index.NumberOfThreads=${NUM_THREADS} \
    Index.BKTNeighborhoodSize=${BKT_NEIGHBORHOOD_SIZE} \
    Index.BKTLeafSize=${BKT_LEAF_SIZE} \
    Index.NeighborhoodSize=${NEIGHBORHOOD_SIZE} \
    Index.CEF=${CEF} \
    Index.MaxCheck=${MAX_CHECK}

if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "Index build completed successfully!"
    echo "Index location: ${INDEX_PATH}"
    echo "=========================================="
else
    echo ""
    echo "Error: Index build failed!"
    exit 1
fi

