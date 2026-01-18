#!/bin/bash
# Script to test SPTAG client connection to server

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR" || exit 1

SERVER_ADDR="${1:-localhost}"
SERVER_PORT="${2:-8000}"
DATASET="siftsmall"
QUERY_FILE="data/${DATASET}/${DATASET}_query.bin"
GROUNDTRUTH_FILE="data/${DATASET}/${DATASET}_groundtruth.bin"
K=100  # Check recall@100

echo "=========================================="
echo "Testing SPTAG Client with Groundtruth"
echo "=========================================="
echo "Server: ${SERVER_ADDR}:${SERVER_PORT}"
echo "Query file: ${QUERY_FILE}"
echo "Groundtruth file: ${GROUNDTRUTH_FILE}"
echo "K: ${K} (Recall@${K})"
echo "=========================================="

# Check if client executable exists
if [ ! -f "./Release/client" ]; then
    echo "Error: SPTAG client not found at ./Release/client"
    echo "Please build SPTAG first"
    exit 1
fi

# Check if query file exists
if [ ! -f "${QUERY_FILE}" ]; then
    echo "Error: Query file not found: ${QUERY_FILE}"
    exit 1
fi

# Check if groundtruth file exists
if [ ! -f "${GROUNDTRUTH_FILE}" ]; then
    echo "Error: Groundtruth file not found: ${GROUNDTRUTH_FILE}"
    exit 1
fi

echo ""
echo "Connecting to server and processing all queries..."
echo ""

# Set library path so Python can find the shared libraries
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:$(pwd)/Release"

# Test connection using Python client (simpler for testing)
if [ -f "./Release/SPTAGClient.py" ]; then
    python3 << EOF
import sys
import os
sys.path.insert(0, './Release')
import SPTAGClient
import numpy as np
import time

print("Connecting to server ${SERVER_ADDR}:${SERVER_PORT}...")
client = SPTAGClient.AnnClient('${SERVER_ADDR}', '${SERVER_PORT}')

# Wait for connection
max_wait = 30
waited = 0
while not client.IsConnected() and waited < max_wait:
    time.sleep(1)
    waited += 1
    if waited % 5 == 0:
        print(f"Waiting for connection... ({waited}s)")

if not client.IsConnected():
    print("Error: Failed to connect to server after ${max_wait} seconds")
    sys.exit(1)

print("Connected successfully!")

# Load queries and groundtruth from binary files
query_file = '${QUERY_FILE}'
groundtruth_file = '${GROUNDTRUTH_FILE}'
k = ${K}

print(f"\\nLoading queries from {query_file}...")
with open(query_file, 'rb') as f:
    num_queries = np.frombuffer(f.read(4), dtype=np.uint32)[0]
    dim = np.frombuffer(f.read(4), dtype=np.uint32)[0]
    queries = np.frombuffer(f.read(num_queries * dim * 4), dtype=np.float32).reshape(num_queries, dim)

print(f"Loading groundtruth from {groundtruth_file}...")
with open(groundtruth_file, 'rb') as f:
    gt_num_queries = np.frombuffer(f.read(4), dtype=np.uint32)[0]
    gt_k = np.frombuffer(f.read(4), dtype=np.uint32)[0]
    groundtruth = np.frombuffer(f.read(gt_num_queries * gt_k * 4), dtype=np.uint32).reshape(gt_num_queries, gt_k)

print(f"Loaded {num_queries} queries of dimension {dim}")
print(f"Groundtruth has {gt_num_queries} queries with K={gt_k}")
print(f"Using K={k} for search (computing Recall@{k})\\n")

# Process all queries
total_recall = 0.0
recall_values = []

for i in range(num_queries):
    query = queries[i]
    expected_ids = set(groundtruth[i])
    
    try:
        result = client.Search(query, k, 'Float', False)
        if result and len(result) >= 2:
            result_ids = result[0]
            result_ids_set = set(result_ids)
            
            # Calculate recall
            matches = len(result_ids_set.intersection(expected_ids))
            recall = matches / len(expected_ids) if len(expected_ids) > 0 else 0.0
            total_recall += recall
            recall_values.append(recall)
            
            if (i + 1) % 100 == 0 or i == 0:
                print(f"Query {i+1}/{num_queries}: Recall@{k} = {recall:.4f} ({matches}/{len(expected_ids)} matches)")
        else:
            print(f"Warning: Query {i+1} returned invalid result")
            recall_values.append(0.0)
    except Exception as e:
        print(f"Error processing query {i+1}: {e}")
        recall_values.append(0.0)

# Calculate and print statistics
avg_recall = total_recall / num_queries if num_queries > 0 else 0.0
recall_values = np.array(recall_values)
min_recall = np.min(recall_values)
max_recall = np.max(recall_values)
median_recall = np.median(recall_values)

print(f"\\n==========================================")
print(f"Results Summary")
print(f"==========================================")
print(f"Total queries processed: {num_queries}")
print(f"Average Recall@{k}: {avg_recall:.4f}")
print(f"Min Recall@{k}: {min_recall:.4f}")
print(f"Max Recall@{k}: {max_recall:.4f}")
print(f"Median Recall@{k}: {median_recall:.4f}")
print(f"==========================================")
EOF
else
    echo "Python client not found. Please build SPTAG with Python bindings."
    exit 1
fi

