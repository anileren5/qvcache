#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# Change to the project root directory (one level above the script's location)
cd "$(dirname "$0")/../../.." || exit 1

# Configuration parameters
dataset="siftsmall"
data_type="float"
R=128
L=128
B=8
M=8
T=16
similarity="l2"  # Distance metric: "l2", "cosine" etc.
single_file_index=0
sector_len=4096

# Input and output paths
base_file="./data/${dataset}/${dataset}_base.bin"
index_dir="./index/${dataset}"
index_prefix="${index_dir}/${dataset}"

# Create index directory if it doesn't exist
mkdir -p "$index_dir"

# Run the build_disk_index command with all parameters
./build/benchmarks/build_index "$data_type" \
  --data_file "$base_file" \
  --index_prefix_path "$index_prefix" \
  --R "$R" \
  --L "$L" \
  --B "$B" \
  --M "$M" \
  --T "$T" \
  --dist_metric "$similarity" \
  --single_file_index "$single_file_index" \
  --sector_len "$sector_len"
