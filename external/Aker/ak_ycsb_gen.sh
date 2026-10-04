#!/usr/bin/env bash

set -euo pipefail

# Convenience wrapper to run the Zipfian key-sequence generator.
#
# Expected usage (example):
#   ./ak_ycsb_gen.sh -t 1000000 -q 100000
#
# Outputs are written under:
#   <project root>/apps/ak_ycsb_gen/

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${SCRIPT_DIR}"

BIN_PATH_DEFAULT="${PROJECT_ROOT}/build/bin/ak_ycsb_gen"
BIN_PATH="${AK_YCSB_GEN_BIN_PATH:-${BIN_PATH_DEFAULT}}"

if [[ ! -x "${BIN_PATH}" ]]; then
    printf "[ak_ycsb_gen] ERROR: generator binary not found or not executable: %s\n" "${BIN_PATH}" >&2
    printf "[ak_ycsb_gen] HINT: build first (from repo root):\n" >&2
    printf "[ak_ycsb_gen]   cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DFAISS_ROOT=/usr/local\n" >&2
    printf "[ak_ycsb_gen]   cmake --build build -j\n" >&2
    exit 1
fi

# libaker.so is built under <project root>/build/lib. The project intentionally
# does not embed build-tree RPATH, so add it to the loader search path.
export LD_LIBRARY_PATH="${PROJECT_ROOT}/build/lib:${LD_LIBRARY_PATH:-}"

cd "${PROJECT_ROOT}"

"${BIN_PATH}" "$@"

printf "[ak_ycsb_gen] Output directory: %s\n" "${PROJECT_ROOT}/apps/ak_ycsb_gen"
