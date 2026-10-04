#!/bin/bash
# Build Aker (libaker.so) and then QVCache, including the Aker benchmark binaries.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT}"

# Aker must be installed before this CMake run. The Aker benchmark targets
# are added only when libaker.so is already on the library path.
./scripts/aker/build.sh

mkdir -p build
cmake -S "${ROOT}" -B "${ROOT}/build"
cmake --build "${ROOT}/build" -j"$(nproc)"
