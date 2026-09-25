#!/usr/bin/env bash
# Aker: build libaker.so from the bind-mounted Aker tree. Does not edit Aker sources.
#
# Prerequisites (image build runs scripts/aker/install_deps.sh):
#   Boost >= 1.80 at $BOOST_ROOT (default /opt/boost_1_86)
#   FAISS C++ headers/libs under $FAISS_ROOT (default /usr/local)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
AKER_ROOT="${AKER_ROOT:-${REPO_ROOT}/Aker}"
BOOST_ROOT="${BOOST_ROOT:-/opt/boost_1_86}"
FAISS_ROOT="${FAISS_ROOT:-/usr/local}"
AKER_MODE="${AKER_MODE:-standard}"
BUILD_DIR="${AKER_ROOT}/build"

if [[ ! -d "${AKER_ROOT}" ]]; then
    echo "ERROR: Aker source not found at ${AKER_ROOT}" >&2
    exit 1
fi

# Aker: compile-time mode (exactly one; default is Standard)
case "${AKER_MODE}" in
    standard)   MODE_FLAGS="-DAKER_ENABLE_PROXIMITY_MODE=OFF -DAKER_ENABLE_POTLUCK_MODE=OFF" ;;
    proximity)  MODE_FLAGS="-DAKER_ENABLE_PROXIMITY_MODE=ON  -DAKER_ENABLE_POTLUCK_MODE=OFF" ;;
    potluck)    MODE_FLAGS="-DAKER_ENABLE_PROXIMITY_MODE=OFF -DAKER_ENABLE_POTLUCK_MODE=ON" ;;
    *)
        echo "ERROR: unknown AKER_MODE=${AKER_MODE} (expected: standard|proximity|potluck)" >&2
        exit 1
        ;;
esac

# Aker: CMakeLists skips RPATH, so libaker.so must be on the loader path
export LD_LIBRARY_PATH="${AKER_ROOT}/build/lib:${BOOST_ROOT}/lib:${FAISS_ROOT}/lib:${FAISS_ROOT}/lib64:${LD_LIBRARY_PATH:-}"

# Aker: init xxHash / YCSB-C submodules if they are still empty
if git -C "${AKER_ROOT}" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    git config --global --add safe.directory "${AKER_ROOT}" || true
    git -C "${AKER_ROOT}" submodule update --init --recursive
fi

if [[ ! -f "${AKER_ROOT}/extern/xxHash/xxh3.h" ]]; then
    echo "ERROR: missing Aker submodule headers (extern/xxHash). From the host, run:" >&2
    echo "  git -C ${AKER_ROOT} submodule update --init --recursive" >&2
    exit 1
fi

if [[ ! -f "${BOOST_ROOT}/include/boost/version.hpp" ]]; then
    echo "ERROR: Boost not found at ${BOOST_ROOT}. Rebuild the qvcache image (scripts/aker/install_deps.sh)." >&2
    exit 1
fi

if [[ ! -f "${FAISS_ROOT}/include/faiss/Index.h" ]]; then
    echo "ERROR: FAISS headers not found under ${FAISS_ROOT}. Rebuild the qvcache image (scripts/aker/install_deps.sh)." >&2
    exit 1
fi

echo "Building Aker (${AKER_MODE}) from ${AKER_ROOT}"
# Aker: compile out Boost.Log per-query debug spam (header switch in Aker, no source edits)
cmake -S "${AKER_ROOT}" -B "${BUILD_DIR}" \
    -DCMAKE_BUILD_TYPE=Release \
    -DBOOST_ROOT="${BOOST_ROOT}" \
    -DFAISS_ROOT="${FAISS_ROOT}" \
    -DCMAKE_CXX_FLAGS="-DAKER_ENABLE_LOGGING=0" \
    ${MODE_FLAGS}

cmake --build "${BUILD_DIR}" -j"$(nproc)"

# Aker: install libaker.so into /usr/local/lib (already on the image LD_LIBRARY_PATH)
cmake --install "${BUILD_DIR}"
ldconfig

echo
echo "Aker artifacts:"
echo "  ${AKER_ROOT}/build/lib/libaker.so"
echo "  ${AKER_ROOT}/build/bin/aker-random-cache-test"
echo "  ${AKER_ROOT}/build/bin/ak_ycsb_gen"
echo
echo "This shell already has LD_LIBRARY_PATH set. In a new shell, run:"
echo "  export LD_LIBRARY_PATH=\"${AKER_ROOT}/build/lib:${BOOST_ROOT}/lib:\\\$LD_LIBRARY_PATH\""
echo
echo "Smoke test:"
echo "  \"${AKER_ROOT}/build/bin/aker-random-cache-test\" --entries 100 --queries 1000"
