#!/usr/bin/env bash
# Aker: install C++ dependencies into the qvcache image/container.
# Does not modify Aker sources. Installs:
#   - extra apt packages
#   - CMake >= 3.23 (jammy apt CMake is 3.22; current FAISS needs 3.23+)
#   - Boost 1.86 (Aker requires Boost >= 1.80; jammy ships 1.74)
#   - FAISS C++ shared library + headers (pip faiss-cpu does not provide these)
set -euo pipefail

# Aker: match Aker README / docker/images/Dockerfile.aker_test where possible
BOOST_VERSION="${BOOST_VERSION:-1.86.0}"
BOOST_ROOT="${BOOST_ROOT:-/opt/boost_1_86}"
CMAKE_VERSION="${CMAKE_VERSION:-3.28.6}"
FAISS_PREFIX="${FAISS_PREFIX:-/usr/local}"
FAISS_SRC="${FAISS_SRC:-/opt/src/faiss}"
FAISS_REPO_URL="${FAISS_REPO_URL:-https://github.com/facebookresearch/faiss.git}"
# Aker: empty = clone default branch, same as Aker's Dockerfile.aker_test
FAISS_REF="${FAISS_REF:-}"
NPROC="$(nproc)"

export DEBIAN_FRONTEND=noninteractive

log() { printf '[aker-deps] %s\n' "$*"; }

version_ge() {
    # usage: version_ge 3.28.6 3.23.1
    printf '%s\n%s\n' "$2" "$1" | sort -V | head -n1 | grep -qx "$2"
}

install_apt_packages() {
    log "installing apt packages"
    apt-get update
    apt-get install -y --no-install-recommends \
        ca-certificates \
        wget \
        ninja-build \
        pkg-config \
        gfortran \
        libopenblas-dev \
        libgflags-dev
}

install_cmake() {
    local cmake_ver
    cmake_ver="$(cmake --version 2>/dev/null | awk 'NR==1 {print $3}')" || cmake_ver="0"
    if version_ge "${cmake_ver}" "3.23.1"; then
        log "CMake ${cmake_ver} already meets FAISS requirement (>= 3.23.1)"
        return
    fi

    local installer="/tmp/cmake-${CMAKE_VERSION}-linux-x86_64.sh"
    local url="https://github.com/Kitware/CMake/releases/download/v${CMAKE_VERSION}/cmake-${CMAKE_VERSION}-linux-x86_64.sh"
    log "installing CMake ${CMAKE_VERSION} over jammy cmake ${cmake_ver}"
    wget -q -O "${installer}" "${url}"
    chmod +x "${installer}"
    bash "${installer}" --prefix=/usr/local --skip-license
    rm -f "${installer}"
    hash -r || true
    log "CMake now: $(cmake --version | awk 'NR==1 {print $3}')"
}

boost_ready() {
    [[ -f "${BOOST_ROOT}/include/boost/version.hpp" ]] && \
        [[ -e "${BOOST_ROOT}/lib/libboost_log.so" || -e "${BOOST_ROOT}/lib64/libboost_log.so" ]]
}

install_boost() {
    if boost_ready; then
        log "Boost already installed at ${BOOST_ROOT}"
        return
    fi

    local ver_us src_dir tarball url
    ver_us="${BOOST_VERSION//./_}"
    src_dir="/opt/src/boost_${ver_us}"
    tarball="/opt/src/boost_${ver_us}.tar.gz"
    url="https://archives.boost.io/release/${BOOST_VERSION}/source/boost_${ver_us}.tar.gz"

    log "building Boost ${BOOST_VERSION} -> ${BOOST_ROOT}"
    mkdir -p /opt/src
    if [[ ! -f "${tarball}" ]]; then
        wget -q -O "${tarball}" "${url}"
    fi
    if [[ ! -d "${src_dir}" ]]; then
        tar -xzf "${tarball}" -C /opt/src
    fi

    pushd "${src_dir}" >/dev/null
    ./bootstrap.sh --prefix="${BOOST_ROOT}" \
        --with-libraries=program_options,log,date_time,thread,system,filesystem,regex,atomic,chrono
    ./b2 -j"${NPROC}" variant=release link=shared threading=multi install
    popd >/dev/null

    log "Boost installed at ${BOOST_ROOT}"
}

faiss_ready() {
    [[ -f "${FAISS_PREFIX}/include/faiss/Index.h" ]] && \
        { [[ -e "${FAISS_PREFIX}/lib/libfaiss.so" ]] || [[ -e "${FAISS_PREFIX}/lib64/libfaiss.so" ]]; }
}

install_faiss() {
    if faiss_ready; then
        log "FAISS already installed under ${FAISS_PREFIX}"
        return
    fi

    log "building FAISS -> ${FAISS_PREFIX}"
    mkdir -p /opt/src

    # Aker: drop a previous failed clone (e.g. cmake too old) so configure is clean
    if [[ -d "${FAISS_SRC}" ]] && ! faiss_ready; then
        rm -rf "${FAISS_SRC}"
    fi

    if [[ ! -d "${FAISS_SRC}/.git" ]]; then
        git clone --recursive "${FAISS_REPO_URL}" "${FAISS_SRC}"
        if [[ -n "${FAISS_REF}" ]]; then
            git -C "${FAISS_SRC}" checkout -f "${FAISS_REF}"
            git -C "${FAISS_SRC}" submodule update --init --recursive
        fi
    fi

    # Aker: flags copied from Aker/docker/images/Dockerfile.aker_test
    cmake -S "${FAISS_SRC}" -B "${FAISS_SRC}/build" \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX="${FAISS_PREFIX}" \
        -DFAISS_ENABLE_GPU=OFF \
        -DFAISS_ENABLE_PYTHON=OFF \
        -DFAISS_ENABLE_CUVS=OFF \
        -DBUILD_SHARED_LIBS=ON \
        -DFAISS_ENABLE_C_API=ON \
        -DFAISS_ENABLE_MKL=OFF

    make -C "${FAISS_SRC}/build" -j"${NPROC}" faiss
    # Aker: optional ISA variants; ignore if the target/CPU does not support them
    make -C "${FAISS_SRC}/build" -j"${NPROC}" faiss_avx2 || true
    make -C "${FAISS_SRC}/build" -j"${NPROC}" faiss_avx512 || true
    make -C "${FAISS_SRC}/build" install
    ldconfig

    log "FAISS installed under ${FAISS_PREFIX}"
}

install_apt_packages
install_cmake
install_boost
install_faiss

log "done"
log "BOOST_ROOT=${BOOST_ROOT}"
log "FAISS_PREFIX=${FAISS_PREFIX}"
log "Add to LD_LIBRARY_PATH: ${BOOST_ROOT}/lib:${FAISS_PREFIX}/lib"
