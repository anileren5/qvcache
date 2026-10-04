FROM ubuntu:jammy

# Install system dependencies
# Aker: also wget, ninja, gfortran, OpenBLAS, gflags for Boost 1.86 + FAISS source builds
RUN apt update && \
    apt install -y software-properties-common && \
    add-apt-repository -y ppa:git-core/ppa && \
    add-apt-repository -y ppa:deadsnakes/ppa && \
    apt update && \
    DEBIAN_FRONTEND=noninteractive apt install -y \
        git make cmake g++ libaio-dev libgoogle-perftools-dev libunwind-dev \
        clang-format libboost-dev libboost-program-options-dev \
        libmkl-full-dev libcpprest-dev python3.10 python3.10-dev python3-pip \
        python3.8 python3.8-dev libpython3.8 \
        libeigen3-dev \
        libspdlog-dev libnuma-dev libtbb-dev libtbb2 \
        libpq-dev \
        ca-certificates wget ninja-build pkg-config gfortran \
        libopenblas-dev libgflags-dev && \
    # Install Python dependencies for bindings
    python3 -m pip install --upgrade pip setuptools wheel && \
    python3 -m pip install "protobuf<5.0.0" && \
    python3 -m pip install pybind11 numpy matplotlib psycopg2-binary

# Aker: Ubuntu 22.04 CMake (3.22) and Boost (1.74) are too old for Aker/FAISS.
# Install CMake 3.28, Boost 1.86, and FAISS C++ into the image. libaker.so itself
# is built later inside the container (bind-mounted source at /app/external/Aker).
COPY scripts/aker/install_deps.sh /tmp/install_aker_deps.sh
RUN bash /tmp/install_aker_deps.sh && rm -f /tmp/install_aker_deps.sh

# Aker: Boost 1.86, FAISS, and Aker's libaker.so (Aker skips RPATH; loader needs this path)
ENV BOOST_ROOT=/opt/boost_1_86
ENV FAISS_ROOT=/usr/local
ENV AKER_CONFIG_PATH=/app/external/Aker/bootstrap/aker-standard.ini
ENV LD_LIBRARY_PATH=/app/external/Aker/build/lib:/opt/boost_1_86/lib:/usr/lib/x86_64-linux-gnu:/usr/local/lib:${LD_LIBRARY_PATH}

WORKDIR /app
