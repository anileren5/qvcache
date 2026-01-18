FROM ubuntu:jammy

# Install system dependencies
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
        libspdlog-dev libnuma-dev libtbb-dev libtbb2 && \
    # Install Python dependencies for bindings
    python3 -m pip install --upgrade pip setuptools wheel && \
    python3 -m pip install "protobuf<5.0.0" && \
    python3 -m pip install pybind11 numpy matplotlib qdrant-client pinecone psycopg2-binary faiss-cpu

# Set up LD_LIBRARY_PATH to include Python library directory and TBB libraries for SPTAG client
# TBB libraries are typically in /usr/lib/x86_64-linux-gnu
# Also check common TBB installation locations
ENV LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:/usr/local/lib:${LD_LIBRARY_PATH}

WORKDIR /app
