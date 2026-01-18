#!/bin/bash
# Build script for SPTAG
# This script automates the build process including zstd dependency setup

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR" || exit 1

echo "=========================================="
echo "Building SPTAG"
echo "=========================================="

# Step 0: Check for SWIG (required for Python wrappers)
echo "Checking for SWIG..."
if ! command -v swig &> /dev/null; then
    echo "SWIG not found. Installing SWIG..."
    if command -v apt-get &> /dev/null; then
        apt-get update && apt-get install -y swig || {
            echo "Error: Failed to install SWIG"
            echo "Please install SWIG manually: apt-get install -y swig"
            exit 1
        }
    else
        echo "Error: SWIG is required but not found and cannot be auto-installed"
        echo "Please install SWIG: apt-get install -y swig (or equivalent for your system)"
        exit 1
    fi
else
    echo "SWIG found: $(which swig)"
    swig -version | head -1 || true
fi

# Step 1: Set up zstd dependency
echo "Setting up zstd dependency..."
if [ ! -d "ThirdParty/zstd" ] || [ -z "$(ls -A ThirdParty/zstd 2>/dev/null)" ] || [ ! -d "ThirdParty/zstd/build/cmake" ]; then
    echo "Cloning zstd repository..."
    rm -rf ThirdParty/zstd
    mkdir -p ThirdParty/zstd
    cd ThirdParty/zstd
    # Try cloning specific versions that have build/cmake directory
    if git clone --depth 1 --branch v1.5.5 https://github.com/facebook/zstd.git . 2>/dev/null; then
        echo "Cloned zstd v1.5.5"
    elif git clone --depth 1 --branch v1.4.9 https://github.com/facebook/zstd.git . 2>/dev/null; then
        echo "Cloned zstd v1.4.9"
    elif git clone --depth 1 --branch release https://github.com/facebook/zstd.git . 2>/dev/null; then
        echo "Cloned zstd release branch"
    elif git clone --depth 1 https://github.com/facebook/zstd.git . 2>/dev/null; then
        echo "Cloned zstd main branch"
    else
        echo "Error: Failed to clone zstd repository"
        echo "Please ensure git is installed: apt-get update && apt-get install -y git"
        exit 1
    fi
    # If build/cmake still doesn't exist, check if it's in contrib/cmake or root
    if [ ! -d "build/cmake" ]; then
        if [ -d "contrib/cmake" ]; then
            mkdir -p build
            cp -r contrib/cmake build/cmake
        elif [ -f "CMakeLists.txt" ]; then
            mkdir -p build/cmake
            cp CMakeLists.txt build/cmake/ 2>/dev/null || true
        fi
    fi
    cd "$SCRIPT_DIR"
else
    echo "zstd dependency already exists, skipping clone..."
fi

# Step 2: Set compiler environment variables (if available)
if [ -f "/usr/bin/gcc-8" ] && [ -f "/usr/bin/g++-8" ]; then
    echo "Using gcc-8 and g++-8..."
    export CC=/usr/bin/gcc-8
    export CXX=/usr/bin/g++-8
else
    echo "gcc-8/g++-8 not found, using system default compilers..."
fi

# Step 3: Create build directory and Release directory
echo "Creating build directory..."
rm -rf build
mkdir -p build
# Create Release directory for output files
mkdir -p Release
cd build

# Step 4: Configure with CMake
echo "Configuring with CMake..."
if ! cmake -DSPDK=OFF -DROCKSDB=OFF ..; then
    echo "Error: CMake configuration failed"
    exit 1
fi

# Step 5: Ensure Release directory exists and is writable
echo "Ensuring Release directory exists..."
cd ..
mkdir -p Release
chmod 755 Release
# Verify we can write to the Release directory
if [ ! -w "Release" ]; then
    echo "Warning: Release directory is not writable. Attempting to fix permissions..."
    chmod u+w Release
fi
cd build

# Step 6: Build
echo "Building SPTAG (this may take a while)..."
if ! make -j$(nproc); then
    echo "Error: Build failed"
    exit 1
fi

# Step 7: Go back to root
cd ..

echo ""
echo "=========================================="
echo "Build completed successfully!"
echo "=========================================="
echo "Binaries should be in the Release/ directory"
echo "To verify, run: ls -lh Release/"
