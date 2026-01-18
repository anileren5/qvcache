#pragma once

/**
 * SPTAG Client Backend for QVCache
 * 
 * This backend uses SPTAG's client API to communicate with a remote SPTAG server,
 * avoiding Python binding issues and complex CMake integration.
 * 
 * Architecture:
 *   - SPTAG server runs in a separate container/service with the index loaded
 *   - QVCache uses this backend to connect to the server via socket communication
 *   - All communication happens in C++ (no Python bindings)
 */

#include "qvcache/backend_interface.h"
#include "diskann/utils.h"
#include <vector>
#include <cstdint>
#include <iostream>
#include <fstream>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <thread>
#include <chrono>

// Forward declaration - we'll use PIMPL to avoid exposing SPTAG headers in header file
class SPTAGClientImpl;

namespace qvcache {

template <typename T, typename TagT = uint32_t>
class SPTAGClientBackend : public BackendInterface<T, TagT> {
private:
    std::unique_ptr<SPTAGClientImpl> pimpl_;
    size_t dim;
    size_t num_vectors;
    T* vector_data; // For fetch_vectors_by_ids - loaded from data file

public:
    // Constructor: connects to SPTAG server
    // server_addr: SPTAG server address (e.g., "localhost" or "sptag-service")
    // server_port: SPTAG server port (e.g., "8000")
    // dim: Vector dimension (if 0, will be read from data_path if provided; otherwise required)
    // data_path: Path to the data file for fetch_vectors_by_ids (optional, can be empty if not needed)
    SPTAGClientBackend(const std::string& server_addr, const std::string& server_port, 
                      size_t dim = 0, const std::string& data_path = "");

    ~SPTAGClientBackend();

    void search(
        const T *query,
        uint64_t K,
        TagT* result_tags,
        float* result_distances,
        void* search_parameters = nullptr,
        void* stats = nullptr) override;

    std::vector<std::vector<T>> fetch_vectors_by_ids(const std::vector<TagT>& ids) override;
};

} // namespace qvcache

