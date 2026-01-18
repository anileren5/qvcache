#pragma once

#include "qvcache/backend_interface.h"
#include "diskann/utils.h"
#include <vector>
#include <cstdint>
#include <algorithm>
#include <iostream>
#include <fstream>
#include <cstring>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace qvcache {

template <typename T, typename TagT = uint32_t>
class BruteforceBackend : public BackendInterface<T, TagT> {

private:
    T* data; // All vectors stored in memory
    size_t num_vectors; // Number of vectors
    size_t dim; // Dimension of vectors

public: 
    BruteforceBackend(const std::string& data_path)
        : data(nullptr), num_vectors(0), dim(0) 
    {
        // Load metadata
        diskann::get_bin_metadata(data_path, num_vectors, dim);

        // Allocate memory
        data = new T[num_vectors * dim];

        // Load the data
        std::ifstream reader(data_path, std::ios::binary);
        if (!reader.is_open()) {
            throw std::runtime_error("Failed to open file: " + data_path);
        }

        // Skip metadata (2 * sizeof(uint32_t))
        reader.seekg(2 * sizeof(uint32_t), std::ios::beg);

        // Read all vectors
        reader.read(reinterpret_cast<char*>(data), num_vectors * dim * sizeof(T));
        reader.close();

        std::cout << "BruteforceBackend initialized." <<  std::endl;
        std::cout << "Loaded " << num_vectors << " vectors of dimension " << dim << " from " << data_path << std::endl;
    }


    ~BruteforceBackend() {
        if (data != nullptr) {
            delete[] data;
        }
    }
    
    void search(
        const T *query,
        uint64_t K,
        TagT* result_tags,
        float* result_distances,
        void* search_parameters = nullptr,
        void* stats = nullptr) override 
    {
        // Vector to store all distances
        std::vector<std::pair<float, TagT>> distance_tag_pairs;
        distance_tag_pairs.reserve(num_vectors);

        // Compute  L2 distance to all vectors
        for (size_t i = 0; i < num_vectors; ++i) {
            float sum = 0.0f;
            for (size_t d = 0; d < dim; ++d) {
                float diff = query[d] - data[i * dim + d];
                sum += diff * diff; 
            }
            distance_tag_pairs.emplace_back(sum, static_cast<TagT>(i));
        }

        // Partial sort to get top K
        size_t actual_K = std::min(static_cast<size_t>(K), num_vectors);
        std::partial_sort(
            distance_tag_pairs.begin(),
            distance_tag_pairs.begin() + actual_K,
            distance_tag_pairs.end(),
            [](const auto& a, const auto& b) {
                return a.first < b.first;  
            }
        );

        // Copy results to output arrays
        for (size_t k = 0; k < actual_K; ++k) {
            result_distances[k] = distance_tag_pairs[k].first;
            result_tags[k] = distance_tag_pairs[k].second; 
        }

        // Fill remaining results if K > num_vectors
        for (size_t k = actual_K; k < K; ++k) {
            result_distances[k] = std::numeric_limits<float>::infinity();
            result_tags[k] = 0;
        }
    }

    std::vector<std::vector<T>> fetch_vectors_by_ids(const std::vector<TagT>& ids) override {
        std::vector<std::vector<T>> result;
        result.reserve(ids.size());

        for (const TagT& id : ids) {
            std::vector<T> vec(dim);
            if (id < num_vectors) {
                std::memcpy(vec.data(), &data[id * dim], dim * sizeof(T));
            } else {
                std::fill(vec.begin(), vec.end(), static_cast<T>(0));
            }
            result.push_back(std::move(vec));
        }
        return result;
    }

};

}