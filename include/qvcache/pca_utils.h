#pragma once

#include <Eigen/Dense>
#include <vector>
#include <string>
#include <fstream>
#include <filesystem>
#include <type_traits>
#include <unordered_map>
#include <mutex>
#include <cstdint>
#include <iostream>
#include <limits>
#include <list>
#include "diskann/distance.h"

namespace qvcache {

    // Hash for std::vector<uint8_t>
    struct ArrayHash {
        std::size_t operator()(const std::vector<uint8_t>& arr) const {
            std::size_t h = 0;
            for (auto v : arr) h = h * 31 + v;
            return h;
        }
    };

    template <typename T>
    class PCAUtils {
    private:
        size_t dim;
        size_t PCA_DIM;
        size_t BUCKETS_PER_DIM;
        std::string disk_index_prefix;
        diskann::Metric metric = diskann::L2; // Distance metric (default: L2)
        
        using RegionKey = std::vector<uint8_t>;
        // Map: region -> (K -> theta)
        std::unordered_map<RegionKey, std::unordered_map<uint32_t, double>, ArrayHash> region_theta_map;
        std::mutex region_theta_map_mutex;
        // Track insertion order for FIFO eviction
        std::list<RegionKey> region_insertion_order;
        size_t max_regions;
        
        // PCA projection matrix and min/max for bucketing
        Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> pca_components; // [dim, PCA_DIM]
        Eigen::Matrix<T, 1, Eigen::Dynamic> pca_mean; // [1, dim]
        std::vector<T> pca_min, pca_max; // min/max for each PCA dim
        
        // PCA float storage for int8/uint8 types
        // Only used if T is not floating point
        Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> pca_components_float;
        Eigen::Matrix<float, 1, Eigen::Dynamic> pca_mean_float;
        std::vector<float> pca_min_float, pca_max_float;

        // Helper methods
        std::string get_pca_filename() const {
            return disk_index_prefix + ".pca.bin";
        }
        
        bool file_exists(const std::string& filename) const {
            return std::filesystem::exists(filename);
        }

    public:
        PCAUtils(size_t dim, size_t pca_dim, size_t buckets_per_dim, const std::string& disk_index_prefix, diskann::Metric metric_ = diskann::L2, size_t max_regions_ = std::numeric_limits<size_t>::max())
            : dim(dim), PCA_DIM(pca_dim), BUCKETS_PER_DIM(buckets_per_dim), disk_index_prefix(disk_index_prefix), metric(metric_), max_regions(max_regions_) {}

        // Save PCA data to file
        void save_pca_to_file(bool is_float) {
            std::ofstream ofs(get_pca_filename(), std::ios::binary);
            if (!ofs) return;
            ofs.write((char*)&dim, sizeof(dim));
            ofs.write((char*)&PCA_DIM, sizeof(PCA_DIM));
            ofs.write((char*)&BUCKETS_PER_DIM, sizeof(BUCKETS_PER_DIM));
            if (is_float) {
                // Save pca_mean
                ofs.write(reinterpret_cast<const char*>(pca_mean_float.data()), sizeof(float) * dim);
                // Save pca_components (row-major)
                ofs.write(reinterpret_cast<const char*>(pca_components_float.data()), sizeof(float) * dim * PCA_DIM);
                // Save pca_min, pca_max
                ofs.write(reinterpret_cast<const char*>(pca_min_float.data()), sizeof(float) * PCA_DIM);
                ofs.write(reinterpret_cast<const char*>(pca_max_float.data()), sizeof(float) * PCA_DIM);
            } else {
                // Save pca_mean
                ofs.write(reinterpret_cast<const char*>(pca_mean.data()), sizeof(T) * dim);
                // Save pca_components (row-major)
                ofs.write(reinterpret_cast<const char*>(pca_components.data()), sizeof(T) * dim * PCA_DIM);
                // Save pca_min, pca_max
                ofs.write(reinterpret_cast<const char*>(pca_min.data()), sizeof(T) * PCA_DIM);
                ofs.write(reinterpret_cast<const char*>(pca_max.data()), sizeof(T) * PCA_DIM);
            }
        }

        // Load PCA data from file
        bool load_pca_from_file(bool is_float) {
            std::ifstream ifs(get_pca_filename(), std::ios::binary);
            if (!ifs) return false;
            size_t file_dim, file_pca_dim, file_buckets_per_dim;
            ifs.read((char*)&file_dim, sizeof(file_dim));
            ifs.read((char*)&file_pca_dim, sizeof(file_pca_dim));
            ifs.read((char*)&file_buckets_per_dim, sizeof(file_buckets_per_dim));
            if (file_dim != dim || file_pca_dim != PCA_DIM || file_buckets_per_dim != BUCKETS_PER_DIM) return false;
            if (is_float) {
                pca_mean_float.resize(dim);
                ifs.read(reinterpret_cast<char*>(pca_mean_float.data()), sizeof(float) * dim);
                pca_components_float.resize(dim, PCA_DIM);
                ifs.read(reinterpret_cast<char*>(pca_components_float.data()), sizeof(float) * dim * PCA_DIM);
                pca_min_float.resize(PCA_DIM);
                pca_max_float.resize(PCA_DIM);
                ifs.read(reinterpret_cast<char*>(pca_min_float.data()), sizeof(float) * PCA_DIM);
                ifs.read(reinterpret_cast<char*>(pca_max_float.data()), sizeof(float) * PCA_DIM);
            } else {
                pca_mean.resize(dim);
                ifs.read(reinterpret_cast<char*>(pca_mean.data()), sizeof(T) * dim);
                pca_components.resize(dim, PCA_DIM);
                ifs.read(reinterpret_cast<char*>(pca_components.data()), sizeof(T) * dim * PCA_DIM);
                pca_min.resize(PCA_DIM);
                pca_max.resize(PCA_DIM);
                ifs.read(reinterpret_cast<char*>(pca_min.data()), sizeof(T) * PCA_DIM);
                ifs.read(reinterpret_cast<char*>(pca_max.data()), sizeof(T) * PCA_DIM);
            }
            return true;
        }

        // Construct PCA from data
        void construct_pca_from_data(const T* data, size_t num_points, size_t aligned_dim, const std::string& disk_index_prefix) {
            this->disk_index_prefix = disk_index_prefix;
            
            if constexpr (std::is_floating_point<T>::value) {
                std::cout << "[PCAUtils] Starting PCA construction (float/double)..." << std::endl;
                // Copy to Eigen matrix (only the first 'dim' of each vector)
                std::cout << "[PCAUtils] Copying data to Eigen matrix..." << std::endl;
                Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> data_mat(num_points, dim);
                for (size_t i = 0; i < num_points; ++i) {
                    for (size_t j = 0; j < dim; ++j) {
                        data_mat(i, j) = data[i * aligned_dim + j];
                    }
                }
                std::cout << "[PCAUtils] Mean centering..." << std::endl;
                pca_mean = data_mat.colwise().mean();
                Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> centered = data_mat.rowwise() - pca_mean;
                std::cout << "[PCAUtils] Running SVD..." << std::endl;
                Eigen::JacobiSVD<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>> svd(centered, Eigen::ComputeThinU | Eigen::ComputeThinV);
                pca_components = svd.matrixV().leftCols(PCA_DIM);
                std::cout << "[PCAUtils] Projecting data and computing min/max for each PCA dim..." << std::endl;
                // Project all data to PCA and compute min/max for each dim
                Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> projected = centered * pca_components.leftCols(PCA_DIM);
                pca_min.resize(PCA_DIM);
                pca_max.resize(PCA_DIM);
                for (size_t i = 0; i < PCA_DIM; ++i) {
                    pca_min[i] = projected.col(i).minCoeff();
                    pca_max[i] = projected.col(i).maxCoeff();
                }
                std::cout << "[PCAUtils] PCA construction complete." << std::endl;
                save_pca_to_file(false);
            } else {
                std::cout << "[PCAUtils] Starting PCA construction (int8/uint8 branch, using float)..." << std::endl;
                // Convert to float for PCA
                std::cout << "[PCAUtils] Converting data to float..." << std::endl;
                std::vector<float> float_data(num_points * dim);
                for (size_t i = 0; i < num_points; ++i) {
                    for (size_t j = 0; j < dim; ++j) {
                        float_data[i * dim + j] = static_cast<float>(data[i * aligned_dim + j]);
                    }
                }
                std::cout << "[PCAUtils] Copying float data to Eigen matrix..." << std::endl;
                Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> data_mat(num_points, dim);
                for (size_t i = 0; i < num_points; ++i) {
                    for (size_t j = 0; j < dim; ++j) {
                        data_mat(i, j) = float_data[i * dim + j];
                    }
                }
                std::cout << "[PCAUtils] Mean centering..." << std::endl;
                pca_mean_float = data_mat.colwise().mean();
                Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> centered = data_mat.rowwise() - pca_mean_float;
                std::cout << "[PCAUtils] Running SVD..." << std::endl;
                Eigen::JacobiSVD<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>> svd(centered, Eigen::ComputeThinU | Eigen::ComputeThinV);
                pca_components_float = svd.matrixV().leftCols(PCA_DIM);
                std::cout << "[PCAUtils] Projecting data and computing min/max for each PCA dim..." << std::endl;
                // Project all data to PCA and compute min/max for each dim
                Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> projected = centered * pca_components_float.leftCols(PCA_DIM);
                pca_min_float.resize(PCA_DIM);
                pca_max_float.resize(PCA_DIM);
                for (size_t i = 0; i < PCA_DIM; ++i) {
                    pca_min_float[i] = projected.col(i).minCoeff();
                    pca_max_float[i] = projected.col(i).maxCoeff();
                }
                std::cout << "[PCAUtils] PCA construction complete." << std::endl;
                save_pca_to_file(true);
            }
        }

        // Project vector to PCA and compute region key
        RegionKey compute_region_key(const T* vec) {
            RegionKey key(PCA_DIM);
            if constexpr (std::is_floating_point<T>::value) {
                Eigen::Map<const Eigen::Matrix<T, 1, Eigen::Dynamic>> v(vec, dim);
                Eigen::Matrix<T, 1, Eigen::Dynamic> proj = (v - pca_mean) * pca_components.leftCols(PCA_DIM);
                for (size_t i = 0; i < PCA_DIM; ++i) {
                    T val = proj(0, i);
                    T minv = pca_min[i], maxv = pca_max[i];
                    if (maxv == minv) key[i] = 0;
                    else {
                        T norm = (val - minv) / (maxv - minv);
                        size_t bucket = std::min<size_t>(BUCKETS_PER_DIM - 1, static_cast<size_t>(norm * BUCKETS_PER_DIM));
                        key[i] = static_cast<uint8_t>(bucket);
                    }
                }
            } else {
                std::vector<float> float_vec(dim);
                for (size_t j = 0; j < dim; ++j) float_vec[j] = static_cast<float>(vec[j]);
                Eigen::Map<const Eigen::Matrix<float, 1, Eigen::Dynamic>> v(float_vec.data(), dim);
                Eigen::Matrix<float, 1, Eigen::Dynamic> proj = (v - pca_mean_float) * pca_components_float.leftCols(PCA_DIM);
                for (size_t i = 0; i < PCA_DIM; ++i) {
                    float val = proj(0, i);
                    float minv = pca_min_float[i], maxv = pca_max_float[i];
                    if (maxv == minv) key[i] = 0;
                    else {
                        float norm = (val - minv) / (maxv - minv);
                        size_t bucket = std::min<size_t>(BUCKETS_PER_DIM - 1, static_cast<size_t>(norm * BUCKETS_PER_DIM));
                        key[i] = static_cast<uint8_t>(bucket);
                    }
                }
            }
            return key;
        }

        // Lazy initialize region theta map
        void lazy_init_region(const RegionKey& key) {
            std::lock_guard<std::mutex> lock(region_theta_map_mutex);
            if (region_theta_map.find(key) == region_theta_map.end()) {
                // Check if we need to evict the oldest region
                if (region_theta_map.size() >= max_regions && !region_insertion_order.empty()) {
                    // Evict the oldest region (first in the list)
                    RegionKey oldest_key = region_insertion_order.front();
                    region_insertion_order.pop_front();
                    region_theta_map.erase(oldest_key);
                }
                
                // Initialize so that initially everything is a miss
                // For cosine: use negative infinity so that distances[K-1] > -infinity is always true (MISS)
                // For L2: use max double so that distances[K-1] > max is always false initially,
                //   but we'll check for uninitialized and force MISS anyway
                double init_value = (metric == diskann::COSINE) ? -std::numeric_limits<double>::infinity() : std::numeric_limits<double>::max();
                region_theta_map[key][1] = init_value;
                region_theta_map[key][5] = init_value;
                region_theta_map[key][10] = init_value;
                region_theta_map[key][100] = init_value;
                
                // Add to insertion order list (at the end)
                region_insertion_order.push_back(key);
            }
        }

        // Check if query is a hit using regional theta
        bool isHit(const T* query_ptr, uint32_t K, const float* distances, size_t num_vectors_in_memory, double deviation_factor) {
            if (num_vectors_in_memory < K) {
                return false;
            }
            
            RegionKey region = compute_region_key(query_ptr);
            lazy_init_region(region);
            
            std::lock_guard<std::mutex> lock(region_theta_map_mutex);
            double threshold = region_theta_map[region][K];
            
            // Handle uninitialized thresholds
            // For cosine: if threshold is -infinity, it means it hasn't been updated yet, so always miss
            // For L2: if threshold is max, it means it hasn't been updated yet, so always miss
            bool is_uninitialized = (metric == diskann::COSINE) 
                ? (threshold == -std::numeric_limits<double>::infinity()) 
                : (threshold >= std::numeric_limits<double>::max() * 0.5);
            
            if (is_uninitialized) {
                return false;
            }
            
            // Use multiplicative tolerance for both L2 and cosine
            double cache_distance = static_cast<double>(distances[K - 1]);
            double tolerance_threshold = (1.0 + deviation_factor) * threshold;
            
            // Check: if cache distance is worse than tolerance threshold, it's a miss
            if (cache_distance > tolerance_threshold) {
                return false;
            }
            return true;
        }

        // Update theta for a region
        void update_theta(const T* query_ptr, uint32_t K, float query_distance, double p) {
            RegionKey region = compute_region_key(query_ptr);
            lazy_init_region(region);
            
            std::lock_guard<std::mutex> lock(region_theta_map_mutex);
            double current_theta = region_theta_map[region][K];
            
            // Handle initialization: if current_theta is the uninitialized value, replace it completely
            bool is_uninitialized = (metric == diskann::COSINE)
                ? (current_theta == -std::numeric_limits<double>::infinity())
                : (current_theta >= std::numeric_limits<double>::max() * 0.5);
            
            if (is_uninitialized) {
                region_theta_map[region][K] = static_cast<double>(query_distance);
            } else {
                region_theta_map[region][K] = p * static_cast<double>(query_distance) + (1 - p) * current_theta;
            }
        }

        // Get PCA filename for logging
        std::string get_pca_filename_for_logging() const {
            return get_pca_filename();
        }

        // Get number of active regions (regions that have been initialized)
        size_t get_number_of_active_regions() const {
            std::lock_guard<std::mutex> lock(const_cast<std::mutex&>(region_theta_map_mutex));
            return region_theta_map.size();
        }
    };

} // namespace qvcache 