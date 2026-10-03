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
#include <algorithm>
#include <cstdlib>
#include <cstddef>
#include "diskann/distance.h"

namespace qvcache {

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
        diskann::Metric metric = diskann::L2;

        using RegionKey = std::vector<uint8_t>;
        std::unordered_map<RegionKey, std::unordered_map<uint32_t, double>, ArrayHash> region_theta_map;
        // Saturating write penalty: 1 = fully trusted; α after one write.
        std::unordered_map<RegionKey, double, ArrayHash> region_discount_map;
        std::mutex region_theta_map_mutex;
        std::list<RegionKey> region_insertion_order;
        size_t max_regions;
        size_t write_l1_radius = 1;

        Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> pca_components;
        Eigen::Matrix<T, 1, Eigen::Dynamic> pca_mean;
        std::vector<T> pca_min, pca_max;

        Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> pca_components_float;
        Eigen::Matrix<float, 1, Eigen::Dynamic> pca_mean_float;
        std::vector<float> pca_min_float, pca_max_float;

        std::string get_pca_filename() const {
            return disk_index_prefix + ".pca.bin";
        }

        bool file_exists(const std::string& filename) const {
            return std::filesystem::exists(filename);
        }

    public:
        PCAUtils(size_t dim, size_t pca_dim, size_t buckets_per_dim, const std::string& disk_index_prefix,
                 diskann::Metric metric_ = diskann::L2,
                 size_t max_regions_ = std::numeric_limits<size_t>::max())
            : dim(dim), PCA_DIM(pca_dim), BUCKETS_PER_DIM(buckets_per_dim),
              disk_index_prefix(disk_index_prefix), metric(metric_), max_regions(max_regions_) {}

        void set_write_l1_radius(size_t r) { write_l1_radius = r; }
        size_t get_write_l1_radius() const { return write_l1_radius; }

        void save_pca_to_file(bool is_float) {
            std::ofstream ofs(get_pca_filename(), std::ios::binary);
            if (!ofs) return;
            ofs.write((char*)&dim, sizeof(dim));
            ofs.write((char*)&PCA_DIM, sizeof(PCA_DIM));
            ofs.write((char*)&BUCKETS_PER_DIM, sizeof(BUCKETS_PER_DIM));
            if (is_float) {
                ofs.write(reinterpret_cast<const char*>(pca_mean_float.data()), sizeof(float) * dim);
                ofs.write(reinterpret_cast<const char*>(pca_components_float.data()), sizeof(float) * dim * PCA_DIM);
                ofs.write(reinterpret_cast<const char*>(pca_min_float.data()), sizeof(float) * PCA_DIM);
                ofs.write(reinterpret_cast<const char*>(pca_max_float.data()), sizeof(float) * PCA_DIM);
            } else {
                ofs.write(reinterpret_cast<const char*>(pca_mean.data()), sizeof(T) * dim);
                ofs.write(reinterpret_cast<const char*>(pca_components.data()), sizeof(T) * dim * PCA_DIM);
                ofs.write(reinterpret_cast<const char*>(pca_min.data()), sizeof(T) * PCA_DIM);
                ofs.write(reinterpret_cast<const char*>(pca_max.data()), sizeof(T) * PCA_DIM);
            }
        }

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

        void construct_pca_from_data(const T* data, size_t num_points, size_t aligned_dim,
                                     const std::string& disk_index_prefix) {
            this->disk_index_prefix = disk_index_prefix;

            if constexpr (std::is_floating_point<T>::value) {
                std::cout << "[PCAUtils] Starting PCA construction (float/double)..." << std::endl;
                Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> data_mat(num_points, dim);
                for (size_t i = 0; i < num_points; ++i) {
                    for (size_t j = 0; j < dim; ++j) {
                        data_mat(i, j) = data[i * aligned_dim + j];
                    }
                }
                pca_mean = data_mat.colwise().mean();
                Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> centered = data_mat.rowwise() - pca_mean;
                Eigen::JacobiSVD<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>> svd(
                    centered, Eigen::ComputeThinU | Eigen::ComputeThinV);
                pca_components = svd.matrixV().leftCols(PCA_DIM);
                Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> projected =
                    centered * pca_components.leftCols(PCA_DIM);
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
                std::vector<float> float_data(num_points * dim);
                for (size_t i = 0; i < num_points; ++i) {
                    for (size_t j = 0; j < dim; ++j) {
                        float_data[i * dim + j] = static_cast<float>(data[i * aligned_dim + j]);
                    }
                }
                Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> data_mat(num_points, dim);
                for (size_t i = 0; i < num_points; ++i) {
                    for (size_t j = 0; j < dim; ++j) {
                        data_mat(i, j) = float_data[i * dim + j];
                    }
                }
                pca_mean_float = data_mat.colwise().mean();
                Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> centered = data_mat.rowwise() - pca_mean_float;
                Eigen::JacobiSVD<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>> svd(
                    centered, Eigen::ComputeThinU | Eigen::ComputeThinV);
                pca_components_float = svd.matrixV().leftCols(PCA_DIM);
                Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> projected =
                    centered * pca_components_float.leftCols(PCA_DIM);
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

        RegionKey compute_region_key(const T* vec) {
            RegionKey key(PCA_DIM);
            if constexpr (std::is_floating_point<T>::value) {
                Eigen::Map<const Eigen::Matrix<T, 1, Eigen::Dynamic>> v(vec, dim);
                Eigen::Matrix<T, 1, Eigen::Dynamic> proj = (v - pca_mean) * pca_components.leftCols(PCA_DIM);
                for (size_t i = 0; i < PCA_DIM; ++i) {
                    T val = proj(0, i);
                    T minv = pca_min[i], maxv = pca_max[i];
                    if (maxv == minv) {
                        key[i] = 0;
                    } else {
                        T norm = (val - minv) / (maxv - minv);
                        if (norm < static_cast<T>(0)) norm = static_cast<T>(0);
                        if (norm > static_cast<T>(1)) norm = static_cast<T>(1);
                        size_t bucket = std::min<size_t>(BUCKETS_PER_DIM - 1,
                                                        static_cast<size_t>(norm * BUCKETS_PER_DIM));
                        key[i] = static_cast<uint8_t>(bucket);
                    }
                }
            } else {
                std::vector<float> float_vec(dim);
                for (size_t j = 0; j < dim; ++j) float_vec[j] = static_cast<float>(vec[j]);
                Eigen::Map<const Eigen::Matrix<float, 1, Eigen::Dynamic>> v(float_vec.data(), dim);
                Eigen::Matrix<float, 1, Eigen::Dynamic> proj =
                    (v - pca_mean_float) * pca_components_float.leftCols(PCA_DIM);
                for (size_t i = 0; i < PCA_DIM; ++i) {
                    float val = proj(0, i);
                    float minv = pca_min_float[i], maxv = pca_max_float[i];
                    if (maxv == minv) {
                        key[i] = 0;
                    } else {
                        float norm = (val - minv) / (maxv - minv);
                        if (norm < 0.0f) norm = 0.0f;
                        if (norm > 1.0f) norm = 1.0f;
                        size_t bucket = std::min<size_t>(BUCKETS_PER_DIM - 1,
                                                        static_cast<size_t>(norm * BUCKETS_PER_DIM));
                        key[i] = static_cast<uint8_t>(bucket);
                    }
                }
            }
            return key;
        }

        double uninitialized_theta() const {
            return (metric == diskann::COSINE) ? -std::numeric_limits<double>::infinity()
                                               : std::numeric_limits<double>::max();
        }

        void reset_theta_entry(std::unordered_map<uint32_t, double>& entry) {
            const double init_value = uninitialized_theta();
            entry[1] = init_value;
            entry[5] = init_value;
            entry[10] = init_value;
            entry[100] = init_value;
        }

        bool is_uninitialized_theta(double threshold) const {
            return (metric == diskann::COSINE)
                ? (threshold == -std::numeric_limits<double>::infinity())
                : (threshold >= std::numeric_limits<double>::max() * 0.5);
        }

        bool entry_has_live_theta(const std::unordered_map<uint32_t, double>& entry) const {
            for (const auto& kv : entry) {
                if (!is_uninitialized_theta(kv.second)) {
                    return true;
                }
            }
            return false;
        }

        void invalidate_theta(const T* vec, size_t /*n_dims*/) {
            if (vec == nullptr) {
                return;
            }
            const RegionKey key = compute_region_key(vec);
            std::lock_guard<std::mutex> lock(region_theta_map_mutex);
            auto it = region_theta_map.find(key);
            if (it != region_theta_map.end()) {
                reset_theta_entry(it->second);
                region_discount_map[key] = 1.0;
            }
        }

        size_t tighten_one_region_locked(const RegionKey& fine, double alpha) {
            auto rit = region_theta_map.find(fine);
            if (rit == region_theta_map.end() || !entry_has_live_theta(rit->second)) {
                return 0;
            }
            auto dit = region_discount_map.find(fine);
            const double already = (dit == region_discount_map.end()) ? 1.0 : dit->second;
            if (already <= alpha) {
                return 0;
            }
            for (auto& kv : rit->second) {
                if (!is_uninitialized_theta(kv.second)) {
                    kv.second *= alpha;
                }
            }
            region_discount_map[fine] = alpha;
            return 1;
        }

        // Walk every 16-D code with L1 distance <= left from center.
        void tighten_l1_ball_locked(const RegionKey& center, size_t dim_i, size_t left,
                                    RegionKey& cur, size_t& n, double alpha) {
            if (dim_i == center.size()) {
                n += tighten_one_region_locked(cur, alpha);
                return;
            }
            const int c = static_cast<int>(center[dim_i]);
            const int B = static_cast<int>(BUCKETS_PER_DIM);
            const int maxd = static_cast<int>(left);
            for (int delta = -maxd; delta <= maxd; ++delta) {
                const int b = c + delta;
                if (b < 0 || b >= B) {
                    continue;
                }
                cur[dim_i] = static_cast<uint8_t>(b);
                tighten_l1_ball_locked(center, dim_i + 1, left - static_cast<size_t>(std::abs(delta)),
                                       cur, n, alpha);
            }
            cur[dim_i] = center[dim_i];
        }

        // Write-path only. θ ← α·θ on v's 16-D cell and every learned
        // cell within write_l1_radius (Manhattan on the bucket code).
        // Radius 0 = only v's cell. Saturates until a miss relearns.
        size_t tighten_theta_for_vector(const T* vec, double alpha) {
            if (vec == nullptr || alpha >= 1.0 || alpha <= 0.0) {
                return 0;
            }
            const RegionKey key = compute_region_key(vec);
            RegionKey cur = key;
            std::lock_guard<std::mutex> lock(region_theta_map_mutex);
            size_t n = 0;
            tighten_l1_ball_locked(key, 0, write_l1_radius, cur, n, alpha);
            return n;
        }

        void lazy_init_region(const RegionKey& key) {
            std::lock_guard<std::mutex> lock(region_theta_map_mutex);
            if (region_theta_map.find(key) == region_theta_map.end()) {
                if (region_theta_map.size() >= max_regions && !region_insertion_order.empty()) {
                    RegionKey oldest_key = region_insertion_order.front();
                    region_insertion_order.pop_front();
                    region_theta_map.erase(oldest_key);
                    region_discount_map.erase(oldest_key);
                }
                const double init_value = uninitialized_theta();
                region_theta_map[key][1] = init_value;
                region_theta_map[key][5] = init_value;
                region_theta_map[key][10] = init_value;
                region_theta_map[key][100] = init_value;
                region_insertion_order.push_back(key);
            }
        }

        bool isHit(const T* query_ptr, uint32_t K, const float* distances, size_t num_vectors_in_memory,
                   double deviation_factor) {
            if (num_vectors_in_memory < K || distances == nullptr || K == 0) {
                return false;
            }

            RegionKey region = compute_region_key(query_ptr);
            lazy_init_region(region);

            std::lock_guard<std::mutex> lock(region_theta_map_mutex);
            double threshold = region_theta_map[region][K];
            if (is_uninitialized_theta(threshold)) {
                return false;
            }
            const double cache_distance = static_cast<double>(distances[K - 1]);
            const double tolerance_threshold = (1.0 + deviation_factor) * threshold;
            return cache_distance <= tolerance_threshold;
        }

        void update_theta(const T* query_ptr, uint32_t K, float query_distance, double p) {
            RegionKey region = compute_region_key(query_ptr);
            lazy_init_region(region);

            std::lock_guard<std::mutex> lock(region_theta_map_mutex);
            double current_theta = region_theta_map[region][K];
            if (is_uninitialized_theta(current_theta)) {
                region_theta_map[region][K] = static_cast<double>(query_distance);
            } else {
                region_theta_map[region][K] = p * static_cast<double>(query_distance) + (1 - p) * current_theta;
            }
            region_discount_map[region] = 1.0;
        }

        std::string get_pca_filename_for_logging() const {
            return get_pca_filename();
        }

        size_t get_number_of_active_regions() const {
            std::lock_guard<std::mutex> lock(const_cast<std::mutex&>(region_theta_map_mutex));
            return region_theta_map.size();
        }
    };

} // namespace qvcache
