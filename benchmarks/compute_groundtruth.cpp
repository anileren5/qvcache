#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <chrono>
#include <cstring>
#include <cctype>
#include <limits>
#include <omp.h>

struct GroundtruthEntry {
    uint32_t id;
    float distance;
    
    bool operator<(const GroundtruthEntry& other) const {
        return distance < other.distance;
    }
};

// Read binary file format: [npts (int32), dim (int32), data (float32)]
template<typename T>
void load_bin(const std::string& filename, T*& data, size_t& npts, size_t& dim) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file: " + filename);
    }
    
    int32_t npts_i32, dim_i32;
    file.read(reinterpret_cast<char*>(&npts_i32), sizeof(int32_t));
    file.read(reinterpret_cast<char*>(&dim_i32), sizeof(int32_t));
    
    npts = static_cast<size_t>(npts_i32);
    dim = static_cast<size_t>(dim_i32);
    
    std::cout << "Reading " << filename << "..." << std::endl;
    std::cout << "  npts: " << npts << ", dim: " << dim << std::endl;
    
    size_t data_size = static_cast<size_t>(npts) * static_cast<size_t>(dim) * sizeof(T);
    data = new T[npts * dim];
    
    // Read in chunks for large files
    const size_t chunk_size = 100 * 1024 * 1024; // 100MB chunks
    size_t bytes_read = 0;
    char* data_ptr = reinterpret_cast<char*>(data);
    
    while (bytes_read < data_size) {
        size_t to_read = std::min(chunk_size, data_size - bytes_read);
        file.read(data_ptr + bytes_read, to_read);
        bytes_read += file.gcount();
        
        if (file.gcount() < static_cast<std::streamsize>(to_read)) {
            break;
        }
        
        std::cout << "\r  Progress: " << std::fixed << std::setprecision(1) 
                  << (bytes_read / (1024.0 * 1024.0)) << " MB / " 
                  << (data_size / (1024.0 * 1024.0)) << " MB" << std::flush;
    }
    std::cout << std::endl;
    
    if (bytes_read != data_size) {
        throw std::runtime_error("Failed to read all data: expected " + 
                                std::to_string(data_size) + " bytes, got " + 
                                std::to_string(bytes_read));
    }
    
    std::cout << "  Read " << npts << " vectors of dimension " << dim << std::endl;
}

// Compute L2 distance between query vector and all base vectors
template<typename T>
void compute_distances_l2(const T* query, const T* base, size_t n_base, size_t dim,
                           std::vector<float>& distances) {
    distances.resize(n_base);
    
    // Convert query to float for computation
    std::vector<float> query_float(dim);
    for (size_t d = 0; d < dim; ++d) {
        query_float[d] = static_cast<float>(query[d]);
    }
    
    // Compute ||query||^2 once
    float query_norm_sq = 0.0f;
    for (size_t d = 0; d < dim; ++d) {
        query_norm_sq += query_float[d] * query_float[d];
    }
    
    // Compute distances: ||q - b||^2 = ||q||^2 + ||b||^2 - 2*q*b
    #pragma omp parallel for
    for (size_t i = 0; i < n_base; ++i) {
        const T* base_vec = base + i * dim;
        
        // Convert base vector to float and compute ||base_vec||^2
        float base_norm_sq = 0.0f;
        float dot_product = 0.0f;
        for (size_t d = 0; d < dim; ++d) {
            float base_val = static_cast<float>(base_vec[d]);
            base_norm_sq += base_val * base_val;
            dot_product += query_float[d] * base_val;
        }
        
        // Compute squared distance
        float dist_sq = query_norm_sq + base_norm_sq - 2.0f * dot_product;
        distances[i] = std::sqrt(std::max(0.0f, dist_sq));
    }
}

// Compute inner product distance (matching DiskANN: returns -dot(a, b))
template<typename T>
void compute_distances_inner_product(const T* query, const T* base, size_t n_base, size_t dim,
                                     std::vector<float>& distances) {
    distances.resize(n_base);
    
    // Convert query to float for computation
    std::vector<float> query_float(dim);
    for (size_t d = 0; d < dim; ++d) {
        query_float[d] = static_cast<float>(query[d]);
    }
    
    #pragma omp parallel for
    for (size_t i = 0; i < n_base; ++i) {
        const T* base_vec = base + i * dim;
        
        // Compute dot product
        float dot_product = 0.0f;
        for (size_t d = 0; d < dim; ++d) {
            dot_product += query_float[d] * static_cast<float>(base_vec[d]);
        }
        
        // Inner product distance = -dot_product (matching DiskANN's DistanceInnerProduct::compare)
        distances[i] = -dot_product;
    }
}

// Compute cosine distance (matching DiskANN: returns 1.0 - (dot(a,b) / (sqrt(||a||^2) * sqrt(||b||^2))))
template<typename T>
void compute_distances_cosine(const T* query, const T* base, size_t n_base, size_t dim,
                              std::vector<float>& distances) {
    distances.resize(n_base);
    
    // Convert query to float for computation
    std::vector<float> query_float(dim);
    for (size_t d = 0; d < dim; ++d) {
        query_float[d] = static_cast<float>(query[d]);
    }
    
    // Compute ||query||^2 once
    float query_norm_sq = 0.0f;
    for (size_t d = 0; d < dim; ++d) {
        query_norm_sq += query_float[d] * query_float[d];
    }
    float query_mag = std::sqrt(query_norm_sq);
    
    #pragma omp parallel for
    for (size_t i = 0; i < n_base; ++i) {
        const T* base_vec = base + i * dim;
        
        // Compute ||base_vec||^2
        float base_norm_sq = 0.0f;
        for (size_t d = 0; d < dim; ++d) {
            float base_val = static_cast<float>(base_vec[d]);
            base_norm_sq += base_val * base_val;
        }
        float base_mag = std::sqrt(base_norm_sq);
        
        // Compute dot product
        float dot_product = 0.0f;
        for (size_t d = 0; d < dim; ++d) {
            dot_product += query_float[d] * static_cast<float>(base_vec[d]);
        }
        
        // Cosine distance = 1.0 - (dot(a,b) / (||a|| * ||b||)) (matching DiskANN's DistanceCosineFloat::compare)
        if (query_mag > 0.0f && base_mag > 0.0f) {
            distances[i] = 1.0f - (dot_product / (query_mag * base_mag));
        } else {
            distances[i] = std::numeric_limits<float>::max();
        }
    }
}

// Find top K nearest neighbors
void find_top_k(const std::vector<float>& distances, size_t k, 
                std::vector<uint32_t>& top_k_ids, std::vector<float>& top_k_dists) {
    std::vector<GroundtruthEntry> entries;
    entries.reserve(distances.size());
    
    for (size_t i = 0; i < distances.size(); ++i) {
        entries.push_back({static_cast<uint32_t>(i), distances[i]});
    }
    
    // Use partial sort for efficiency
    std::partial_sort(entries.begin(), entries.begin() + k, entries.end());
    
    top_k_ids.resize(k);
    top_k_dists.resize(k);
    
    for (size_t i = 0; i < k; ++i) {
        top_k_ids[i] = entries[i].id;
        top_k_dists[i] = entries[i].distance;
    }
}

// Save groundtruth to binary format
void save_groundtruth(const std::string& filename, 
                      const std::vector<std::vector<uint32_t>>& all_ids,
                      const std::vector<std::vector<float>>& all_distances,
                      size_t n_queries, size_t k) {
    std::ofstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file for writing: " + filename);
    }
    
    std::cout << "\nWriting groundtruth to " << filename << "..." << std::endl;
    
    // Write header: npts (int32), dim (int32)
    int32_t npts_i32 = static_cast<int32_t>(n_queries);
    int32_t dim_i32 = static_cast<int32_t>(k);
    file.write(reinterpret_cast<const char*>(&npts_i32), sizeof(int32_t));
    file.write(reinterpret_cast<const char*>(&dim_i32), sizeof(int32_t));
    
    // Write IDs (uint32_t)
    for (size_t q = 0; q < n_queries; ++q) {
        file.write(reinterpret_cast<const char*>(all_ids[q].data()), 
                   k * sizeof(uint32_t));
    }
    
    // Write distances (float32)
    for (size_t q = 0; q < n_queries; ++q) {
        file.write(reinterpret_cast<const char*>(all_distances[q].data()), 
                   k * sizeof(float));
    }
    
    file.close();
    
    size_t file_size = std::ifstream(filename, std::ios::binary | std::ios::ate).tellg();
    std::cout << "  File size: " << std::fixed << std::setprecision(2) 
              << (file_size / (1024.0 * 1024.0)) << " MB" << std::endl;
    std::cout << "  Done!" << std::endl;
}

template<typename T>
void run_compute_groundtruth(const std::string& base_file, const std::string& query_file,
                             const std::string& output_file, size_t k, const std::string& metric_str) {
    enum Metric { L2, INNER_PRODUCT, COSINE };
    Metric metric;
    if (metric_str == "l2" || metric_str == "l2_distance") {
        metric = L2;
    } else if (metric_str == "inner_product" || metric_str == "innerproduct" || metric_str == "ip") {
        metric = INNER_PRODUCT;
    } else if (metric_str == "cosine") {
        metric = COSINE;
    } else {
        std::cerr << "ERROR: Unknown metric '" << metric_str << "'. Must be l2, inner_product, or cosine." << std::endl;
        return;
    }
    
    std::cout << "=" << std::string(60, '=') << std::endl;
    std::cout << "Computing Groundtruth (C++)" << std::endl;
    std::cout << "=" << std::string(60, '=') << std::endl;
    std::cout << "\nBase file: " << base_file << std::endl;
    std::cout << "Query file: " << query_file << std::endl;
    std::cout << "Output file: " << output_file << std::endl;
    std::cout << "K: " << k << std::endl;
    std::cout << "Metric: " << metric_str << std::endl;
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // Load base vectors
    std::cout << "\n1. Loading base vectors..." << std::endl;
    T* base_data = nullptr;
    size_t n_base, base_dim;
    load_bin<T>(base_file, base_data, n_base, base_dim);
    
    // Load query vectors
    std::cout << "\n2. Loading query vectors..." << std::endl;
    T* query_data = nullptr;
    size_t n_queries, query_dim;
    load_bin<T>(query_file, query_data, n_queries, query_dim);
    
    if (base_dim != query_dim) {
        std::cerr << "ERROR: Dimension mismatch! Base dim=" << base_dim 
                  << ", Query dim=" << query_dim << std::endl;
        delete[] base_data;
        delete[] query_data;
        return;
    }
    
    size_t dim = base_dim;
    
    // Compute groundtruth
    std::cout << "\n3. Computing groundtruth..." << std::endl;
    std::cout << "  Processing " << n_queries << " queries against " << n_base 
              << " base vectors..." << std::endl;
    std::cout << "  K=" << k << ", processing queries one at a time" << std::endl;
    std::cout << "  Using " << metric_str << " distance" << std::endl;
    
    std::vector<std::vector<uint32_t>> all_ids(n_queries);
    std::vector<std::vector<float>> all_distances(n_queries);
    std::vector<float> distances(n_base);
    
    auto compute_start = std::chrono::high_resolution_clock::now();
    
    for (size_t q = 0; q < n_queries; ++q) {
        const T* query_vec = query_data + q * dim;
        
        // Compute distances based on metric
        if (metric == L2) {
            compute_distances_l2<T>(query_vec, base_data, n_base, dim, distances);
        } else if (metric == INNER_PRODUCT) {
            compute_distances_inner_product<T>(query_vec, base_data, n_base, dim, distances);
        } else if (metric == COSINE) {
            compute_distances_cosine<T>(query_vec, base_data, n_base, dim, distances);
        }
        
        // Find top K
        find_top_k(distances, k, all_ids[q], all_distances[q]);
        
        // Progress update
        if ((q + 1) % 100 == 0 || q == n_queries - 1) {
            std::cout << "\r  Processing query " << (q + 1) << "/" << n_queries 
                      << "..." << std::flush;
        }
    }
    std::cout << std::endl;
    
    auto compute_end = std::chrono::high_resolution_clock::now();
    auto compute_duration = std::chrono::duration_cast<std::chrono::seconds>(
        compute_end - compute_start).count();
    std::cout << "  Done! Time: " << compute_duration << " seconds" << std::endl;
    
    // Save groundtruth
    std::cout << "\n4. Saving groundtruth..." << std::endl;
    save_groundtruth(output_file, all_ids, all_distances, n_queries, k);
    
    // Cleanup
    delete[] base_data;
    delete[] query_data;
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto total_duration = std::chrono::duration_cast<std::chrono::seconds>(
        end_time - start_time).count();
    
    std::cout << "\n" << "=" << std::string(60, '=') << std::endl;
    std::cout << "Groundtruth computation completed successfully!" << std::endl;
    std::cout << "=" << std::string(60, '=') << std::endl;
    std::cout << "\nOutput file: " << output_file << std::endl;
    std::cout << "\nDataset statistics:" << std::endl;
    std::cout << "  Base vectors: " << n_base << " x " << dim << std::endl;
    std::cout << "  Query vectors: " << n_queries << " x " << dim << std::endl;
    std::cout << "  Groundtruth: " << n_queries << " queries x " << k << " neighbors" << std::endl;
    std::cout << "  Total time: " << total_duration << " seconds" << std::endl;
}

int main(int argc, char* argv[]) {
    if (argc < 5) {
        std::cerr << "Usage: " << argv[0] << " <base_file> <query_file> <output_file> <data_type> [K] [metric]" << std::endl;
        std::cerr << "  base_file: Path to base vectors (.bin format)" << std::endl;
        std::cerr << "  query_file: Path to query vectors (.bin format)" << std::endl;
        std::cerr << "  output_file: Path to output groundtruth file (.bin format)" << std::endl;
        std::cerr << "  data_type: Data type - float, int8, or uint8" << std::endl;
        std::cerr << "  K: Number of nearest neighbors (default: 100)" << std::endl;
        std::cerr << "  metric: Distance metric - l2, inner_product, or cosine (default: l2)" << std::endl;
        return 1;
    }
    
    std::string base_file = argv[1];
    std::string query_file = argv[2];
    std::string output_file = argv[3];
    std::string data_type = argv[4];
    size_t k = (argc > 5) ? std::stoul(argv[5]) : 100;
    std::string metric_str = (argc > 6) ? argv[6] : "l2";
    
    // Normalize data_type and metric strings
    std::transform(data_type.begin(), data_type.end(), data_type.begin(), ::tolower);
    std::transform(metric_str.begin(), metric_str.end(), metric_str.begin(), ::tolower);
    
    // Run with appropriate data type
    if (data_type == "float") {
        run_compute_groundtruth<float>(base_file, query_file, output_file, k, metric_str);
    } else if (data_type == "int8") {
        run_compute_groundtruth<int8_t>(base_file, query_file, output_file, k, metric_str);
    } else if (data_type == "uint8") {
        run_compute_groundtruth<uint8_t>(base_file, query_file, output_file, k, metric_str);
    } else {
        std::cerr << "ERROR: Unknown data type '" << data_type << "'. Must be float, int8, or uint8." << std::endl;
        return 1;
    }
    
    return 0;
}

