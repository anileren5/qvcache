#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <pybind11/functional.h>
#include <memory>
#include <vector>
#include <string>

#include "qvcache/qvcache.h"
#include "python_backend_wrapper.h"
#include "qvcache/utils.h"

namespace py = pybind11;
using namespace qvcache;

// Helper function to create Python backend wrapper
template<typename T>
std::unique_ptr<BackendInterface<T, uint32_t>> create_python_backend(
    py::object python_backend, size_t dim) {
    PyObject* backend_obj = python_backend.ptr();
    return std::make_unique<PythonBackendWrapper<T, uint32_t>>(backend_obj, dim);
}

// Helper to get dimension from data path
size_t get_dim_from_data_path(const std::string& data_path) {
    size_t num_points, dim;
    diskann::get_bin_metadata(data_path, num_points, dim);
    return dim;
}

// Helper function to initialize numpy (returns int to match import_array macro)
static int init_numpy() {
    import_array();
    return 0;
}

// Python module definition
PYBIND11_MODULE(qvcache, m) {
    // Initialize Python threading (required for GIL when called from C++ threads)
    if (!PyEval_ThreadsInitialized()) {
        PyEval_InitThreads();
    }
    
    // Initialize numpy array API
    if (PyArray_API == nullptr) {
        init_numpy();
    }

    m.doc() = "QVCache Python bindings";

    // Bind DiskANN Metric enum
    py::enum_<diskann::Metric>(m, "Metric")
        .value("L2", diskann::Metric::L2)
        .value("COSINE", diskann::Metric::COSINE)
        .value("INNER_PRODUCT", diskann::Metric::INNER_PRODUCT)
        .value("FAST_L2", diskann::Metric::FAST_L2);

    // Bind SearchStrategy enum
    py::enum_<QVCache<float>::SearchStrategy>(m, "SearchStrategy")
        .value("SEQUENTIAL_LRU_STOP_FIRST_HIT", QVCache<float>::SearchStrategy::SEQUENTIAL_LRU_STOP_FIRST_HIT)
        .value("SEQUENTIAL_LRU_ADAPTIVE", QVCache<float>::SearchStrategy::SEQUENTIAL_LRU_ADAPTIVE)
        .value("SEQUENTIAL_ALL", QVCache<float>::SearchStrategy::SEQUENTIAL_ALL)
        .value("PARALLEL", QVCache<float>::SearchStrategy::PARALLEL);

    // Bind QVCache for float
    py::class_<QVCache<float>>(m, "QVCache")
        .def(py::init([](const std::string& data_path,
                         const std::string& pca_prefix,
                         uint32_t R, uint32_t memory_L, uint32_t B, uint32_t M,
                         float alpha, uint32_t build_threads, uint32_t search_threads,
                         bool use_reconstructed_vectors, double p, double deviation_factor,
                         size_t memory_index_max_points, uint32_t beamwidth,
                         bool use_regional_theta, size_t pca_dim, size_t buckets_per_dim,
                         size_t max_regions,
                         uint32_t n_async_insert_threads, bool lazy_theta_updates,
                         size_t number_of_mini_indexes, bool search_mini_indexes_in_parallel,
                         size_t max_search_threads, diskann::Metric metric,
                         py::object python_backend) {
            if (python_backend.is_none()) {
                return std::make_unique<QVCache<float>>(
                    data_path, pca_prefix, R, memory_L, B, M, alpha,
                    build_threads, search_threads, use_reconstructed_vectors,
                    p, deviation_factor, memory_index_max_points, beamwidth,
                    use_regional_theta, pca_dim, buckets_per_dim, max_regions,
                    n_async_insert_threads, lazy_theta_updates,
                    number_of_mini_indexes, search_mini_indexes_in_parallel,
                    max_search_threads, metric, nullptr
                );
            } else {
                size_t dim = get_dim_from_data_path(data_path);
                auto backend = create_python_backend<float>(python_backend, dim);
                return std::make_unique<QVCache<float>>(
                    data_path, pca_prefix, R, memory_L, B, M, alpha,
                    build_threads, search_threads, use_reconstructed_vectors,
                    p, deviation_factor, memory_index_max_points, beamwidth,
                    use_regional_theta, pca_dim, buckets_per_dim, max_regions,
                    n_async_insert_threads, lazy_theta_updates,
                    number_of_mini_indexes, search_mini_indexes_in_parallel,
                    max_search_threads, metric, std::move(backend)
                );
            }
        }),
            py::arg("data_path"),
            py::arg("pca_prefix"),
            py::arg("R"),
            py::arg("memory_L"),
            py::arg("B"),
            py::arg("M"),
            py::arg("alpha"),
            py::arg("build_threads"),
            py::arg("search_threads"),
            py::arg("use_reconstructed_vectors"),
            py::arg("p"),
            py::arg("deviation_factor"),
            py::arg("memory_index_max_points"),
            py::arg("beamwidth"),
            py::arg("use_regional_theta") = true,
            py::arg("pca_dim") = 16,
            py::arg("buckets_per_dim") = 4,
            py::arg("max_regions") = std::numeric_limits<size_t>::max(),
            py::arg("n_async_insert_threads") = 4,
            py::arg("lazy_theta_updates") = true,
            py::arg("number_of_mini_indexes") = 2,
            py::arg("search_mini_indexes_in_parallel") = false,
            py::arg("max_search_threads") = 32,
            py::arg("metric") = diskann::Metric::L2,
            py::arg("backend") = py::none(),
            "Construct QVCache with optional Python backend")
        .def("search", [](QVCache<float>& index, 
                          py::array_t<float> query,
                          uint32_t K) {
            py::buffer_info query_buf = query.request();
            if (query_buf.ndim != 1) {
                throw std::runtime_error("Query must be a 1D array");
            }
            
            // Ensure query is contiguous
            py::array_t<float> query_contiguous = py::array_t<float>::ensure(query);
            py::buffer_info query_contig_buf = query_contiguous.request();
            float* query_ptr = static_cast<float*>(query_contig_buf.ptr);
            
            std::vector<uint32_t> result_tags(K);
            std::vector<float> result_dists(K);
            std::vector<float*> res;
            
            bool hit = index.search(query_ptr, K, result_tags.data(), res, result_dists.data(), nullptr);
            
            // Convert to numpy arrays (copy data)
            // Based on Reddit post: numpy 2.x compatibility issue with pybind11
            // Solution: Use py::cast to convert vectors to numpy arrays - this handles the copy correctly
            // and avoids the issue where all elements become copies of the first element
            py::array_t<uint32_t> tags_array = py::cast(result_tags);
            py::array_t<float> dists_array = py::cast(result_dists);
            
            return py::make_tuple(
                py::cast(hit),
                tags_array,
                dists_array
            );
        }, "Search for K nearest neighbors")
        .def("set_search_strategy", &QVCache<float>::set_search_strategy)
        .def("get_search_strategy", &QVCache<float>::get_search_strategy)
        .def("enable_adaptive_strategy", &QVCache<float>::enable_adaptive_strategy)
        .def("set_hit_ratio_window_size", &QVCache<float>::set_hit_ratio_window_size)
        .def("set_hit_ratio_threshold", &QVCache<float>::set_hit_ratio_threshold)
        .def("get_number_of_vectors_in_memory_index", &QVCache<float>::get_number_of_vectors_in_memory_index)
        .def("get_number_of_max_points_in_memory_index", &QVCache<float>::get_number_of_max_points_in_memory_index)
        .def("get_number_of_mini_indexes", &QVCache<float>::get_number_of_mini_indexes)
        .def("get_index_vector_count", &QVCache<float>::get_index_vector_count)
        .def("get_number_of_active_pca_regions", &QVCache<float>::get_number_of_active_pca_regions);

    // Helper function to create QVCache with Python backend
    m.def("create_qvcache_with_python_backend", 
        [](const std::string& data_path,
           const std::string& pca_prefix,
           uint32_t R, uint32_t memory_L, uint32_t B, uint32_t M,
           float alpha, uint32_t build_threads, uint32_t search_threads,
           bool use_reconstructed_vectors, double p, double deviation_factor,
           size_t memory_index_max_points, uint32_t beamwidth,
           bool use_regional_theta, size_t pca_dim, size_t buckets_per_dim,
           size_t max_regions,
           uint32_t n_async_insert_threads, bool lazy_theta_updates,
           size_t number_of_mini_indexes, bool search_mini_indexes_in_parallel,
           size_t max_search_threads, diskann::Metric metric,
           py::object python_backend) {
            
            size_t dim = get_dim_from_data_path(data_path);
            auto backend = create_python_backend<float>(python_backend, dim);
            
            return std::make_unique<QVCache<float>>(
                data_path, pca_prefix, R, memory_L, B, M, alpha,
                build_threads, search_threads, use_reconstructed_vectors,
                p, deviation_factor, memory_index_max_points, beamwidth,
                use_regional_theta, pca_dim, buckets_per_dim, max_regions,
                n_async_insert_threads, lazy_theta_updates,
                number_of_mini_indexes, search_mini_indexes_in_parallel,
                max_search_threads, metric, std::move(backend)
            );
        },
        "Create QVCache with a Python backend implementation");

    // Utility functions
    m.def("load_aligned_binary_data", [](const std::string& file_path) {
        float* data = nullptr;
        size_t num, dim, aligned_dim;
        load_aligned_binary_data<float>(file_path, data, num, dim, aligned_dim);
        
        // Create numpy array (reshape to num x dim)
        py::array_t<float> result({static_cast<py::ssize_t>(num), static_cast<py::ssize_t>(dim)});
        py::buffer_info buf = result.request();
        
        // Copy data (only the actual dim, not aligned_dim)
        for (size_t i = 0; i < num; ++i) {
            std::memcpy(static_cast<float*>(buf.ptr) + i * dim, 
                       data + i * aligned_dim, 
                       dim * sizeof(float));
        }
        
        // Use aligned_free for aligned memory allocated by load_aligned_bin
        diskann::aligned_free(data);
        
        return py::make_tuple(result, static_cast<int>(dim), static_cast<int>(aligned_dim));
    }, "Load binary data file");

    m.def("load_ground_truth_data", [](const std::string& file_path) {
        uint32_t* ids = nullptr;
        float* dists = nullptr;
        size_t num, dim;
        load_ground_truth_data<uint32_t>(file_path, ids, dists, num, dim);
        
        // Create numpy arrays with explicit writeable flag
        py::array_t<uint32_t> ids_array = py::array_t<uint32_t>({static_cast<py::ssize_t>(num), static_cast<py::ssize_t>(dim)});
        ids_array.attr("flags").attr("writeable") = true;
        
        py::array_t<float> dists_array = py::array_t<float>({static_cast<py::ssize_t>(num), static_cast<py::ssize_t>(dim)});
        dists_array.attr("flags").attr("writeable") = true;
        
        // Get mutable data pointers
        uint32_t* ids_data = static_cast<uint32_t*>(ids_array.mutable_data());
        float* dists_data = static_cast<float*>(dists_array.mutable_data());
        
        // Copy data
        if (ids && ids_data) {
            std::memcpy(ids_data, ids, num * dim * sizeof(uint32_t));
        }
        if (dists && dists_data) {
            std::memcpy(dists_data, dists, num * dim * sizeof(float));
        }
        
        // Free original memory
        if (ids) {
            delete[] ids;
        }
        if (dists) {
            delete[] dists;
        }
        
        return py::make_tuple(ids_array, dists_array);
    }, "Load ground truth data file");

    m.def("calculate_recall", [](size_t K, py::array_t<uint32_t> groundtruth_ids,
                                  py::array_t<uint32_t> query_result_tags,
                                  size_t query_num, size_t groundtruth_dim) {
        py::buffer_info gt_buf = groundtruth_ids.request();
        py::buffer_info res_buf = query_result_tags.request();
        
        uint32_t* gt_ptr = static_cast<uint32_t*>(gt_buf.ptr);
        uint32_t* res_ptr = static_cast<uint32_t*>(res_buf.ptr);
        
        std::vector<uint32_t> result_tags_vec(res_ptr, res_ptr + query_num * K);
        calculate_recall<float, uint32_t>(K, gt_ptr, result_tags_vec, query_num, groundtruth_dim);
    }, "Calculate recall");

    m.def("calculate_hit_recall", [](size_t K, py::array_t<uint32_t> groundtruth_ids,
                                      py::array_t<uint32_t> query_result_tags,
                                      const std::vector<bool>& hit_results,
                                      size_t query_num, size_t groundtruth_dim) {
        py::buffer_info gt_buf = groundtruth_ids.request();
        py::buffer_info res_buf = query_result_tags.request();
        
        uint32_t* gt_ptr = static_cast<uint32_t*>(gt_buf.ptr);
        uint32_t* res_ptr = static_cast<uint32_t*>(res_buf.ptr);
        
        std::vector<uint32_t> result_tags_vec(res_ptr, res_ptr + query_num * K);
        calculate_hit_recall<float, uint32_t>(K, gt_ptr, result_tags_vec, hit_results, query_num, groundtruth_dim);
    }, "Calculate hit recall");
}

