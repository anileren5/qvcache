#pragma once

#include "qvcache/backend_interface.h"
#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>
#include <vector>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <limits>
#include <cstring>

namespace qvcache {

// Python backend wrapper that allows implementing backends in Python
template <typename T, typename TagT = uint32_t>
class PythonBackendWrapper : public BackendInterface<T, TagT> {
private:
    PyObject* python_backend_obj;  // Python object that implements the backend
    size_t dim;  // Dimension of vectors
    bool owns_python_obj;  // Whether we should decref the Python object

public:
    // Constructor that takes a Python object
    // The Python object must have:
    //   - search(query: numpy array, K: int) -> tuple of (tags: numpy array, distances: numpy array)
    //   - fetch_vectors_by_ids(ids: list) -> list of numpy arrays
    PythonBackendWrapper(PyObject* python_backend, size_t vector_dim)
        : python_backend_obj(python_backend), dim(vector_dim), owns_python_obj(false) {
        if (python_backend_obj == nullptr) {
            throw std::runtime_error("Python backend object cannot be nullptr");
        }
        Py_INCREF(python_backend_obj);  // Increment reference count
        owns_python_obj = true;
    }

    ~PythonBackendWrapper() {
        if (owns_python_obj && python_backend_obj != nullptr) {
            Py_DECREF(python_backend_obj);
        }
    }

    // Non-copyable
    PythonBackendWrapper(const PythonBackendWrapper&) = delete;
    PythonBackendWrapper& operator=(const PythonBackendWrapper&) = delete;

    // Movable
    PythonBackendWrapper(PythonBackendWrapper&& other) noexcept
        : python_backend_obj(other.python_backend_obj),
          dim(other.dim),
          owns_python_obj(other.owns_python_obj) {
        other.python_backend_obj = nullptr;
        other.owns_python_obj = false;
    }

    // Helper method to get search results as vectors (easier to work with)
    std::pair<std::vector<TagT>, std::vector<float>> search_impl(
        const T *query,
        uint64_t K) {
        
        if (python_backend_obj == nullptr) {
            throw std::runtime_error("Python backend object is null");
        }

        // Acquire Python GIL (critical when called from C++ threads)
        PyGILState_STATE gstate = PyGILState_Ensure();

        std::vector<TagT> tags_vec;
        std::vector<float> dists_vec;

        try {
            // Ensure numpy array API is initialized (must be called once at module init)
            // This is handled by the PYBIND11_MODULE macro, so we just check
            if (PyArray_API == nullptr) {
                PyGILState_Release(gstate);
                throw std::runtime_error("NumPy array API not initialized. Call import_array() in module init.");
            }
        
        // Create numpy array from query (copy to ensure it's writable)
        npy_intp dims[1] = {static_cast<npy_intp>(dim)};
        int numpy_type = std::is_same<T, float>::value ? NPY_FLOAT32 :
                         std::is_same<T, int8_t>::value ? NPY_INT8 :
                         std::is_same<T, uint8_t>::value ? NPY_UINT8 : NPY_FLOAT32;
        
        PyObject* query_array = PyArray_SimpleNew(1, dims, numpy_type);
        if (query_array == nullptr) {
            PyErr_Print();
            throw std::runtime_error("Failed to create query numpy array");
        }
        
        // Copy query data
        void* query_data = PyArray_DATA((PyArrayObject*)query_array);
        std::memcpy(query_data, query, dim * sizeof(T));

        // Call Python search method
        PyObject* search_method = PyObject_GetAttrString(python_backend_obj, "search");
        if (search_method == nullptr || !PyCallable_Check(search_method)) {
            Py_DECREF(query_array);
            PyErr_Print();
            throw std::runtime_error("Python backend object does not have a callable 'search' method");
        }

        PyObject* args = PyTuple_New(2);
        PyTuple_SetItem(args, 0, query_array);
        PyTuple_SetItem(args, 1, PyLong_FromUnsignedLongLong(K));

        PyObject* result = PyObject_CallObject(search_method, args);
        Py_DECREF(args);
        Py_DECREF(search_method);

        if (result == nullptr) {
            PyErr_Print();
            throw std::runtime_error("Python search method call failed");
        }

        // Extract results from tuple (tags, distances)
        if (!PyTuple_Check(result) || PyTuple_Size(result) != 2) {
            Py_DECREF(result);
            throw std::runtime_error("Python search method must return a tuple of (tags, distances)");
        }

        // Get items from tuple - these are borrowed references, so we need to incref
        PyObject* tags_obj = PyTuple_GetItem(result, 0);
        PyObject* dists_obj = PyTuple_GetItem(result, 1);
        
        // Increment reference counts since PyTuple_GetItem returns borrowed references
        // This is important to ensure the objects stay alive during conversion
        Py_INCREF(tags_obj);
        Py_INCREF(dists_obj);

        // Convert tags to C array - first check if it's already a numpy array
        PyArrayObject* tags_array = nullptr;
        if (PyArray_Check(tags_obj)) {
            // It's already a numpy array - get a contiguous view/copy
            tags_array = (PyArrayObject*)PyArray_GETCONTIGUOUS((PyArrayObject*)tags_obj);
            // If GETCONTIGUOUS returned a new array, we need to manage it
            // If it returned the same array, we incremented the refcount
            if (tags_array != (PyArrayObject*)tags_obj) {
                // New array was created, tags_obj refcount unchanged
            } else {
                // Same array returned, refcount was incremented
            }
        } else {
            // Convert to numpy array
            tags_array = (PyArrayObject*)PyArray_FromAny(
                tags_obj, PyArray_DescrFromType(NPY_UINT32), 1, 1, 
                NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_FORCECAST, nullptr);
        }
        if (tags_array == nullptr) {
            Py_DECREF(tags_obj);
            Py_DECREF(dists_obj);
            Py_DECREF(result);
            PyErr_Print();
            throw std::runtime_error("Failed to convert tags to numpy array");
        }

        // Convert distances to C array
        PyArrayObject* dists_array = nullptr;
        if (PyArray_Check(dists_obj)) {
            dists_array = (PyArrayObject*)PyArray_GETCONTIGUOUS((PyArrayObject*)dists_obj);
        } else {
            dists_array = (PyArrayObject*)PyArray_FromAny(
                dists_obj, PyArray_DescrFromType(NPY_FLOAT32), 1, 1,
                NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_FORCECAST, nullptr);
        }
        if (dists_array == nullptr) {
            Py_DECREF(tags_obj);
            Py_DECREF(dists_obj);
            Py_DECREF(tags_array);
            Py_DECREF(result);
            PyErr_Print();
            throw std::runtime_error("Failed to convert distances to numpy array");
        }
        
        // Now we can decref the original objects (we increfed them earlier)
        Py_DECREF(tags_obj);
        Py_DECREF(dists_obj);

        // Copy results
        // Verify arrays are 1D
        if (PyArray_NDIM(tags_array) != 1 || PyArray_NDIM(dists_array) != 1) {
            Py_DECREF(tags_array);
            Py_DECREF(dists_array);
            Py_DECREF(result);
            throw std::runtime_error("Python backend must return 1D arrays for tags and distances");
        }
        
        // Get the actual size - use PyArray_SIZE for total elements (more reliable)
        npy_intp tags_size = PyArray_SIZE(tags_array);
        npy_intp dists_size = PyArray_SIZE(dists_array);
        npy_intp tags_len = PyArray_DIM(tags_array, 0);
        npy_intp dists_len = PyArray_DIM(dists_array, 0);
        
        // For 1D arrays, size should equal the first dimension
        if (tags_size != tags_len || dists_size != dists_len) {
            Py_DECREF(tags_array);
            Py_DECREF(dists_array);
            Py_DECREF(result);
            throw std::runtime_error("Array size mismatch - expected 1D arrays");
        }
        
        size_t result_size = std::min(static_cast<size_t>(K), 
                                     std::min(static_cast<size_t>(tags_size), 
                                             static_cast<size_t>(dists_size)));
        
        // Verify we have the expected size
        if (result_size == 0) {
            Py_DECREF(tags_array);
            Py_DECREF(dists_array);
            Py_DECREF(result);
            throw std::runtime_error("Backend returned empty results");
        }
        
        // Ensure we copy at least K elements if available
        if (tags_size >= static_cast<npy_intp>(K) && dists_size >= static_cast<npy_intp>(K)) {
            result_size = static_cast<size_t>(K);
        }
        
        // Get data pointers - arrays are guaranteed contiguous after PyArray_FromAny with NPY_ARRAY_C_CONTIGUOUS
        TagT* tags_data = static_cast<TagT*>(PyArray_DATA(tags_array));
        float* dists_data = static_cast<float*>(PyArray_DATA(dists_array));
        
        // Verify the data pointer is valid
        if (tags_data == nullptr || dists_data == nullptr) {
            Py_DECREF(tags_array);
            Py_DECREF(dists_array);
            Py_DECREF(result);
            throw std::runtime_error("Backend returned null data pointers");
        }
        
        // Copy results to vectors - resize first, then copy
        tags_vec.resize(result_size);
        dists_vec.resize(result_size);
        
        
        // Copy element by element - use memcpy for efficiency since arrays are contiguous
        std::memcpy(tags_vec.data(), tags_data, result_size * sizeof(TagT));
        std::memcpy(dists_vec.data(), dists_data, result_size * sizeof(float));
        
        // Verify the copy worked - check if all elements are the same
        if (result_size > 1) {
            bool all_same = true;
            for (size_t i = 1; i < result_size; ++i) {
                if (tags_vec[i] != tags_vec[0]) {
                    all_same = false;
                    break;
                }
            }
            // If all are the same and we expected different values, something is wrong
            // But we can't throw here as this might be valid (unlikely for search results)
        }

        // Clean up
        Py_DECREF(tags_array);
        Py_DECREF(dists_array);
        Py_DECREF(result);
        
        } catch (...) {
            // Release GIL before rethrowing
            PyGILState_Release(gstate);
            throw;
        }
        
        // Release Python GIL
        PyGILState_Release(gstate);
        
        return std::make_pair(std::move(tags_vec), std::move(dists_vec));
    }

    void search(
        const T *query,
        uint64_t K,
        TagT* result_tags,
        float* result_distances,
        void* search_parameters = nullptr,
        void* stats = nullptr) override {
        
        // Get results as vectors
        auto [tags_vec, dists_vec] = search_impl(query, K);
        
        // Verify vectors have correct size
        if (tags_vec.size() == 0 || dists_vec.size() == 0) {
            // Fill with defaults
            for (size_t i = 0; i < static_cast<size_t>(K); ++i) {
                result_tags[i] = 0;
                result_distances[i] = std::numeric_limits<float>::infinity();
            }
            return;
        }
        
        // Copy to output arrays
        size_t result_size = std::min(tags_vec.size(), static_cast<size_t>(K));
        result_size = std::min(result_size, dists_vec.size());
        
        // Copy element by element to ensure correctness
        for (size_t i = 0; i < result_size; ++i) {
            result_tags[i] = tags_vec[i];
            result_distances[i] = dists_vec[i];
        }
        
        // Fill remaining if needed
        for (size_t i = result_size; i < static_cast<size_t>(K); ++i) {
            result_tags[i] = 0;
            result_distances[i] = std::numeric_limits<float>::infinity();
        }
    }

    std::vector<std::vector<T>> fetch_vectors_by_ids(
        const std::vector<TagT> &ids) override {
        
        if (python_backend_obj == nullptr) {
            throw std::runtime_error("Python backend object is null");
        }

        // Acquire Python GIL (critical when called from C++ threads)
        PyGILState_STATE gstate = PyGILState_Ensure();

        try {
            // Create Python list from ids
            PyObject* ids_list = PyList_New(ids.size());
            for (size_t i = 0; i < ids.size(); ++i) {
                PyList_SetItem(ids_list, i, PyLong_FromUnsignedLong(ids[i]));
            }

            // Call Python fetch_vectors_by_ids method
            PyObject* fetch_method = PyObject_GetAttrString(python_backend_obj, "fetch_vectors_by_ids");
            if (fetch_method == nullptr || !PyCallable_Check(fetch_method)) {
                Py_DECREF(ids_list);
                PyGILState_Release(gstate);
                PyErr_Print();
                throw std::runtime_error("Python backend object does not have a callable 'fetch_vectors_by_ids' method");
            }

            PyObject* args = PyTuple_New(1);
            PyTuple_SetItem(args, 0, ids_list);

            PyObject* result = PyObject_CallObject(fetch_method, args);
            Py_DECREF(args);
            Py_DECREF(fetch_method);

            if (result == nullptr) {
                PyGILState_Release(gstate);
                PyErr_Print();
                throw std::runtime_error("Python fetch_vectors_by_ids method call failed");
            }

            if (!PyList_Check(result)) {
                Py_DECREF(result);
                PyGILState_Release(gstate);
                throw std::runtime_error("Python fetch_vectors_by_ids must return a list");
            }

            // Convert Python list of arrays to C++ vector of vectors
            std::vector<std::vector<T>> vectors;
            Py_ssize_t num_vectors = PyList_Size(result);
            vectors.reserve(num_vectors);

            for (Py_ssize_t i = 0; i < num_vectors; ++i) {
                PyObject* vec_obj = PyList_GetItem(result, i);
                int numpy_type = std::is_same<T, float>::value ? NPY_FLOAT32 :
                                 std::is_same<T, int8_t>::value ? NPY_INT8 :
                                 std::is_same<T, uint8_t>::value ? NPY_UINT8 : NPY_FLOAT32;
                PyArrayObject* vec_array = (PyArrayObject*)PyArray_FromAny(
                    vec_obj, PyArray_DescrFromType(numpy_type), 0, 0, NPY_ARRAY_IN_ARRAY, nullptr);

                if (vec_array == nullptr) {
                    Py_DECREF(result);
                    PyGILState_Release(gstate);
                    PyErr_Print();
                    throw std::runtime_error("Failed to convert vector to numpy array");
                }

                std::vector<T> vec(dim);
                T* vec_data = (T*)PyArray_DATA(vec_array);
                std::memcpy(vec.data(), vec_data, dim * sizeof(T));
                vectors.push_back(std::move(vec));

                Py_DECREF(vec_array);
            }

            Py_DECREF(result);
            
            // Release GIL before returning
            PyGILState_Release(gstate);
            return vectors;
            
        } catch (...) {
            // Release GIL before rethrowing
            PyGILState_Release(gstate);
            throw;
        }
    }
};

}  // namespace qvcache

