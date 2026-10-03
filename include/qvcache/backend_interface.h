#pragma once

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace qvcache
{

template <typename T, typename TagT>
class BackendInterface
{
  public:
    virtual ~BackendInterface() = default;

    virtual void search(
        const T *query,
        uint64_t K,
        TagT* result_tags,
        float* result_distances,
        void* search_parameters = nullptr,
        void* stats = nullptr) = 0;

    virtual std::vector<std::vector<T>> fetch_vectors_by_ids(
        const std::vector<TagT> &ids) = 0;

    virtual bool supports_updates() const { return false; }

    virtual void insert(TagT id, const T* vector) {
        (void)id;
        (void)vector;
        throw std::runtime_error("backend does not support insert");
    }

    virtual void remove(TagT id) {
        (void)id;
        throw std::runtime_error("backend does not support delete");
    }
};

} 
