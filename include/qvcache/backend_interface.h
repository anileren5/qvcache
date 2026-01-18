#pragma once

#include <vector>
#include <cstdint>

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
};

} 
