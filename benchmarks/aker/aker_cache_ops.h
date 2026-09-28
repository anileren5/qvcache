#pragma once

#include <chrono>
#include <cstdint>
#include <cstring>
#include <type_traits>
#include <vector>

#include "ak_anns_cache_c_wrapper.h"
#include "aker_c_abi_helpers.h"
#include "diskann/distance.h"
#include "pgvector_backend.h"

namespace aker_ops {

inline void noop_transform(uint64_t, uint8_t*, size_t, uint64_t, uint64_t) {}

template <typename T>
bool (*transform_for_type())(void*, size_t, size_t, void*, uint8_t*) {
    if constexpr (std::is_same_v<T, float>)
        return aker_transform_float;
    else if constexpr (std::is_same_v<T, int8_t>)
        return aker_transform_int8;
    else
        return aker_transform_uint8;
}

template <typename T>
float (*distance_for_type(diskann::Metric metric))(uint8_t*, uint8_t*, size_t) {
    if constexpr (std::is_same_v<T, float>) {
        if (metric == diskann::Metric::INNER_PRODUCT)
            return aker_ip_float_bytes;
        return aker_l2_float_bytes;
    } else if constexpr (std::is_same_v<T, int8_t>) {
        return aker_l2_int8_bytes;
    } else {
        return aker_l2_uint8_bytes;
    }
}

template <typename T>
void apply_insert(
    anns_cache_c_wrapper_t* cache,
    qvcache::PgVectorBackend<T>& backend,
    uint32_t id,
    const T* vec,
    uint32_t dim,
    size_t vector_in_bytes,
    diskann::Metric metric,
    bool process_log
) {
    backend.insert(id, vec);
    const uint64_t vid = static_cast<uint64_t>(id) + 1;
    char* slot = akerCreateVectorSlot(
        vid, vector_in_bytes, reinterpret_cast<char*>(const_cast<T*>(vec)), 0, 0, 0.0f);
    char* view = akerCreateVectorView(
        slot, static_cast<size_t>(dim), vector_in_bytes, transform_for_type<T>());
    akerInsertWriteLogEntry(cache, view, distance_for_type<T>(metric), noop_transform);
    if (process_log) {
        akerProcessWriteLogEntries(cache, distance_for_type<T>(metric), noop_transform);
    }
    akerDestroyVectorView(view);
    akerDestroyVectorSlot(slot);
}

template <typename T>
void apply_delete(
    anns_cache_c_wrapper_t* cache,
    qvcache::PgVectorBackend<T>& backend,
    uint32_t id
) {
    backend.remove(id);
    akerMarkVectorDeleted(cache, static_cast<uint64_t>(id) + 1);
}

template <typename T>
void process_log(anns_cache_c_wrapper_t* cache, diskann::Metric metric) {
    akerProcessWriteLogEntries(cache, distance_for_type<T>(metric), noop_transform);
}

template <typename T>
bool search(
    anns_cache_c_wrapper_t* cache,
    qvcache::PgVectorBackend<T>& backend,
    const T* query,
    uint32_t dim,
    uint32_t K,
    uint32_t slot_list_size,
    size_t vector_in_bytes,
    diskann::Metric metric,
    uint32_t* out_tags,
    double* miss_penalty_ms = nullptr
) {
    const auto lookup_start = std::chrono::high_resolution_clock::now();
    std::vector<T> packed(dim);
    std::memcpy(packed.data(), query, dim * sizeof(T));
    uint64_t query_id = akerDefaultHash(reinterpret_cast<char*>(packed.data()), vector_in_bytes);
    if (query_id == 0) query_id = 1;

    char* query_slot = akerCreateVectorSlot(
        query_id, vector_in_bytes, reinterpret_cast<char*>(packed.data()), 0, 0, 0.0f);
    char* query_view = akerCreateVectorView(
        query_slot, static_cast<size_t>(dim), vector_in_bytes, transform_for_type<T>());

    bool similar = false, invalid = false;
    char* entry = akerGetCacheEntry(cache, query_view, &similar, &invalid, distance_for_type<T>(metric));
    const bool hit = (entry != nullptr && !invalid);
    if (hit) {
        for (uint32_t j = 0; j < K; ++j) {
            char* slot = akerGetResultVectorSlotAt(entry, static_cast<int>(j));
            out_tags[j] = slot ? static_cast<uint32_t>(akerGetVectorIdFromVectorSlot(slot)) : 0;
        }
        akerDestroyCacheEntry(entry);
        if (miss_penalty_ms != nullptr) {
            *miss_penalty_ms = 0.0;
        }
    } else {
        if (entry) akerDestroyCacheEntry(entry);
        if (miss_penalty_ms != nullptr) {
            *miss_penalty_ms = std::chrono::duration<double, std::milli>(
                std::chrono::high_resolution_clock::now() - lookup_start).count();
        }
        std::vector<uint32_t> tags(slot_list_size, 0);
        std::vector<float> dists(slot_list_size, 0.0f);
        backend.search(query, slot_list_size, tags.data(), dists.data(), nullptr, nullptr);
        for (uint32_t j = 0; j < K; ++j) out_tags[j] = tags[j] + 1;

        std::vector<char*> neighbor_slots(slot_list_size, nullptr);
        auto vecs = backend.fetch_vectors_by_ids(std::vector<uint32_t>(tags.begin(), tags.begin() + slot_list_size));
        for (uint32_t j = 0; j < slot_list_size; ++j) {
            neighbor_slots[j] = akerCreateVectorSlot(
                static_cast<uint64_t>(tags[j]) + 1, vector_in_bytes,
                reinterpret_cast<char*>(vecs[j].data()), 0, 0, dists[j]);
        }
        char* new_entry = akerCreateCacheEntry(cache, query_slot, slot_list_size, neighbor_slots.data());
        if (new_entry != nullptr && !akerInsertCacheEntry(cache, query_id, new_entry, query_view)) {
            akerDestroyCacheEntry(new_entry);
        }
        for (char* s : neighbor_slots) akerDestroyVectorSlot(s);
    }
    akerDestroyVectorView(query_view);
    akerDestroyVectorSlot(query_slot);
    return hit;
}

}  // namespace aker_ops
