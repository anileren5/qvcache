/**
 * @file ak_anns_cache_c_wrapper.cc
 * @brief C ABI wrapper implementations for Aker ANNSCache.
 */

#include "ak_anns_cache_c_wrapper.h"

#include "core/ak_anns_cache.hh"
#include "utils/ak_param_parser.hh"
#include "utils/ak_distance_funcs.hh"

#include "xxHash/xxh3.h"

#include <cassert>
#include <cstdlib>
#include <cstring>

#include <map>
#include <string>

/**
 * This helper map is used by C-side tests/tools to store opaque pointers.
 */
static std::map<uint64_t, char*> helper_map;

/**
 * Pointer conversion helpers.
 */
static inline aker::ANNSCache* asCache(char* p)
{
    return reinterpret_cast<aker::ANNSCache*>(p);
}

static inline char* asChar(aker::ANNSCache* c)
{
    return reinterpret_cast<char*>(c);
}

static inline aker::VectorSlot* asVectorSlot(char* p)
{
    return reinterpret_cast<aker::VectorSlot*>(p);
}

static inline char* asChar(aker::VectorSlot* v)
{
    return reinterpret_cast<char*>(v);
}

static inline aker::vector_view_t* asVectorView(char* p)
{
    return reinterpret_cast<aker::vector_view_t*>(p);
}

static inline char* asChar(aker::vector_view_t* v)
{
    return reinterpret_cast<char*>(v);
}

static inline aker::anns_cache_entry_t* asCacheEntry(char* e)
{
    return reinterpret_cast<aker::anns_cache_entry_t*>(e);
}

static inline char* asChar(aker::anns_cache_entry_t* e)
{
    return reinterpret_cast<char*>(e);
}

/**
 * Adapts a C ABI distance function pointer to the internal std::function signature.
 *
 * The internal cache treats vector buffers as read-only (const pointers).
 * The C ABI historically used non-const pointers, so we forward using const_cast.
 * Callers must NOT mutate the input buffers.
 */
static inline aker::distance_function_t adaptDistanceFunction(
    float (*distance_function)(uint8_t*, uint8_t*, size_t))
{
    if (distance_function == nullptr)
        return aker::distance_function_t{};

    return [distance_function](
        const aker::vector_data_t* vector1,
        const aker::vector_data_t* vector2,
        size_t dimension) -> float
    {
        return distance_function(
            const_cast<aker::vector_data_t*>(vector1),
            const_cast<aker::vector_data_t*>(vector2),
            dimension);
    };
}

void akerImportAnnsCacheConfig(char* path, anns_cache_parameter_c_t* parameter)
{
    if (path == nullptr || parameter == nullptr)
        return;

    /* Load config from a parameter file. */
    std::string file_path(path);
    aker::ParameterParser parser(file_path);
    aker::anns_cache_config_t parameter_info = parser.getParameter();

    parameter->vector_format.dimension = parameter_info.vector_format.dimension;
    parameter->vector_format.vector_in_bytes = parameter_info.vector_format.vector_in_bytes;

    parameter->capacity.pool_size = parameter_info.capacity.pool_size;
    parameter->capacity.in_topk = parameter_info.capacity.in_topk;
    parameter->capacity.top_delta = parameter_info.capacity.top_delta;

    parameter->tuning.global_thresh = parameter_info.tuning.global_thresh;
    parameter->tuning.dropout = parameter_info.tuning.dropout;
    parameter->tuning.risk_thresh = parameter_info.tuning.risk_thresh;
    parameter->tuning.alpha_tighten = parameter_info.tuning.alpha_tighten;
    parameter->tuning.alpha_loosen = parameter_info.tuning.alpha_loosen;

    parameter->distance_metric = static_cast<uint8_t>(parameter_info.distance_metric);
}

bool akerDefaultConversionFunction(void* src, size_t src_size, size_t dim, void* dst, uint8_t* aux)
{
    (void)src;
    (void)src_size;
    (void)dim;
    (void)dst;
    (void)aux;

    /* Dummy conversion for tests. */
    return true;
}

anns_cache_c_wrapper_t* akerCreateAnnsCache(anns_cache_parameter_c_t parameter)
{
    /* Convert C config struct to C++ config struct. */
    aker::anns_cache_config_t parameter_info;
    parameter_info.vector_format.dimension = parameter.vector_format.dimension;
    parameter_info.vector_format.vector_in_bytes = parameter.vector_format.vector_in_bytes;

    parameter_info.capacity.pool_size = parameter.capacity.pool_size;
    parameter_info.capacity.in_topk = parameter.capacity.in_topk;
    parameter_info.capacity.top_delta = parameter.capacity.top_delta;

    parameter_info.tuning.global_thresh = parameter.tuning.global_thresh;
    parameter_info.tuning.dropout = parameter.tuning.dropout;
    parameter_info.tuning.risk_thresh = parameter.tuning.risk_thresh;
    parameter_info.tuning.alpha_tighten = parameter.tuning.alpha_tighten;
    parameter_info.tuning.alpha_loosen = parameter.tuning.alpha_loosen;

    parameter_info.distance_metric = static_cast<aker::distance_metric_t>(parameter.distance_metric);

    /* Create wrapper and cache instance. */
    anns_cache_c_wrapper_t* wrapper = new anns_cache_c_wrapper_t;
    wrapper->anns_cache = asChar(new aker::ANNSCache(parameter_info));

    /* Echo the resolved parameters back to the wrapper. */
    wrapper->parameter.vector_format.dimension = parameter_info.vector_format.dimension;
    wrapper->parameter.vector_format.vector_in_bytes = parameter_info.vector_format.vector_in_bytes;

    wrapper->parameter.capacity.pool_size = parameter_info.capacity.pool_size;
    wrapper->parameter.capacity.in_topk = parameter_info.capacity.in_topk;
    wrapper->parameter.capacity.top_delta = parameter_info.capacity.top_delta;

    wrapper->parameter.tuning.global_thresh = parameter_info.tuning.global_thresh;
    wrapper->parameter.tuning.dropout = parameter_info.tuning.dropout;
    wrapper->parameter.tuning.risk_thresh = parameter_info.tuning.risk_thresh;
    wrapper->parameter.tuning.alpha_tighten = parameter_info.tuning.alpha_tighten;
    wrapper->parameter.tuning.alpha_loosen = parameter_info.tuning.alpha_loosen;

    wrapper->parameter.distance_metric = static_cast<uint8_t>(parameter_info.distance_metric);

    return wrapper;
}

void akerDestroyAnnsCache(anns_cache_c_wrapper_t* wrapper)
{
    if (wrapper == nullptr)
        return;

    delete asCache(wrapper->anns_cache);
    delete wrapper;
}

char* akerCreateVectorSlot(
    uint64_t vector_id,
    size_t vector_size,
    char* vector_data,
    uint64_t aux_data_1,
    uint64_t aux_data_2,
    float distance)
{
    /* Create a standalone VectorSlot wrapper. */
    aker::VectorSlot* vector = new aker::VectorSlot(vector_size);
    vector->setVectorId(vector_id);
    vector->setAuxData1(aux_data_1);
    vector->setAuxData2(aux_data_2);

    if (vector_data != nullptr)
        std::memcpy(vector->getVectorData(), vector_data, vector_size);

    vector->setDistance(distance);
    return asChar(vector);
}

void akerDestroyVectorSlot(char* vector_slot_wrapper)
{
    if (vector_slot_wrapper == nullptr)
        return;

    delete asVectorSlot(vector_slot_wrapper);
}

uint64_t akerGetVectorIdFromVectorSlot(char* vector_slot_wrapper)
{
    aker::VectorSlot* vector = asVectorSlot(vector_slot_wrapper);
    return vector->getVectorId();
}

uint64_t akerGetVectorIdFromVectorView(char* vector_view_wrapper)
{
    aker::vector_view_t* vector_view = asVectorView(vector_view_wrapper);
    return vector_view->vector_id;
}

uint64_t akerGetQueryVectorIdFromCacheEntry(char* cache_entry_wrapper)
{
    aker::anns_cache_entry_t* entry = asCacheEntry(cache_entry_wrapper);
    return entry->query_vector->getVectorId();
}

char* akerGetVectorDataFromVectorSlot(char* vector_slot_wrapper)
{
    aker::VectorSlot* vector = asVectorSlot(vector_slot_wrapper);
    return reinterpret_cast<char*>(vector->getVectorData());
}

void akerSetDistanceForVectorSlot(char* vector_slot_wrapper, float distance)
{
    aker::VectorSlot* vector = asVectorSlot(vector_slot_wrapper);
    vector->setDistance(distance);
}

char* akerCreateVectorView(
    char* vector_slot_wrapper,
    size_t dim,
    size_t vector_in_bytes,
    bool (*transform_callback)(void*, size_t, size_t, void*, uint8_t*))
{
    /* Wrap a VectorSlot as a VectorView for FAISS input conversion. */
    aker::VectorSlot* vector = asVectorSlot(vector_slot_wrapper);
    aker::vector_view_t* vector_view = new aker::vector_view_t;

    vector_view->vector_id = vector->getVectorId();
    vector_view->vector_data = vector->getVectorData();
    vector_view->dimension = dim;
    vector_view->vector_in_bytes = vector_in_bytes;
    vector_view->transform_callback = transform_callback;

    vector_view->aux_data_1 = vector->getAuxData1();
    vector_view->aux_data_2 = vector->getAuxData2();

    return asChar(vector_view);
}

void akerDestroyVectorView(char* vector_view_wrapper)
{
    if (vector_view_wrapper == nullptr)
        return;

    delete asVectorView(vector_view_wrapper);
}

char* akerCreateCacheEntry(
    anns_cache_c_wrapper_t* wrapper,
    char* query_vector_slot_wrapper,
    size_t neighbors,
    char** neighbors_list)
{
    aker::ANNSCache* cache = asCache(wrapper->anns_cache);
    aker::anns_cache_entry_t* entry =
        cache->createCacheEntry(
            asVectorSlot(query_vector_slot_wrapper),
            static_cast<std::uint32_t>(neighbors),
            reinterpret_cast<aker::VectorSlot**>(neighbors_list));
    return asChar(entry);
}

void akerDestroyCacheEntry(char* cache_entry_wrapper)
{
    /* Frees a cache entry returned to the caller.
     * This mirrors ANNSCache::destroyCacheEntry() without requiring a cache instance.
     */
    if (cache_entry_wrapper == nullptr)
        return;

    aker::anns_cache_entry_t* entry = asCacheEntry(cache_entry_wrapper);

    if (entry->entry_kind == aker::ANNS_CACHE_ENTRY_KIND_RETURNED_COPY)
    {
        if (entry->neighbors_list != nullptr)
        {
            for (size_t i = 0; i < entry->neighbors; i++)
                delete entry->neighbors_list[i];
        }
    }

    delete entry->query_vector;
    entry->query_vector = nullptr;

    free(entry->neighbors_list);
    entry->neighbors_list = nullptr;

    delete entry;
}

char* akerGetCacheEntry(
    anns_cache_c_wrapper_t* wrapper,
    char* query_vector_view_wrapper,
    bool* similar_entry,
    bool* is_invalid,
    float (*distance_function)(uint8_t*, uint8_t*, size_t))
{
    if (query_vector_view_wrapper == nullptr)
        return nullptr;

    assert(wrapper != nullptr);
    assert(similar_entry != nullptr);
    assert(is_invalid != nullptr);

    aker::vector_view_t* query_vector_data = asVectorView(query_vector_view_wrapper);
    aker::ANNSCache* cache = asCache(wrapper->anns_cache);

    aker::distance_function_t distance_function_cb = adaptDistanceFunction(distance_function);

    aker::anns_cache_entry_t* found_entry =
        cache->getCacheEntry(
            *query_vector_data,
            *similar_entry,
            *is_invalid,
            distance_function_cb);

    return reinterpret_cast<char*>(found_entry);
}

char* akerGetResultVectorSlotAt(char* cache_entry_wrapper, int index)
{
    if (cache_entry_wrapper == nullptr)
        return nullptr;

    aker::anns_cache_entry_t* entry = asCacheEntry(cache_entry_wrapper);
    return reinterpret_cast<char*>(entry->neighbors_list[index]);
}

char* akerGetResultVectorSlots(char* cache_entry_wrapper)
{
    if (cache_entry_wrapper == nullptr)
        return nullptr;

    aker::anns_cache_entry_t* entry = asCacheEntry(cache_entry_wrapper);
    return reinterpret_cast<char*>(entry->neighbors_list);
}

void akerFormatCacheEntryStatus(char* cache_entry_wrapper, char* status_string)
{
    if (cache_entry_wrapper == nullptr || status_string == nullptr)
        return;

    /* Build a compact status string for debugging. */
    aker::anns_cache_entry_t* cache_entry = asCacheEntry(cache_entry_wrapper);
    std::string status;

    status += "{";
    status += "Entry ID: " + std::to_string(cache_entry->query_vector->getVectorId()) + ", ";
    status += "Entry Size: " + std::to_string(cache_entry->neighbors) + ", ";
    status += "Entry Thresh: " + std::to_string(cache_entry->thresh) + ", ";
    status += "Entry Min Distance: " + std::to_string(cache_entry->min_distance) + ", ";
    status += "Entry Max Distance: " + std::to_string(cache_entry->max_distance) + ", ";
    status += "Entry Risk Factor: " + std::to_string(cache_entry->risk_factor) + ", ";

    status += "List: [";
    for (size_t i = 0; i < cache_entry->neighbors; i++)
    {
        if (i > 0)
            status += ", ";

        status += "{";
        aker::VectorSlot* element = cache_entry->neighbors_list[i];
        status += "VID: ";

        if (element == nullptr)
        {
            status += "NULL";
            status += "}";
            continue;
        }

        if (!element->isValid())
        {
            status += "INVALID";
            status += "}";
            continue;
        }

        status += std::to_string(element->getVectorId());
        status += "}";
    }
    status += "]}";

    std::strcpy(status_string, status.c_str());
}

bool akerInsertCacheEntry(
    anns_cache_c_wrapper_t* wrapper,
    uint64_t vector_id,
    char* new_cache_entry_wrapper,
    char* query_vector_view_wrapper)
{
    aker::vector_view_t* query_vector_data = asVectorView(query_vector_view_wrapper);
    aker::ANNSCache* cache = asCache(wrapper->anns_cache);

    return cache->insertCacheEntry(
        vector_id,
        asCacheEntry(new_cache_entry_wrapper),
        *query_vector_data);
}

bool akerLinkCacheEntry(anns_cache_c_wrapper_t* wrapper, char* new_cache_entry_wrapper, uint64_t found_id)
{
    aker::ANNSCache* cache = asCache(wrapper->anns_cache);
    aker::anns_cache_entry_t* entry = asCacheEntry(new_cache_entry_wrapper);

    if (entry->neighbors_list == nullptr)
        entry->neighbors_list = nullptr;

    return cache->linkCacheEntry(entry, found_id);
}

void akerInsertWriteLogEntry(
    anns_cache_c_wrapper_t* wrapper,
    char* vector_view_wrapper,
    float (*distance_function)(uint8_t*, uint8_t*, size_t),
    void (*result_transform_callback)(uint64_t, uint8_t*, size_t, uint64_t, uint64_t))
{
    aker::ANNSCache* cache = asCache(wrapper->anns_cache);
    cache->insertWriteLogEntry(
        *asVectorView(vector_view_wrapper),
        adaptDistanceFunction(distance_function),
        result_transform_callback);
}

void akerProcessWriteLogEntries(
    anns_cache_c_wrapper_t* wrapper,
    float (*distance_function)(uint8_t*, uint8_t*, size_t),
    void (*result_transform_callback)(uint64_t, uint8_t*, size_t, uint64_t, uint64_t))
{
    aker::ANNSCache* cache = asCache(wrapper->anns_cache);
    cache->processWriteLogEntries(adaptDistanceFunction(distance_function), result_transform_callback);
}

void akerMarkVectorDeleted(anns_cache_c_wrapper_t* wrapper, uint64_t vector_id)
{
    aker::ANNSCache* cache = asCache(wrapper->anns_cache);
    cache->markVectorDeleted(vector_id);
}

char* akerGetCacheStatusText(anns_cache_c_wrapper_t* wrapper)
{
    if (wrapper == nullptr)
        return nullptr;

    aker::ANNSCache* cache = asCache(wrapper->anns_cache);
    static thread_local std::string tl_cache_string;
    tl_cache_string = cache->getStatusText();
    return const_cast<char*>(tl_cache_string.c_str());
}

void akerExportTraceToFiles(anns_cache_c_wrapper_t* wrapper)
{
    if (wrapper == nullptr)
        return;

    aker::ANNSCache* cache = asCache(wrapper->anns_cache);
    cache->exportTraceToFiles();
}

void akerStressTestInvalidateRandom(anns_cache_c_wrapper_t* wrapper, float percent)
{
    if (wrapper == nullptr)
        return;

    aker::ANNSCache* cache = asCache(wrapper->anns_cache);
    cache->stressTestInvalidateRandom(percent);
}

uint64_t akerDefaultHash(char* data, size_t nbytes)
{
    if (data == nullptr || nbytes == 0)
        return 0;

    uint64_t hash = XXH3_64bits(data, nbytes);
    return hash;
}

float akerL2Distance(float* a, float* b, size_t dim)
{
    return l2_dist(a, b, dim);
}

float akerInnerProductDistance(float* a, float* b, size_t dim)
{
    return inner_product_dist(a, b, dim);
}

void akerHelperMapInit()
{
    helper_map.clear();
}

void akerHelperMapInsert(uint64_t key, char* value, size_t value_size)
{
    char* value_copy = new char[value_size];
    std::memcpy(value_copy, value, value_size);
    helper_map[key] = value_copy;
}

char* akerHelperMapGet(uint64_t key)
{
    std::map<uint64_t, char*>::iterator it = helper_map.find(key);
    if (it != helper_map.end())
        return it->second;
    return nullptr;
}

void akerHelperMapClear()
{
    for (std::map<uint64_t, char*>::iterator it = helper_map.begin(); it != helper_map.end(); ++it)
        delete[] it->second;
    helper_map.clear();
}

/*
 * Legacy API implementations.
 */#if defined(AKER_ENABLE_LEGACY_C_API) && (AKER_ENABLE_LEGACY_C_API != 0)

void import_aker_parameter(char* path, anns_cache_parameter_c_t* parameter)
{
    akerImportAnnsCacheConfig(path, parameter);
}

bool transform_callback_c_wrapper(void* src, size_t src_size, size_t dim, void* dst, uint8_t* aux)
{
    return akerDefaultConversionFunction(src, src_size, dim, dst, aux);
}

anns_cache_c_wrapper_t* create_ann_cache_2_c_wrapper(anns_cache_parameter_c_t parameter)
{
    return akerCreateAnnsCache(parameter);
}

void destroy_ann_cache_2_c_wrapper(anns_cache_c_wrapper_t* wrapper)
{
    akerDestroyAnnsCache(wrapper);
}

char* create_vector_slot_c_wrapper(
    uint64_t vector_id,
    size_t vector_size,
    char* vector_data,
    uint64_t aux_data_1,
    uint64_t aux_data_2,
    float distance)
{
    return akerCreateVectorSlot(vector_id, vector_size, vector_data, aux_data_1, aux_data_2, distance);
}

uint64_t get_vector_id_entry_2_c_wrapper(char* searched_entry)
{
    return akerGetQueryVectorIdFromCacheEntry(searched_entry);
}

uint64_t get_vector_id_vector_slot_c_wrapper(char* vector_2_wrapper)
{
    return akerGetVectorIdFromVectorSlot(vector_2_wrapper);
}

uint64_t get_vector_id_vector_view_c_wrapper(char* float_vector_2_wrapper)
{
    return akerGetVectorIdFromVectorView(float_vector_2_wrapper);
}

char* get_vector_data_vector_slot_c_wrapper(char* vector_2_wrapper)
{
    return akerGetVectorDataFromVectorSlot(vector_2_wrapper);
}

void set_distance_vector_slot_c_wrapper(char* vector_2_wrapper, float distance)
{
    akerSetDistanceForVectorSlot(vector_2_wrapper, distance);
}

void destroy_vector_slot_c_wrapper(char* vector_2_wrapper)
{
    akerDestroyVectorSlot(vector_2_wrapper);
}

char* create_vector_view_c_wrapper(
    char* vector_2_wrapper,
    size_t dim,
    size_t vector_in_bytes,
    bool (*transform_callback)(void*, size_t, size_t, void*, uint8_t*))
{
    return akerCreateVectorView(vector_2_wrapper, dim, vector_in_bytes, transform_callback);
}

void destroy_vector_view_c_wrapper(char* float_vector_2_wrapper)
{
    akerDestroyVectorView(float_vector_2_wrapper);
}

char* make_cache_entry_c_wrapper(
    anns_cache_c_wrapper_t* wrapper,
    char* query_vector,
    size_t vector_list_size,
    char** neighbors_list)
{
    return akerCreateCacheEntry(wrapper, query_vector, vector_list_size, neighbors_list);
}

void free_cache_entry_c_wrapper(char* cache_entry_ptr)
{
    akerDestroyCacheEntry(cache_entry_ptr);
}

char* sim_search_c_wrapper(
    anns_cache_c_wrapper_t* wrapper,
    char* float_query_vector,
    bool* similar_entry,
    bool* is_invalid,
    float (*distance_function)(uint8_t*, uint8_t*, size_t))
{
    return akerGetCacheEntry(wrapper, float_query_vector, similar_entry, is_invalid, distance_function);
}

char* get_result_c_wrapper(char* searched_entry, int index)
{
    return akerGetResultVectorSlotAt(searched_entry, index);
}

char* get_result_sets_c_wrapper(char* searched_entry)
{
    return akerGetResultVectorSlots(searched_entry);
}

void debug_print_cache_entry_c_wrapper(char* entry, char* status_string)
{
    akerFormatCacheEntryStatus(entry, status_string);
}

bool insert_cache_entry_c_wrapper(
    anns_cache_c_wrapper_t* wrapper,
    uint64_t vector_id,
    char* new_cache_entry,
    char* float_query_vector)
{
    return akerInsertCacheEntry(wrapper, vector_id, new_cache_entry, float_query_vector);
}

bool link_cache_entry_c_wrapper(anns_cache_c_wrapper_t* wrapper, char* new_cache_entry, uint64_t found_id)
{
    return akerLinkCacheEntry(wrapper, new_cache_entry, found_id);
}

void insert_wl_entry_c_wrapper(
    anns_cache_c_wrapper_t* wrapper,
    char* float_vector,
    float (*distance_function)(uint8_t*, uint8_t*, size_t),
    void (*result_transform_callback)(uint64_t, uint8_t*, size_t, uint64_t, uint64_t))
{
    akerInsertWriteLogEntry(wrapper, float_vector, distance_function, result_transform_callback);
}

void consume_wl_entry_c_wrapper(
    anns_cache_c_wrapper_t* wrapper,
    float (*distance_function)(uint8_t*, uint8_t*, size_t),
    void (*result_transform_callback)(uint64_t, uint8_t*, size_t, uint64_t, uint64_t))
{
    akerProcessWriteLogEntries(wrapper, distance_function, result_transform_callback);
}

void mark_deleted_c_wrapper(anns_cache_c_wrapper_t* wrapper, uint64_t vector_id)
{
    akerMarkVectorDeleted(wrapper, vector_id);
}

char* print_cache_c_wrapper(anns_cache_c_wrapper_t* wrapper)
{
    return akerGetCacheStatusText(wrapper);
}

void export_call_c_wrapper(anns_cache_c_wrapper_t* wrapper)
{
    akerExportTraceToFiles(wrapper);
}

void stress_test_invalidate_random_c_wrapper(anns_cache_c_wrapper_t* wrapper, float percent)
{
    akerStressTestInvalidateRandom(wrapper, percent);
}

uint64_t default_hash(char* buffer, size_t size)
{
    return akerDefaultHash(buffer, size);
}

float l2_dist_c_wrapper(float* a, float* b, size_t dim)
{
    return akerL2Distance(a, b, dim);
}

float ip_dist_c_wrapper(float* a, float* b, size_t dim)
{
    return akerInnerProductDistance(a, b, dim);
}

void init_helpermap_c_wrapper()
{
    akerHelperMapInit();
}

void insert_helpermap_c_wrapper(uint64_t key, char* value, size_t value_size)
{
    akerHelperMapInsert(key, value, value_size);
}

char* get_helpermap_c_wrapper(uint64_t key)
{
    return akerHelperMapGet(key);
}

void clear_helpermap_c_wrapper()
{
    akerHelperMapClear();
}

#endif // AKER_ENABLE_LEGACY_C_API
