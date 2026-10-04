/**
 * @file ak_anns_cache_c_wrapper.h
 * @brief C ABI wrappers for Aker ANNSCache.
 */

#ifndef AKER_ANNS_CACHE_C_WRAPPER_H
#define AKER_ANNS_CACHE_C_WRAPPER_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#define AKER_START
#define AKER_END

/**
 * @brief Compiler-specific deprecation attribute.
 */
#if defined(__GNUC__) || defined(__clang__)
#define AKER_DEPRECATED(msg) __attribute__((deprecated(msg)))
#elif defined(_MSC_VER)
#define AKER_DEPRECATED(msg) __declspec(deprecated(msg))
#else
#define AKER_DEPRECATED(msg)
#endif

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief C configuration struct for ANNSCache.
 *
 * This struct mirrors the C++ configuration layout (ANNSCacheConfig):
 * - vector_format
 * - capacity
 * - tuning
 * - distance_metric
 */

typedef struct VectorFormatConfigC {
    uint32_t dimension;
    size_t vector_in_bytes;
} anns_cache_vector_format_config_c_t;

typedef struct CacheCapacityConfigC {
    size_t pool_size;
    size_t in_topk;
    size_t top_delta;
} anns_cache_capacity_config_c_t;

typedef struct AlgorithmTuningConfigC {
    float global_thresh;
    float dropout;

    float risk_thresh;
    float alpha_tighten;
    float alpha_loosen;
} anns_cache_tuning_config_c_t;

typedef struct ANNSCacheConfigC {
    anns_cache_vector_format_config_c_t vector_format;
    anns_cache_capacity_config_c_t capacity;
    anns_cache_tuning_config_c_t tuning;

    /** 0: L2, 1: Inner Product */
    uint8_t distance_metric;
} anns_cache_parameter_c_t;

/**
 * @brief Opaque wrapper for C callers.
 */
typedef struct ANNSCacheCWrapper {
    char* anns_cache;
    anns_cache_parameter_c_t parameter;
} anns_cache_c_wrapper_t;

/**
 * @brief Loads ANNS cache configuration from a file.
 *
 * @param path      Path to the parameter file.
 * @param parameter Output parameter struct.
 */
void akerImportAnnsCacheConfig(char* path, anns_cache_parameter_c_t* parameter);

/**
 * @brief Default conversion function stub (for testing).
 */
bool akerDefaultConversionFunction(
    void* src,
    size_t src_size,
    size_t dim,
    void* dst,
    uint8_t* aux);

/**
 * @brief Creates a cache instance wrapper.
 */
anns_cache_c_wrapper_t* akerCreateAnnsCache(anns_cache_parameter_c_t parameter);

/**
 * @brief Destroys a cache instance wrapper.
 */
void akerDestroyAnnsCache(anns_cache_c_wrapper_t* wrapper);

/**
 * @brief Creates a VectorSlot wrapper object.
 */
char* akerCreateVectorSlot(
    uint64_t vector_id,
    size_t vector_size,
    char* vector_data,
    uint64_t aux_data_1,
    uint64_t aux_data_2,
    float distance);

/**
 * @brief Destroys a VectorSlot wrapper object.
 */
void akerDestroyVectorSlot(char* vector_slot_wrapper);

/**
 * @brief Returns the vector id stored in a VectorSlot.
 */
uint64_t akerGetVectorIdFromVectorSlot(char* vector_slot_wrapper);

/**
 * @brief Returns the vector id stored in a VectorView.
 */
uint64_t akerGetVectorIdFromVectorView(char* vector_view_wrapper);

/**
 * @brief Returns the query vector id from a cache entry.
 */
uint64_t akerGetQueryVectorIdFromCacheEntry(char* cache_entry_wrapper);

/**
 * @brief Returns the raw vector data pointer from a VectorSlot.
 */
char* akerGetVectorDataFromVectorSlot(char* vector_slot_wrapper);

/**
 * @brief Sets the distance field for a VectorSlot.
 */
void akerSetDistanceForVectorSlot(char* vector_slot_wrapper, float distance);

/**
 * @brief Creates a VectorView wrapper from a VectorSlot.
 */
char* akerCreateVectorView(
    char* vector_slot_wrapper,
    size_t dim,
    size_t vector_in_bytes,
    bool (*transform_callback)(void*, size_t, size_t, void*, uint8_t*));

/**
 * @brief Destroys a VectorView wrapper.
 */
void akerDestroyVectorView(char* vector_view_wrapper);

/**
 * @brief Creates a cache entry (prepared entry) from query + result vectors.
 */
char* akerCreateCacheEntry(
    anns_cache_c_wrapper_t* wrapper,
    char* query_vector_slot_wrapper,
    size_t neighbors,
    char** neighbors_list);

/**
 * @brief Destroys a cache entry returned to the caller.
 */
void akerDestroyCacheEntry(char* cache_entry_wrapper);

/**
 * @brief Looks up a cache entry (exact-hit or approx-hit).
 */
char* akerGetCacheEntry(
    anns_cache_c_wrapper_t* wrapper,
    char* query_vector_view_wrapper,
    bool* similar_entry,
    bool* is_invalid,
    float (*distance_function)(uint8_t*, uint8_t*, size_t));

/**
 * @brief Returns the result VectorSlot at the given index.
 */
char* akerGetResultVectorSlotAt(char* cache_entry_wrapper, int index);

/**
 * @brief Returns the raw result VectorSlot list pointer.
 */
char* akerGetResultVectorSlots(char* cache_entry_wrapper);

/**
 * @brief Formats a cache entry status into the caller-provided buffer.
 */
void akerFormatCacheEntryStatus(char* cache_entry_wrapper, char* status_string);

/**
 * @brief Inserts a prepared cache entry into the cache.
 */
bool akerInsertCacheEntry(
    anns_cache_c_wrapper_t* wrapper,
    uint64_t vector_id,
    char* new_cache_entry_wrapper,
    char* query_vector_view_wrapper);

/**
 * @brief Links a prepared cache entry to an existing representative entry.
 */
bool akerLinkCacheEntry(
    anns_cache_c_wrapper_t* wrapper,
    char* new_cache_entry_wrapper,
    uint64_t found_id);

/**
 * @brief Inserts a write-log entry and runs fast-path update.
 */
void akerInsertWriteLogEntry(
    anns_cache_c_wrapper_t* wrapper,
    char* vector_view_wrapper,
    float (*distance_function)(uint8_t*, uint8_t*, size_t),
    void (*result_transform_callback)(uint64_t, uint8_t*, size_t, uint64_t, uint64_t));

/**
 * @brief Processes write-log slow-path updates.
 */
void akerProcessWriteLogEntries(
    anns_cache_c_wrapper_t* wrapper,
    float (*distance_function)(uint8_t*, uint8_t*, size_t),
    void (*result_transform_callback)(uint64_t, uint8_t*, size_t, uint64_t, uint64_t));

/**
 * @brief Marks a vector as deleted (tombstone) in the cache slot pool.
 */
void akerMarkVectorDeleted(anns_cache_c_wrapper_t* wrapper, uint64_t vector_id);

/**
 * @brief Returns a human-readable cache status text (thread-local buffer).
 */
char* akerGetCacheStatusText(anns_cache_c_wrapper_t* wrapper);

/**
 * @brief Exports trace/telemetry files.
 */
void akerExportTraceToFiles(anns_cache_c_wrapper_t* wrapper);

/**
 * @brief Stress-test helper for invalidation.
 */
void akerStressTestInvalidateRandom(anns_cache_c_wrapper_t* wrapper, float percent);

/**
 * @brief Default hash helper.
 */
uint64_t akerDefaultHash(char* buffer, size_t size);

/**
 * @brief L2 distance helper.
 */
float akerL2Distance(float* a, float* b, size_t dim);

/**
 * @brief Inner product distance helper.
 */
float akerInnerProductDistance(float* a, float* b, size_t dim);

/**
 * @brief Initializes helper map storage.
 */
void akerHelperMapInit();

/**
 * @brief Inserts into helper map.
 */
void akerHelperMapInsert(uint64_t key, char* value, size_t value_size);

/**
 * @brief Gets a value from helper map.
 */
char* akerHelperMapGet(uint64_t key);

/**
 * @brief Clears helper map storage.
 */
void akerHelperMapClear();

#if defined(AKER_ENABLE_LEGACY_C_API) && (AKER_ENABLE_LEGACY_C_API != 0)
/**
 * @brief Legacy C wrapper APIs.
 *
 * These are preserved for backward compatibility and are disabled by default. Prefer the aker* APIs.
 * Define AKER_ENABLE_LEGACY_C_API=1 to enable them.
 */
void import_aker_parameter(char* path, anns_cache_parameter_c_t* parameter)
    AKER_DEPRECATED("Use akerImportAnnsCacheConfig()");

bool transform_callback_c_wrapper(void* src, size_t src_size, size_t dim, void* dst, uint8_t* aux)
    AKER_DEPRECATED("Use akerDefaultConversionFunction()");

anns_cache_c_wrapper_t* create_ann_cache_2_c_wrapper(anns_cache_parameter_c_t parameter)
    AKER_DEPRECATED("Use akerCreateAnnsCache()");

void destroy_ann_cache_2_c_wrapper(anns_cache_c_wrapper_t* wrapper)
    AKER_DEPRECATED("Use akerDestroyAnnsCache()");

char* create_vector_slot_c_wrapper(uint64_t vector_id, size_t vector_size, char* vector_data,
                                  uint64_t aux_data_1, uint64_t aux_data_2, float distance)
    AKER_DEPRECATED("Use akerCreateVectorSlot()");

uint64_t get_vector_id_entry_2_c_wrapper(char* searched_entry)
    AKER_DEPRECATED("Use akerGetQueryVectorIdFromCacheEntry()");

uint64_t get_vector_id_vector_slot_c_wrapper(char* vector_2_wrapper)
    AKER_DEPRECATED("Use akerGetVectorIdFromVectorSlot()");

uint64_t get_vector_id_vector_view_c_wrapper(char* float_vector_2_wrapper)
    AKER_DEPRECATED("Use akerGetVectorIdFromVectorView()");

char* get_vector_data_vector_slot_c_wrapper(char* vector_2_wrapper)
    AKER_DEPRECATED("Use akerGetVectorDataFromVectorSlot()");

void set_distance_vector_slot_c_wrapper(char* vector_2_wrapper, float distance)
    AKER_DEPRECATED("Use akerSetDistanceForVectorSlot()");

void destroy_vector_slot_c_wrapper(char* vector_2_wrapper)
    AKER_DEPRECATED("Use akerDestroyVectorSlot()");

char* create_vector_view_c_wrapper(char* vector_2_wrapper, size_t dim, size_t vector_in_bytes,
                                 bool (*transform_callback)(void*, size_t, size_t, void*, uint8_t*))
    AKER_DEPRECATED("Use akerCreateVectorView()");

void destroy_vector_view_c_wrapper(char* float_vector_2_wrapper)
    AKER_DEPRECATED("Use akerDestroyVectorView()");

char* make_cache_entry_c_wrapper(anns_cache_c_wrapper_t* wrapper, char* query_vector,
                               size_t neighbors, char** neighbors_list)
    AKER_DEPRECATED("Use akerCreateCacheEntry()");

void free_cache_entry_c_wrapper(char* cache_entry_ptr)
    AKER_DEPRECATED("Use akerDestroyCacheEntry()");

char* sim_search_c_wrapper(anns_cache_c_wrapper_t* wrapper, char* float_query_vector,
                         bool* similar_entry, bool* is_invalid,
                         float (*distance_function)(uint8_t*, uint8_t*, size_t))
    AKER_DEPRECATED("Use akerGetCacheEntry()");

char* get_result_c_wrapper(char* searched_entry, int index)
    AKER_DEPRECATED("Use akerGetResultVectorSlotAt()");

char* get_result_sets_c_wrapper(char* searched_entry)
    AKER_DEPRECATED("Use akerGetResultVectorSlots()");

void debug_print_cache_entry_c_wrapper(char* entry, char* status_string)
    AKER_DEPRECATED("Use akerFormatCacheEntryStatus()");

bool insert_cache_entry_c_wrapper(anns_cache_c_wrapper_t* wrapper, uint64_t vector_id,
                                char* new_cache_entry, char* float_query_vector)
    AKER_DEPRECATED("Use akerInsertCacheEntry()");

bool link_cache_entry_c_wrapper(anns_cache_c_wrapper_t* wrapper, char* new_cache_entry, uint64_t found_id)
    AKER_DEPRECATED("Use akerLinkCacheEntry()");

void insert_wl_entry_c_wrapper(anns_cache_c_wrapper_t* wrapper, char* float_vector,
                             float (*distance_function)(uint8_t*, uint8_t*, size_t),
                             void (*result_transform_callback)(uint64_t, uint8_t*, size_t, uint64_t, uint64_t))
    AKER_DEPRECATED("Use akerInsertWriteLogEntry()");

void consume_wl_entry_c_wrapper(anns_cache_c_wrapper_t* wrapper,
                              float (*distance_function)(uint8_t*, uint8_t*, size_t),
                              void (*result_transform_callback)(uint64_t, uint8_t*, size_t, uint64_t, uint64_t))
    AKER_DEPRECATED("Use akerProcessWriteLogEntries()");

void mark_deleted_c_wrapper(anns_cache_c_wrapper_t* wrapper, uint64_t vector_id)
    AKER_DEPRECATED("Use akerMarkVectorDeleted()");

char* print_cache_c_wrapper(anns_cache_c_wrapper_t* wrapper)
    AKER_DEPRECATED("Use akerGetCacheStatusText()");

void export_call_c_wrapper(anns_cache_c_wrapper_t* wrapper)
    AKER_DEPRECATED("Use akerExportTraceToFiles()");

void stress_test_invalidate_random_c_wrapper(anns_cache_c_wrapper_t* wrapper, float percent)
    AKER_DEPRECATED("Use akerStressTestInvalidateRandom()");

uint64_t default_hash(char* buffer, size_t size)
    AKER_DEPRECATED("Use akerDefaultHash()");

float l2_dist_c_wrapper(float* a, float* b, size_t dim)
    AKER_DEPRECATED("Use akerL2Distance()");

float ip_dist_c_wrapper(float* a, float* b, size_t dim)
    AKER_DEPRECATED("Use akerInnerProductDistance()");

void init_helpermap_c_wrapper()
    AKER_DEPRECATED("Use akerHelperMapInit()");

void insert_helpermap_c_wrapper(uint64_t key, char* value, size_t value_size)
    AKER_DEPRECATED("Use akerHelperMapInsert()");

char* get_helpermap_c_wrapper(uint64_t key)
    AKER_DEPRECATED("Use akerHelperMapGet()");

void clear_helpermap_c_wrapper()
    AKER_DEPRECATED("Use akerHelperMapClear()");

#endif // AKER_ENABLE_LEGACY_C_API

#ifdef __cplusplus
} // extern "C"
#endif

#endif // AKER_ANNS_CACHE_C_WRAPPER_H
