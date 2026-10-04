// DiskANN (FreshVamana) headers
#include "diskann/utils.h"
#include "diskann/index_factory.h"

// QVCache headers
#include "qvcache/insert_thread_pool.h"
#include "qvcache/search_thread_pool.h"
#include "qvcache/pca_utils.h"
#include "qvcache/lru_cache.h"
#include "qvcache/backend_interface.h"
#include "qvcache/hit_rate_tracker.h" 

// System headers
#include <cstdint>
#include <unordered_map>
#include <mutex>
#include <type_traits>
#include <memory>
#include <cstring>
#include <atomic>
#include <future>
#include <vector>
#include <algorithm>
#include <deque>
#include <limits>
#include <chrono>
#include <unordered_set>
#include <string>
#include <stdexcept>
#include <iostream>

namespace qvcache {

    template <typename T, typename TagT = uint32_t>
    class QVCache {
        
        private:
            // Backend vector database
            std::unique_ptr<BackendInterface<T, TagT>> backend;
            
            // LRU-managed memory indices: n memory indices managed by LRU eviction
            std::vector<std::unique_ptr<diskann::AbstractIndex>> memory_indices;
            std::atomic<size_t> active_insert_index_id; // Index ID for new insertions

            // --- LRU Cache for managing mini-index access patterns ---
            std::unique_ptr<LRUCache<size_t>> lru_cache; // Tracks mini-index usage

            // --- Index parameters ---
            std::string data_path;
            std::string pca_prefix;
            size_t dim, aligned_dim;
            size_t num_points;
            size_t memory_index_max_points_per_index; // Capacity per index
            size_t number_of_mini_indexes; // Number of mini indexes
            uint32_t search_threads;
            bool use_reconstructed_vectors;
            std::unique_ptr<qvcache::InsertThreadPool<T, TagT>> insert_pool;
            std::unordered_map<uint32_t, double> theta_map;
            std::mutex theta_map_mutex;
            double p, deviation_factor;
            uint32_t memory_L; 
            uint32_t beamwidth;
            bool use_regional_theta = true;

            uint32_t n_async_insert_threads = 4;
            bool lazy_theta_updates = true;
            bool search_mini_indexes_in_parallel = false; // Control parallel vs sequential search
            size_t max_search_threads = 32; // Maximum threads for parallel search (should be > query processing threads)
            // Declared after memory_indices so the pool joins before those indexes are destroyed.
            std::unique_ptr<SearchThreadPool> search_pool;
            diskann::Metric metric = diskann::L2; // Distance metric (default: L2)

            // --- LRU eviction state ---
            std::atomic<bool> eviction_in_progress{false};
            std::future<void> eviction_future;
            std::mutex eviction_mutex;

            uint32_t miss_admit_delta = 0;
            double write_theta_discount = 0.80;
            size_t write_l1_radius = 1;
            double global_theta_discount = 1.0;

            // Live-consistency L1: eager tombstones. Backend ids are 0-based.
            std::mutex deleted_mutex;
            std::unordered_set<TagT> deleted_ids;

            // Admission-order FIFO of memory tags (id+1). Miss-admit evicts
            // oldest points, not a whole mini-index.
            std::mutex admit_fifo_mutex;
            std::deque<TagT> admit_fifo;
            std::unordered_set<TagT> admit_set;
            std::atomic<uint64_t> point_evictions{0};
            std::atomic<uint64_t> region_invalidations{0};

            // Aker: cache-layer time up to (not including) backend search, for miss_penalty.
            inline static thread_local std::chrono::high_resolution_clock::time_point tls_search_t0{};
            inline static thread_local double tls_cache_lookup_ms{0.0};

            void note_cache_lookup_end() {
                tls_cache_lookup_ms = std::chrono::duration<double, std::milli>(
                    std::chrono::high_resolution_clock::now() - tls_search_t0).count();
            }

            // --- PCA utilities ---
            std::unique_ptr<PCAUtils<T>> pca_utils;

        public:
            // Configuration option to choose search strategy
            enum class SearchStrategy {
                SEQUENTIAL_LRU_STOP_FIRST_HIT,  // Problematic: Stop at first hit in LRU order (causes recall drops)
                SEQUENTIAL_LRU_ADAPTIVE,  // Adaptive: Monitor hit ratio, switch to SEQUENTIAL_ALL when low
                SEQUENTIAL_ALL,           // Search all indices sequentially, pick best result
                PARALLEL                  // Parallel search (existing implementation)
            };

        private:
            // Search strategy for the public search method
            SearchStrategy search_strategy = SearchStrategy::SEQUENTIAL_LRU_STOP_FIRST_HIT;
            
            // Hit ratio monitoring for adaptive strategy - using HitRateTracker (already thread-safe)
            mutable std::mutex hit_rate_tracker_mutex;  // Protect tracker pointer and initialization
            std::unique_ptr<HitRateTracker> hit_rate_tracker;  // Thread-safe hit rate tracker
            std::atomic<bool> use_adaptive_strategy{false};  // Whether to use adaptive behavior
            std::atomic<double> hit_ratio_threshold{0.90};  // Threshold for switching strategies (cached for fast access)

            // Helper function to create a memory index with given max points
            std::unique_ptr<diskann::AbstractIndex> create_memory_index(size_t max_points) {
                diskann::IndexWriteParameters memory_index_write_params = diskann::IndexWriteParametersBuilder(memory_L, aligned_dim)
                                                                    .with_alpha(1.2f) // Default alpha
                                                                    .with_num_threads(4) // Default threads
                                                                    .build();

                diskann::IndexSearchParams memory_index_search_params = diskann::IndexSearchParams(memory_L, search_threads);

                diskann::IndexConfig memory_index_config = diskann::IndexConfigBuilder()
                                                            .with_metric(metric)
                                                            .with_dimension(dim)
                                                            .with_max_points(max_points)
                                                            .is_dynamic_index(true)
                                                            .with_index_write_params(memory_index_write_params)
                                                            .with_index_search_params(memory_index_search_params)
                                                            .with_data_type(diskann_type_to_name<T>())
                                                            .with_tag_type(diskann_type_to_name<TagT>())
                                                            .with_data_load_store_strategy(diskann::DataStoreStrategy::MEMORY)
                                                            .with_graph_load_store_strategy(diskann::GraphStoreStrategy::MEMORY)
                                                            .is_enable_tags(true)
                                                            .is_filtered(false)
                                                            .with_num_frozen_pts(0)
                                                            .is_concurrent_consolidate(true)
                                                            .build();
                
                diskann::IndexFactory memory_index_factory = diskann::IndexFactory(memory_index_config);
                auto index = memory_index_factory.create_instance();
                index->set_start_points_at_random(static_cast<T>(0));
                return index;
            }

            void memory_index_insert_sync(std::unique_ptr<diskann::AbstractIndex>& index, std::vector<TagT> to_be_inserted) {
                // Use backend to fetch vectors by IDs
                std::vector<std::vector<T>> fetched_vectors = backend->fetch_vectors_by_ids(to_be_inserted);
                
                // Allocate aligned memory for vectors and copy from fetched vectors
                std::vector<T*> vectors;
                vectors.reserve(to_be_inserted.size());
                for (size_t i = 0; i < fetched_vectors.size(); ++i) {
                    T* vector = nullptr;
                    diskann::alloc_aligned((void**)&vector, aligned_dim * sizeof(T), 8 * sizeof(T));
                    // Copy the fetched vector data
                    std::memcpy(vector, fetched_vectors[i].data(), dim * sizeof(T));
                    // Zero out padding if needed
                    if (aligned_dim > dim) {
                        std::memset(vector + dim, 0, (aligned_dim - dim) * sizeof(T));
                    }
                    vectors.push_back(vector);
                }
                
                size_t successful_inserts = 0;
                for (size_t i = 0; i < to_be_inserted.size(); ++i) {
                    int ret = index->insert_point(vectors[i], 1 + to_be_inserted[i]);
                    if (ret == 0) ++successful_inserts;
                }

                const size_t current_active_id = active_insert_index_id.load();
                if (index.get() == memory_indices[current_active_id].get() &&
                    !shard_has_insert_slots(current_active_id) &&
                    !eviction_in_progress.load()) {
                    trigger_eviction();
                }

                for (auto v : vectors) {
                    diskann::aligned_free(v);
                }
            }

            void memory_index_insert_reconstructed_sync(std::unique_ptr<diskann::AbstractIndex>& index, std::vector<TagT> to_be_inserted) {
                // Use disk_backend to fetch vectors
                std::vector<std::vector<T>> reconstructed_vectors = this->backend->fetch_vectors_by_ids(to_be_inserted);
                size_t successful_inserts = 0;
                
                // Insert the reconstructed vectors into the provided index
                for (size_t i = 0; i < to_be_inserted.size(); i++) {
                    const auto& reconstructed_vec = reconstructed_vectors[i];
                    int ret = index->insert_point(reconstructed_vec.data(), 1 + to_be_inserted[i]);
                    if (ret == 0) ++successful_inserts;
                }
                const size_t current_active_id = active_insert_index_id.load();
                if (index.get() == memory_indices[current_active_id].get() &&
                    !shard_has_insert_slots(current_active_id) &&
                    !eviction_in_progress.load()) {
                    trigger_eviction();
                }
            }

            // LRU eviction: evict the least recently used index and replace with fresh one
            void trigger_eviction() {
                std::lock_guard<std::mutex> lock(eviction_mutex);
                
                if (eviction_in_progress.load()) {
                    return; // Already in progress
                }
                
                eviction_in_progress.store(true);
                
                // Start background eviction
                eviction_future = std::async(std::launch::async, [this]() {
                    this->perform_eviction();
                });
            }

            void perform_eviction() {
                // Debug: Show current LRU order before eviction
                std::vector<size_t> current_lru_order = lru_cache->get_all_tags();
                std::cout << "[LRU Eviction] Current LRU order before eviction: ";
                for (size_t idx : current_lru_order) {
                    std::cout << idx << " ";
                }
                std::cout << std::endl;
                
                // Get the least recently used index ID from LRU cache
                std::vector<size_t> lru_tags = lru_cache->get_lru_tags(1);
                if (lru_tags.empty()) {
                    // Fallback to cycling if LRU cache is empty
                    size_t current_active_id = active_insert_index_id.load();
                    size_t next_active_id = (current_active_id + 1) % number_of_mini_indexes;
                    std::cout << "[LRU Eviction] Fallback: Evicting index " << next_active_id << std::endl;
                    memory_indices[next_active_id] = create_memory_index(memory_index_max_points_per_index);
                    active_insert_index_id.store(next_active_id);
                } else {
                    size_t lru_index_id = lru_tags[0];
                    
                    // Check if this is the only index in the cache
                    if (current_lru_order.size() == 1) {
                        // If only one index remains, don't evict it - this prevents infinite eviction loop
                        std::cout << "[LRU Eviction] Only one index remaining (" << lru_index_id << "), skipping eviction to prevent infinite loop" << std::endl;
                        eviction_in_progress.store(false);
                        return;
                    }
                    
                    std::cout << "[LRU Eviction] Evicting least recently used index: " << lru_index_id << std::endl;
                    memory_indices[lru_index_id] = create_memory_index(memory_index_max_points_per_index);
                    active_insert_index_id.store(lru_index_id);
                    // STEP 3: Remove the evicted index from LRU cache and add the new active index
                    // Note: We don't need to explicitly evict since we're replacing the same index ID
                    lru_cache->access(lru_index_id); // This will move the index to front (most recently used)
                }
                
                // Debug: Show new LRU order after eviction
                std::vector<size_t> new_lru_order = lru_cache->get_all_tags();
                std::cout << "[LRU Eviction] New LRU order after eviction: ";
                for (size_t idx : new_lru_order) {
                    std::cout << idx << " ";
                }
                std::cout << std::endl;
                std::cout << "[LRU Eviction] LRU cache size: " << lru_cache->size() << "/" << lru_cache->max_capacity() << std::endl;
                
                // Ensure all indices are present in the LRU cache
                for (size_t i = 0; i < number_of_mini_indexes; ++i) {
                    if (!lru_cache->contains(i)) {
                        std::cout << "[LRU Eviction] Warning: Index " << i << " missing from LRU cache, adding it" << std::endl;
                        lru_cache->access(i);
                    }
                }
                
                // Eviction complete
                eviction_in_progress.store(false);
            }

            bool shard_has_capacity(size_t i) const {
                return i < memory_indices.size() && memory_indices[i] &&
                       memory_indices[i]->get_number_of_active_vectors() < memory_index_max_points_per_index;
            }

            // DiskANN insert uses occupied slots (_nd), not just live tags.
            // Lazy deletes keep a slot until consolidate_deletes.
            bool shard_has_insert_slots(size_t i) const {
                if (i >= memory_indices.size() || !memory_indices[i]) {
                    return false;
                }
                const size_t live = memory_indices[i]->get_number_of_active_vectors();
                const size_t dead = memory_indices[i]->get_number_of_lazy_deleted_points();
                return live + dead < memory_index_max_points_per_index;
            }

            diskann::IndexWriteParameters make_index_write_params() const {
                return diskann::IndexWriteParametersBuilder(memory_L, aligned_dim)
                    .with_alpha(1.2f)
                    .with_num_threads(4)
                    .build();
            }

            void note_admitted_tag(TagT mem_tag) {
                std::lock_guard<std::mutex> lock(admit_fifo_mutex);
                if (admit_set.insert(mem_tag).second) {
                    admit_fifo.push_back(mem_tag);
                }
            }

            void forget_admitted_tag(TagT mem_tag) {
                std::lock_guard<std::mutex> lock(admit_fifo_mutex);
                admit_set.erase(mem_tag);
            }

            void consolidate_shard(size_t i) {
                if (i >= memory_indices.size() || !memory_indices[i]) {
                    return;
                }
                if (memory_indices[i]->get_number_of_lazy_deleted_points() == 0) {
                    return;
                }
                memory_indices[i]->consolidate_deletes(make_index_write_params());
            }

            void consolidate_dirty_shards() {
                for (size_t i = 0; i < memory_indices.size(); ++i) {
                    consolidate_shard(i);
                }
            }

            // Reclaim tombstone slots. Not an eviction.
            void reclaim_insert_slots(size_t i) {
                if (shard_has_capacity(i) && !shard_has_insert_slots(i)) {
                    if (insert_pool) {
                        insert_pool->wait_idle();
                    }
                    consolidate_shard(i);
                }
            }

            // Evict oldest admitted points (one or more neighborhoods), then
            // consolidate only shards that lost points. Does not drop a
            // mini-index and does not wipe θ.
            size_t evict_oldest_points(size_t n) {
                if (n == 0) {
                    return 0;
                }
                if (insert_pool) {
                    insert_pool->wait_idle();
                }
                std::lock_guard<std::mutex> lock(eviction_mutex);
                std::vector<TagT> victims;
                victims.reserve(n);
                {
                    std::lock_guard<std::mutex> fifo_lock(admit_fifo_mutex);
                    while (victims.size() < n && !admit_fifo.empty()) {
                        const TagT tag = admit_fifo.front();
                        admit_fifo.pop_front();
                        if (admit_set.erase(tag)) {
                            victims.push_back(tag);
                        }
                    }
                }
                if (victims.empty()) {
                    return 0;
                }
                std::vector<char> dirty(memory_indices.size(), 0);
                for (size_t i = 0; i < memory_indices.size(); ++i) {
                    if (!memory_indices[i]) {
                        continue;
                    }
                    for (TagT tag : victims) {
                        if (memory_indices[i]->lazy_delete(tag) == 0) {
                            dirty[i] = 1;
                        }
                    }
                    if (dirty[i]) {
                        consolidate_shard(i);
                    }
                }
                const uint64_t total = point_evictions.fetch_add(victims.size()) + victims.size();
                if ((total - victims.size()) / 1000 != total / 1000) {
                    std::cout << "[PointEvict] cumulative=" << total
                              << " last_batch=" << victims.size()
                              << " |S|=" << get_number_of_vectors_in_memory_index()
                              << std::endl;
                }
                return victims.size();
            }

            // Prefer an empty shard, then the least-loaded shard that still
            // has insert slots. Data-plane inserts never evict; miss-admit
            // may evict oldest points after this returns "none".
            size_t find_shard_with_capacity() const {
                size_t best = std::numeric_limits<size_t>::max();
                size_t best_load = std::numeric_limits<size_t>::max();
                for (size_t i = 0; i < memory_indices.size(); ++i) {
                    if (!shard_has_insert_slots(i)) {
                        continue;
                    }
                    const size_t load = memory_indices[i]->get_number_of_active_vectors();
                    if (load < best_load) {
                        best = i;
                        best_load = load;
                    }
                }
                return best;
            }

            size_t make_room_for_admit(size_t n_needed) {
                if (insert_pool) {
                    insert_pool->wait_idle();
                }
                consolidate_dirty_shards();
                size_t chosen = find_shard_with_capacity();
                if (chosen != std::numeric_limits<size_t>::max()) {
                    return chosen;
                }
                evict_oldest_points(n_needed);
                return find_shard_with_capacity();
            }

            size_t evict_one_shard_sync() {
                if (insert_pool) {
                    insert_pool->wait_idle();
                }
                std::lock_guard<std::mutex> lock(eviction_mutex);
                std::vector<size_t> lru_tags = lru_cache->get_lru_tags(1);
                if (lru_tags.empty()) {
                    return std::numeric_limits<size_t>::max();
                }
                const size_t id = lru_tags[0];
                replace_shard(id);
                lru_cache->access(id);
                active_insert_index_id.store(id);
                return id;
            }

            // Miss-admit must keep |S| moving. Tombstones occupy DiskANN
            // slots, so after deletes a shard can look "not full" on live
            // count while insert_point fails. Reclaim, then rotate like warmup.
            size_t ensure_admit_shard() {
                size_t shard = active_insert_index_id.load();
                reclaim_insert_slots(shard);
                if (shard_has_insert_slots(shard)) {
                    return shard;
                }
                for (size_t i = 0; i < memory_indices.size(); ++i) {
                    if (i != shard) {
                        reclaim_insert_slots(i);
                    }
                }
                const size_t spare = find_shard_with_capacity();
                if (spare < memory_indices.size()) {
                    active_insert_index_id.store(spare);
                    lru_cache->access(spare);
                    return spare;
                }
                if (insert_pool) {
                    insert_pool->wait_idle();
                }
                std::lock_guard<std::mutex> lock(eviction_mutex);
                if (!eviction_in_progress.load()) {
                    eviction_in_progress.store(true);
                    perform_eviction();
                }
                return active_insert_index_id.load();
            }

            void reset_global_theta_map() {
                std::lock_guard<std::mutex> lock(theta_map_mutex);
                const double init_value = (metric == diskann::COSINE)
                    ? -std::numeric_limits<double>::infinity()
                    : std::numeric_limits<double>::max();
                for (auto& kv : theta_map) {
                    kv.second = init_value;
                }
                global_theta_discount = 1.0;
            }

            // Write-path only. θ ← α·θ on v's 16-D cell and L1 neighbors.
            // Saturates until a miss EMA-updates that cell.
            void stale_write_regions(const T* vec) {
                if (vec == nullptr || !use_regional_theta || !pca_utils) {
                    return;
                }
                const size_t n = pca_utils->tighten_theta_for_vector(vec, write_theta_discount);
                if (n > 0) {
                    region_invalidations.fetch_add(n, std::memory_order_relaxed);
                }
            }

            std::vector<T> fetch_backend_vector(TagT id) {
                if (!backend) {
                    return {};
                }
                try {
                    auto vs = backend->fetch_vectors_by_ids({id});
                    if (vs.empty() || vs[0].empty()) {
                        return {};
                    }
                    return std::move(vs[0]);
                } catch (...) {
                    return {};
                }
            }

            // Global θ only. Regional thresholds stay with their PCA cells.
            void invalidate_thetas_for_shard(size_t /*shard*/) {
                if (!use_regional_theta || !pca_utils) {
                    reset_global_theta_map();
                }
            }

            void replace_shard(size_t id) {
                invalidate_thetas_for_shard(id);
                memory_indices[id] = create_memory_index(memory_index_max_points_per_index);
            }

            // Miss-admit lands on the active insert shard. If it is full,
            // evict oldest admitted points and use a shard that has room.
            size_t assign_write_shard(const T* /*vec*/, size_t n_admit) {
                const size_t shard = active_insert_index_id.load();
                reclaim_insert_slots(shard);
                if (shard_has_insert_slots(shard)) {
                    return shard;
                }
                return make_room_for_admit(n_admit);
            }

            bool is_deleted_id(TagT id) {
                std::lock_guard<std::mutex> lock(deleted_mutex);
                return deleted_ids.find(id) != deleted_ids.end();
            }

            void insert_point_into_shard(size_t shard, TagT backend_id, const T* vec) {
                if (shard >= memory_indices.size() || vec == nullptr || is_deleted_id(backend_id)) {
                    return;
                }
                if (!shard_has_insert_slots(shard)) {
                    return;
                }
                T* aligned = nullptr;
                diskann::alloc_aligned((void**)&aligned, aligned_dim * sizeof(T), 8 * sizeof(T));
                std::memcpy(aligned, vec, dim * sizeof(T));
                if (aligned_dim > dim) {
                    std::memset(aligned + dim, 0, (aligned_dim - dim) * sizeof(T));
                }
                const TagT mem_tag = static_cast<TagT>(1) + backend_id;
                if (memory_indices[shard]->insert_point(aligned, mem_tag) == 0) {
                    note_admitted_tag(mem_tag);
                }
                diskann::aligned_free(aligned);
            }

            // Data-plane admit: never evict and never touch θ. Land v only
            // when a shard already has a free slot.
            bool try_admit_data_vector(TagT id, const T* vector) {
                if (vector == nullptr) {
                    return false;
                }
                size_t shard = active_insert_index_id.load();
                reclaim_insert_slots(shard);
                if (!shard_has_insert_slots(shard)) {
                    shard = find_shard_with_capacity();
                    if (shard == std::numeric_limits<size_t>::max()) {
                        return false;
                    }
                }
                insert_point_into_shard(shard, id, vector);
                lru_cache->access(shard);
                return true;
            }

            // Admit n_admit backend ids into S. θ is committed by the caller
            // from the live K-th distance (clears the write discount).
            void admit_miss_neighbors(const T* query_ptr, uint32_t n_admit, uint32_t* tags) {
                std::vector<TagT> tags_to_insert;
                tags_to_insert.reserve(n_admit);
                {
                    std::lock_guard<std::mutex> lock(deleted_mutex);
                    for (uint32_t j = 0; j < n_admit; ++j) {
                        const TagT t = static_cast<TagT>(tags[j]);
                        if (t == std::numeric_limits<TagT>::max()) {
                            continue;
                        }
                        if (deleted_ids.find(t) == deleted_ids.end()) {
                            tags_to_insert.push_back(t);
                        }
                    }
                }
                const size_t shard = assign_write_shard(query_ptr, n_admit);
                const bool can_admit = shard < memory_indices.size() && shard_has_insert_slots(shard)
                                       && !tags_to_insert.empty();
                if (can_admit) {
                    insert_pool->submit(memory_indices[shard], tags_to_insert, data_path, this->dim);
                }
            }

            void handle_backend_miss(const T* query_ptr, uint32_t K, uint32_t* tags, float* dists, void* backend_stats) {
                note_cache_lookup_end();
                this->backend->search(query_ptr, static_cast<uint64_t>(K), tags, dists, nullptr, backend_stats);
                std::vector<TagT> tags_to_insert;
                tags_to_insert.reserve(K);
                {
                    std::lock_guard<std::mutex> lock(deleted_mutex);
                    for (uint32_t j = 0; j < K; ++j) {
                        const TagT t = static_cast<TagT>(tags[j]);
                        if (t == std::numeric_limits<TagT>::max()) {
                            continue;
                        }
                        if (deleted_ids.find(t) == deleted_ids.end()) {
                            tags_to_insert.push_back(t);
                        }
                    }
                }
                const size_t current_active_id = ensure_admit_shard();
                const float kth = (dists != nullptr && K > 0) ? dists[K - 1] : 0.0f;
                if (lazy_theta_updates) {
                    T* query_copy = nullptr;
                    diskann::alloc_aligned((void**)&query_copy, this->aligned_dim * sizeof(T), 8 * sizeof(T));
                    std::memcpy(query_copy, query_ptr, this->aligned_dim * sizeof(T));
                    insert_pool->submit(memory_indices[current_active_id], tags_to_insert,
                                        data_path, this->dim, K, kth, query_copy);
                } else {
                    update_theta(query_ptr, K, kth);
                    insert_pool->submit(memory_indices[current_active_id], tags_to_insert,
                                        data_path, this->dim, K, kth);
                }
                for (uint32_t j = 0; j < K; ++j) {
                    tags[j] += 1;
                }
            }

            bool has_k_live_results(uint32_t K, const uint32_t* tags, size_t num_results) {
                if (num_results < K) {
                    return false;
                }
                if (tags == nullptr) {
                    return true;
                }
                for (uint32_t j = 0; j < K; ++j) {
                    const uint32_t tag = tags[j];
                    if (tag == 0 || tag == std::numeric_limits<uint32_t>::max()) {
                        return false;
                    }
                    if (is_deleted_id(static_cast<TagT>(tag - 1))) {
                        return false;
                    }
                }
                return true;
            }

            bool isHit(const T* query_ptr, uint32_t K, const float* distances,
                       const uint32_t* tags = nullptr, size_t num_results = std::numeric_limits<size_t>::max()) {
                if (!has_k_live_results(K, tags, num_results)) {
                    return false;
                }
                std::lock_guard<std::mutex> lock(theta_map_mutex);
                
                if (use_regional_theta) {
                    return pca_utils->isHit(query_ptr, K, distances, this->get_number_of_vectors_in_memory_index(),
                                            deviation_factor);
                }
                if (this->get_number_of_vectors_in_memory_index() < K || distances == nullptr || K == 0) {
                    return false;
                }
                double threshold = theta_map[K];
                const bool is_uninitialized = (metric == diskann::COSINE)
                    ? (threshold == -std::numeric_limits<double>::infinity())
                    : (threshold >= std::numeric_limits<double>::max() * 0.5);
                if (is_uninitialized) {
                    return false;
                }
                const double cache_distance = static_cast<double>(distances[K - 1]);
                const double tolerance_threshold = (1.0 + deviation_factor) * threshold;
                return cache_distance <= tolerance_threshold;
            }

            void update_theta(const T* query_ptr, uint32_t K, float query_distance) {
                std::lock_guard<std::mutex> lock(theta_map_mutex);
                if (use_regional_theta) {
                    pca_utils->update_theta(query_ptr, K, query_distance, p);
                } else {
                    double current_theta = theta_map[K];
                    // Handle initialization: if current_theta is uninitialized, replace it completely
                    bool is_uninitialized = (metric == diskann::COSINE)
                        ? (current_theta == -std::numeric_limits<double>::infinity())
                        : (current_theta >= std::numeric_limits<double>::max() * 0.5);
                    if (is_uninitialized) {
                        theta_map[K] = static_cast<double>(query_distance);
                    } else {
                        theta_map[K] = p * static_cast<double>(query_distance) + (1 - p) * current_theta;
                    }
                    global_theta_discount = 1.0;
                }
            }


            // Helper function to search a single memory index
            bool search_single_index(size_t index_id, const T* query_ptr, uint32_t K,  
                                   uint32_t* query_result_tags_ptr, std::vector<T*>& res, 
                                   float* query_result_dists_ptr) {
                if (memory_indices[index_id]->get_number_of_active_vectors() > 0) {
                    const size_t num_results = memory_indices[index_id]->search_with_tags(
                        query_ptr, K, memory_L, query_result_tags_ptr, query_result_dists_ptr, res);
                    bool is_hit = this->isHit(query_ptr, K, query_result_dists_ptr,
                                              query_result_tags_ptr, num_results);
                    
                    // Only update LRU cache if this search resulted in a hit
                    if (is_hit) {
                        lru_cache->access(index_id);
                    }
                    
                    return is_hit;
                }
                return false;
            }

            // Helper function to merge and re-rank results from multiple indices
            // Takes results from all indices, merges by tag (keeping best distance), and returns top-K
            struct MergedResult {
                uint32_t tag;
                float distance;
                T* vector_ptr;
                size_t source_index_id;
            };

            void merge_and_rerank_results(
                const std::vector<std::vector<uint32_t>>& all_tags,
                const std::vector<std::vector<float>>& all_dists,
                const std::vector<std::vector<T*>>& all_res,
                const std::vector<size_t>& index_ids,
                uint32_t K,
                std::vector<uint32_t>& merged_tags,
                std::vector<float>& merged_dists,
                std::vector<T*>& merged_res,
                std::vector<size_t>& contributing_indices_ordered) {
                
                // Map: tag -> (best_distance, vector_ptr, source_index_id)
                std::unordered_map<uint32_t, std::tuple<float, T*, size_t>> tag_to_best;
                
                // Merge results from all indices
                for (size_t idx = 0; idx < index_ids.size(); ++idx) {
                    size_t index_id = index_ids[idx];
                    
                    if (index_id >= all_tags.size() || index_id >= all_dists.size() || index_id >= all_res.size()) {
                        continue;
                    }
                    
                    const auto& tags = all_tags[index_id];
                    const auto& dists = all_dists[index_id];
                    const auto& res_vec = all_res[index_id];
                    
                    // Process ALL results from this index (tags and dists should have same size)
                    // Note: res_vec might be empty (DiskANN skips populating it if empty)
                    // We don't actually need the vectors for merging, only tags and distances
                    size_t num_results = std::min(tags.size(), dists.size());
                    // Ensure we process all available results, not just first K
                    for (size_t i = 0; i < num_results; ++i) {
                        uint32_t tag = tags[i];
                        float dist = dists[i];
                        T* vec_ptr = (i < res_vec.size()) ? res_vec[i] : nullptr;  // May be null if res_vec wasn't populated
                        
                        // For L2: smaller is better, for COSINE: smaller is better (distance)
                        // For INNER_PRODUCT: smaller is better (negative inner product)
                        // Deduplicate by tag, keeping the best (smallest) distance
                        auto it = tag_to_best.find(tag);
                        if (it == tag_to_best.end()) {
                            // New tag, add it
                            tag_to_best[tag] = std::make_tuple(dist, vec_ptr, index_id);
                        } else {
                            // Tag already exists, keep the better (smaller) distance
                            float existing_dist = std::get<0>(it->second);
                            if (dist < existing_dist) {
                                tag_to_best[tag] = std::make_tuple(dist, vec_ptr, index_id);
                            }
                        }
                    }
                }
                
                // Convert map to vector and sort by distance
                std::vector<MergedResult> merged_candidates;
                merged_candidates.reserve(tag_to_best.size());
                for (const auto& [tag, tuple_val] : tag_to_best) {
                    merged_candidates.push_back({
                        tag,
                        std::get<0>(tuple_val),
                        std::get<1>(tuple_val),
                        std::get<2>(tuple_val)
                    });
                }
                
                // Sort by distance (ascending - smaller is better)
                std::sort(merged_candidates.begin(), merged_candidates.end(),
                    [](const MergedResult& a, const MergedResult& b) {
                        return a.distance < b.distance;
                    });
                
                // Take top-K
                size_t result_size = std::min(static_cast<size_t>(K), merged_candidates.size());
                merged_tags.resize(result_size);
                merged_dists.resize(result_size);
                merged_res.resize(result_size);
                
                // Track contribution counts per index
                std::unordered_map<size_t, size_t> contribution_counts;
                
                for (size_t i = 0; i < result_size; ++i) {
                    merged_tags[i] = merged_candidates[i].tag;
                    merged_dists[i] = merged_candidates[i].distance;
                    merged_res[i] = merged_candidates[i].vector_ptr;
                    
                    // Count contributions to final top-K
                    size_t source_index_id = merged_candidates[i].source_index_id;
                    contribution_counts[source_index_id]++;
                }
                
                // Create ordered list of contributing indices: least to most contributing
                // This ensures that when we call access() in this order, the most contributing
                // index will be at the front (most recently used) position
                contributing_indices_ordered.clear();
                contributing_indices_ordered.reserve(contribution_counts.size());
                
                // Convert to vector of pairs for sorting
                std::vector<std::pair<size_t, size_t>> contributions;
                contributions.reserve(contribution_counts.size());
                for (const auto& [index_id, count] : contribution_counts) {
                    contributions.push_back({index_id, count});
                }
                
                // Sort by contribution count (ascending: least to most)
                std::sort(contributions.begin(), contributions.end(),
                    [](const std::pair<size_t, size_t>& a, const std::pair<size_t, size_t>& b) {
                        return a.second < b.second;
                    });
                
                // Extract ordered index IDs
                for (const auto& [index_id, count] : contributions) {
                    contributing_indices_ordered.push_back(index_id);
                }
            }

            // Parallel search across all memory indices using thread pool with merge and re-rank
            bool parallel_search_memory_indices(const T* query_ptr, uint32_t K, 
                                              uint32_t* query_result_tags_ptr, std::vector<T*>& res,
                                              float* query_result_dists_ptr) {
                if (number_of_mini_indexes == 1) {
                    // Single index case - no need for parallelization
                    return search_single_index(0, query_ptr, K, query_result_tags_ptr, res, query_result_dists_ptr);
                }

                // Prepare results for each index. Each index id is claimed by one worker.
                std::vector<std::vector<uint32_t>> all_tags(number_of_mini_indexes);
                std::vector<std::vector<float>> all_dists(number_of_mini_indexes);
                std::vector<std::vector<T*>> all_res(number_of_mini_indexes);

                auto search_worker = [&](size_t index_id) {
                    if (index_id >= memory_indices.size()) {
                        return;
                    }

                    if (memory_indices[index_id]->get_number_of_active_vectors() > 0) {
                        std::vector<uint32_t> temp_tags(K);
                        std::vector<float> temp_dists(K);
                        // Don't pre-size temp_res - DiskANN's get_vector expects valid pointers if res_vectors is non-empty.
                        std::vector<T*> temp_res;

                        size_t num_results = memory_indices[index_id]->search_with_tags(query_ptr, K, memory_L, temp_tags.data(), temp_dists.data(), temp_res);

                        temp_tags.resize(num_results);
                        temp_dists.resize(num_results);
                        all_tags[index_id] = std::move(temp_tags);
                        all_dists[index_id] = std::move(temp_dists);
                        all_res[index_id] = std::move(temp_res);
                    }
                };

                if (search_pool) {
                    search_pool->parallel_for(number_of_mini_indexes, search_worker);
                } else {
                    for (size_t index_id = 0; index_id < number_of_mini_indexes; ++index_id) {
                        search_worker(index_id);
                    }
                }

                // Collect indices that were searched
                std::vector<size_t> searched_index_ids;
                for (size_t i = 0; i < number_of_mini_indexes; ++i) {
                    if (!all_tags[i].empty()) {
                        searched_index_ids.push_back(i);
                    }
                }
                
                // Fast path: if only one index has results, use it directly (no merge needed)
                if (searched_index_ids.size() == 1) {
                    size_t single_index_id = searched_index_ids[0];
                    if (all_tags[single_index_id].size() >= K) {
                        bool is_hit = this->isHit(query_ptr, K, all_dists[single_index_id].data(),
                                                  all_tags[single_index_id].data(), all_tags[single_index_id].size());
                        if (is_hit) {
                            std::copy(all_tags[single_index_id].begin(), all_tags[single_index_id].begin() + K, query_result_tags_ptr);
                            std::copy(all_dists[single_index_id].begin(), all_dists[single_index_id].begin() + K, query_result_dists_ptr);
                            res.resize(K);
                            if (all_res[single_index_id].size() >= K) {
                                std::copy(all_res[single_index_id].begin(), all_res[single_index_id].begin() + K, res.begin());
                            }
                            lru_cache->access(single_index_id);
                            return true;
                        }
                    }
                }

                // Multiple indices have results - always merge and re-rank for better recall
                std::vector<uint32_t> merged_tags;
                std::vector<float> merged_dists;
                std::vector<T*> merged_res;
                std::vector<size_t> contributing_indices_ordered;
                merge_and_rerank_results(all_tags, all_dists, all_res, searched_index_ids, K, merged_tags, merged_dists, merged_res, contributing_indices_ordered);

                // Check if merged result is a hit
                // isHit() requires at least K results (it accesses distances[K-1])
                if (merged_dists.size() >= K) {
                    bool is_hit = this->isHit(query_ptr, K, merged_dists.data(),
                                              merged_tags.data(), merged_tags.size());
                    
                    if (is_hit) {
                        // Copy merged results to output (only first K)
                        size_t copy_size = std::min(static_cast<size_t>(K), merged_tags.size());
                        std::copy(merged_tags.begin(), merged_tags.begin() + copy_size, query_result_tags_ptr);
                        std::copy(merged_dists.begin(), merged_dists.begin() + copy_size, query_result_dists_ptr);
                        res.resize(copy_size);
                        std::copy(merged_res.begin(), merged_res.begin() + copy_size, res.begin());
                        
                        // Update LRU cache in order from least to most contributing
                        // This ensures the most contributing index ends up at the front (most recently used)
                        for (size_t index_id : contributing_indices_ordered) {
                            lru_cache->access(index_id);
                        }
                        
                        return true;
                    }
                }

                return false; // No hits found
            }



            // Problematic search strategy: stop at first hit in LRU order (causes recall drops)
            bool search_sequential_lru_stop_first_hit(const T* query_ptr, uint32_t K, uint32_t* query_result_tags_ptr, std::vector<T *>& res, float* query_result_dists_ptr, void* backend_stats) {
                std::vector<size_t> lru_order = lru_cache->get_all_tags();
                
                for (size_t index_id : lru_order) {
                    if (index_id >= memory_indices.size()) {
                        continue;  // Safety check
                    }
                    
                    if (memory_indices[index_id]->get_number_of_active_vectors() > 0) {
                        // Use local temp res to avoid issues with passed-in res vector
                        // This ensures clean state for each search
                        std::vector<T*> temp_res;
                        size_t num_results = memory_indices[index_id]->search_with_tags(query_ptr, K, memory_L, query_result_tags_ptr, query_result_dists_ptr, temp_res);
                        
                        // Copy to output res if hit (temp_res will be destroyed but pointers are still valid)
                        bool is_hit = this->isHit(query_ptr, K, query_result_dists_ptr,
                                                  query_result_tags_ptr, num_results);
                        
                        // Only update LRU cache if this search resulted in a hit
                        if (is_hit) {
                            // Copy temp_res to res (pointers are still valid at this point)
                            res = temp_res;
                            lru_cache->access(index_id);
                            return true; // Found a hit, stop immediately (this was the problem!)
                        }
                    }
                }
                
                // No hit found - ensure res is empty
                res.clear();
                handle_backend_miss(query_ptr, K, query_result_tags_ptr, query_result_dists_ptr, backend_stats);
                return false;
            }

            // Adaptive search strategy: monitor hit ratio and switch to SEQUENTIAL_ALL when low
            bool search_adaptive_hit_ratio(const T* query_ptr, uint32_t K, uint32_t* query_result_tags_ptr, std::vector<T *>& res, float* query_result_dists_ptr, void* backend_stats) {
                // Check if we should use adaptive strategy (hit ratio is low)
                bool use_adaptive = should_use_adaptive_strategy();
                
                bool was_hit;
                
                // Clear res vector to ensure clean state regardless of previous call
                // This prevents stale pointers when switching between strategies
                res.clear();
                
                if (use_adaptive) {
                    // Use SEQUENTIAL_ALL strategy when hit ratio is low
                    was_hit = search_sequential_all_impl(query_ptr, K, query_result_tags_ptr, res, query_result_dists_ptr, backend_stats);
                } else {
                    // Use SEQUENTIAL_LRU_STOP_FIRST_HIT strategy when hit ratio is good
                    was_hit = search_sequential_lru_stop_first_hit(query_ptr, K, query_result_tags_ptr, res, query_result_dists_ptr, backend_stats);
                }
                
                // Update hit history for monitoring
                update_hit_history(was_hit);
                
                return was_hit;
            }
            
            // Helper method for SEQUENTIAL_ALL implementation with merge and re-rank
            bool search_sequential_all_impl(const T* query_ptr, uint32_t K, uint32_t* query_result_tags_ptr, std::vector<T *>& res, float* query_result_dists_ptr, void* backend_stats) {
                std::vector<size_t> lru_order = lru_cache->get_all_tags();
                
                // Collect results from all indices
                std::vector<std::vector<uint32_t>> all_tags(number_of_mini_indexes);
                std::vector<std::vector<float>> all_dists(number_of_mini_indexes);
                std::vector<std::vector<T*>> all_res(number_of_mini_indexes);
                std::vector<size_t> searched_index_ids;
                
                // Search all indices in LRU order
                for (size_t index_id : lru_order) {
                    if (index_id >= memory_indices.size()) {
                        continue;
                    }
                    
                    if (memory_indices[index_id]->get_number_of_active_vectors() > 0) {
                        // Allocate temporary storage for this search
                        std::vector<uint32_t> temp_tags(K);
                        std::vector<float> temp_dists(K);
                        // Don't pre-size temp_res - DiskANN's get_vector expects valid pointers if res_vectors is non-empty
                        // We don't need the vectors for merging anyway, only tags and distances
                        std::vector<T*> temp_res;
                        
                        // Search in this index - returns number of results found
                        size_t num_results = memory_indices[index_id]->search_with_tags(query_ptr, K, memory_L, temp_tags.data(), temp_dists.data(), temp_res);
                        
                        // Resize tags and dists to actual number of results
                        // temp_res stays empty (DiskANN skips populating it if empty, which is fine)
                        temp_tags.resize(num_results);
                        temp_dists.resize(num_results);
                        
                        // Store results for merging
                        all_tags[index_id] = std::move(temp_tags);
                        all_dists[index_id] = std::move(temp_dists);
                        all_res[index_id] = std::move(temp_res);
                        searched_index_ids.push_back(index_id);
                    }
                }
                
                // Fast path: if only one index has results, use it directly (no merge needed)
                // This matches STOP_FIRST behavior exactly for single-index case
                if (searched_index_ids.size() == 1) {
                    size_t single_index_id = searched_index_ids[0];
                    if (single_index_id < all_tags.size() && single_index_id < all_dists.size() && 
                        all_tags[single_index_id].size() >= K && all_dists[single_index_id].size() >= K) {
                        bool is_hit = this->isHit(query_ptr, K, all_dists[single_index_id].data(),
                                                  all_tags[single_index_id].data(), all_tags[single_index_id].size());
                        
                        if (is_hit) {
                            std::copy(all_tags[single_index_id].begin(), all_tags[single_index_id].begin() + K, query_result_tags_ptr);
                            std::copy(all_dists[single_index_id].begin(), all_dists[single_index_id].begin() + K, query_result_dists_ptr);
                            res.resize(K);
                            if (single_index_id < all_res.size() && all_res[single_index_id].size() >= K) {
                                std::copy(all_res[single_index_id].begin(), all_res[single_index_id].begin() + K, res.begin());
                            }
                            lru_cache->access(single_index_id);
                            return true;
                        }
                        // If single index doesn't have hit, continue to disk search below
                    }
                }
                
                // Multiple indices have results - merge and re-rank for better recall
                // We'll check merged results, and if that fails, fall back to checking individual indices
                std::vector<uint32_t> merged_tags;
                std::vector<float> merged_dists;
                std::vector<T*> merged_res;
                std::vector<size_t> contributing_indices_ordered;
                merge_and_rerank_results(all_tags, all_dists, all_res, searched_index_ids, K, merged_tags, merged_dists, merged_res, contributing_indices_ordered);
                
                // Check if merged result is a hit
                // isHit() requires at least K results (it accesses distances[K-1])
                if (merged_dists.size() >= K) {
                    bool merged_is_hit = this->isHit(query_ptr, K, merged_dists.data(),
                                                     merged_tags.data(), merged_tags.size());
                    
                    if (merged_is_hit) {
                        // Copy merged results to output (only first K)
                        size_t copy_size = std::min(static_cast<size_t>(K), merged_tags.size());
                        std::copy(merged_tags.begin(), merged_tags.begin() + copy_size, query_result_tags_ptr);
                        std::copy(merged_dists.begin(), merged_dists.begin() + copy_size, query_result_dists_ptr);
                        res.resize(copy_size);
                        if (merged_res.size() >= copy_size) {
                            std::copy(merged_res.begin(), merged_res.begin() + copy_size, res.begin());
                        }
                        
                        // Update LRU cache in order from least to most contributing
                        // This ensures the most contributing index ends up at the front (most recently used)
                        for (size_t index_id : contributing_indices_ordered) {
                            lru_cache->access(index_id);
                        }
                        
                        return true;
                    }
                }
                
                // Merged result didn't pass hit check, try individual indices as fallback
                // This ensures we don't lose recall compared to STOP_FIRST_HIT
                // Check in LRU order (matching STOP_FIRST_HIT behavior)
                for (size_t index_id : lru_order) {
                    // Skip if this index wasn't searched or doesn't have results
                    if (std::find(searched_index_ids.begin(), searched_index_ids.end(), index_id) == searched_index_ids.end()) {
                        continue;
                    }
                    
                    if (index_id >= all_tags.size() || index_id >= all_dists.size()) {
                        continue;
                    }
                    
                    const auto& tags = all_tags[index_id];
                    const auto& dists = all_dists[index_id];
                    
                    // Check if this index has enough results and is a hit
                    if (tags.size() >= K && dists.size() >= K) {
                        bool is_hit = this->isHit(query_ptr, K, dists.data(), tags.data(), tags.size());
                        
                        if (is_hit) {
                            // Use results from this index directly (it's already a hit)
                            // This matches STOP_FIRST_HIT behavior - use first hit in LRU order
                            std::copy(tags.begin(), tags.begin() + K, query_result_tags_ptr);
                            std::copy(dists.begin(), dists.begin() + K, query_result_dists_ptr);
                            res.resize(K);
                            if (index_id < all_res.size() && all_res[index_id].size() >= K) {
                                std::copy(all_res[index_id].begin(), all_res[index_id].begin() + K, res.begin());
                            }
                            lru_cache->access(index_id);
                            return true;
                        }
                    }
                }
                
                handle_backend_miss(query_ptr, K, query_result_tags_ptr, query_result_dists_ptr, backend_stats);
                return false;
            }

            void load_sampled_data(const std::string& data_path, T*& sampled_data, size_t& sampled_num_points, size_t aligned_dim, size_t total_num_points, size_t sample_rate = 1000) {
                // Calculate the number of sampled points
                sampled_num_points = total_num_points / sample_rate;
                sampled_data = nullptr;

                // Allocate memory for the sampled data
                diskann::alloc_aligned((void**)&sampled_data, sampled_num_points * aligned_dim * sizeof(T), 8 * sizeof(T));

                // Open the binary file
                std::ifstream reader(data_path, std::ios::binary);
                if (!reader.is_open()) {
                    throw std::runtime_error("Failed to open file: " + data_path);
                }

                // Skip metadata (2 * sizeof(uint32_t))
                reader.seekg(2 * sizeof(uint32_t), std::ios::beg);

                // Randomly sample points
                std::default_random_engine generator(std::random_device{}());
                std::uniform_int_distribution<size_t> distribution(0, total_num_points - 1);

                std::unordered_set<size_t> sampled_indices;
                while (sampled_indices.size() < sampled_num_points) {
                    sampled_indices.insert(distribution(generator));
                }

                size_t current_index = 0;
                size_t sampled_index = 0;
                T* buffer = new T[aligned_dim];

                for (size_t i = 0; i < total_num_points; ++i) {
                    if (i % 100 == 0) {
                        std::cout << "Reading point " << i << "/" << total_num_points << "\r" << std::flush;
                    }
                    // Read the vector
                    reader.read(reinterpret_cast<char*>(buffer), dim * sizeof(T));
                    // Skip padding for aligned dimensions
                    reader.seekg((aligned_dim - dim) * sizeof(T), std::ios::cur);

                    // If the current index is in the sampled set, copy it to the sampled data
                    if (sampled_indices.count(i)) {
                        std::memcpy(sampled_data + sampled_index * aligned_dim, buffer, dim * sizeof(T));
                        std::memset(sampled_data + sampled_index * aligned_dim + dim, 0, (aligned_dim - dim) * sizeof(T));
                        ++sampled_index;
                    }

                    if (sampled_index >= sampled_num_points) {
                        break;
                    }
                }

                delete[] buffer;
                reader.close();
            }



        public:
            // Aker: cache-layer time on the last search() of this thread (ms).
            double last_cache_lookup_ms() const { return tls_cache_lookup_ms; }

            template <typename... Args>
            QVCache(const std::string& data_path,
                        const std::string& pca_prefix,
                        uint32_t R, uint32_t memory_L,
                        uint32_t B, uint32_t M,
                        float alpha,
                        uint32_t build_threads,
                        uint32_t search_threads,
                        bool use_reconstructed_vectors,
                        double p,
                        double deviation_factor, 
                        size_t memory_index_max_points,
                        uint32_t beamwidth_,
                        bool use_regional_theta = true,
                        size_t pca_dim = 16,
                        size_t buckets_per_dim = 4,
                        size_t max_regions_ = std::numeric_limits<size_t>::max(),
                        uint32_t n_async_insert_threads_ = 4,
                        bool lazy_theta_updates_ = true,
                        size_t number_of_mini_indexes_ = 2,
                        bool search_mini_indexes_in_parallel_ = false,
                        size_t max_search_threads_ = 32,
                        diskann::Metric metric_ = diskann::L2,
                        std::unique_ptr<BackendInterface<T, TagT>> disk_backend_ptr = nullptr,
                        bool learn_pca_from_queries = false,
                        const std::string& query_path = "")
                        : data_path(data_path),
                        pca_prefix(pca_prefix),
                        search_threads(search_threads),
                        use_reconstructed_vectors(use_reconstructed_vectors),
                        p(p),
                        deviation_factor(deviation_factor),
                        memory_L(memory_L),
                        beamwidth(beamwidth_),
                        use_regional_theta(use_regional_theta),
                        number_of_mini_indexes(number_of_mini_indexes_),
                        memory_index_max_points_per_index(memory_index_max_points / number_of_mini_indexes_), // Equal capacity per index
                        n_async_insert_threads(n_async_insert_threads_),
                        lazy_theta_updates(lazy_theta_updates_),
                        search_mini_indexes_in_parallel(search_mini_indexes_in_parallel_),
                        max_search_threads(max_search_threads_),
                        metric(metric_),
                        backend(std::move(disk_backend_ptr))
            {                
                // Read metadata
                diskann::get_bin_metadata(data_path, num_points, dim);
                aligned_dim = ROUND_UP(dim, 8);

                // Build LRU-managed memory indices
                memory_indices.reserve(number_of_mini_indexes);
                for (size_t i = 0; i < number_of_mini_indexes; ++i) {
                    memory_indices.push_back(create_memory_index(memory_index_max_points_per_index));
                }
                
                // Initialize LRU cache for managing mini-index access patterns
                lru_cache = std::make_unique<LRUCache<size_t>>(number_of_mini_indexes);
                
                // Set index 0 as active initially for insertions
                active_insert_index_id.store(0);
                
                // Initialize LRU cache with ALL indices (not just the active one)
                // This ensures we have a complete LRU order from the start
                for (size_t i = 0; i < number_of_mini_indexes; ++i) {
                    lru_cache->access(i);
                }
            
                std::cout << "QVCache LRU-managed memory indices built successfully!" << std::endl;
                std::cout << "Created " << number_of_mini_indexes << " indices, each can hold up to " << memory_index_max_points_per_index << " vectors" << std::endl;
                std::cout << "LRU eviction policy enabled" << std::endl;
                if (search_mini_indexes_in_parallel && number_of_mini_indexes > 1) {
                    const size_t pool_threads = std::min(max_search_threads, number_of_mini_indexes);
                    if (pool_threads > 1) {
                        search_pool = std::make_unique<SearchThreadPool>(pool_threads);
                    }
                    std::cout << "Parallel search enabled with a fixed pool of "
                              << (search_pool ? search_pool->size() : 0) << " threads" << std::endl;
                }

                std::cout << "QVCache disk index built successfully!" << std::endl;

                // Initialize insert thread pool for async insertions
                if (use_reconstructed_vectors) {
                    auto task = [this](std::unique_ptr<diskann::AbstractIndex>& index, std::vector<TagT> to_be_inserted, const std::string& data_path, const size_t dim, uint32_t K, float query_distance) {
                        this->memory_index_insert_reconstructed_sync(index, to_be_inserted);
                    };
                    if (lazy_theta_updates) {
                        auto theta_update_task = [this](T* query_ptr, uint32_t K, float query_distance) {
                            this->update_theta(query_ptr, K, query_distance);
                        };
                        insert_pool = std::make_unique<qvcache::InsertThreadPool<T, TagT>>(n_async_insert_threads, task, theta_update_task);
                    } else {
                        insert_pool = std::make_unique<qvcache::InsertThreadPool<T, TagT>>(n_async_insert_threads, task);
                    }
                } else {
                    auto task = [this](std::unique_ptr<diskann::AbstractIndex>& index, std::vector<TagT> to_be_inserted, const std::string& data_path, const size_t dim, uint32_t K, float query_distance) {
                        this->memory_index_insert_sync(index, to_be_inserted);
                    };
                    if (lazy_theta_updates) {
                        auto theta_update_task = [this](T* query_ptr, uint32_t K, float query_distance) {
                            this->update_theta(query_ptr, K, query_distance);
                        };
                        insert_pool = std::make_unique<qvcache::InsertThreadPool<T, TagT>>(n_async_insert_threads, task, theta_update_task);
                    } else {
                        insert_pool = std::make_unique<qvcache::InsertThreadPool<T, TagT>>(n_async_insert_threads, task);
                    }
                }

                std::cout << "QVCache built successfully with LRU eviction policy!" << std::endl;

                // PCA is constructed at construction time using Eigen. Eigen is required.
                // Query-fit PCA uses a separate file so it never reuses a data-fit .pca.bin.
                if (use_regional_theta) {
                    const std::string pca_file_prefix = learn_pca_from_queries
                        ? (pca_prefix + ".query")
                        : pca_prefix;
                    pca_utils = std::make_unique<PCAUtils<T>>(dim, pca_dim, buckets_per_dim, pca_file_prefix, metric, max_regions_);
                    pca_utils->set_write_l1_radius(write_l1_radius);
                    bool loaded = false;
                    if constexpr (std::is_floating_point<T>::value) {
                        loaded = pca_utils->load_pca_from_file(false);
                    } else {
                        loaded = pca_utils->load_pca_from_file(true);
                    }
                    if (loaded) {
                        std::cout << "[QVCache] Loaded PCA from file: " << pca_utils->get_pca_filename_for_logging() << std::endl;
                    } else {
                        T* pca_data = nullptr;
                        size_t sampled_num_points = 0;
                        size_t pca_aligned_dim = aligned_dim;
                        if (learn_pca_from_queries) {
                            if (query_path.empty()) {
                                throw std::runtime_error("[QVCache] learn_pca_from_queries=true but query_path is empty");
                            }
                            size_t q_num = 0, q_dim = 0;
                            diskann::load_aligned_bin<T>(query_path, pca_data, q_num, q_dim, pca_aligned_dim);
                            if (q_dim != dim) {
                                diskann::aligned_free(pca_data);
                                throw std::runtime_error("[QVCache] query dim (" + std::to_string(q_dim) +
                                                         ") != data dim (" + std::to_string(dim) + ")");
                            }
                            sampled_num_points = q_num;
                            std::cout << "[QVCache] Learning PCA from " << sampled_num_points
                                      << " queries in " << query_path << std::endl;
                        } else {
                            std::cout << "[QVCache] No PCA file found or mismatch, running PCA on data..." << std::endl;
                            diskann::get_bin_metadata(data_path, num_points, dim);
                            pca_aligned_dim = ROUND_UP(dim, 8);
                            load_sampled_data(data_path, pca_data, sampled_num_points, pca_aligned_dim, num_points);
                            std::cout << "[QVCache] Loaded " << sampled_num_points << " sampled points from " << data_path << std::endl;
                        }
                        pca_utils->construct_pca_from_data(pca_data, sampled_num_points, pca_aligned_dim, pca_file_prefix);
                        diskann::aligned_free(pca_data);
                    }
                } else {
                    std::cout << "[QVCache] Skipping PCA construction (use_regional_theta is false)." << std::endl;
                }

                if (!use_regional_theta) {
                    // Initialize global theta_map for K=1,5,10,100
                    // Use values so that everything is a miss initially
                    // For cosine: use negative infinity so distances[K-1] > -infinity is always true (MISS)
                    // For L2: use max double (will check for uninitialized and force MISS anyway)
                    double init_value = (metric == diskann::COSINE) ? -std::numeric_limits<double>::infinity() : std::numeric_limits<double>::max();
                    theta_map[1] = init_value;
                    theta_map[5] = init_value;
                    theta_map[10] = init_value;
                    theta_map[100] = init_value;
                }
                
                // Initialize hit rate tracker (lazy initialization - only when adaptive strategy is enabled)
                // Will be created when enable_adaptive_strategy(true) is called
                hit_rate_tracker = nullptr;
            }


            bool search(const T* query_ptr, uint32_t K, uint32_t* query_result_tags_ptr, std::vector<T *>& res, float* query_result_dists_ptr, void* backend_stats) {
                tls_search_t0 = std::chrono::high_resolution_clock::now();
                tls_cache_lookup_ms = 0.0;
                size_t current_active_id = active_insert_index_id.load();
                (void)current_active_id;
                
                // Search all memory indices based on configured strategy
                bool is_hit = false;
                
                if (search_strategy == SearchStrategy::PARALLEL && search_mini_indexes_in_parallel && number_of_mini_indexes > 1) {
                    is_hit = parallel_search_memory_indices(query_ptr, K, query_result_tags_ptr, res, query_result_dists_ptr);

                } else if (search_strategy == SearchStrategy::SEQUENTIAL_LRU_STOP_FIRST_HIT) {
                    return search_sequential_lru_stop_first_hit(query_ptr, K, query_result_tags_ptr, res, query_result_dists_ptr, backend_stats);
                } else if (search_strategy == SearchStrategy::SEQUENTIAL_LRU_ADAPTIVE) {
                    return search_adaptive_hit_ratio(query_ptr, K, query_result_tags_ptr, res, query_result_dists_ptr, backend_stats);
                } else if (search_strategy == SearchStrategy::SEQUENTIAL_ALL) {
                    return search_sequential_all_impl(query_ptr, K, query_result_tags_ptr, res, query_result_dists_ptr, backend_stats);
                } else {
                    // Fallback: use SEQUENTIAL_ALL implementation
                    return search_sequential_all_impl(query_ptr, K, query_result_tags_ptr, res, query_result_dists_ptr, backend_stats);
                }
                
                if (is_hit) {
                    return true;
                }
                handle_backend_miss(query_ptr, K, query_result_tags_ptr, query_result_dists_ptr, backend_stats);
                return false;
            }

            size_t get_number_of_vectors_in_memory_index() const {
                size_t total = 0;
                for (const auto& index : memory_indices) {
                    total += index->get_number_of_active_vectors();
                }
                return total;
            }

            size_t get_number_of_max_points_in_memory_index() const {
                return memory_index_max_points_per_index * number_of_mini_indexes; // Total capacity across all indices
            }

            uint64_t get_point_evictions() const {
                return point_evictions.load();
            }

            uint64_t get_region_invalidations() const {
                return region_invalidations.load();
            }

            // New methods for LRU eviction status
            bool is_eviction_in_progress() const {
                return eviction_in_progress.load();
            }

            size_t get_index_vector_count(size_t index_id) const {
                if (index_id < number_of_mini_indexes) {
                    return memory_indices[index_id]->get_number_of_active_vectors();
                }
                return 0;
            }

            size_t get_active_index_id() const {
                return active_insert_index_id.load();
            }

            size_t get_number_of_mini_indexes() const {
                return number_of_mini_indexes;
            }

            bool is_parallel_search_enabled() const {
                return search_mini_indexes_in_parallel;
            }

            size_t get_max_search_threads() const {
                return max_search_threads;
            }

            size_t get_number_of_active_pca_regions() const {
                if (use_regional_theta && pca_utils) {
                    return pca_utils->get_number_of_active_regions();
                }
                return 0;
            }

            // LRU management methods
            std::vector<size_t> get_lru_order() const {
                return lru_cache->get_all_tags();
            }

            std::vector<size_t> get_most_recently_used_indices(size_t n) const {
                return lru_cache->get_mru_tags(n);
            }

            std::vector<size_t> get_least_recently_used_indices(size_t n) const {
                return lru_cache->get_lru_tags(n);
            }

            size_t get_lru_cache_size() const {
                return lru_cache->size();
            }



            void set_search_strategy(SearchStrategy strategy) {
                search_strategy = strategy;
            }

            SearchStrategy get_search_strategy() const {
                return search_strategy;
            }
            
            // Hit ratio monitoring methods - direct approach using HitRateTracker (already thread-safe)
            // HitRateTracker uses its own internal mutex, so it's safe to call from multiple threads
            // But we need to protect the pointer itself from being deleted while in use
            void update_hit_history(bool was_hit) {
                std::lock_guard<std::mutex> lock(hit_rate_tracker_mutex);
                if (!hit_rate_tracker) {
                    return;  // Tracker not initialized, skip silently
                }
                
                // Direct call - HitRateTracker is already thread-safe with its own mutex
                // Lock protects the pointer, HitRateTracker's mutex protects its data
                try {
                    hit_rate_tracker->record_request(was_hit);
                } catch (...) {
                    // Ignore any exceptions - don't let hit tracking break the search
                }
            }
            
            double get_current_hit_ratio() const {
                std::lock_guard<std::mutex> lock(hit_rate_tracker_mutex);
                if (!hit_rate_tracker) {
                    return 1.0;  // Assume good hit ratio if tracker not initialized
                }
                
                try {
                    return hit_rate_tracker->get_hit_rate();
                } catch (...) {
                    // If tracker access fails, assume good hit ratio (conservative)
                    return 1.0;
                }
            }
            
            bool should_use_adaptive_strategy() const {
                // Read atomic flag without lock (lock-free read)
                if (!use_adaptive_strategy.load(std::memory_order_acquire)) {
                    return false;
                }
                
                // Get local copy of tracker pointer while holding lock
                HitRateTracker* tracker_ptr = nullptr;
                {
                    std::lock_guard<std::mutex> lock(hit_rate_tracker_mutex);
                    if (!hit_rate_tracker) {
                        // Tracker not initialized, don't switch to adaptive
                        return false;
                    }
                    tracker_ptr = hit_rate_tracker.get();  // Get raw pointer while mutex is held
                }
                // Mutex released here, but tracker_ptr is valid as long as we're not recreating
                
                try {
                    // Use local pointer - HitRateTracker is thread-safe internally
                    // Note: There's a small window where tracker could be recreated, but that's rare
                    // and HitRateTracker's internal mutex protects its data
                    double current_hit_ratio = tracker_ptr->get_hit_rate();
                    double threshold = hit_ratio_threshold.load(std::memory_order_acquire);
                    
                    // Sanity check on threshold
                    if (threshold <= 0.0 || threshold > 1.0) {
                        return false;  // Invalid threshold, don't switch
                    }
                    
                    // Sanity check on hit ratio (should be between 0 and 1)
                    if (current_hit_ratio < 0.0 || current_hit_ratio > 1.0) {
                        return false;  // Invalid hit ratio, don't switch
                    }
                    
                    return current_hit_ratio < threshold;
                } catch (...) {
                    // If any error occurs, don't switch to adaptive (conservative)
                    return false;
                }
            }
            
            // Configuration methods for adaptive strategy
            void set_hit_ratio_window_size(size_t new_window_size) {
                std::lock_guard<std::mutex> lock(hit_rate_tracker_mutex);
                double threshold = hit_ratio_threshold.load(std::memory_order_acquire);
                // Create new tracker with new window size (protected by mutex)
                hit_rate_tracker = std::make_unique<HitRateTracker>(new_window_size, threshold);
            }
            
            void set_hit_ratio_threshold(double threshold) {
                hit_ratio_threshold.store(threshold, std::memory_order_release);
                std::lock_guard<std::mutex> lock(hit_rate_tracker_mutex);
                // Update tracker if it exists (recreate with new threshold, protected by mutex)
                if (hit_rate_tracker) {
                    // Save window size before destroying tracker
                    size_t window_size = hit_rate_tracker->get_window_size();
                    // Destroy old tracker and create new one (protected by mutex)
                    hit_rate_tracker.reset();
                    hit_rate_tracker = std::make_unique<HitRateTracker>(window_size, threshold);
                }
            }
            
            // Writes do not evict to make room. θ ← α·θ on v's 16-D cell.
            void insert(TagT id, const T* vector) {
                {
                    std::lock_guard<std::mutex> lock(deleted_mutex);
                    deleted_ids.erase(id);
                }
                if (backend && backend->supports_updates()) {
                    backend->insert(id, vector);
                }
                try_admit_data_vector(id, vector);
                stale_write_regions(vector);
            }

            void remove(TagT id, const T* vector = nullptr) {
                std::vector<T> owned;
                const T* vec = vector;
                if (vec == nullptr) {
                    owned = fetch_backend_vector(id);
                    if (!owned.empty()) {
                        vec = owned.data();
                    }
                }
                stale_write_regions(vec);
                {
                    std::lock_guard<std::mutex> lock(deleted_mutex);
                    deleted_ids.insert(id);
                }
                const TagT mem_tag = static_cast<TagT>(1) + id;
                forget_admitted_tag(mem_tag);
                for (auto& index : memory_indices) {
                    if (index) {
                        index->lazy_delete(mem_tag);
                    }
                }
                if (backend && backend->supports_updates()) {
                    backend->remove(id);
                }
            }

            void set_miss_admit_delta(uint32_t delta) { miss_admit_delta = delta; }
            uint32_t get_miss_admit_delta() const { return miss_admit_delta; }
            void set_write_theta_discount(double alpha) { write_theta_discount = alpha; }
            double get_write_theta_discount() const { return write_theta_discount; }
            void set_write_l1_radius(size_t r) {
                write_l1_radius = r;
                if (pca_utils) {
                    pca_utils->set_write_l1_radius(r);
                }
            }
            size_t get_write_l1_radius() const {
                return pca_utils ? pca_utils->get_write_l1_radius() : write_l1_radius;
            }

            void wait_for_pending_inserts() {
                if (insert_pool) {
                    insert_pool->wait_idle();
                }
            }

            bool backend_supports_updates() const {
                return backend && backend->supports_updates();
            }

            void enable_adaptive_strategy(bool enable) {
                use_adaptive_strategy.store(enable, std::memory_order_release);
                
                std::lock_guard<std::mutex> lock(hit_rate_tracker_mutex);
                if (enable && !hit_rate_tracker) {
                    // Initialize tracker with default values (will be configured by set_hit_ratio_window_size/threshold)
                    double threshold = hit_ratio_threshold.load(std::memory_order_acquire);
                    hit_rate_tracker = std::make_unique<HitRateTracker>(100, threshold);  // Default window size 100
                }
                // Don't clear tracker when disabled - keep it for potential re-enabling
            }
            


    };
}