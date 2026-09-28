// Mixed search/insert/delete workload on Aker + pgvector.
// Same CLI / metrics as qvcache_update_pgvector.

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <memory>
#include <random>
#include <string>
#include <type_traits>
#include <unordered_set>
#include <vector>

#include <boost/program_options.hpp>
#include <omp.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/spdlog.h>

#include "ak_anns_cache_c_wrapper.h"
#include "aker_c_abi_helpers.h"
#include "diskann/distance.h"
#include "diskann/utils.h"
#include "greator/utils.h"
#include "pgvector_backend.h"
#include "update/update_common.h"

namespace po = boost::program_options;

static void aker_noop_transform(uint64_t, uint8_t*, size_t, uint64_t, uint64_t) {}

template <typename T>
bool (*aker_transform_for_type())(void*, size_t, size_t, void*, uint8_t*) {
    if constexpr (std::is_same_v<T, float>)
        return aker_transform_float;
    else if constexpr (std::is_same_v<T, int8_t>)
        return aker_transform_int8;
    else
        return aker_transform_uint8;
}

template <typename T>
float (*aker_distance_for_type(diskann::Metric metric))(uint8_t*, uint8_t*, size_t) {
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
void aker_apply_insert(
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
        slot, static_cast<size_t>(dim), vector_in_bytes, aker_transform_for_type<T>());
    akerInsertWriteLogEntry(cache, view, aker_distance_for_type<T>(metric), aker_noop_transform);
    if (process_log) {
        akerProcessWriteLogEntries(cache, aker_distance_for_type<T>(metric), aker_noop_transform);
    }
    akerDestroyVectorView(view);
    akerDestroyVectorSlot(slot);
}

template <typename T>
void aker_apply_delete(
    anns_cache_c_wrapper_t* cache,
    qvcache::PgVectorBackend<T>& backend,
    uint32_t id
) {
    backend.remove(id);
    akerMarkVectorDeleted(cache, static_cast<uint64_t>(id) + 1);
}

template <typename T>
bool aker_search(
    anns_cache_c_wrapper_t* cache,
    qvcache::PgVectorBackend<T>& backend,
    const T* query,
    uint32_t dim,
    uint32_t K,
    uint32_t slot_list_size,
    size_t vector_in_bytes,
    diskann::Metric metric,
    uint32_t* out_tags
) {
    std::vector<T> packed(dim);
    std::memcpy(packed.data(), query, dim * sizeof(T));
    uint64_t query_id = akerDefaultHash(reinterpret_cast<char*>(packed.data()), vector_in_bytes);
    if (query_id == 0) query_id = 1;

    char* query_slot = akerCreateVectorSlot(
        query_id, vector_in_bytes, reinterpret_cast<char*>(packed.data()), 0, 0, 0.0f);
    char* query_view = akerCreateVectorView(
        query_slot, static_cast<size_t>(dim), vector_in_bytes, aker_transform_for_type<T>());

    bool similar = false, invalid = false;
    char* entry = akerGetCacheEntry(cache, query_view, &similar, &invalid, aker_distance_for_type<T>(metric));
    const bool hit = (entry != nullptr && !invalid);
    if (hit) {
        for (uint32_t j = 0; j < K; ++j) {
            char* slot = akerGetResultVectorSlotAt(entry, static_cast<int>(j));
            out_tags[j] = slot ? static_cast<uint32_t>(akerGetVectorIdFromVectorSlot(slot)) : 0;
        }
        akerDestroyCacheEntry(entry);
    } else {
        if (entry) akerDestroyCacheEntry(entry);
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

template <typename T>
int run_update_bench(
    const std::string& data_path,
    const std::string& query_path,
    const std::string& aker_config_path,
    const std::string& table_name,
    const std::string& db_host, int db_port,
    const std::string& db_name, const std::string& db_user, const std::string& db_password,
    int hnsw_ef_search,
    uint32_t K, uint32_t aker_top_delta, size_t aker_pool_size,
    uint32_t search_threads, diskann::Metric metric,
    size_t n_ops, double insert_ratio, double delete_ratio,
    int report_interval, uint32_t seed, bool check_visibility, bool process_log
) {
    aker_pin_openmp(static_cast<int>(search_threads));

    size_t nq = 0, qdim = 0, qaligned = 0;
    T* queries = nullptr;
    greator::load_aligned_bin<T>(query_path, queries, nq, qdim, qaligned);
    const uint32_t dim = static_cast<uint32_t>(qdim);
    const size_t vector_in_bytes = static_cast<size_t>(dim) * sizeof(T);
    const uint32_t slot_list_size = K + aker_top_delta;

    size_t n_base = 0, bdim = 0, baligned = 0;
    T* base = nullptr;
    greator::load_aligned_bin<T>(data_path, base, n_base, bdim, baligned);

    anns_cache_parameter_c_t parameter{};
    akerImportAnnsCacheConfig(const_cast<char*>(aker_config_path.c_str()), &parameter);
    parameter.vector_format.dimension = dim;
    parameter.vector_format.vector_in_bytes = vector_in_bytes;
    parameter.capacity.in_topk = K;
    parameter.capacity.top_delta = aker_top_delta;
    if (aker_pool_size == 0) {
        aker_pool_size = static_cast<size_t>(slot_list_size + 1) * 20000;
    }
    parameter.capacity.pool_size = aker_pool_size;
    parameter.distance_metric = (metric == diskann::Metric::INNER_PRODUCT) ? 1 : 0;

    anns_cache_c_wrapper_t* cache = akerCreateAnnsCache(parameter);
    if (cache == nullptr) {
        std::cerr << "Failed to create Aker cache\n";
        return 1;
    }

    qvcache::PgVectorBackend<T> backend(
        table_name, data_path, db_host, db_port, db_name, db_user, db_password,
        (metric == diskann::Metric::COSINE) ? "cosine" : "l2", hnsw_ef_search);

    std::vector<uint32_t> live(n_base);
    for (size_t i = 0; i < n_base; ++i) live[i] = static_cast<uint32_t>(i);
    std::vector<uint32_t> retired;
    uint32_t next_id = static_cast<uint32_t>(n_base);

    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> u01(0.0, 1.0);
    qvcache_update::UpdateMetrics cum, interval;
    const uint32_t INVALID = std::numeric_limits<uint32_t>::max();
    auto t0 = std::chrono::high_resolution_clock::now();

    spdlog::info("{{\"event\": \"update_start\", \"cache\": \"aker\", \"n_ops\": {}, "
                 "\"insert_ratio\": {}, \"delete_ratio\": {}, \"K\": {}}}",
                 n_ops, insert_ratio, delete_ratio, K);

    for (size_t i = 0; i < n_ops; ++i) {
        const int op = qvcache_update::pick_op(u01(rng), insert_ratio, delete_ratio);
        if (op == 2 && live.size() > static_cast<size_t>(K) + 1) {
            std::uniform_int_distribution<size_t> pick(0, live.size() - 1);
            const size_t idx = pick(rng);
            const uint32_t id = live[idx];
            live[idx] = live.back();
            live.pop_back();
            retired.push_back(id);
            auto s = std::chrono::high_resolution_clock::now();
            aker_apply_delete<T>(cache, backend, id);
            auto e = std::chrono::high_resolution_clock::now();
            const double ms = std::chrono::duration<double, std::milli>(e - s).count();
            interval.n_delete++;
            interval.delete_ms += ms;
            cum.n_delete++;
            cum.delete_ms += ms;
        } else if (op == 1) {
            uint32_t id;
            const T* vec = nullptr;
            if (!retired.empty()) {
                id = retired.back();
                retired.pop_back();
                live.push_back(id);
                if (id < n_base) vec = base + static_cast<size_t>(id) * baligned;
            } else {
                id = next_id++;
                live.push_back(id);
                vec = queries + (i % nq) * qaligned;
            }
            std::vector<T> owned;
            if (vec == nullptr) {
                owned.assign(dim, static_cast<T>(0));
                vec = owned.data();
            }
            auto s = std::chrono::high_resolution_clock::now();
            aker_apply_insert<T>(cache, backend, id, vec, dim, vector_in_bytes, metric, process_log);
            auto e = std::chrono::high_resolution_clock::now();
            const double ms = std::chrono::duration<double, std::milli>(e - s).count();
            interval.n_insert++;
            interval.insert_ms += ms;
            cum.n_insert++;
            cum.insert_ms += ms;
            if (check_visibility) {
                std::vector<uint32_t> tags(K, INVALID);
                aker_search<T>(cache, backend, vec, dim, K, slot_list_size, vector_in_bytes, metric, tags.data());
                interval.visibility_checks++;
                cum.visibility_checks++;
                for (uint32_t j = 0; j < K; ++j) {
                    if (tags[j] == id + 1) {
                        interval.visibility_hits++;
                        cum.visibility_hits++;
                        break;
                    }
                }
            }
        } else {
            std::vector<uint32_t> tags(K, INVALID);
            auto s = std::chrono::high_resolution_clock::now();
            const bool hit = aker_search<T>(
                cache, backend, queries + (i % nq) * qaligned, dim, K, slot_list_size,
                vector_in_bytes, metric, tags.data());
            auto e = std::chrono::high_resolution_clock::now();
            const double ms = std::chrono::duration<double, std::milli>(e - s).count();
            interval.n_search++;
            interval.search_ms += ms;
            cum.n_search++;
            cum.search_ms += ms;
            if (hit) {
                interval.search_hits++;
                interval.hit_ms += ms;
                cum.search_hits++;
                cum.hit_ms += ms;
                std::unordered_set<uint32_t> live_set(live.begin(), live.end());
                for (uint32_t j = 0; j < K; ++j) {
                    if (tags[j] != INVALID && live_set.count(tags[j] - 1) == 0) {
                        interval.stale_hits++;
                        cum.stale_hits++;
                        break;
                    }
                }
            }
        }

        if (report_interval > 0 && ((i + 1) % static_cast<size_t>(report_interval) == 0 || i + 1 == n_ops)) {
            qvcache_update::log_interval("aker", i + 1, n_ops, interval, cum);
            interval = {};
        }
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    spdlog::info("{{\"event\": \"update_end\", \"cache\": \"aker\", \"elapsed_ms\": {}}}",
                 std::chrono::duration<double, std::milli>(t1 - t0).count());

    akerDestroyAnnsCache(cache);
    greator::aligned_free(base);
    greator::aligned_free(queries);
    return 0;
}

int main(int argc, char** argv) {
    std::string data_type, data_path, query_path;
    std::string aker_config_path = "Aker/bootstrap/aker-standard.ini";
    std::string table_name = "spacev_1m";
    std::string db_host = "postgres", db_name = "postgres", db_user = "postgres", db_password = "postgres";
    int db_port = 5432, hnsw_ef_search = 200;
    uint32_t K = 10, aker_top_delta = 5, search_threads = 1;
    size_t aker_pool_size = 200000;
    std::string metric_str = "l2";
    size_t n_ops = 10000;
    double insert_ratio = 0.05, delete_ratio = 0.05;
    int report_interval = 100;
    uint32_t seed = 1;
    bool check_visibility = true, process_log = true;

    po::options_description desc("Aker update workload (pgvector)");
    desc.add_options()
        ("help,h", "help")
        ("data_type", po::value<std::string>(&data_type)->required(), "float|int8|uint8")
        ("data_path", po::value<std::string>(&data_path)->required(), "base bin")
        ("query_path", po::value<std::string>(&query_path)->required(), "query bin")
        ("aker_config", po::value<std::string>(&aker_config_path)->default_value(aker_config_path))
        ("table_name", po::value<std::string>(&table_name)->default_value(table_name))
        ("db_host", po::value<std::string>(&db_host)->default_value(db_host))
        ("db_port", po::value<int>(&db_port)->default_value(db_port))
        ("db_name", po::value<std::string>(&db_name)->default_value(db_name))
        ("db_user", po::value<std::string>(&db_user)->default_value(db_user))
        ("db_password", po::value<std::string>(&db_password)->default_value(db_password))
        ("hnsw_ef_search", po::value<int>(&hnsw_ef_search)->default_value(hnsw_ef_search))
        ("K", po::value<uint32_t>(&K)->default_value(K))
        ("aker_top_delta", po::value<uint32_t>(&aker_top_delta)->default_value(aker_top_delta))
        ("aker_pool_size", po::value<size_t>(&aker_pool_size)->default_value(aker_pool_size))
        ("search_threads", po::value<uint32_t>(&search_threads)->default_value(search_threads))
        ("metric", po::value<std::string>(&metric_str)->default_value(metric_str))
        ("n_ops", po::value<size_t>(&n_ops)->default_value(n_ops))
        ("insert_ratio", po::value<double>(&insert_ratio)->default_value(insert_ratio))
        ("delete_ratio", po::value<double>(&delete_ratio)->default_value(delete_ratio))
        ("report_interval", po::value<int>(&report_interval)->default_value(report_interval))
        ("seed", po::value<uint32_t>(&seed)->default_value(seed))
        ("check_visibility", po::value<bool>(&check_visibility)->default_value(check_visibility))
        ("aker_process_log", po::value<bool>(&process_log)->default_value(process_log),
         "Run Aker slow-path after each insert");

    po::variables_map vm;
    try {
        po::store(po::parse_command_line(argc, argv, desc), vm);
        if (vm.count("help")) {
            std::cout << desc << "\n";
            return 0;
        }
        po::notify(vm);
    } catch (const std::exception& ex) {
        std::cerr << ex.what() << "\n" << desc << "\n";
        return 1;
    }

    diskann::Metric metric = diskann::Metric::L2;
    if (metric_str == "cosine") metric = diskann::Metric::COSINE;
    else if (metric_str == "inner_product") metric = diskann::Metric::INNER_PRODUCT;

    auto logger = spdlog::stdout_color_mt("console");
    spdlog::set_pattern("%v");

    if (data_type == "float") {
        return run_update_bench<float>(
            data_path, query_path, aker_config_path, table_name, db_host, db_port, db_name, db_user, db_password,
            hnsw_ef_search, K, aker_top_delta, aker_pool_size, search_threads, metric,
            n_ops, insert_ratio, delete_ratio, report_interval, seed, check_visibility, process_log);
    }
    if (data_type == "int8") {
        return run_update_bench<int8_t>(
            data_path, query_path, aker_config_path, table_name, db_host, db_port, db_name, db_user, db_password,
            hnsw_ef_search, K, aker_top_delta, aker_pool_size, search_threads, metric,
            n_ops, insert_ratio, delete_ratio, report_interval, seed, check_visibility, process_log);
    }
    if (data_type == "uint8") {
        return run_update_bench<uint8_t>(
            data_path, query_path, aker_config_path, table_name, db_host, db_port, db_name, db_user, db_password,
            hnsw_ef_search, K, aker_top_delta, aker_pool_size, search_threads, metric,
            n_ops, insert_ratio, delete_ratio, report_interval, seed, check_visibility, process_log);
    }
    std::cerr << "Unsupported data_type: " << data_type << "\n";
    return 1;
}
