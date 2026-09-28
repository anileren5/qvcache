// Mixed search/insert/delete workload on QVCache + pgvector.
// Same CLI shape as aker_update_pgvector so the two can be compared.

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <memory>
#include <random>
#include <string>
#include <unordered_set>
#include <vector>

#include <boost/program_options.hpp>
#include <omp.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/spdlog.h>

#include "diskann/distance.h"
#include "diskann/utils.h"
#include "pgvector_backend.h"
#include "qvcache/qvcache.h"
#include "update/update_common.h"

namespace po = boost::program_options;

template <typename T>
int run_update_bench(
    const std::string& data_path,
    const std::string& query_path,
    const std::string& disk_index_prefix,
    const std::string& table_name,
    const std::string& db_host, int db_port,
    const std::string& db_name, const std::string& db_user, const std::string& db_password,
    int hnsw_ef_search,
    uint32_t R, uint32_t memory_L, uint32_t K, uint32_t B, uint32_t M,
    float alpha, uint32_t build_threads, uint32_t search_threads,
    uint32_t beamwidth, double p, double deviation_factor,
    size_t memory_index_max_points, bool use_regional_theta,
    uint32_t pca_dim, uint32_t buckets_per_dim, size_t max_regions,
    uint32_t n_async_insert_threads, bool lazy_theta_updates,
    size_t number_of_mini_indexes, const std::string& search_strategy,
    diskann::Metric metric,
    size_t n_ops, double insert_ratio, double delete_ratio,
    int report_interval, uint32_t seed, bool check_visibility
) {
    omp_set_num_threads(static_cast<int>(search_threads));

    auto backend = std::make_unique<qvcache::PgVectorBackend<T>>(
        table_name, data_path, db_host, db_port, db_name, db_user, db_password,
        (metric == diskann::Metric::COSINE) ? "cosine" : "l2", hnsw_ef_search);

    qvcache::QVCache<T> cache(
        data_path, disk_index_prefix,
        R, memory_L, B, M, alpha,
        build_threads, search_threads,
        false, p, deviation_factor, memory_index_max_points, beamwidth,
        use_regional_theta, pca_dim, buckets_per_dim, max_regions,
        n_async_insert_threads, lazy_theta_updates, number_of_mini_indexes,
        false, 32, metric, std::move(backend), false, query_path);

    if (search_strategy == "SEQUENTIAL_ALL") {
        cache.set_search_strategy(qvcache::QVCache<T>::SearchStrategy::SEQUENTIAL_ALL);
    } else if (search_strategy == "SEQUENTIAL_LRU_ADAPTIVE") {
        cache.set_search_strategy(qvcache::QVCache<T>::SearchStrategy::SEQUENTIAL_LRU_ADAPTIVE);
        cache.enable_adaptive_strategy(true);
    } else {
        cache.set_search_strategy(qvcache::QVCache<T>::SearchStrategy::SEQUENTIAL_LRU_STOP_FIRST_HIT);
    }

    size_t n_base = 0, dim = 0;
    diskann::get_bin_metadata(data_path, n_base, dim);
    T* base = nullptr;
    size_t base_npts = 0, base_dim = 0, base_aligned = 0;
    diskann::load_aligned_bin<T>(data_path, base, base_npts, base_dim, base_aligned);

    size_t nq = 0, qdim = 0, qaligned = 0;
    T* queries = nullptr;
    diskann::load_aligned_bin<T>(query_path, queries, nq, qdim, qaligned);

    std::vector<uint32_t> live(n_base);
    for (size_t i = 0; i < n_base; ++i) live[i] = static_cast<uint32_t>(i);
    std::vector<uint32_t> retired;
    uint32_t next_id = static_cast<uint32_t>(n_base);

    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> u01(0.0, 1.0);

    qvcache_update::UpdateMetrics cum, interval;
    const uint32_t INVALID = std::numeric_limits<uint32_t>::max();
    auto t0 = std::chrono::high_resolution_clock::now();

    spdlog::info("{{\"event\": \"update_start\", \"cache\": \"qvcache\", \"n_ops\": {}, "
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
            const T* vec = (id < base_npts) ? (base + static_cast<size_t>(id) * base_aligned) : nullptr;
            cache.remove(id, vec);
            auto e = std::chrono::high_resolution_clock::now();
            const double ms = std::chrono::duration<double, std::milli>(e - s).count();
            interval.n_delete++;
            interval.delete_ms += ms;
            cum.n_delete++;
            cum.delete_ms += ms;
        } else if (op == 1) {
            uint32_t id;
            const T* vec = nullptr;
            std::vector<T> owned;
            if (!retired.empty()) {
                id = retired.back();
                retired.pop_back();
                live.push_back(id);
                if (id < base_npts) {
                    vec = base + static_cast<size_t>(id) * base_aligned;
                }
            } else {
                id = next_id++;
                live.push_back(id);
                const size_t qi = i % nq;
                vec = queries + qi * qaligned;
            }
            if (vec == nullptr) {
                owned.assign(dim, static_cast<T>(0));
                vec = owned.data();
            }
            auto s = std::chrono::high_resolution_clock::now();
            cache.insert(id, vec);
            auto e = std::chrono::high_resolution_clock::now();
            const double ms = std::chrono::duration<double, std::milli>(e - s).count();
            interval.n_insert++;
            interval.insert_ms += ms;
            cum.n_insert++;
            cum.insert_ms += ms;

            if (check_visibility) {
                std::vector<uint32_t> tags(K, INVALID);
                std::vector<float> dists(K, 0.0f);
                std::vector<T*> res;
                cache.search(vec, K, tags.data(), res, dists.data(), nullptr);
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
            const size_t qi = i % nq;
            std::vector<uint32_t> tags(K, INVALID);
            std::vector<float> dists(K, 0.0f);
            std::vector<T*> res;
            auto s = std::chrono::high_resolution_clock::now();
            const bool hit = cache.search(queries + qi * qaligned, K, tags.data(), res, dists.data(), nullptr);
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
                bool stale = false;
                std::unordered_set<uint32_t> live_set(live.begin(), live.end());
                for (uint32_t j = 0; j < K; ++j) {
                    if (tags[j] != INVALID && live_set.count(tags[j] - 1) == 0) {
                        stale = true;
                        break;
                    }
                }
                if (stale) {
                    interval.stale_hits++;
                    cum.stale_hits++;
                }
            }
        }

        if (report_interval > 0 && ((i + 1) % static_cast<size_t>(report_interval) == 0 || i + 1 == n_ops)) {
            qvcache_update::log_interval("qvcache", i + 1, n_ops, interval, cum,
                                        cache.get_number_of_vectors_in_memory_index());
            interval = {};
        }
    }

    cache.wait_for_pending_inserts();
    auto t1 = std::chrono::high_resolution_clock::now();
    spdlog::info("{{\"event\": \"update_end\", \"cache\": \"qvcache\", \"elapsed_ms\": {}, "
                 "\"memory_active_vectors\": {}, \"region_directory_size\": {}}}",
                 std::chrono::duration<double, std::milli>(t1 - t0).count(),
                 cache.get_number_of_vectors_in_memory_index(),
                 cache.get_region_directory_size());

    diskann::aligned_free(base);
    diskann::aligned_free(queries);
    return 0;
}

int main(int argc, char** argv) {
    std::string data_type, data_path, query_path, disk_index_prefix;
    std::string table_name = "spacev_1m";
    std::string db_host = "postgres", db_name = "postgres", db_user = "postgres", db_password = "postgres";
    int db_port = 5432, hnsw_ef_search = 200;
    uint32_t R = 64, memory_L = 16, K = 10, B = 8, M = 8;
    float alpha = 1.2f;
    uint32_t build_threads = 8, search_threads = 1, beamwidth = 2;
    double p = 0.9, deviation_factor = 0.0;
    size_t memory_index_max_points = 200000, max_regions = 1000000, number_of_mini_indexes = 4;
    bool use_regional_theta = true, lazy_theta_updates = true;
    uint32_t pca_dim = 16, buckets_per_dim = 8, n_async_insert_threads = 4;
    std::string search_strategy = "SEQUENTIAL_LRU_STOP_FIRST_HIT";
    std::string metric_str = "l2";
    size_t n_ops = 10000;
    double insert_ratio = 0.05, delete_ratio = 0.05;
    int report_interval = 100;
    uint32_t seed = 1;
    bool check_visibility = true;

    po::options_description desc("QVCache update workload (pgvector)");
    desc.add_options()
        ("help,h", "help")
        ("data_type", po::value<std::string>(&data_type)->required(), "float|int8|uint8")
        ("data_path", po::value<std::string>(&data_path)->required(), "base bin")
        ("query_path", po::value<std::string>(&query_path)->required(), "query bin")
        ("disk_index_prefix", po::value<std::string>(&disk_index_prefix)->required(), "PCA prefix")
        ("table_name", po::value<std::string>(&table_name)->default_value(table_name))
        ("db_host", po::value<std::string>(&db_host)->default_value(db_host))
        ("db_port", po::value<int>(&db_port)->default_value(db_port))
        ("db_name", po::value<std::string>(&db_name)->default_value(db_name))
        ("db_user", po::value<std::string>(&db_user)->default_value(db_user))
        ("db_password", po::value<std::string>(&db_password)->default_value(db_password))
        ("hnsw_ef_search", po::value<int>(&hnsw_ef_search)->default_value(hnsw_ef_search))
        ("R", po::value<uint32_t>(&R)->default_value(R))
        ("memory_L", po::value<uint32_t>(&memory_L)->default_value(memory_L))
        ("K", po::value<uint32_t>(&K)->default_value(K))
        ("B", po::value<uint32_t>(&B)->default_value(B))
        ("M", po::value<uint32_t>(&M)->default_value(M))
        ("alpha", po::value<float>(&alpha)->default_value(alpha))
        ("build_threads", po::value<uint32_t>(&build_threads)->default_value(build_threads))
        ("search_threads", po::value<uint32_t>(&search_threads)->default_value(search_threads))
        ("beamwidth", po::value<uint32_t>(&beamwidth)->default_value(beamwidth))
        ("p", po::value<double>(&p)->default_value(p))
        ("deviation_factor", po::value<double>(&deviation_factor)->default_value(deviation_factor))
        ("memory_index_max_points", po::value<size_t>(&memory_index_max_points)->default_value(memory_index_max_points))
        ("use_regional_theta", po::value<bool>(&use_regional_theta)->default_value(use_regional_theta))
        ("pca_dim", po::value<uint32_t>(&pca_dim)->default_value(pca_dim))
        ("buckets_per_dim", po::value<uint32_t>(&buckets_per_dim)->default_value(buckets_per_dim))
        ("max_regions", po::value<size_t>(&max_regions)->default_value(max_regions))
        ("n_async_insert_threads", po::value<uint32_t>(&n_async_insert_threads)->default_value(n_async_insert_threads))
        ("lazy_theta_updates", po::value<bool>(&lazy_theta_updates)->default_value(lazy_theta_updates))
        ("number_of_mini_indexes", po::value<size_t>(&number_of_mini_indexes)->default_value(number_of_mini_indexes))
        ("search_strategy", po::value<std::string>(&search_strategy)->default_value(search_strategy))
        ("metric", po::value<std::string>(&metric_str)->default_value(metric_str))
        ("n_ops", po::value<size_t>(&n_ops)->default_value(n_ops), "mixed operations")
        ("insert_ratio", po::value<double>(&insert_ratio)->default_value(insert_ratio))
        ("delete_ratio", po::value<double>(&delete_ratio)->default_value(delete_ratio))
        ("report_interval", po::value<int>(&report_interval)->default_value(report_interval))
        ("seed", po::value<uint32_t>(&seed)->default_value(seed))
        ("check_visibility", po::value<bool>(&check_visibility)->default_value(check_visibility));

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
            data_path, query_path, disk_index_prefix, table_name, db_host, db_port, db_name, db_user, db_password,
            hnsw_ef_search, R, memory_L, K, B, M, alpha, build_threads, search_threads, beamwidth, p, deviation_factor,
            memory_index_max_points, use_regional_theta, pca_dim, buckets_per_dim, max_regions,
            n_async_insert_threads, lazy_theta_updates, number_of_mini_indexes, search_strategy, metric,
            n_ops, insert_ratio, delete_ratio, report_interval, seed, check_visibility);
    }
    if (data_type == "int8") {
        return run_update_bench<int8_t>(
            data_path, query_path, disk_index_prefix, table_name, db_host, db_port, db_name, db_user, db_password,
            hnsw_ef_search, R, memory_L, K, B, M, alpha, build_threads, search_threads, beamwidth, p, deviation_factor,
            memory_index_max_points, use_regional_theta, pca_dim, buckets_per_dim, max_regions,
            n_async_insert_threads, lazy_theta_updates, number_of_mini_indexes, search_strategy, metric,
            n_ops, insert_ratio, delete_ratio, report_interval, seed, check_visibility);
    }
    if (data_type == "uint8") {
        return run_update_bench<uint8_t>(
            data_path, query_path, disk_index_prefix, table_name, db_host, db_port, db_name, db_user, db_password,
            hnsw_ef_search, R, memory_L, K, B, M, alpha, build_threads, search_threads, beamwidth, p, deviation_factor,
            memory_index_max_points, use_regional_theta, pca_dim, buckets_per_dim, max_regions,
            n_async_insert_threads, lazy_theta_updates, number_of_mini_indexes, search_strategy, metric,
            n_ops, insert_ratio, delete_ratio, report_interval, seed, check_visibility);
    }
    std::cerr << "Unsupported data_type: " << data_type << "\n";
    return 1;
}
