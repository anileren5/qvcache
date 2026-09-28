// Aker §5.4 refresh stress-test on QVCache + pgvector (SPACEV-1M, simZipf 0.99).
// Warmup is search-only. Re-search recall uses an in-memory exact top-k
// over base + inserts − deletes. Existing GT files are not touched.

#include <algorithm>
#include <chrono>
#include <cmath>
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
#include "update/refresh_common.h"

namespace po = boost::program_options;

template <typename T>
int run_refresh_bench(
    const std::string& data_path,
    const std::string& query_path,
    const std::string& live_gt_path,
    const std::string& warmup_gt_path,
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
    double insert_frac, double delete_rate,
    size_t n_warmup, size_t n_eval, int report_interval, uint32_t seed, int gt_threads,
    double write_theta_discount, size_t write_l1_radius
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
    cache.set_write_theta_discount(write_theta_discount);
    cache.set_write_l1_radius(write_l1_radius);

    T* base = nullptr;
    size_t n_base = 0, base_dim = 0, base_aligned = 0;
    diskann::load_aligned_bin<T>(data_path, base, n_base, base_dim, base_aligned);

    T* queries = nullptr;
    size_t nq = 0, qdim = 0, qaligned = 0;
    diskann::load_aligned_bin<T>(query_path, queries, nq, qdim, qaligned);

    if (n_warmup == 0 || n_warmup > nq) n_warmup = nq;
    if (n_eval == 0 || n_eval > nq) n_eval = nq;

    qvcache_refresh::LiveVectors<T> live;
    live.base = base;
    live.n_base = n_base;
    live.dim = base_dim;
    live.aligned = base_aligned;

    const uint32_t INVALID = std::numeric_limits<uint32_t>::max();
    const size_t n_insert = static_cast<size_t>(std::llround(insert_frac * static_cast<double>(n_base)));

    spdlog::info("{{\"event\": \"refresh_start\", \"cache\": \"qvcache\", "
                 "\"n_base\": {}, \"n_queries\": {}, \"n_warmup\": {}, \"n_eval\": {}, "
                 "\"insert_frac\": {}, \"n_insert\": {}, \"delete_rate\": {}, \"K\": {}, "
                 "\"memory_index_max_points\": {}, \"write_theta_discount\": {}, "
                 "\"write_l1_radius\": {}}}",
                 n_base, nq, n_warmup, n_eval, insert_frac, n_insert, delete_rate, K,
                 memory_index_max_points, cache.get_write_theta_discount(),
                 cache.get_write_l1_radius());

    auto snapshot = [&]() {
        qvcache_refresh::CacheSnapshot snap;
        snap.memory_active_vectors = cache.get_number_of_vectors_in_memory_index();
        snap.memory_max_points = cache.get_number_of_max_points_in_memory_index();
        snap.pca_active_regions = cache.get_number_of_active_pca_regions();
        snap.region_directory_size = cache.get_region_directory_size();
        snap.point_evictions = cache.get_point_evictions();
        snap.region_invalidations = cache.get_region_invalidations();
        const size_t nidx = cache.get_number_of_mini_indexes();
        for (size_t idx = 0; idx < nidx; ++idx) {
            snap.index_vectors[idx] = cache.get_index_vector_count(idx);
        }
        return snap;
    };

    auto run_searches = [&](size_t n, const char* phase, const qvcache_refresh::LiveGroundtruth* gt) {
        qvcache_refresh::PhaseMetrics m;
        const int interval = report_interval;
        size_t interval_begin = 0;
        for (size_t i = 0; i < n; ++i) {
            std::vector<uint32_t> tags(K, INVALID);
            std::vector<float> dists(K, 0.0f);
            std::vector<T*> res;
            auto s = std::chrono::high_resolution_clock::now();
            const bool hit = cache.search(queries + i * qaligned, K, tags.data(), res, dists.data(), nullptr);
            auto e = std::chrono::high_resolution_clock::now();
            const double ms = std::chrono::duration<double, std::milli>(e - s).count();
            const double miss_penalty = hit ? 0.0 : cache.last_cache_lookup_ms();
            double rec = -1.0;
            if (gt != nullptr) {
                rec = qvcache_refresh::recall_of(gt->topk(i), tags.data(), K, INVALID);
            }
            bool stale = false;
            if (hit) {
                for (uint32_t j = 0; j < K; ++j) {
                    if (tags[j] != INVALID && !live.is_live(tags[j] - 1)) {
                        stale = true;
                        break;
                    }
                }
            }
            m.add(ms, hit, rec, miss_penalty, stale);
            if (interval > 0 && ((i + 1) % static_cast<size_t>(interval) == 0 || i + 1 == n)) {
                qvcache_refresh::log_search_progress(
                    "qvcache", phase, i + 1, n, m, gt != nullptr, K, snapshot(),
                    interval_begin, search_threads);
                interval_begin = i + 1;
            }
        }
        qvcache_refresh::log_search_phase("qvcache", phase, m, snapshot(), gt != nullptr, K, search_threads);
        return m;
    };

    qvcache_refresh::LiveGroundtruth warmup_gt;
    const qvcache_refresh::LiveGroundtruth* warmup_gt_ptr = nullptr;
    if (!warmup_gt_path.empty()) {
        warmup_gt = qvcache_refresh::require_search_only_gt(warmup_gt_path, n_warmup, K);
        warmup_gt_ptr = &warmup_gt;
    }
    const auto warmup = run_searches(n_warmup, "warmup", warmup_gt_ptr);
    cache.wait_for_pending_inserts();

    qvcache_refresh::synthesize_extra(live, n_insert, seed);
    auto t_ins0 = std::chrono::high_resolution_clock::now();
    for (uint32_t id : live.extra_ids) {
        cache.insert(id, live.ptr(id));
    }
    cache.wait_for_pending_inserts();
    auto t_ins1 = std::chrono::high_resolution_clock::now();
    const double insert_ms = std::chrono::duration<double, std::milli>(t_ins1 - t_ins0).count();
    qvcache_refresh::log_mutate_phase("qvcache", "insert", live.extra_ids.size(), insert_ms,
                                     -1.0, cache.get_region_invalidations());

    std::vector<uint32_t> candidates;
    candidates.reserve(n_base + live.extra_ids.size());
    for (uint32_t id = 0; id < n_base; ++id) candidates.push_back(id);
    for (uint32_t id : live.extra_ids) candidates.push_back(id);
    std::mt19937 rng(seed + 1);
    std::shuffle(candidates.begin(), candidates.end(), rng);
    const size_t n_delete = static_cast<size_t>(
        std::llround(delete_rate * static_cast<double>(candidates.size())));
    auto t_del0 = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < n_delete; ++i) {
        const uint32_t id = candidates[i];
        const T* vec = live.ptr(id);
        cache.remove(id, vec);
        live.mark_deleted(id);
    }
    cache.wait_for_pending_inserts();
    auto t_del1 = std::chrono::high_resolution_clock::now();
    const double delete_ms = std::chrono::duration<double, std::milli>(t_del1 - t_del0).count();
    qvcache_refresh::log_mutate_phase("qvcache", "delete", n_delete, delete_ms, delete_rate,
                                     cache.get_region_invalidations());

    omp_set_num_threads(gt_threads);
    qvcache_refresh::LiveGroundtruth gt;
    const bool gt_reused = qvcache_refresh::load_live_groundtruth(live_gt_path, n_eval, K, gt);
    auto t_gt0 = std::chrono::high_resolution_clock::now();
    if (!gt_reused) {
        gt = qvcache_refresh::compute_exact_gt(queries, qaligned, n_eval, live, K);
        qvcache_refresh::save_live_groundtruth(live_gt_path, gt);
    }
    auto t_gt1 = std::chrono::high_resolution_clock::now();
    omp_set_num_threads(static_cast<int>(search_threads));
    const double gt_ms = std::chrono::duration<double, std::milli>(t_gt1 - t_gt0).count();
    const size_t n_live = n_base + live.extra_ids.size() - live.deleted.size();
    qvcache_refresh::log_gt_phase("qvcache", n_eval, n_live, gt_ms, gt_reused);

    const auto eval = run_searches(n_eval, "eval", &gt);

    spdlog::info("{{\"event\": \"refresh_end\", \"cache\": \"qvcache\", "
                 "\"warmup_elapsed_ms\": {}, \"insert_elapsed_ms\": {}, "
                 "\"delete_elapsed_ms\": {}, \"gt_elapsed_ms\": {}, \"eval_elapsed_ms\": {}, "
                 "\"warmup_hit_ratio\": {}, \"warmup_recall_all\": {}, "
                 "\"warmup_recall_hits\": {}, \"warmup_recall_misses\": {}, "
                 "\"eval_hit_ratio\": {}, \"eval_miss_ratio\": {}, "
                 "\"eval_recall_all\": {}, \"eval_recall_hits\": {}, \"eval_recall_misses\": {}, "
                 "\"eval_qps\": {}, \"memory_active_vectors\": {}, \"region_directory_size\": {}, "
                 "\"region_invalidations\": {}}}",
                 warmup.search_ms, insert_ms, delete_ms, gt_ms, eval.search_ms,
                 warmup.hit_ratio(), warmup.recall_all(), warmup.recall_hits(),
                 warmup.recall_misses(), eval.hit_ratio(), eval.miss_ratio(),
                 eval.recall_all(), eval.recall_hits(), eval.recall_misses(),
                 eval.search_ms > 0.0 ? static_cast<double>(eval.n_search) / (eval.search_ms / 1000.0) : 0.0,
                 cache.get_number_of_vectors_in_memory_index(),
                 cache.get_region_directory_size(),
                 cache.get_region_invalidations());

    diskann::aligned_free(base);
    diskann::aligned_free(queries);
    return 0;
}

int main(int argc, char** argv) {
    std::string data_type, data_path, query_path, live_gt_path, warmup_gt_path, disk_index_prefix;
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
    double insert_frac = 0.05, delete_rate = 0.05;
    size_t n_warmup = 0, n_eval = 0;
    int report_interval = 100, gt_threads = 8;
    uint32_t seed = 1;
    double write_theta_discount = 0.80;
    size_t write_l1_radius = 1;

    po::options_description desc("QVCache Aker-style refresh stress-test (pgvector)");
    desc.add_options()
        ("help,h", "help")
        ("data_type", po::value<std::string>(&data_type)->required(), "float|int8|uint8")
        ("data_path", po::value<std::string>(&data_path)->required())
        ("query_path", po::value<std::string>(&query_path)->required())
        ("live_gt_path", po::value<std::string>(&live_gt_path)->default_value(""),
         "Sidecar truthset; reused if present, else computed. Does not overwrite search-only GT")
        ("warmup_gt_path", po::value<std::string>(&warmup_gt_path)->default_value(""),
         "Existing search-only truthset for warmup recall. Loaded only, never computed")
        ("disk_index_prefix", po::value<std::string>(&disk_index_prefix)->required())
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
        ("insert_frac", po::value<double>(&insert_frac)->default_value(insert_frac),
         "Extra vectors to insert after warmup (paper: 0.05 of |P|)")
        ("delete_rate", po::value<double>(&delete_rate)->default_value(delete_rate),
         "Fraction of the live set deleted after inserts (paper: 0.05, 0.10, 0.25)")
        ("n_warmup", po::value<size_t>(&n_warmup)->default_value(n_warmup), "0 = all queries")
        ("n_eval", po::value<size_t>(&n_eval)->default_value(n_eval), "0 = all queries")
        ("report_interval", po::value<int>(&report_interval)->default_value(report_interval))
        ("gt_threads", po::value<int>(&gt_threads)->default_value(gt_threads))
        ("seed", po::value<uint32_t>(&seed)->default_value(seed))
        ("write_theta_discount", po::value<double>(&write_theta_discount)->default_value(write_theta_discount),
         "θ ← this · θ on the L1 neighborhood (e.g. 0.80). 1.0 = no penalty")
        ("write_l1_radius", po::value<size_t>(&write_l1_radius)->default_value(write_l1_radius),
         "Manhattan radius on the 16-D region code. 0 = only v's cell; 1 = self + 32 neighbors");

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
    if (gt_threads < 1) gt_threads = 1;

    if (data_type == "float") {
        return run_refresh_bench<float>(
            data_path, query_path, live_gt_path, warmup_gt_path, disk_index_prefix, table_name,
            db_host, db_port, db_name, db_user, db_password, hnsw_ef_search,
            R, memory_L, K, B, M, alpha, build_threads, search_threads, beamwidth,
            p, deviation_factor, memory_index_max_points, use_regional_theta,
            pca_dim, buckets_per_dim, max_regions, n_async_insert_threads,
            lazy_theta_updates, number_of_mini_indexes, search_strategy, metric,
            insert_frac, delete_rate, n_warmup, n_eval, report_interval, seed, gt_threads,
            write_theta_discount, write_l1_radius);
    }
    if (data_type == "int8") {
        return run_refresh_bench<int8_t>(
            data_path, query_path, live_gt_path, warmup_gt_path, disk_index_prefix, table_name,
            db_host, db_port, db_name, db_user, db_password, hnsw_ef_search,
            R, memory_L, K, B, M, alpha, build_threads, search_threads, beamwidth,
            p, deviation_factor, memory_index_max_points, use_regional_theta,
            pca_dim, buckets_per_dim, max_regions, n_async_insert_threads,
            lazy_theta_updates, number_of_mini_indexes, search_strategy, metric,
            insert_frac, delete_rate, n_warmup, n_eval, report_interval, seed, gt_threads,
            write_theta_discount, write_l1_radius);
    }
    if (data_type == "uint8") {
        return run_refresh_bench<uint8_t>(
            data_path, query_path, live_gt_path, warmup_gt_path, disk_index_prefix, table_name,
            db_host, db_port, db_name, db_user, db_password, hnsw_ef_search,
            R, memory_L, K, B, M, alpha, build_threads, search_threads, beamwidth,
            p, deviation_factor, memory_index_max_points, use_regional_theta,
            pca_dim, buckets_per_dim, max_regions, n_async_insert_threads,
            lazy_theta_updates, number_of_mini_indexes, search_strategy, metric,
            insert_frac, delete_rate, n_warmup, n_eval, report_interval, seed, gt_threads,
            write_theta_discount, write_l1_radius);
    }
    std::cerr << "Unsupported data_type: " << data_type << "\n";
    return 1;
}
