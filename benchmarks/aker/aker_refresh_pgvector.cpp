// Aker §5.4 refresh stress-test on Aker + pgvector (SPACEV-1M, simZipf 0.99).
// Warmup is search-only. Re-search recall uses an in-memory exact top-k
// over base + inserts − deletes (insert_frac=0 is delete-only). Existing
// GT files are not touched.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <random>
#include <string>
#include <unordered_set>
#include <vector>

#include <boost/program_options.hpp>
#include <omp.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/spdlog.h>

#include "aker_cache_ops.h"
#include "diskann/distance.h"
#include "diskann/utils.h"
#include "greator/utils.h"
#include "pgvector_backend.h"
#include "update/refresh_common.h"

namespace po = boost::program_options;

template <typename T>
int run_refresh_bench(
    const std::string& data_path,
    const std::string& query_path,
    const std::string& live_gt_path,
    const std::string& warmup_gt_path,
    const std::string& aker_config_path,
    const std::string& table_name,
    const std::string& db_host, int db_port,
    const std::string& db_name, const std::string& db_user, const std::string& db_password,
    int hnsw_ef_search,
    uint32_t K, uint32_t aker_top_delta, size_t aker_pool_size,
    uint32_t search_threads, diskann::Metric metric,
    double insert_frac, double delete_rate,
    size_t n_warmup, size_t n_eval, int report_interval, uint32_t seed,
    int gt_threads, bool process_log
) {
    aker_pin_openmp(static_cast<int>(search_threads));

    T* queries = nullptr;
    size_t nq = 0, qdim = 0, qaligned = 0;
    greator::load_aligned_bin<T>(query_path, queries, nq, qdim, qaligned);
    const uint32_t dim = static_cast<uint32_t>(qdim);
    const size_t vector_in_bytes = static_cast<size_t>(dim) * sizeof(T);
    const uint32_t slot_list_size = K + aker_top_delta;

    T* base = nullptr;
    size_t n_base = 0, bdim = 0, baligned = 0;
    greator::load_aligned_bin<T>(data_path, base, n_base, bdim, baligned);

    if (n_warmup == 0 || n_warmup > nq) n_warmup = nq;
    if (n_eval == 0 || n_eval > nq) n_eval = nq;

    anns_cache_parameter_c_t parameter{};
    akerImportAnnsCacheConfig(const_cast<char*>(aker_config_path.c_str()), &parameter);
    parameter.vector_format.dimension = dim;
    parameter.vector_format.vector_in_bytes = vector_in_bytes;
    parameter.capacity.in_topk = K;
    parameter.capacity.top_delta = aker_top_delta;
    if (aker_pool_size == 0) {
        aker_pool_size = static_cast<size_t>(n_base) / 100;
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

    qvcache_refresh::LiveVectors<T> live;
    live.base = base;
    live.n_base = n_base;
    live.dim = bdim;
    live.aligned = baligned;

    const uint32_t INVALID = std::numeric_limits<uint32_t>::max();
    const size_t n_insert = static_cast<size_t>(std::llround(insert_frac * static_cast<double>(n_base)));

    spdlog::info("{{\"event\": \"refresh_start\", \"cache\": \"aker\", "
                 "\"n_base\": {}, \"n_queries\": {}, \"n_warmup\": {}, \"n_eval\": {}, "
                 "\"insert_frac\": {}, \"n_insert\": {}, \"delete_only\": {}, \"delete_rate\": {}, \"K\": {}, "
                 "\"aker_pool_size\": {}, \"aker_top_delta\": {}}}",
                 n_base, nq, n_warmup, n_eval, insert_frac, n_insert, n_insert == 0, delete_rate, K,
                 aker_pool_size, aker_top_delta);

    auto snapshot = [&]() {
        qvcache_refresh::CacheSnapshot snap;
        snap.memory_max_points = aker_pool_size;
        return snap;
    };

    auto run_searches = [&](size_t n, const char* phase, const qvcache_refresh::LiveGroundtruth* gt) {
        qvcache_refresh::PhaseMetrics m;
        size_t interval_begin = 0;
        for (size_t i = 0; i < n; ++i) {
            std::vector<uint32_t> tags(K, INVALID);
            double miss_penalty = 0.0;
            auto s = std::chrono::high_resolution_clock::now();
            const bool hit = aker_ops::search<T>(
                cache, backend, queries + i * qaligned, dim, K, slot_list_size,
                vector_in_bytes, metric, tags.data(), &miss_penalty);
            auto e = std::chrono::high_resolution_clock::now();
            const double ms = std::chrono::duration<double, std::milli>(e - s).count();
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
            if (report_interval > 0 &&
                ((i + 1) % static_cast<size_t>(report_interval) == 0 || i + 1 == n)) {
                qvcache_refresh::log_search_progress(
                    "aker", phase, i + 1, n, m, gt != nullptr, K, snapshot(),
                    interval_begin, search_threads);
                interval_begin = i + 1;
            }
        }
        qvcache_refresh::log_search_phase("aker", phase, m, snapshot(), gt != nullptr, K, search_threads);
        return m;
    };

    qvcache_refresh::LiveGroundtruth warmup_gt;
    const qvcache_refresh::LiveGroundtruth* warmup_gt_ptr = nullptr;
    if (!warmup_gt_path.empty()) {
        warmup_gt = qvcache_refresh::require_search_only_gt(warmup_gt_path, n_warmup, K);
        warmup_gt_ptr = &warmup_gt;
    }
    const auto warmup = run_searches(n_warmup, "warmup", warmup_gt_ptr);

    double insert_ms = 0.0;
    if (n_insert > 0) {
        qvcache_refresh::synthesize_extra(live, n_insert, seed);
        auto t_ins0 = std::chrono::high_resolution_clock::now();
        for (uint32_t id : live.extra_ids) {
            aker_ops::apply_insert<T>(cache, backend, id, live.ptr(id), dim, vector_in_bytes, metric, false);
        }
        if (process_log) {
            aker_ops::process_log<T>(cache, metric);
        }
        auto t_ins1 = std::chrono::high_resolution_clock::now();
        insert_ms = std::chrono::duration<double, std::milli>(t_ins1 - t_ins0).count();
        qvcache_refresh::log_mutate_phase("aker", "insert", live.extra_ids.size(), insert_ms);
    } else {
        spdlog::info("{{\"event\": \"refresh_insert\", \"cache\": \"aker\", "
                     "\"n\": 0, \"skipped\": true, \"reason\": \"delete_only\"}}");
    }

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
        live.mark_deleted(id);
        aker_ops::apply_delete<T>(cache, backend, id);
    }
    auto t_del1 = std::chrono::high_resolution_clock::now();
    const double delete_ms = std::chrono::duration<double, std::milli>(t_del1 - t_del0).count();
    qvcache_refresh::log_mutate_phase("aker", "delete", n_delete, delete_ms, delete_rate);

    omp_set_num_threads(gt_threads);
    qvcache_refresh::LiveGroundtruth gt;
    const bool gt_reused = qvcache_refresh::load_live_groundtruth(live_gt_path, n_eval, K, gt);
    auto t_gt0 = std::chrono::high_resolution_clock::now();
    if (!gt_reused) {
        gt = qvcache_refresh::compute_exact_gt(queries, qaligned, n_eval, live, K);
        qvcache_refresh::save_live_groundtruth(live_gt_path, gt);
    }
    auto t_gt1 = std::chrono::high_resolution_clock::now();
    aker_pin_openmp(static_cast<int>(search_threads));
    const double gt_ms = std::chrono::duration<double, std::milli>(t_gt1 - t_gt0).count();
    const size_t n_live = n_base + live.extra_ids.size() - live.deleted.size();
    qvcache_refresh::log_gt_phase("aker", n_eval, n_live, gt_ms, gt_reused);

    const auto eval = run_searches(n_eval, "eval", &gt);

    spdlog::info("{{\"event\": \"refresh_end\", \"cache\": \"aker\", "
                 "\"warmup_elapsed_ms\": {}, \"insert_elapsed_ms\": {}, "
                 "\"delete_elapsed_ms\": {}, \"gt_elapsed_ms\": {}, \"eval_elapsed_ms\": {}, "
                 "\"warmup_hit_ratio\": {}, \"warmup_recall_all\": {}, "
                 "\"warmup_recall_hits\": {}, \"warmup_recall_misses\": {}, "
                 "\"eval_hit_ratio\": {}, \"eval_miss_ratio\": {}, "
                 "\"eval_recall_all\": {}, \"eval_recall_hits\": {}, \"eval_recall_misses\": {}, "
                 "\"eval_qps\": {}}}",
                 warmup.search_ms, insert_ms, delete_ms, gt_ms, eval.search_ms,
                 warmup.hit_ratio(), warmup.recall_all(), warmup.recall_hits(),
                 warmup.recall_misses(), eval.hit_ratio(), eval.miss_ratio(),
                 eval.recall_all(), eval.recall_hits(), eval.recall_misses(),
                 eval.search_ms > 0.0 ? static_cast<double>(eval.n_search) / (eval.search_ms / 1000.0) : 0.0);

    akerDestroyAnnsCache(cache);
    greator::aligned_free(base);
    greator::aligned_free(queries);
    return 0;
}

int main(int argc, char** argv) {
    std::string data_type, data_path, query_path, live_gt_path, warmup_gt_path;
    std::string aker_config_path = "Aker/bootstrap/aker-standard.ini";
    std::string table_name = "spacev_1m";
    std::string db_host = "postgres", db_name = "postgres", db_user = "postgres", db_password = "postgres";
    int db_port = 5432, hnsw_ef_search = 200;
    uint32_t K = 10, aker_top_delta = 10, search_threads = 1;
    size_t aker_pool_size = 200000;
    std::string metric_str = "l2";
    double insert_frac = 0.05, delete_rate = 0.05;
    size_t n_warmup = 0, n_eval = 0;
    int report_interval = 100, gt_threads = 8;
    uint32_t seed = 1;
    bool process_log = true;

    po::options_description desc("Aker refresh stress-test (pgvector)");
    desc.add_options()
        ("help,h", "help")
        ("data_type", po::value<std::string>(&data_type)->required())
        ("data_path", po::value<std::string>(&data_path)->required())
        ("query_path", po::value<std::string>(&query_path)->required())
        ("live_gt_path", po::value<std::string>(&live_gt_path)->default_value(""),
         "Sidecar truthset; reused if present, else computed. Does not overwrite search-only GT")
        ("warmup_gt_path", po::value<std::string>(&warmup_gt_path)->default_value(""),
         "Existing search-only truthset for warmup recall. Loaded only, never computed")
        ("aker_config", po::value<std::string>(&aker_config_path)->default_value(aker_config_path))
        ("table_name", po::value<std::string>(&table_name)->default_value(table_name))
        ("db_host", po::value<std::string>(&db_host)->default_value(db_host))
        ("db_port", po::value<int>(&db_port)->default_value(db_port))
        ("db_name", po::value<std::string>(&db_name)->default_value(db_name))
        ("db_user", po::value<std::string>(&db_user)->default_value(db_user))
        ("db_password", po::value<std::string>(&db_password)->default_value(db_password))
        ("hnsw_ef_search", po::value<int>(&hnsw_ef_search)->default_value(hnsw_ef_search))
        ("K", po::value<uint32_t>(&K)->default_value(K))
        ("aker_top_delta", po::value<uint32_t>(&aker_top_delta)->default_value(aker_top_delta),
         "Aker Δ reserve (paper AK-D10 = 10)")
        ("aker_pool_size", po::value<size_t>(&aker_pool_size)->default_value(aker_pool_size),
         "Neighbor object pool; paper refresh uses 1% of |P|")
        ("search_threads", po::value<uint32_t>(&search_threads)->default_value(search_threads))
        ("metric", po::value<std::string>(&metric_str)->default_value(metric_str))
        ("insert_frac", po::value<double>(&insert_frac)->default_value(insert_frac),
         "Extra vectors after warmup (paper: 0.05 of |P|). 0 = delete-only")
        ("delete_rate", po::value<double>(&delete_rate)->default_value(delete_rate))
        ("n_warmup", po::value<size_t>(&n_warmup)->default_value(n_warmup))
        ("n_eval", po::value<size_t>(&n_eval)->default_value(n_eval))
        ("report_interval", po::value<int>(&report_interval)->default_value(report_interval))
        ("gt_threads", po::value<int>(&gt_threads)->default_value(gt_threads))
        ("seed", po::value<uint32_t>(&seed)->default_value(seed))
        ("aker_process_log", po::value<bool>(&process_log)->default_value(process_log),
         "Process the insert log once after the insert batch (no-op if insert_frac=0)");

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
            data_path, query_path, live_gt_path, warmup_gt_path, aker_config_path, table_name,
            db_host, db_port, db_name, db_user, db_password, hnsw_ef_search,
            K, aker_top_delta, aker_pool_size, search_threads, metric,
            insert_frac, delete_rate, n_warmup, n_eval, report_interval, seed,
            gt_threads, process_log);
    }
    if (data_type == "int8") {
        return run_refresh_bench<int8_t>(
            data_path, query_path, live_gt_path, warmup_gt_path, aker_config_path, table_name,
            db_host, db_port, db_name, db_user, db_password, hnsw_ef_search,
            K, aker_top_delta, aker_pool_size, search_threads, metric,
            insert_frac, delete_rate, n_warmup, n_eval, report_interval, seed,
            gt_threads, process_log);
    }
    if (data_type == "uint8") {
        return run_refresh_bench<uint8_t>(
            data_path, query_path, live_gt_path, warmup_gt_path, aker_config_path, table_name,
            db_host, db_port, db_name, db_user, db_password, hnsw_ef_search,
            K, aker_top_delta, aker_pool_size, search_threads, metric,
            insert_frac, delete_rate, n_warmup, n_eval, report_interval, seed,
            gt_threads, process_log);
    }
    std::cerr << "Unsupported data_type: " << data_type << "\n";
    return 1;
}
