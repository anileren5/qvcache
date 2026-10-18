// Refresh insert/delete latency with no cache layer.
// Same synthesized inserts and the same delete sample as the QVCache and
// Aker refresh benches (seed, insert_frac, delete_rate). Each operation is
// only the pgvector INSERT or DELETE.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include <boost/program_options.hpp>
#include <spdlog/spdlog.h>

#include "diskann/utils.h"
#include "pgvector/refresh_common.h"
#include "pgvector_backend.h"

namespace po = boost::program_options;

namespace {

double percentile(std::vector<double> values, double p) {
    if (values.empty()) {
        return 0.0;
    }
    std::sort(values.begin(), values.end());
    size_t idx = static_cast<size_t>(std::ceil(p * static_cast<double>(values.size())));
    if (idx == 0) {
        idx = 1;
    }
    --idx;
    if (idx >= values.size()) {
        idx = values.size() - 1;
    }
    return values[idx];
}

void log_mutate(const char* phase, size_t n, double elapsed_ms, const std::vector<double>& samples,
                double extra = -1.0) {
    const double avg = n ? elapsed_ms / static_cast<double>(n) : 0.0;
    const double sec = elapsed_ms / 1000.0;
    const double qps = sec > 0.0 ? static_cast<double>(n) / sec : 0.0;
    const double p50 = percentile(samples, 0.50);
    const double p90 = percentile(samples, 0.90);
    const double p95 = percentile(samples, 0.95);
    const double p99 = percentile(samples, 0.99);
    if (extra >= 0.0) {
        spdlog::info("{{\"event\": \"refresh_{}\", \"cache\": \"pgvector\", \"n\": {}, "
                     "\"delete_rate\": {}, \"elapsed_ms\": {}, \"avg_ms\": {}, \"qps\": {}, "
                     "\"tail_latency_ms\": {{\"p50\": {}, \"p90\": {}, \"p95\": {}, \"p99\": {}}}}}",
                     phase, n, extra, elapsed_ms, avg, qps, p50, p90, p95, p99);
    } else {
        spdlog::info("{{\"event\": \"refresh_{}\", \"cache\": \"pgvector\", \"n\": {}, "
                     "\"elapsed_ms\": {}, \"avg_ms\": {}, \"qps\": {}, "
                     "\"tail_latency_ms\": {{\"p50\": {}, \"p90\": {}, \"p95\": {}, \"p99\": {}}}}}",
                     phase, n, elapsed_ms, avg, qps, p50, p90, p95, p99);
    }
}

template <typename T>
int run(const std::string& data_path, const std::string& table_name,
        const std::string& db_host, int db_port, const std::string& db_name,
        const std::string& db_user, const std::string& db_password,
        double insert_frac, double delete_rate, uint32_t seed, int report_interval) {
    auto backend = std::make_unique<qvcache::PgVectorBackend<T>>(
        table_name, data_path, db_host, db_port, db_name, db_user, db_password, "l2", 200);

    T* base = nullptr;
    size_t n_base = 0, dim = 0, aligned = 0;
    diskann::load_aligned_bin<T>(data_path, base, n_base, dim, aligned);

    qvcache_refresh::LiveVectors<T> live;
    live.base = base;
    live.n_base = n_base;
    live.dim = dim;
    live.aligned = aligned;

    const size_t n_insert = static_cast<size_t>(std::llround(insert_frac * static_cast<double>(n_base)));
    spdlog::info("{{\"event\": \"refresh_start\", \"cache\": \"pgvector\", "
                 "\"n_base\": {}, \"insert_frac\": {}, \"n_insert\": {}, \"delete_rate\": {}, "
                 "\"seed\": {}}}",
                 n_base, insert_frac, n_insert, delete_rate, seed);

    double insert_ms = 0.0;
    std::vector<double> insert_samples;
    if (n_insert > 0) {
        qvcache_refresh::synthesize_extra(live, n_insert, seed);
        insert_samples.reserve(live.extra_ids.size());
        auto t0 = std::chrono::high_resolution_clock::now();
        for (size_t i = 0; i < live.extra_ids.size(); ++i) {
            const uint32_t id = live.extra_ids[i];
            auto s = std::chrono::high_resolution_clock::now();
            backend->insert(id, live.ptr(id));
            auto e = std::chrono::high_resolution_clock::now();
            insert_samples.push_back(std::chrono::duration<double, std::milli>(e - s).count());
            if (report_interval > 0 &&
                ((i + 1) % static_cast<size_t>(report_interval) == 0 || i + 1 == live.extra_ids.size())) {
                double sum = 0.0;
                for (double v : insert_samples) {
                    sum += v;
                }
                spdlog::info("{{\"event\": \"refresh_mutate_progress\", \"cache\": \"pgvector\", "
                             "\"phase\": \"insert\", \"completed\": {}, \"total\": {}, \"avg_ms\": {}}}",
                             i + 1, live.extra_ids.size(), sum / static_cast<double>(insert_samples.size()));
            }
        }
        auto t1 = std::chrono::high_resolution_clock::now();
        insert_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        log_mutate("insert", live.extra_ids.size(), insert_ms, insert_samples);
    } else {
        spdlog::info("{{\"event\": \"refresh_insert\", \"cache\": \"pgvector\", "
                     "\"n\": 0, \"skipped\": true, \"reason\": \"delete_only\"}}");
    }

    std::vector<uint32_t> candidates;
    candidates.reserve(n_base + live.extra_ids.size());
    for (uint32_t id = 0; id < static_cast<uint32_t>(n_base); ++id) {
        candidates.push_back(id);
    }
    for (uint32_t id : live.extra_ids) {
        candidates.push_back(id);
    }
    std::mt19937 rng(seed + 1);
    std::shuffle(candidates.begin(), candidates.end(), rng);
    const size_t n_delete = static_cast<size_t>(
        std::llround(delete_rate * static_cast<double>(candidates.size())));

    std::vector<double> delete_samples;
    delete_samples.reserve(n_delete);
    auto d0 = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < n_delete; ++i) {
        const uint32_t id = candidates[i];
        auto s = std::chrono::high_resolution_clock::now();
        backend->remove(id);
        live.mark_deleted(id);
        auto e = std::chrono::high_resolution_clock::now();
        delete_samples.push_back(std::chrono::duration<double, std::milli>(e - s).count());
        if (report_interval > 0 &&
            ((i + 1) % static_cast<size_t>(report_interval) == 0 || i + 1 == n_delete)) {
            double sum = 0.0;
            for (double v : delete_samples) {
                sum += v;
            }
            spdlog::info("{{\"event\": \"refresh_mutate_progress\", \"cache\": \"pgvector\", "
                         "\"phase\": \"delete\", \"completed\": {}, \"total\": {}, \"avg_ms\": {}}}",
                         i + 1, n_delete, sum / static_cast<double>(delete_samples.size()));
        }
    }
    auto d1 = std::chrono::high_resolution_clock::now();
    const double delete_ms = std::chrono::duration<double, std::milli>(d1 - d0).count();
    log_mutate("delete", n_delete, delete_ms, delete_samples, delete_rate);

    const size_t n_live = n_base + live.extra_ids.size() - live.deleted.size();
    spdlog::info("{{\"event\": \"refresh_end\", \"cache\": \"pgvector\", "
                 "\"insert_elapsed_ms\": {}, \"delete_elapsed_ms\": {}, \"n_insert\": {}, "
                 "\"n_delete\": {}, \"n_live\": {}}}",
                 insert_ms, delete_ms, live.extra_ids.size(), n_delete, n_live);

    diskann::aligned_free(base);
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    std::string data_type, data_path, table_name = "spacev_1m";
    std::string db_host = "postgres", db_name = "postgres", db_user = "postgres", db_password = "postgres";
    int db_port = 5432;
    double insert_frac = 0.05, delete_rate = 0.05;
    uint32_t seed = 1;
    int report_interval = 1000;

    po::options_description desc("pgvector refresh insert/delete latency (no cache)");
    desc.add_options()
        ("help,h", "help")
        ("data_type", po::value<std::string>(&data_type)->required(), "float|int8|uint8")
        ("data_path", po::value<std::string>(&data_path)->required())
        ("table_name", po::value<std::string>(&table_name)->default_value(table_name))
        ("db_host", po::value<std::string>(&db_host)->default_value(db_host))
        ("db_port", po::value<int>(&db_port)->default_value(db_port))
        ("db_name", po::value<std::string>(&db_name)->default_value(db_name))
        ("db_user", po::value<std::string>(&db_user)->default_value(db_user))
        ("db_password", po::value<std::string>(&db_password)->default_value(db_password))
        ("insert_frac", po::value<double>(&insert_frac)->default_value(insert_frac))
        ("delete_rate", po::value<double>(&delete_rate)->default_value(delete_rate))
        ("seed", po::value<uint32_t>(&seed)->default_value(seed))
        ("report_interval", po::value<int>(&report_interval)->default_value(report_interval));

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

    spdlog::set_pattern("%v");
    spdlog::flush_on(spdlog::level::info);

    if (data_type == "float") {
        return run<float>(data_path, table_name, db_host, db_port, db_name, db_user, db_password,
                          insert_frac, delete_rate, seed, report_interval);
    }
    if (data_type == "int8") {
        return run<int8_t>(data_path, table_name, db_host, db_port, db_name, db_user, db_password,
                           insert_frac, delete_rate, seed, report_interval);
    }
    if (data_type == "uint8") {
        return run<uint8_t>(data_path, table_name, db_host, db_port, db_name, db_user, db_password,
                            insert_frac, delete_rate, seed, report_interval);
    }
    std::cerr << "Unsupported data_type: " << data_type << "\n";
    return 1;
}
