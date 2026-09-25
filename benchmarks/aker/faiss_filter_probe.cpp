// Aker: standalone timing probe for the query-filter cost.
//
// Reproduces Aker's ApproxFilterHnsw2 configuration (IndexHNSWFlat, M=4,
// efSearch=8, wrapped in IndexIDMap, searched one query at a time, two filters
// per lookup) and isolates the variables that could explain the ~9-11 ms
// per-lookup cost seen in aker_sim_*.log:
//   - being called from inside an OpenMP parallel region (nested regions)
//   - single-vector add_with_ids interleaved with searches
//   - index size
//
// Links only against FAISS; no Aker, DiskANN or qvcache code involved.

#include <faiss/IndexHNSW.h>
#include <faiss/IndexIDMap.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include <omp.h>

namespace {

constexpr int kDim = 100;          // SPACEV dimensionality
constexpr int kHnswM = 4;          // Aker: k_hnsw_m
constexpr int kHnswEfSearch = 8;   // Aker: k_hnsw_ef_search
constexpr int kTopK = 1;           // Aker searches the filter for the nearest query

using Clock = std::chrono::high_resolution_clock;

struct Stats {
    double median_ms;
    double p99_ms;
    double min_ms;
    double max_ms;
    double mean_ms;
};

Stats summarize(std::vector<double> samples) {
    std::sort(samples.begin(), samples.end());
    Stats s{};
    const size_t n = samples.size();
    s.min_ms = samples.front();
    s.max_ms = samples.back();
    s.median_ms = samples[n / 2];
    s.p99_ms = samples[std::min(n - 1, static_cast<size_t>(0.99 * n))];
    double sum = 0.0;
    for (double v : samples) sum += v;
    s.mean_ms = sum / static_cast<double>(n);
    return s;
}

void report(const std::string& label, const Stats& s) {
    std::printf("%-56s med=%9.4f  p99=%9.4f  min=%9.4f  max=%9.4f  mean=%9.4f\n",
                label.c_str(), s.median_ms, s.p99_ms, s.min_ms, s.max_ms, s.mean_ms);
}

// Aker: IndexIDMap owning an IndexHNSWFlat, same params as ApproxFilterHnsw2.
struct Filter {
    std::unique_ptr<faiss::IndexIDMap> idmap;

    Filter() {
        auto* hnsw = new faiss::IndexHNSWFlat(kDim, kHnswM, faiss::METRIC_L2);
        hnsw->hnsw.efSearch = kHnswEfSearch;
        idmap.reset(new faiss::IndexIDMap(hnsw));
    }

    void add_one(const float* x, faiss::idx_t id) { idmap->add_with_ids(1, x, &id); }

    void search_one(const float* x, float* d, faiss::idx_t* l) {
        idmap->search(1, x, kTopK, d, l);
    }
};

std::vector<float> random_vectors(size_t n, std::mt19937& rng) {
    std::uniform_real_distribution<float> dist(-128.0f, 127.0f);
    std::vector<float> out(n * kDim);
    for (auto& v : out) v = dist(rng);
    return out;
}

// Fill a filter by repeated single-vector adds, mirroring Aker's insert path.
void fill_filter(Filter& f, const std::vector<float>& data, size_t count) {
    for (size_t i = 0; i < count; ++i)
        f.add_one(data.data() + i * kDim, static_cast<faiss::idx_t>(i + 1));
}

// Scenario: pure search cost, optionally from inside an OpenMP parallel region.
Stats bench_search(Filter& f0, Filter& f1, const std::vector<float>& probes,
                   size_t n_probe, bool inside_omp, bool dual) {
    std::vector<double> samples(n_probe, 0.0);

    auto one = [&](size_t i) {
        float d0[kTopK], d1[kTopK];
        faiss::idx_t l0[kTopK], l1[kTopK];
        const float* x = probes.data() + i * kDim;
        auto t0 = Clock::now();
        f0.search_one(x, d0, l0);
        if (dual) f1.search_one(x, d1, l1);
        auto t1 = Clock::now();
        samples[i] = std::chrono::duration<double, std::milli>(t1 - t0).count();
    };

    if (inside_omp) {
        // Aker benchmark: query loop is `omp parallel for num_threads(search_threads)`.
        #pragma omp parallel for num_threads(1) schedule(dynamic)
        for (size_t i = 0; i < n_probe; ++i) one(i);
    } else {
        for (size_t i = 0; i < n_probe; ++i) one(i);
    }
    return summarize(std::move(samples));
}

// Scenario: search interleaved with single-vector adds, as the real run does
// (every miss appends one query vector to the filter).
Stats bench_interleaved(const std::vector<float>& data, size_t n_seed, size_t n_probe,
                        bool inside_omp, std::vector<double>* add_samples) {
    Filter f0, f1;
    std::mt19937 rng(7);
    fill_filter(f0, data, n_seed);

    std::vector<double> search_samples(n_probe, 0.0);
    std::vector<double> adds(n_probe, 0.0);

    auto one = [&](size_t i) {
        float d0[kTopK], d1[kTopK];
        faiss::idx_t l0[kTopK], l1[kTopK];
        const float* x = data.data() + ((n_seed + i) % (data.size() / kDim)) * kDim;

        auto t0 = Clock::now();
        f0.search_one(x, d0, l0);
        f1.search_one(x, d1, l1);
        auto t1 = Clock::now();
        search_samples[i] = std::chrono::duration<double, std::milli>(t1 - t0).count();

        auto t2 = Clock::now();
        f0.add_one(x, static_cast<faiss::idx_t>(n_seed + i + 1));
        auto t3 = Clock::now();
        adds[i] = std::chrono::duration<double, std::milli>(t3 - t2).count();
    };

    if (inside_omp) {
        #pragma omp parallel for num_threads(1) schedule(dynamic)
        for (size_t i = 0; i < n_probe; ++i) one(i);
    } else {
        for (size_t i = 0; i < n_probe; ++i) one(i);
    }

    if (add_samples) *add_samples = adds;
    return summarize(std::move(search_samples));
}

}  // namespace

int main(int argc, char** argv) {
    size_t n_probe = 2000;
    if (argc > 1) n_probe = static_cast<size_t>(std::atol(argv[1]));

    std::printf("FAISS query-filter probe\n");
    std::printf("  dim=%d  M=%d  efSearch=%d  k=%d\n", kDim, kHnswM, kHnswEfSearch, kTopK);
    std::printf("  omp_get_max_threads=%d  omp_get_max_active_levels=%d\n",
                omp_get_max_threads(), omp_get_max_active_levels());
    std::printf("  probes per scenario=%zu\n\n", n_probe);

    std::mt19937 rng(1234);
    const size_t n_pool = 40000;
    std::vector<float> data = random_vectors(n_pool, rng);
    std::vector<float> probes = random_vectors(n_probe, rng);

    // ---- Pure search cost vs index size, outside and inside an OMP region ----
    for (size_t n_seed : {size_t(2000), size_t(20000)}) {
        Filter f0, f1;
        fill_filter(f0, data, n_seed);
        fill_filter(f1, data, n_seed);

        char buf[128];
        std::snprintf(buf, sizeof(buf), "search single-filter  N=%-6zu outside omp", n_seed);
        report(buf, bench_search(f0, f1, probes, n_probe, false, false));

        std::snprintf(buf, sizeof(buf), "search single-filter  N=%-6zu inside  omp", n_seed);
        report(buf, bench_search(f0, f1, probes, n_probe, true, false));

        std::snprintf(buf, sizeof(buf), "search dual-filter    N=%-6zu outside omp", n_seed);
        report(buf, bench_search(f0, f1, probes, n_probe, false, true));

        std::snprintf(buf, sizeof(buf), "search dual-filter    N=%-6zu inside  omp", n_seed);
        report(buf, bench_search(f0, f1, probes, n_probe, true, true));
        std::printf("\n");
    }

    // ---- Search interleaved with single-vector adds (the real run's pattern) ----
    for (size_t n_seed : {size_t(2000), size_t(20000)}) {
        for (bool inside : {false, true}) {
            std::vector<double> adds;
            Stats s = bench_interleaved(data, n_seed, n_probe, inside, &adds);
            char buf[128];
            std::snprintf(buf, sizeof(buf), "interleaved search    N=%-6zu %s omp",
                          n_seed, inside ? "inside " : "outside");
            report(buf, s);
            std::snprintf(buf, sizeof(buf), "interleaved add_one   N=%-6zu %s omp",
                          n_seed, inside ? "inside " : "outside");
            report(buf, summarize(std::move(adds)));
        }
        std::printf("\n");
    }

    return 0;
}
