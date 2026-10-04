// FAISS HNSW latency micro-benchmark (ms, CSV, no CLI args)
// - Insert latency per element (ms)  -> insert.csv
// - Average search latency per step (ms) over NQ random queries -> search.csv
//
// Build example:
//   g++ -std=cpp17 -O3 hnsw_bench_ms.cpp -o hnsw_bench_ms -lfaiss
//
// Notes:
// * Uses L2 distance with IndexHNSWFlat.
// * Times only FAISS calls (add/search).
// * CSVs are comma-separated, no header: "element_count,latency_ms".

#include <faiss/IndexHNSW.h>

#include "Timer.hh"   // uses topkache::ElapsedPair as provided

#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <random>
#include <string>
#include <utility>
#include <vector>

namespace config {
    // ======= Fixed configuration (edit and rebuild) =======
    constexpr size_t        N  = 100000;   // total elements to insert
    constexpr size_t        D  = 100;      // vector dimension
    constexpr int           M  = 8;       // HNSW M
    constexpr int           EF_CONSTRUCTION = 32;
    constexpr int           EF_SEARCH       = 16;
    constexpr size_t        K  = 1;       // top-k for search
    constexpr size_t        NQ = 1000;      // queries per step
    constexpr uint64_t      SEED_BUILD = 42;
    constexpr uint64_t      SEED_QUERY = 1337;
    constexpr const char*   INSERT_CSV = "insert.csv";
    constexpr const char*   SEARCH_CSV = "search.csv";
    // =======================================================
}

class RandomInsert {
public:
    using LatRec = std::pair<size_t, double>; // (element_count, latency_ms)

    RandomInsert(size_t d, int M, int efC, uint64_t seed)
        : d_(d),
          rng_(seed),
          dist_(0.0f, 1.0f),
          tmp_(d, 0.0f) {
        index_ = std::make_unique<faiss::IndexHNSWFlat>(static_cast<int>(d_), M);
        index_->hnsw.efConstruction = efC;
        if (!index_) throw std::runtime_error("Failed to create IndexHNSWFlat");
    }

    faiss::IndexHNSWFlat& index() noexcept { return *index_; }
    const faiss::IndexHNSWFlat& index() const noexcept { return *index_; }

    // Insert one random vector and measure add() latency in milliseconds.
    double insert_one_and_measure_ms() {
        const float* x = make_random_vec_();
        topkache::ElapsedPair t;
        t.start();
        index_->add(1, x);
        t.end();
        t.elapsedMs();
        double ms = t.getElapsedMs();
        insert_latency_.emplace_back(static_cast<size_t>(index_->ntotal), ms);
        return ms;
    }

    const std::vector<LatRec>& insert_latency_records() const noexcept {
        return insert_latency_;
    }

private:
    const float* make_random_vec_() {
        for (size_t i = 0; i < d_; ++i) tmp_[i] = dist_(rng_);
        return tmp_.data();
    }

    size_t d_;
    std::unique_ptr<faiss::IndexHNSWFlat> index_;
    std::mt19937_64 rng_;
    std::uniform_real_distribution<float> dist_;
    std::vector<float> tmp_;
    std::vector<LatRec> insert_latency_;
};

class RandomRead {
public:
    using LatRec = std::pair<size_t, double>; // (element_count, avg_search_latency_ms)

    RandomRead(RandomInsert& builder, size_t k, size_t nq, int efS, uint64_t seed)
        : builder_(builder),
          k_(k),
          nq_(nq),
          rng_(seed),
          dist_(0.0f, 1.0f),
          qtmp_(builder.index().d, 0.0f),
          D_(k),
          I_(k) {
        builder_.index().hnsw.efSearch = efS;
    }

    // After each insert, run nq random queries and record average search latency (ms).
    double measure_and_record_ms() {
        auto& index = builder_.index();
        if (index.ntotal == 0) {
            search_latency_.emplace_back(0, 0.0);
            return 0.0;
        }

        topkache::ElapsedPair t;
        t.start();
        for (size_t i = 0; i < nq_; ++i) {
            const float* q = make_random_query_();
            index.search(1, q, static_cast<faiss::idx_t>(k_), D_.data(), I_.data());
        }
        t.end();
        t.elapsedMs();
        double avg_ms = t.getElapsedMs() / static_cast<double>(nq_);
        search_latency_.emplace_back(static_cast<size_t>(index.ntotal), avg_ms);
        return avg_ms;
    }

    const std::vector<LatRec>& search_latency_records() const noexcept {
        return search_latency_;
    }

private:
    const float* make_random_query_() {
        for (size_t i = 0; i < qtmp_.size(); ++i) qtmp_[i] = dist_(rng_);
        return qtmp_.data();
    }

    RandomInsert& builder_;
    size_t k_;
    size_t nq_;
    std::mt19937_64 rng_;
    std::uniform_real_distribution<float> dist_;
    std::vector<float> qtmp_;
    std::vector<float> D_;
    std::vector<faiss::idx_t> I_;
    std::vector<LatRec> search_latency_;
};

static void write_csv(const std::string& path,
                      const std::vector<std::pair<size_t, double>>& rows) {
    std::ofstream ofs(path, std::ios::out | std::ios::trunc);
    if (!ofs) throw std::runtime_error("Failed to open output file: " + path);
    ofs.setf(std::ios::fixed);
    ofs << std::setprecision(6); // ms with microsecond fraction
    for (const auto& r : rows) {
        ofs << r.first << "," << r.second << "\n"; // element_count,latency_ms
    }
    ofs.flush();
    if (!ofs) throw std::runtime_error("Failed to write output file: " + path);
}

int main() {
    try {
        std::cout << "HNSW benchmark (ms, CSV)\n"
                  << " N=" << config::N
                  << " D=" << config::D
                  << " M=" << config::M
                  << " efC=" << config::EF_CONSTRUCTION
                  << " efS=" << config::EF_SEARCH
                  << " K=" << config::K
                  << " NQ=" << config::NQ
                  << "\n";

        RandomInsert builder(config::D, config::M, config::EF_CONSTRUCTION, config::SEED_BUILD);
        RandomRead reader(builder, config::K, config::NQ, config::EF_SEARCH, config::SEED_QUERY);

        for (size_t i = 0; i < config::N; ++i) {
            builder.insert_one_and_measure_ms();
            reader.measure_and_record_ms();
            if ((i + 1) % 10000 == 0) {
                std::cout << "Inserted " << (i + 1) << " / " << config::N << "\n";
            }
        }

        write_csv(config::INSERT_CSV, builder.insert_latency_records());
        write_csv(config::SEARCH_CSV, reader.search_latency_records());

        std::cout << "Wrote: " << config::INSERT_CSV << ", " << config::SEARCH_CSV << "\n";
        return 0;

    } catch (const std::exception& ex) {
        std::cerr << "Error: " << ex.what() << "\n";
        return 1;
    }
}