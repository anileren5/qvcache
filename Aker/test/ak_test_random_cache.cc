/**
 * @file ak_test_random_cache.cc
 * @brief Lightweight smoke test for Aker ANNSCache using the C ABI.
 *
 * This test:
 *  - Fills the cache with randomly generated entries.
 *  - Issues random queries with a configurable exact-hit ratio.
 *  - Prints a short hit/miss summary.
 *
 * NOTE: This is a functional test harness, not a performance benchmark.
 */

#include <cstdint>
#include <cstdlib>
#include <cstring>

#include <cerrno>

#include <iostream>
#include <fstream>
#include <random>
#include <string>
#include <string_view>
#include <vector>

#include "ak_anns_cache_c_wrapper.h"

namespace
{
    static constexpr std::uint32_t k_default_dimension = 128;
    static constexpr std::size_t k_default_entry_count = 1000;
    static constexpr std::size_t k_default_in_topk = 10;
    static constexpr std::size_t k_default_top_delta = 0;
    static constexpr std::size_t k_default_query_count = 50000;
    static constexpr double k_default_exact_hit_ratio = 0.30;
    static constexpr std::uint64_t k_default_seed = 1;

    struct TestOptions
    {
        std::string config_path;
        std::uint32_t dimension = k_default_dimension;

        std::size_t entry_count = k_default_entry_count;
        std::size_t in_topk = k_default_in_topk;
        std::size_t top_delta = k_default_top_delta;
        std::size_t pool_size = 0;

        std::size_t query_count = k_default_query_count;
        double exact_hit_ratio = k_default_exact_hit_ratio;
        std::uint64_t seed = k_default_seed;
    };

    struct InsertedQuery
    {
        std::uint64_t query_id = 0;
        std::vector<float> query_vector;
    };

    void printUsage(const char* argv0)
    {
        std::cerr
            << "Usage: " << argv0 << " [options]\n\n"
            << "Options:\n"
            << "  --config <path>         Bootstrap config path (default: $AKER_CONFIG_PATH or bootstrap/aker-standard.ini)\n"
            << "  --dimension <u32>       Vector dimension (default: 128)\n"
            << "  --entries <n>           Number of cache entries to insert\n"
            << "  --in-topk <n>           In-topK size (default: 10)\n"
            << "  --top-delta <n>         Top-delta size (default: 0)\n"
            << "  --pool-size <n>         Vector pool capacity (default: entries * (in_topk + top_delta))\n"
            << "  --queries <n>           Number of queries to run\n"
            << "  --exact-hit-ratio <f>   Fraction of queries that are exact hits in [0,1] (default: 0.30)\n"
            << "  --seed <u64>            RNG seed (default: 1)\n";
    }

    bool parseUnsigned(const char* s, std::uint64_t& out)
    {
        if (s == nullptr || *s == '\0')
            return false;

        char* end = nullptr;
        errno = 0;
        unsigned long long v = std::strtoull(s, &end, 10);
        if (errno != 0 || end == s || *end != '\0')
            return false;

        out = static_cast<std::uint64_t>(v);
        return true;
    }

    bool parseDouble(const char* s, double& out)
    {
        if (s == nullptr || *s == '\0')
            return false;

        char* end = nullptr;
        errno = 0;
        double v = std::strtod(s, &end);
        if (errno != 0 || end == s || *end != '\0')
            return false;

        out = v;
        return true;
    }

    bool parseArgs(int argc, char** argv, TestOptions& opt)
    {
        const char* env_config = std::getenv("AKER_CONFIG_PATH");
        if (env_config != nullptr)
            opt.config_path = env_config;
        else
            opt.config_path = "bootstrap/aker-standard.ini";

        for (int i = 1; i < argc; i++)
        {
            std::string_view arg(argv[i]);
            auto requireValue = [&](const char* name) -> const char*
            {
                if (i + 1 >= argc)
                {
                    std::cerr << "Missing value for " << name << "\n";
                    return nullptr;
                }
                return argv[++i];
            };

            if (arg == "--config")
            {
                const char* value = requireValue("--config");
                if (value == nullptr)
                    return false;
                opt.config_path = value;
            }
            else if (arg == "--dimension")
            {
                const char* value = requireValue("--dimension");
                std::uint64_t tmp = 0;
                if (value == nullptr || !parseUnsigned(value, tmp) || tmp == 0 || tmp > UINT32_MAX)
                {
                    std::cerr << "Invalid --dimension\n";
                    return false;
                }
                opt.dimension = static_cast<std::uint32_t>(tmp);
            }
            else if (arg == "--entries")
            {
                const char* value = requireValue("--entries");
                std::uint64_t tmp = 0;
                if (value == nullptr || !parseUnsigned(value, tmp) || tmp == 0)
                {
                    std::cerr << "Invalid --entries\n";
                    return false;
                }
                opt.entry_count = static_cast<std::size_t>(tmp);
            }
            else if (arg == "--in-topk")
            {
                const char* value = requireValue("--in-topk");
                std::uint64_t tmp = 0;
                if (value == nullptr || !parseUnsigned(value, tmp) || tmp == 0)
                {
                    std::cerr << "Invalid --in-topk\n";
                    return false;
                }
                opt.in_topk = static_cast<std::size_t>(tmp);
            }
            else if (arg == "--top-delta")
            {
                const char* value = requireValue("--top-delta");
                std::uint64_t tmp = 0;
                if (value == nullptr || !parseUnsigned(value, tmp))
                {
                    std::cerr << "Invalid --top-delta\n";
                    return false;
                }
                opt.top_delta = static_cast<std::size_t>(tmp);
            }
            else if (arg == "--pool-size")
            {
                const char* value = requireValue("--pool-size");
                std::uint64_t tmp = 0;
                if (value == nullptr || !parseUnsigned(value, tmp) || tmp == 0)
                {
                    std::cerr << "Invalid --pool-size\n";
                    return false;
                }
                opt.pool_size = static_cast<std::size_t>(tmp);
            }
            else if (arg == "--queries")
            {
                const char* value = requireValue("--queries");
                std::uint64_t tmp = 0;
                if (value == nullptr || !parseUnsigned(value, tmp) || tmp == 0)
                {
                    std::cerr << "Invalid --queries\n";
                    return false;
                }
                opt.query_count = static_cast<std::size_t>(tmp);
            }
            else if (arg == "--exact-hit-ratio")
            {
                const char* value = requireValue("--exact-hit-ratio");
                double tmp = 0.0;
                if (value == nullptr || !parseDouble(value, tmp) || tmp < 0.0 || tmp > 1.0)
                {
                    std::cerr << "Invalid --exact-hit-ratio (expected 0..1)\n";
                    return false;
                }
                opt.exact_hit_ratio = tmp;
            }
            else if (arg == "--seed")
            {
                const char* value = requireValue("--seed");
                std::uint64_t tmp = 0;
                if (value == nullptr || !parseUnsigned(value, tmp))
                {
                    std::cerr << "Invalid --seed\n";
                    return false;
                }
                opt.seed = tmp;
            }
            else
            {
                std::cerr << "Unknown argument: " << arg << "\n";
                return false;
            }
        }

        return true;
    }

    bool floatCopyTransform(void* src, size_t src_size, size_t dim, void* dst, uint8_t* aux)
    {
        (void)aux;

        if (src == nullptr || dst == nullptr)
            return false;

        const size_t required = dim * sizeof(float);
        if (src_size < required)
            return false;

        std::memcpy(dst, src, required);
        return true;
    }

    float l2DistanceBytes(uint8_t* a, uint8_t* b, size_t dim)
    {
        return akerL2Distance(reinterpret_cast<float*>(a), reinterpret_cast<float*>(b), dim);
    }

    void fillRandomVector(std::vector<float>& v, std::mt19937_64& rng)
    {
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
        for (float& x : v)
            x = dist(rng);
    }

    std::uint64_t makeResultVectorId(std::uint64_t query_id, std::uint32_t rank)
    {
        /* Construct a deterministic, non-zero vector ID for result vectors.
         */
        const std::uint64_t hi = (query_id & 0xFFFFFFFFull) << 32;
        const std::uint64_t lo = static_cast<std::uint64_t>(rank) + 1;
        return hi | lo;
    }

    bool insertCacheEntryForQuery(
        anns_cache_c_wrapper_t* cache,
        std::uint64_t query_id,
        const std::vector<float>& query_vector,
        char* query_slot,
        char* query_view,
        std::size_t neighbors,
        std::size_t vector_in_bytes,
        std::uint32_t dimension,
        std::mt19937_64& rng)
    {
        /* Simulate an index lookup by generating synthetic neighbor vectors,
         * computing distances, and inserting the prepared cache entry.
         */
        std::vector<char*> neighbor_slots;
        neighbor_slots.reserve(neighbors);

        for (std::size_t j = 0; j < neighbors; j++)
        {
            std::vector<float> result_vec(static_cast<std::size_t>(dimension));
            fillRandomVector(result_vec, rng);

            /* The C wrapper exposes akerL2Distance(float*, float*, size_t) even though it does
             * not mutate inputs. This test keeps the query vector const and casts for the call.
             */
            float* mutable_query_ptr = const_cast<float*>(query_vector.data());
            const float dist = akerL2Distance(mutable_query_ptr, result_vec.data(), dimension);
            const std::uint64_t result_id = makeResultVectorId(query_id, static_cast<std::uint32_t>(j));

            char* slot = akerCreateVectorSlot(
                result_id,
                vector_in_bytes,
                reinterpret_cast<char*>(result_vec.data()),
                0,
                0,
                dist);
            neighbor_slots.push_back(slot);
        }

        char* entry = akerCreateCacheEntry(
            cache,
            query_slot,
            neighbors,
            neighbor_slots.data());

        const bool inserted = akerInsertCacheEntry(cache, query_id, entry, query_view);
        if (!inserted)
            akerDestroyCacheEntry(entry);

        for (char* slot : neighbor_slots)
            akerDestroyVectorSlot(slot);

        return inserted;
    }
}

int main(int argc, char** argv)
{
    TestOptions opt;
    if (!parseArgs(argc, argv, opt))
    {
        printUsage(argv[0]);
        return 2;
    }

    if (opt.in_topk == 0)
    {
        std::cerr << "in_topk must be >= 1\n";
        return 2;
    }

    const std::size_t neighbors = opt.in_topk + opt.top_delta;
    if (neighbors < opt.in_topk)
    {
        std::cerr << "Invalid neighbors configuration\n";
        return 2;
    }

    if (opt.pool_size == 0)
        opt.pool_size = opt.entry_count * neighbors;

    const std::size_t vector_in_bytes = static_cast<std::size_t>(opt.dimension) * sizeof(float);

    /* Load configuration and override the vector format + capacity parameters.
     */
    {
        std::ifstream config_stream(opt.config_path);
        if (!config_stream.good())
        {
            std::cerr << "Config file not found: " << opt.config_path << "\n";
            std::cerr << "Tip: set AKER_CONFIG_PATH or pass --config <path>\n";
            return 2;
        }
    }

    anns_cache_parameter_c_t parameter{};
    akerImportAnnsCacheConfig(const_cast<char*>(opt.config_path.c_str()), &parameter);
    parameter.vector_format.dimension = opt.dimension;
    parameter.vector_format.vector_in_bytes = vector_in_bytes;
    parameter.capacity.pool_size = opt.pool_size;
    parameter.capacity.in_topk = opt.in_topk;
    parameter.capacity.top_delta = opt.top_delta;

    anns_cache_c_wrapper_t* cache = akerCreateAnnsCache(parameter);
    if (cache == nullptr)
    {
        std::cerr << "Failed to create cache\n";
        return 1;
    }

    std::mt19937_64 rng(opt.seed);

    /* Keep the original query vectors so exact-hit queries can reuse valid payload bytes.
     */
    std::vector<InsertedQuery> inserted_queries;
    inserted_queries.reserve(opt.entry_count);

    bool insert_failed = false;

    /* Seed the cache with synthetic entries so the query loop can request exact hits.
     */
    for (std::size_t i = 0; i < opt.entry_count; i++)
    {
        const std::uint64_t query_id = static_cast<std::uint64_t>(i + 1);

        std::vector<float> query_vec(static_cast<std::size_t>(opt.dimension));
        fillRandomVector(query_vec, rng);

        char* query_slot = akerCreateVectorSlot(
            query_id,
            vector_in_bytes,
            reinterpret_cast<char*>(query_vec.data()),
            0,
            0,
            0.0f);

        char* query_view = akerCreateVectorView(
            query_slot,
            static_cast<size_t>(opt.dimension),
            vector_in_bytes,
            floatCopyTransform);

        const bool inserted = insertCacheEntryForQuery(
            cache,
            query_id,
            query_vec,
            query_slot,
            query_view,
            neighbors,
            vector_in_bytes,
            opt.dimension,
            rng);
        if (!inserted)
        {
            insert_failed = true;
        }
        else
        {
            InsertedQuery record;
            record.query_id = query_id;
            record.query_vector = std::move(query_vec);
            inserted_queries.emplace_back(std::move(record));
        }

        /* The cache copies query vectors and result payloads into its own storage.
         * Destroy the temporary wrappers allocated for this insertion attempt.
         */
        akerDestroyVectorView(query_view);
        akerDestroyVectorSlot(query_slot);

        if (insert_failed)
            break;
    }

    if (insert_failed)
    {
        std::cerr << "Cache insertion failed (duplicate ID or capacity issue)\n";
        akerDestroyAnnsCache(cache);
        return 1;
    }

    std::uniform_real_distribution<double> hit_dist(0.0, 1.0);

    std::size_t exact_hit = 0;
    std::size_t similar_hit = 0;
    std::size_t miss = 0;
    std::size_t invalid = 0;

    std::size_t miss_inserted = 0;
    std::size_t miss_insert_failed = 0;
    std::size_t requested_hit_but_missed = 0;

    std::uint64_t next_miss_id = static_cast<std::uint64_t>(opt.entry_count + 1);

    /* Issue queries and record whether they hit/miss.
     * For true misses, insert a new entry to exercise eviction/insertion paths.
     */
    for (std::size_t q = 0; q < opt.query_count; q++)
    {
        const bool want_hit = (hit_dist(rng) < opt.exact_hit_ratio);

        std::vector<float> query_vec(static_cast<std::size_t>(opt.dimension));
        std::uint64_t query_id = 0;

        if (want_hit)
        {
            std::uniform_int_distribution<std::size_t> entry_pick(0, inserted_queries.size() - 1);
            const InsertedQuery& picked = inserted_queries[entry_pick(rng)];
            query_id = picked.query_id;
            query_vec = picked.query_vector;
        }
        else
        {
            query_id = next_miss_id++;
            fillRandomVector(query_vec, rng);
        }

        char* query_slot = akerCreateVectorSlot(
            query_id,
            vector_in_bytes,
            reinterpret_cast<char*>(query_vec.data()),
            0,
            0,
            0.0f);
        char* query_view = akerCreateVectorView(
            query_slot,
            static_cast<size_t>(opt.dimension),
            vector_in_bytes,
            floatCopyTransform);

        bool is_similar = false;
        bool is_invalid = false;
        char* result = akerGetCacheEntry(cache, query_view, &is_similar, &is_invalid, l2DistanceBytes);

        if (is_invalid)
            invalid++;

        if (result == nullptr)
        {
            miss++;

            /* Simulate the real integration behavior:
             * on a miss, the caller would query the underlying index, then insert
             * the new result set into the cache (triggering eviction if needed).
             */
            if (!want_hit)
            {
                const bool inserted = insertCacheEntryForQuery(
                    cache,
                    query_id,
                    query_vec,
                    query_slot,
                    query_view,
                    neighbors,
                    vector_in_bytes,
                    opt.dimension,
                    rng);

                if (inserted)
                {
                    InsertedQuery record;
                    record.query_id = query_id;
                    record.query_vector = std::move(query_vec);
                    inserted_queries.emplace_back(std::move(record));
                    miss_inserted++;
                }
                else
                {
                    miss_insert_failed++;
                }
            }
            else
            {
                requested_hit_but_missed++;
            }
        }
        else
        {
            if (is_similar)
                similar_hit++;
            else
                exact_hit++;

            akerDestroyCacheEntry(result);
        }

        akerDestroyVectorView(query_view);
        akerDestroyVectorSlot(query_slot);
    }

    std::cout << "[aker-random-cache-test] dimension=" << opt.dimension
              << " entries=" << opt.entry_count
              << " in_topk=" << opt.in_topk
              << " top_delta=" << opt.top_delta
              << " pool_size=" << opt.pool_size
              << " queries=" << opt.query_count
              << " exact_hit_ratio=" << opt.exact_hit_ratio
              << "\n";

    std::cout << "[aker-random-cache-test] results: exact_hit=" << exact_hit
              << " similar_hit=" << similar_hit
              << " miss=" << miss
              << " miss_inserted=" << miss_inserted
              << " miss_insert_failed=" << miss_insert_failed
              << " requested_hit_but_missed=" << requested_hit_but_missed
              << " invalid=" << invalid
              << "\n";

    const char* status = akerGetCacheStatusText(cache);
    if (status != nullptr)
        std::cout << "\n[aker-random-cache-test] cache status\n" << status << "\n";

    akerDestroyAnnsCache(cache);
    return 0;
}
