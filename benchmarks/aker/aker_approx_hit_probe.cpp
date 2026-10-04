// Aker: probe whether Standard-mode approximate hits can fire.
// Insert one query, then shrink additive/interpolation noise until FAISS+thresh
// reports similar=true (or max_iters). Does not edit Aker sources.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <memory>
#include <random>
#include <string>
#include <type_traits>
#include <vector>

#include <boost/program_options.hpp>

#include "ak_anns_cache_c_wrapper.h"
#include "aker_c_abi_helpers.h"
#include "diskann/distance.h"
#include "greator/utils.h"
#include "greator_backend.h"

namespace po = boost::program_options;

// Aker: store float32 in VectorSlots (same layout as Aker's C test). DiskANN stays T.
template <typename T>
std::vector<float> to_f32(const T* v, uint32_t dim) {
    std::vector<float> out(dim);
    for (uint32_t i = 0; i < dim; ++i)
        out[i] = static_cast<float>(v[i]);
    return out;
}

template <typename T>
T clamp_coord(float x) {
    if constexpr (std::is_same_v<T, float>)
        return x;
    else if constexpr (std::is_same_v<T, int8_t>) {
        x = std::round(x);
        if (x > 127.0f)
            return 127;
        if (x < -128.0f)
            return -128;
        return static_cast<int8_t>(x);
    } else {
        x = std::round(x);
        if (x > 255.0f)
            return 255;
        if (x < 0.0f)
            return 0;
        return static_cast<uint8_t>(x);
    }
}

struct LookupResult {
    bool hit = false;
    bool similar = false;
    bool invalid = false;
};

LookupResult aker_lookup_id(
    anns_cache_c_wrapper_t* cache,
    float* packed_f32,
    uint32_t dim,
    size_t vector_in_bytes,
    uint64_t query_id
) {
    if (query_id == 0)
        query_id = 1;

    char* query_slot = akerCreateVectorSlot(
        query_id, vector_in_bytes, reinterpret_cast<char*>(packed_f32), 0, 0, 0.0f);
    char* query_view = akerCreateVectorView(
        query_slot, static_cast<size_t>(dim), vector_in_bytes, aker_transform_float);

    LookupResult r;
    char* entry = akerGetCacheEntry(cache, query_view, &r.similar, &r.invalid, aker_l2_float_bytes);
    r.hit = (entry != nullptr && !r.invalid);
    if (entry != nullptr)
        akerDestroyCacheEntry(entry);
    akerDestroyVectorView(query_view);
    akerDestroyVectorSlot(query_slot);
    return r;
}

LookupResult aker_lookup(
    anns_cache_c_wrapper_t* cache,
    float* packed_f32,
    uint32_t dim,
    size_t vector_in_bytes
) {
    uint64_t query_id = akerDefaultHash(reinterpret_cast<char*>(packed_f32), vector_in_bytes);
    return aker_lookup_id(cache, packed_f32, dim, vector_in_bytes, query_id);
}

template <typename T>
float packed_l2sq(const T* a, const T* b, uint32_t dim) {
    float sum = 0.0f;
    for (uint32_t i = 0; i < dim; ++i) {
        const float d = static_cast<float>(a[i]) - static_cast<float>(b[i]);
        sum += d * d;
    }
    return sum;
}

template <typename T>
bool vectors_equal(const T* a, const T* b, uint32_t dim) {
    return std::memcmp(a, b, static_cast<size_t>(dim) * sizeof(T)) == 0;
}

template <typename T>
void make_noisy(
    const T* q,
    const T* other,
    uint32_t dim,
    float mu,
    std::mt19937& rng,
    std::vector<T>& out
) {
    out.resize(dim);
    for (uint32_t i = 0; i < dim; ++i) {
        const float mixed = (1.0f - mu) * static_cast<float>(q[i]) + mu * static_cast<float>(other[i]);
        out[i] = clamp_coord<T>(mixed);
    }
    // Aker: exact path is hash(query bytes). Force at least one coordinate change
    // so a tiny mu cannot collapse to an exact hit.
    if (vectors_equal(out.data(), q, dim)) {
        std::uniform_int_distribution<uint32_t> idist(0, dim - 1);
        const uint32_t i = idist(rng);
        if constexpr (std::is_same_v<T, float>) {
            out[i] = q[i] + (q[i] >= 0 ? -1e-3f : 1e-3f);
        } else if constexpr (std::is_same_v<T, int8_t>) {
            out[i] = (q[i] < 127) ? static_cast<int8_t>(q[i] + 1) : static_cast<int8_t>(q[i] - 1);
        } else {
            out[i] = (q[i] < 255) ? static_cast<uint8_t>(q[i] + 1) : static_cast<uint8_t>(q[i] - 1);
        }
    }
}

template <typename T, typename TagT = uint32_t>
int run_probe(
    const std::string& data_path,
    const std::string& query_path,
    const std::string& disk_index_prefix,
    const std::string& aker_config_path,
    uint32_t R, uint32_t disk_L, uint32_t K, uint32_t B, uint32_t M,
    uint32_t build_threads, uint32_t beamwidth, int disk_index_already_built,
    uint32_t sector_len, uint32_t aker_top_delta, size_t aker_pool_size,
    size_t query_index, float mu0, float mu_decay, int max_iters, uint32_t seed
) {
    set_sector_len(sector_len);

    size_t query_num = 0, query_dim = 0, query_aligned_dim = 0;
    T* queries = nullptr;
    greator::load_aligned_bin<T>(query_path, queries, query_num, query_dim, query_aligned_dim);
    if (query_num < 2) {
        std::cerr << "Need at least 2 queries to interpolate noise.\n";
        return 1;
    }
    if (query_index >= query_num) {
        std::cerr << "query_index " << query_index << " >= npts " << query_num << "\n";
        return 1;
    }
    const uint32_t dim = static_cast<uint32_t>(query_dim);
    // Aker: float32 slots (vinb = 4*dim), matching Aker's C test. DiskANN still uses T.
    const size_t vector_in_bytes = static_cast<size_t>(dim) * sizeof(float);
    const uint32_t slot_list_size = K + aker_top_delta;

    if (disk_L < slot_list_size) {
        std::cerr << "disk_L must be >= K + top_delta\n";
        return 1;
    }

    anns_cache_parameter_c_t parameter{};
    akerImportAnnsCacheConfig(const_cast<char*>(aker_config_path.c_str()), &parameter);
    parameter.vector_format.dimension = dim;
    parameter.vector_format.vector_in_bytes = vector_in_bytes;
    parameter.capacity.in_topk = K;
    parameter.capacity.top_delta = aker_top_delta;
    if (aker_pool_size == 0)
        aker_pool_size = static_cast<size_t>(slot_list_size + 1) * 64;
    parameter.capacity.pool_size = aker_pool_size;
    parameter.distance_metric = 0;

    anns_cache_c_wrapper_t* cache = akerCreateAnnsCache(parameter);
    if (cache == nullptr) {
        std::cerr << "Failed to create Aker cache\n";
        return 1;
    }
    std::cout << "cache dim=" << cache->parameter.vector_format.dimension
              << " vinb=" << cache->parameter.vector_format.vector_in_bytes
              << " in_topk=" << cache->parameter.capacity.in_topk
              << " top_delta=" << cache->parameter.capacity.top_delta << "\n";

    qvcache::GreatorBackend<T> backend(
        data_path, disk_index_prefix, R, disk_L, B, M,
        build_threads, disk_index_already_built, beamwidth, diskann::Metric::L2);

    const T* q = queries + query_index * query_aligned_dim;
    const size_t other_idx = (query_index + 1) % query_num;
    const T* other = queries + other_idx * query_aligned_dim;

    std::vector<T> packed(dim);
    std::memcpy(packed.data(), q, dim * sizeof(T));

    std::vector<TagT> tags(slot_list_size, 0);
    std::vector<float> dists(slot_list_size, 0.0f);
    greator::QueryStats stats{};
    backend.search(q, slot_list_size, tags.data(), dists.data(), nullptr, &stats);

    std::vector<float> packed_f32 = to_f32(packed.data(), dim);
    {
        std::vector<float> faiss_buf(static_cast<size_t>(dim), 0.0f);
        const bool ok = aker_transform_float(
            packed_f32.data(), vector_in_bytes, static_cast<size_t>(dim), faiss_buf.data(), nullptr);
        std::cout << "float32 slot vinb=" << vector_in_bytes
                  << " transform ok=" << ok
                  << " dst0=" << faiss_buf[0]
                  << " packed0=" << packed_f32[0] << "\n";
    }
    const uint64_t raw_hash = akerDefaultHash(reinterpret_cast<char*>(packed_f32.data()), vector_in_bytes);
    const int64_t faiss_cast = static_cast<int64_t>(raw_hash);
    // Aker: FAISS IndexIDMap ids are signed; getCacheEntry skips labels < 0.
    // Use a small positive id so the approx path can actually retrieve this entry.
    const uint64_t query_id = 42;

    std::cout << "Inserted query_index=" << query_index
              << " dim=" << dim
              << " npts=" << query_num
              << " nn_dist=" << dists[0]
              << "\n  xxhash=" << raw_hash
              << " as_faiss_idx=" << faiss_cast
              << " (negative => Aker drops approx candidates)"
              << "\n  using cache_id=" << query_id
              << " Standard thresh=min_distance/4=" << (dists[0] / 4.0f)
              << "\n";

    std::vector<TagT> fetch_ids(tags.begin(), tags.end());
    auto neighbor_vecs = backend.fetch_vectors_by_ids(fetch_ids);
    std::vector<char*> neighbor_slots(slot_list_size, nullptr);
    for (uint32_t j = 0; j < slot_list_size; ++j) {
        std::vector<T> nb(dim, 0);
        const size_t copy_n = std::min(static_cast<size_t>(dim), neighbor_vecs[j].size());
        std::memcpy(nb.data(), neighbor_vecs[j].data(), copy_n * sizeof(T));
        std::vector<float> nb_f32 = to_f32(nb.data(), dim);
        neighbor_slots[j] = akerCreateVectorSlot(
            static_cast<uint64_t>(tags[j]) + 1,
            vector_in_bytes,
            reinterpret_cast<char*>(nb_f32.data()),
            0, 0, dists[j]);
    }

    char* query_slot = akerCreateVectorSlot(
        query_id, vector_in_bytes, reinterpret_cast<char*>(packed_f32.data()), 0, 0, 0.0f);
    char* query_view = akerCreateVectorView(
        query_slot, static_cast<size_t>(dim), vector_in_bytes, aker_transform_float);
    char* entry = akerCreateCacheEntry(cache, query_slot, slot_list_size, neighbor_slots.data());
    if (entry == nullptr || !akerInsertCacheEntry(cache, query_id, entry, query_view)) {
        std::cerr << "Aker insert failed\n";
        if (entry)
            akerDestroyCacheEntry(entry);
        return 1;
    }
    akerDestroyVectorView(query_view);
    akerDestroyVectorSlot(query_slot);

    // Aker: extra FAISS points so HNSW is not a 1-node graph. Copies of q so k=1
    // is still the same vector (M=4, efSearch=8 often misses the true NN among far SPACEV queries).
    const int extra_n = 15;
    for (int e = 0; e < extra_n; ++e) {
        std::vector<float> extra = packed_f32;
        const uint64_t extra_id = 1000 + static_cast<uint64_t>(e);
        char* eslot = akerCreateVectorSlot(
            extra_id, vector_in_bytes, reinterpret_cast<char*>(extra.data()), 0, 0, 0.0f);
        char* eview = akerCreateVectorView(
            eslot, static_cast<size_t>(dim), vector_in_bytes, aker_transform_float);
        char* eentry = akerCreateCacheEntry(cache, eslot, slot_list_size, neighbor_slots.data());
        if (eentry != nullptr && !akerInsertCacheEntry(cache, extra_id, eentry, eview))
            akerDestroyCacheEntry(eentry);
        akerDestroyVectorView(eview);
        akerDestroyVectorSlot(eslot);
    }

    for (char* s : neighbor_slots)
        akerDestroyVectorSlot(s);

    if (char* st = akerGetCacheStatusText(cache))
        std::cout << st << "\n";

    LookupResult exact = aker_lookup_id(cache, packed_f32.data(), dim, vector_in_bytes, query_id);
    std::cout << "exact replay id=42: hit=" << exact.hit
              << " similar=" << exact.similar
              << " invalid=" << exact.invalid << "\n";

    // Same bytes as q, but a different id so exact-hash misses and FAISS must hit.
    LookupResult same_bytes = aker_lookup_id(
        cache, packed_f32.data(), dim, vector_in_bytes, /*id*/ 99);
    std::cout << "same bytes id=99 (must be approx): hit=" << same_bytes.hit
              << " similar=" << same_bytes.similar
              << " invalid=" << same_bytes.invalid << "\n";
    std::cout << "transform calls=" << aker_xf_calls.load()
              << " ok=" << aker_xf_ok.load()
              << " last_dim=" << aker_xf_last_dim.load()
              << " last_src=" << aker_xf_last_src.load()
              << " last_n=" << aker_xf_last_n.load() << "\n";

    // Aker: FAISS similar=true never fires through the C ABI (even distance=0 / identical
    // float32). The only non-exact hit we can produce without editing Aker is linkCacheEntry.
    {
        std::vector<char*> link_nbs(slot_list_size, nullptr);
        std::vector<float> z(dim, 0.0f);
        for (uint32_t j = 0; j < slot_list_size; ++j) {
            link_nbs[j] = akerCreateVectorSlot(
                20000 + j, vector_in_bytes, reinterpret_cast<char*>(z.data()), 0, 0,
                (j == 0) ? dists[0] : dists[0] + static_cast<float>(j));
        }
        char* ls = akerCreateVectorSlot(
            99, vector_in_bytes, reinterpret_cast<char*>(packed_f32.data()), 0, 0, 0.0f);
        char* le = akerCreateCacheEntry(cache, ls, slot_list_size, link_nbs.data());
        const bool linked_ok = (le != nullptr) && akerLinkCacheEntry(cache, le, query_id);
        if (!linked_ok && le)
            akerDestroyCacheEntry(le);
        akerDestroyVectorSlot(ls);
        for (char* s : link_nbs)
            akerDestroyVectorSlot(s);
        LookupResult linked = aker_lookup_id(cache, packed_f32.data(), dim, vector_in_bytes, 99);
        std::cout << "linkCacheEntry 99->42 ok=" << linked_ok
                  << " lookup hit=" << linked.hit
                  << " similar=" << linked.similar
                  << " invalid=" << linked.invalid
                  << " (hashmap alias; similar stays 0)\n";
    }

    std::mt19937 rng(seed);
    float mu = mu0;
    bool found = false;
    float hit_mu = 0.0f;
    float hit_dist = 0.0f;

    std::cout << "iter\tmu\tl2sq(q,q')\thit\tsimilar\tinvalid\n";
    for (int iter = 0; iter < max_iters; ++iter) {
        std::vector<T> noisy;
        make_noisy<T>(packed.data(), other, dim, mu, rng, noisy);
        const float d = packed_l2sq(packed.data(), noisy.data(), dim);
        std::vector<float> noisy_f32 = to_f32(noisy.data(), dim);
        LookupResult r = aker_lookup(cache, noisy_f32.data(), dim, vector_in_bytes);
        std::cout << iter << "\t" << mu << "\t" << d << "\t"
                  << r.hit << "\t" << r.similar << "\t" << r.invalid << "\n";
        if (r.hit && r.similar) {
            found = true;
            hit_mu = mu;
            hit_dist = d;
            break;
        }
        if (r.hit && !r.similar) {
            std::cout << "  (hash collided with cache_id; not an approx hit)\n";
        }
        mu *= mu_decay;
    }

    if (found) {
        std::cout << "APPROX HIT at mu=" << hit_mu
                  << " l2sq(q,q')=" << hit_dist
                  << " vs nn_dist=" << dists[0] << "\n";
        std::cout << "Standard accepts when dist(q',q_cached) < min(thresh, min_distance).\n";
    } else {
        std::cout << "NO approx hit in " << max_iters
                  << " iters (final mu=" << mu << ").\n";
        std::cout << "Standard thresh is min_distance/4=" << (dists[0] / 4.0f)
                  << ". FAISS ids from xxhash that are negative as int64 are skipped.\n";
    }

    akerDestroyAnnsCache(cache);
    greator::aligned_free(queries);
    return found ? 0 : 2;
}

int main(int argc, char** argv) {
    std::string data_type, data_path, query_path, disk_index_prefix;
    std::string aker_config_path = "external/Aker/bootstrap/aker-standard.ini";
    uint32_t R = 64, disk_L = 32, K = 10, B = 8, M = 8;
    uint32_t build_threads = 8, beamwidth = 2, sector_len = 4096, aker_top_delta = 5;
    int disk_index_already_built = 1;
    size_t aker_pool_size = 1024;
    size_t query_index = 0;
    float mu0 = 1.0f, mu_decay = 0.5f;
    int max_iters = 24;
    uint32_t seed = 1;

    po::options_description desc("Aker approx-hit probe");
    desc.add_options()
        ("help,h", "help")
        ("data_type", po::value<std::string>(&data_type)->default_value("int8"), "float|int8|uint8")
        ("data_path", po::value<std::string>(&data_path)->required(), "base bin")
        ("query_path", po::value<std::string>(&query_path)->required(), "query bin")
        ("disk_index_prefix", po::value<std::string>(&disk_index_prefix)->required(), "DiskANN prefix")
        ("aker_config", po::value<std::string>(&aker_config_path)->default_value(aker_config_path))
        ("R", po::value<uint32_t>(&R)->default_value(R))
        ("disk_L", po::value<uint32_t>(&disk_L)->default_value(disk_L))
        ("K", po::value<uint32_t>(&K)->default_value(K))
        ("B", po::value<uint32_t>(&B)->default_value(B))
        ("M", po::value<uint32_t>(&M)->default_value(M))
        ("build_threads", po::value<uint32_t>(&build_threads)->default_value(build_threads))
        ("beamwidth", po::value<uint32_t>(&beamwidth)->default_value(beamwidth))
        ("disk_index_already_built", po::value<int>(&disk_index_already_built)->default_value(1))
        ("sector_len", po::value<uint32_t>(&sector_len)->default_value(sector_len))
        ("aker_top_delta", po::value<uint32_t>(&aker_top_delta)->default_value(aker_top_delta))
        ("aker_pool_size", po::value<size_t>(&aker_pool_size)->default_value(aker_pool_size))
        ("query_index", po::value<size_t>(&query_index)->default_value(0), "which query to cache")
        ("mu0", po::value<float>(&mu0)->default_value(1.0f), "start interpolation weight toward another query")
        ("mu_decay", po::value<float>(&mu_decay)->default_value(0.5f), "multiply mu each miss")
        ("max_iters", po::value<int>(&max_iters)->default_value(24))
        ("seed", po::value<uint32_t>(&seed)->default_value(1));

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

    if (data_type == "int8") {
        return run_probe<int8_t>(
            data_path, query_path, disk_index_prefix, aker_config_path,
            R, disk_L, K, B, M, build_threads, beamwidth, disk_index_already_built,
            sector_len, aker_top_delta, aker_pool_size,
            query_index, mu0, mu_decay, max_iters, seed);
    }
    if (data_type == "uint8") {
        return run_probe<uint8_t>(
            data_path, query_path, disk_index_prefix, aker_config_path,
            R, disk_L, K, B, M, build_threads, beamwidth, disk_index_already_built,
            sector_len, aker_top_delta, aker_pool_size,
            query_index, mu0, mu_decay, max_iters, seed);
    }
    if (data_type == "float") {
        return run_probe<float>(
            data_path, query_path, disk_index_prefix, aker_config_path,
            R, disk_L, K, B, M, build_threads, beamwidth, disk_index_already_built,
            sector_len, aker_top_delta, aker_pool_size,
            query_index, mu0, mu_decay, max_iters, seed);
    }
    std::cerr << "Unsupported data_type: " << data_type << "\n";
    return 1;
}
