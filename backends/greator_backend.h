#pragma once

#include "qvcache/backend_interface.h"
#include "greator/pq_flash_index.h"
#include "greator/aux_utils.h"
#include "greator/linux_aligned_file_reader.h"
#include "greator/utils.h"
#include "diskann/distance.h"

namespace qvcache {

// Helper function to convert diskann::Metric to greator::Metric
inline greator::Metric convert_diskann_to_greator_metric(diskann::Metric diskann_metric) {
    switch (diskann_metric) {
        case diskann::Metric::L2:
            return greator::Metric::L2;
        case diskann::Metric::INNER_PRODUCT:
            return greator::Metric::INNER_PRODUCT;
        case diskann::Metric::COSINE:
            return greator::Metric::COSINE;
        case diskann::Metric::FAST_L2:
            return greator::Metric::FAST_L2;
        default:
            return greator::Metric::L2; // Default to L2 for unknown metrics
    }
}

template <typename T, typename TagT = uint32_t>
class GreatorBackend : public BackendInterface<T, TagT> {
public:
    std::unique_ptr<greator::PQFlashIndex<T, TagT>> disk_index;
    std::string data_path;
    std::string disk_index_prefix;
    uint32_t R, disk_L, B, M, build_threads;
    uint32_t beamwidth;
    int disk_index_already_built;
    greator::Metric metric;

    GreatorBackend(const std::string& data_path, const std::string& disk_index_prefix, 
                   uint32_t R, uint32_t disk_L, uint32_t B, uint32_t M, 
                   uint32_t build_threads, int disk_index_already_built, uint32_t beamwidth,
                   diskann::Metric metric = diskann::Metric::L2)
        : data_path(data_path), disk_index_prefix(disk_index_prefix),
          R(R), disk_L(disk_L), B(B), M(M), build_threads(build_threads),
          disk_index_already_built(disk_index_already_built), beamwidth(beamwidth),
          metric(convert_diskann_to_greator_metric(metric))
    {
        // Build disk index if not already built
        if (disk_index_already_built == 0) {
            std::string disk_index_params = std::to_string(R) + " " + std::to_string(disk_L) + " " + std::to_string(B) + " " + std::to_string(M) + " " + std::to_string(build_threads);
            greator::build_disk_index<T>(data_path.c_str(), disk_index_prefix.c_str(), disk_index_params.c_str(), this->metric, false);
        }

        // Load disk index
        std::shared_ptr<greator::AlignedFileReader> reader = nullptr;
        reader.reset(new greator::LinuxAlignedFileReader());
        std::unique_ptr<greator::PQFlashIndex<T>> temp_disk_index(new greator::PQFlashIndex<T>(this->metric, reader, false, false));
        disk_index = std::move(temp_disk_index);
        disk_index->load(disk_index_prefix.c_str(), build_threads);
        
        // Cache vectors near the centroid of the disk index.
        std::vector<uint32_t> node_list;
        disk_index->cache_bfs_levels(500, node_list);
        disk_index->load_cache_list(node_list);
        node_list.clear();
        node_list.shrink_to_fit();

        std::cout << "GreatorBackend disk index loaded successfully!" << std::endl;
    }

    void search(
        const T *query,
        uint64_t K,
        TagT* result_tags,
        float* result_distances,
        void* search_parameters = nullptr,
        void* stats = nullptr) override {
        
        greator::QueryStats* greator_stats = static_cast<greator::QueryStats*>(stats);

        // Use stored disk_L and beamwidth from constructor
        disk_index->cached_beam_search(query, K, disk_L, result_tags, result_distances, beamwidth, greator_stats);
    }

    std::vector<std::vector<T>> fetch_vectors_by_ids(
        const std::vector<TagT> &ids) override {
        return disk_index->inflate_vectors_by_tags(ids);
    }
};

}
