#include "ak_approx_filter.hh"

#include "ak_logger.hh"

#include "utils/ak_stack_alloc.hh"

#include <algorithm>
#include <cassert>
#include <unordered_map>
#include <utility>

#include <faiss/IndexHNSW.h>
#include <faiss/IndexIDMap.h>

namespace aker
{
    namespace detail
    {
        /**
         * @brief Internal wrapper around a single FAISS HNSW index with tombstone deletes.
         *
         * This implementation intentionally does not remove IDs from the underlying HNSW graph,
         * because FAISS HNSW removal is not supported in a safe and general way. Instead, it
         * maintains an external registration map that acts as a tombstone filter.
         */
        class ApproxFilterHnsw2
        {
        public:
            /**
             * @brief Constructs a single HNSW generation.
             */
            ApproxFilterHnsw2(anns_cache_config_t parameter_info, faiss::MetricType metric) noexcept;

            /**
             * @brief Adds a representative vector into this generation.
             */
            void addVector(vector_view_t query) noexcept;

            /**
             * @brief Tombstones a list of IDs.
             */
            int deleteVectors(std::vector<vector_id_t>& vector_id_list) noexcept;

            /**
             * @brief Tombstones one ID.
             */
            int deleteVector(vector_id_t vector_id) noexcept;

            /**
             * @brief Searches the index.
             */
            void searchSimilarVectors(const float* x, faiss::idx_t k, float* distances, faiss::idx_t* labels) noexcept;

            /**
             * @brief Returns whether an ID is registered (not tombstoned).
             */
            bool isRegistered(vector_id_t vector_id) noexcept;

            /**
             * @brief Returns the number of registered representative vectors.
             */
            size_t getRepresentativeVectorNumber() noexcept;

            /**
             * @brief Returns how many vectors were added into this generation.
             */
            size_t getAddedCounts() noexcept;

            /**
             * @brief Clears the registration map and resets the underlying FAISS index.
             */
            void clear() noexcept;

            /**
             * @brief Returns a human-readable status string.
             */
            std::string getStatusText() noexcept;

        private:
            static constexpr int k_hnsw_m = 4;
            static constexpr int k_hnsw_ef_search = 8;
            static constexpr int k_hnsw_ef_construction = 16;

            anns_cache_config_t                    parameter_;

            size_t                                      add_count_;
            std::unordered_map<vector_id_t, bool>        reg_map_;
            std::unique_ptr<faiss::IndexIDMap>          hnsw_index_wrapper_;
        };

        ApproxFilterHnsw2::ApproxFilterHnsw2(
            anns_cache_config_t parameter_info,
            faiss::MetricType metric) noexcept
            : parameter_(parameter_info),
              add_count_(0),
              reg_map_(),
              hnsw_index_wrapper_(nullptr)
        {
            /* Builds a small HNSW index wrapped by IndexIDMap so external IDs are preserved.
             */
            faiss::IndexHNSWFlat* hnsw_index = new faiss::IndexHNSWFlat(parameter_.vector_format.dimension, k_hnsw_m, metric);
            hnsw_index->hnsw.efSearch = k_hnsw_ef_search;
            hnsw_index->hnsw.efConstruction = k_hnsw_ef_construction;

            hnsw_index_wrapper_.reset(new faiss::IndexIDMap(hnsw_index));
            reg_map_.clear();
        }

        void
        ApproxFilterHnsw2::addVector(vector_view_t query) noexcept
        {
            /* Registers the ID and inserts the float-converted vector into the FAISS index.
             */
            const auto emplace_result = reg_map_.emplace(query.vector_id, true);
            const bool inserted = emplace_result.second;
            if (!inserted)
            {
                assert(false);
                return;
            }

            StackFloatBuffer float_query_data(static_cast<size_t>(query.dimension));

            bool converted = query.transform_callback(
                query.vector_data,
                query.vector_in_bytes,
                query.dimension,
                float_query_data.data(),
                query.aux);
            (void)converted;

            faiss::idx_t faiss_id = static_cast<faiss::idx_t>(query.vector_id);
            add_count_++;

            hnsw_index_wrapper_->add_with_ids(1, float_query_data.data(), &faiss_id);
        }

        int
        ApproxFilterHnsw2::deleteVectors(std::vector<vector_id_t>& vector_id_list) noexcept
        {
            /* Applies tombstone deletes by removing IDs from the registration map.
             */
            int deleted = 0;

            for (vector_id_t vector_id : vector_id_list)
            {
                const auto it = reg_map_.find(vector_id);
                if (it == reg_map_.end())
                    continue;

                reg_map_.erase(it);
                ++deleted;
            }

            if (reg_map_.size() == 0)
                hnsw_index_wrapper_->reset();

            return deleted;
        }

        int
        ApproxFilterHnsw2::deleteVector(vector_id_t vector_id) noexcept
        {
            /* Applies a tombstone delete for a single ID.
             */
            const auto it = reg_map_.find(vector_id);
            if (it == reg_map_.end())
                return 0;

            reg_map_.erase(it);

            if (reg_map_.empty())
                hnsw_index_wrapper_->reset();

            return 1;
        }

        void
        ApproxFilterHnsw2::searchSimilarVectors(
            const float* x,
            faiss::idx_t k,
            float* distances,
            faiss::idx_t* labels) noexcept
        {
            /* Delegates to FAISS search. */
            hnsw_index_wrapper_->search(1, x, k, distances, labels);
        }

        bool
        ApproxFilterHnsw2::isRegistered(vector_id_t vector_id) noexcept
        {
            /* Checks tombstone map membership. */
            return (reg_map_.find(vector_id) != reg_map_.end());
        }

        size_t
        ApproxFilterHnsw2::getRepresentativeVectorNumber() noexcept
        {
            return reg_map_.size();
        }

        size_t
        ApproxFilterHnsw2::getAddedCounts() noexcept
        {
            return add_count_;
        }

        void
        ApproxFilterHnsw2::clear() noexcept
        {
            /* Resets this generation completely. */
            reg_map_.clear();
            hnsw_index_wrapper_->reset();
            add_count_ = 0;
        }

        std::string
        ApproxFilterHnsw2::getStatusText() noexcept
        {
            /* Builds a lightweight status summary. */
            std::string status_string;
            status_string += "ApproxFilterHnsw2 Status:\n";
            status_string += "  Registered: " + std::to_string(reg_map_.size()) + "\n";
            status_string += "  Added: " + std::to_string(add_count_) + "\n";
            status_string += "  Index ntotal: " + std::to_string(hnsw_index_wrapper_->ntotal) + "\n";

            return status_string;
        }

    } // namespace detail

    ApproxFilterDualHNSW2::ApproxFilterDualHNSW2(anns_cache_config_t parameter_info) noexcept
        : parameter_(parameter_info),
          filters_()
    {
        /* Initializes two independent generations.
         */
        faiss::MetricType metric_type = faiss::METRIC_L2;
        switch (parameter_.distance_metric)
        {
            case distance_metric_t::DISTANCE_METRIC_L2:
                metric_type = faiss::METRIC_L2;
                break;
            case distance_metric_t::DISTANCE_METRIC_IP:
                metric_type = faiss::METRIC_INNER_PRODUCT;
                break;
            default:
                assert(false);
        }

        filters_[0] = std::make_unique<detail::ApproxFilterHnsw2>(parameter_, metric_type);
        filters_[1] = std::make_unique<detail::ApproxFilterHnsw2>(parameter_, metric_type);
    }

    ApproxFilterDualHNSW2::~ApproxFilterDualHNSW2() noexcept = default;

    void
    ApproxFilterDualHNSW2::switchFilterUnsafe() noexcept
    {
        /* Swaps the active and standby generations. */
        filters_[0].swap(filters_[1]);
    }

    void
    ApproxFilterDualHNSW2::addVector(vector_view_t query) noexcept
    {
        /* Adds the representative vector into the primary generation.
         */
        filters_[0]->addVector(query);

        AKER_LOG_DEBUG << "[ApproxFilter] added representative: vector_id=" << query.vector_id;
    }

    int
    ApproxFilterDualHNSW2::deleteVector(vector_id_t vector_id) noexcept
    {
        /* Applies tombstone delete to both generations.
         */
        int total_deleted = 0;
        for (size_t i = 0; i < k_num_filters; i++)
            total_deleted += filters_[i]->deleteVector(vector_id);

        if (total_deleted > 0)
        {
            AKER_LOG_DEBUG << "[ApproxFilter] tombstoned vector: vector_id=" << vector_id;
        }

        return total_deleted;
    }

    int
    ApproxFilterDualHNSW2::deleteVectors(std::vector<vector_id_t>& vector_id_list) noexcept
    {
        /* Applies tombstone deletes to both generations.
         */
        int total_deleted = 0;
        for (size_t i = 0; i < k_num_filters; i++)
            total_deleted += filters_[i]->deleteVectors(vector_id_list);

        if (!vector_id_list.empty())
        {
            AKER_LOG_DEBUG << "[ApproxFilter] tombstoned vectors: count=" << vector_id_list.size();
        }

        return total_deleted;
    }

    void
    ApproxFilterDualHNSW2::searchSimilarVectors(
        const float* x,
        faiss::idx_t k,
        float* distances,
        faiss::idx_t* labels) noexcept
    {
        /* Searches both generations and merges results.
         *
         * The arrays are expected to have size `k * k_num_filters`.
         */
        const size_t search_number = static_cast<size_t>(k) * k_num_filters;

        std::vector<float> found_distances(search_number);
        std::vector<faiss::idx_t> found_labels(search_number);

        float* distance_local = found_distances.data();
        float* d0 = distance_local;
        float* d1 = distance_local + k;

        faiss::idx_t* labels_local = found_labels.data();
        faiss::idx_t* l0 = labels_local;
        faiss::idx_t* l1 = labels_local + k;

        filters_[0]->searchSimilarVectors(x, k, d0, l0);
        filters_[1]->searchSimilarVectors(x, k, d1, l1);

        /* Normalizes the distance direction for inner-product searches.
         */
        bool negate_distances = false;
        switch (parameter_.distance_metric)
        {
            case distance_metric_t::DISTANCE_METRIC_L2:
                break;
            case distance_metric_t::DISTANCE_METRIC_IP:
                negate_distances = true;
                break;
            default:
                assert(false);
        }

        if (negate_distances)
        {
            for (size_t i = 0; i < search_number; i++)
                distance_local[i] = -distance_local[i];
        }

        /* Applies tombstone filtering via INVALID_DISTANCE and sorts by distance.
         */
        std::vector<std::pair<float, faiss::idx_t>> results;
        results.reserve(search_number);

        for (size_t i = 0; i < search_number; i++)
        {
            const faiss::idx_t label = labels_local[i];

            bool is_valid_label = false;
            if (label >= 0)
            {
                const vector_id_t vector_id = static_cast<vector_id_t>(label);
                if (i < static_cast<size_t>(k))
                    is_valid_label = filters_[0]->isRegistered(vector_id);
                else
                    is_valid_label = filters_[1]->isRegistered(vector_id);
            }

            const float found_distance = is_valid_label ? distance_local[i] : k_invalid_distance;
            results.emplace_back(found_distance, label);
        }

        std::sort(
            results.begin(),
            results.end(),
            [](const std::pair<float, faiss::idx_t>& a, const std::pair<float, faiss::idx_t>& b)
            {
                return a.first < b.first;
            });

        for (size_t i = 0; i < search_number; i++)
        {
            distances[i] = results[i].first;
            labels[i] = results[i].second;
        }
    }

    void
    ApproxFilterDualHNSW2::rotateGeneration() noexcept
    {
        /* Clears the standby generation and swaps roles.
         */
        const size_t before_repr = filters_[0]->getRepresentativeVectorNumber();
        const size_t before_added = filters_[0]->getAddedCounts();

        filters_[1]->clear();
        switchFilterUnsafe();

        AKER_LOG_INFO << "[ApproxFilter] rotated generation: active_repr=" << before_repr
                     << " active_added=" << before_added;
    }

    size_t
    ApproxFilterDualHNSW2::getRepresentativeVectorNumber() noexcept
    {
        return filters_[0]->getRepresentativeVectorNumber() + filters_[1]->getRepresentativeVectorNumber();
    }

    size_t
    ApproxFilterDualHNSW2::getAddedCounts() noexcept
    {
        return filters_[0]->getAddedCounts() + filters_[1]->getAddedCounts();
    }

    std::string
    ApproxFilterDualHNSW2::getStatusText() noexcept
    {
        /* Dumps a simple status for both generations.
         */
        std::string status_string;

        status_string += "ApproxFilterDualHNSW2 Status:\n";
        status_string += "  Generation 0:\n";
        status_string += filters_[0]->getStatusText();
        status_string += "  Generation 1:\n";
        status_string += filters_[1]->getStatusText();

        return status_string;
    }

    bool
    ApproxFilterDualHNSW2::needSwitch() noexcept
    {
        /* Triggers rotation when the active generation becomes stale.
         */
        const size_t per_filter_entry_count =
            parameter_.capacity.pool_size / (parameter_.capacity.getSlotListSize() * k_num_filters);

        const size_t repr_entries = filters_[0]->getRepresentativeVectorNumber();
        const size_t curr_added = filters_[0]->getAddedCounts();

        if (repr_entries < per_filter_entry_count)
            return false;

        return ((repr_entries * 2) < curr_added);
    }

} // namespace aker
