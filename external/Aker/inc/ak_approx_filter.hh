#pragma once

#include <cstddef>
#include <cstdint>

#include <array>
#include <functional>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#include <faiss/Index.h>

#include "ak_anns_cache_config.hh"
#include "ak_vector_slot.hh"

namespace aker
{
    /**
     * @brief User-supplied routine that converts a raw vector blob into a contiguous float array.
     *
     * FAISS `Index` APIs expect input vectors as float32. This callback allows the caller to
     * provide vectors stored in other formats (e.g., int8/uint8) and convert them on demand.
     *
     * @param src            Raw vector blob.
     * @param src_size       Raw blob size in bytes.
     * @param dim            Vector dimension (# of elements).
     * @param dst            Output float32 array with length `dim`.
     * @param aux            Optional caller-provided scratch space (may be null).
     *
     * @return true on successful conversion.
     */
    using transform_callback_t =
        std::function<bool(vector_data_t* src, size_t src_size, std::uint32_t dim, float* dst, std::uint8_t* aux)>;

    /**
     * @brief Sentinel distance used to represent an invalid search candidate.
     */
    static constexpr float k_invalid_distance = std::numeric_limits<float>::max();

    /**
     * @brief View of a query vector and its conversion context.
     *
     * This is a non-owning container that packages raw vector bytes along with the conversion
     * callback required to produce float32 data for FAISS.
     */
    struct VectorView
    {
        vector_id_t           vector_id{0};
        vector_data_t*        vector_data{nullptr};
        std::uint32_t         dimension{0};
        std::uint32_t         vector_in_bytes{0};

        std::uint64_t         aux_data_1{0};
        std::uint64_t         aux_data_2{0};

        transform_callback_t  transform_callback{};
        std::uint8_t*         aux{nullptr};
    };

    /**
     * @brief Alias for the query vector view.
     */
    using vector_view_t = VectorView;

    namespace detail
    {
        class ApproxFilterHnsw2;
    }

    /**
     * @brief Dual-HNSW approximate filter with tombstone deletes and generation rotation.
     *
     * The filter maintains two small HNSW instances.
     * - Insertions go to the primary generation.
     * - Searches consult both generations and return the closest candidate(s).
     * - Deletions are handled as tombstones (ID deregistration). When the invalid ratio grows,
     *   the cache rotates generations by clearing the secondary and swapping roles.
     */
    class ApproxFilterDualHNSW2
    {
    public:
        /**
         * @brief Constructs a dual filter instance.
         */
        explicit ApproxFilterDualHNSW2(anns_cache_config_t parameter_info) noexcept;

        /**
         * @brief Destroys the dual filter instance.
         */
        ~ApproxFilterDualHNSW2() noexcept;

        ApproxFilterDualHNSW2(const ApproxFilterDualHNSW2&) = delete;
        ApproxFilterDualHNSW2& operator=(const ApproxFilterDualHNSW2&) = delete;

        /**
         * @brief Adds a representative vector into the primary filter.
         */
        void addVector(vector_view_t query) noexcept;

        /**
         * @brief Tombstones one vector ID across both generations.
         */
        int deleteVector(vector_id_t vector_id) noexcept;

        /**
         * @brief Tombstones vector IDs across both generations.
         *
         * @return Number of IDs removed from the registration maps.
         */
        int deleteVectors(std::vector<vector_id_t>& vector_id_list) noexcept;

        /**
         * @brief Searches both generations and returns merged candidates.
         *
         * @note The output arrays must be sized to `k * k_num_filters`.
         */
        void searchSimilarVectors(const float* x, faiss::idx_t k, float* distances, faiss::idx_t* labels) noexcept;

        /**
         * @brief Rotates generations by clearing the secondary and swapping roles.
         */
        void rotateGeneration() noexcept;

        /**
         * @brief Legacy name kept for compatibility.
         *
         * Historically, `clear()` performed a generation rotation, not a full reset.
         */
        void clear() noexcept { rotateGeneration(); }

        /**
         * @brief Returns the total number of registered representative vectors.
         */
        size_t getRepresentativeVectorNumber() noexcept;

        /**
         * @brief Returns the cumulative add count across both generations.
         */
        size_t getAddedCounts() noexcept;

        /**
         * @brief Returns a human-readable status string.
         */
        std::string getStatusText() noexcept;

        /**
         * @brief Returns whether a generation rotation is recommended.
         */
        bool needSwitch() noexcept;

    private:
        static constexpr size_t k_num_filters = 2;

        anns_cache_config_t parameter_;

        std::array<std::unique_ptr<detail::ApproxFilterHnsw2>, k_num_filters> filters_;

        /**
         * @brief Swaps the primary and secondary generations.
         */
        void switchFilterUnsafe() noexcept;
    };
}
