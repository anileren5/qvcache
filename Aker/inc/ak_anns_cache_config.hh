#ifndef AKER_ANNS_CACHE_CONFIG_HH
#define AKER_ANNS_CACHE_CONFIG_HH

#include <cstddef>
#include <cstdint>

namespace aker
{
    /**
     * @brief Distance metric selection.
     */
    enum class distance_metric_t : std::uint8_t
    {
        DISTANCE_METRIC_L2 = 0,
        DISTANCE_METRIC_IP
    };

    /**
     * @brief System-wide vector format configuration.
     */
    struct VectorFormatConfig
    {
        std::uint32_t dimension{0};
        size_t vector_in_bytes{0};
    };

    /**
     * @brief Cache capacity configuration.
     */
    struct CacheCapacityConfig
    {
        size_t pool_size{0};
        size_t in_topk{0};
        size_t top_delta{0};

        /**
         * @brief Returns the total number of stored neighbors per entry.
         *
         * This replaces the legacy `slot_list_size` field.
         */
        size_t getSlotListSize() const noexcept { return in_topk + top_delta; }
    };

    /**
     * @brief Algorithmic tuning configuration.
     */
    struct AlgorithmTuningConfig
    {
        /* Global similarity threshold used by Proximity/Potluck modes.
         *
         * - Proximity Mode: fixed global threshold (no adaptive updates)
         * - Potluck Mode  : tuned global threshold (updated at put)
         * - Standard Mode : unused
         */
        float global_thresh{0.0f};

        /* Dropout rate for Potluck mode.
         *
         * Potluck uses random dropout on sim-hit to force revalidation.
         * This value is interpreted as a percentage in [0, 100].
         */
        float dropout{0.0f};

        float risk_thresh{0.0f};
        float alpha_tighten{0.0f};
        float alpha_loosen{0.0f};
    };

    /**
     * @brief Project-wide configuration used to construct the cache.
     */
    struct ANNSCacheConfig
    {
        VectorFormatConfig vector_format;
        CacheCapacityConfig capacity;
        AlgorithmTuningConfig tuning;

        distance_metric_t distance_metric{distance_metric_t::DISTANCE_METRIC_L2};
    };

    /**
     * @brief Backward-compatible typedef used by the cache constructors.
     */
    using anns_cache_config_t = ANNSCacheConfig;
}

#endif // AKER_ANNS_CACHE_CONFIG_HH
