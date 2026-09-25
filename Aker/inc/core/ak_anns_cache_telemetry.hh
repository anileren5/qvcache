#pragma once

#include <string>

#include "core/ak_anns_cache_context.hh"

namespace aker
{
    /**
     * @brief Telemetry module for ANNSCache.
     *
     * This module builds a concise cache summary and coordinates trace export.
     */
    class ANNSCacheTelemetry
    {
    public:
        /**
         * @brief Constructs the module with the shared cache context.
         */
        explicit ANNSCacheTelemetry(ANNSCacheContext* context) noexcept;

        /**
         * @brief Builds a human-readable status string.
         */
        std::string buildStatusText() noexcept;

        /**
         * @brief Builds a concise summary in CSV (key,value) form.
         */
        std::string buildSummaryCsv() noexcept;

        /**
         * @brief Exports all telemetry files under /tmp/aker_trace_<timestamp>_pid<PID>_gen<G>/.
         */
        void exportTraceToFiles() noexcept;

    private:
        ANNSCacheContext* context_;
    };
}
