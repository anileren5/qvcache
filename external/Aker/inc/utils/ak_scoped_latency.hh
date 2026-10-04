// 
// Author: Sukjoon Oh
//
// A small RAII helper for latency measurement.
// This keeps the original ElapsedLatencyPair-based profiling behaviour while
// removing repetitive start/end/push boilerplate.

#ifndef AKER_SCOPED_LATENCY_HH
#define AKER_SCOPED_LATENCY_HH

#include <vector>

#include "ak_timer.hh"

namespace aker
{
    class ScopedLatency
    {
    private:
        std::vector<ElapsedLatencyPair>*          sink;
        ElapsedLatencyPair                       pair;

    public:
        explicit ScopedLatency(std::vector<ElapsedLatencyPair>& sink_ref) noexcept
            : sink(&sink_ref)
        {
            /* Starts timing on construction.
             */
            pair.start();
        }

        ScopedLatency(const ScopedLatency&) = delete;
        ScopedLatency& operator=(const ScopedLatency&) = delete;

        ~ScopedLatency() noexcept
        {
            /* Ends timing and records to the sink on scope exit.
             */
            pair.end();
            sink->push_back(pair);
        }

        ElapsedLatencyPair& getPair() noexcept
        {
            return pair;
        }
    };
}

#endif
