// 
// Author: Sukjoon Oh

#ifndef AKER_TIMER_H
#define AKER_TIMER_H

#include <ctime>
#include <cstdint>

namespace aker {

    /* General purpose timer
     */
    typedef struct timespec                             time_point_t;

    class ElapsedLatencyPair
    {
    private:
        time_point_t                                    start_time;
        time_point_t                                    end_time;

        double                                          elapsed_ms;
        double                                          elapsed_ns;

        std::uint64_t                                   aux_1;
        std::uint64_t                                   aux_2;

    public:
        
        ElapsedLatencyPair() noexcept;
        virtual ~ElapsedLatencyPair() = default;      

        void                                            start() noexcept;
        void                                            end() noexcept;

        void                                            elapsedMs() noexcept;
        void                                            elapsedNs() noexcept;

        void                                            setAux1(std::uint64_t aux) noexcept;
        void                                            setAux2(std::uint64_t aux) noexcept;

        std::uint64_t                                   getAux1() const noexcept;
        std::uint64_t                                   getAux2() const noexcept;

        double                                          getElapsedMs() const noexcept;
        double                                          getElapsedNs() const noexcept;
    };
}

#endif

