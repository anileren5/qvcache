#pragma once

#include <cstddef>
#include <cstdint>

#include <memory>
#include <unordered_map>
#include <queue>

#include "ak_vector_slot.hh"

namespace aker
{
    /**
     * @brief Eviction strategy interface.
     */
    class EvictionStrategy
    {
    public:
        /**
         * @brief Constructor.
         */
        EvictionStrategy() noexcept = default;

        /**
         * @brief Virtual destructor for safe deletion via base pointer.
         */
        virtual ~EvictionStrategy() noexcept = default;

        EvictionStrategy(const EvictionStrategy&) = delete;
        EvictionStrategy& operator=(const EvictionStrategy&) = delete;

        /**
         * @brief Returns the next eviction candidate via output parameter.
         */
        virtual bool nextEvictCandidate(vector_id_t* candidate_key) noexcept = 0;

        /**
         * @brief Adds an eviction candidate.
         */
        virtual bool addEvictCandidate(vector_id_t candidate_key) noexcept = 0;

        /**
         * @brief Marks a candidate as recently accessed.
         */
        virtual bool recentlyAccessed(vector_id_t candidate_key) noexcept = 0;

        /**
         * @brief Convenience wrapper returning the next candidate key, or 0 on failure.
         */
        inline vector_id_t nextEvictCandidate() noexcept
        {
            vector_id_t key = 0;
            bool success = nextEvictCandidate(&key);
            return success ? key : 0;
        }

        /**
         * @brief Returns the current size of the eviction structure.
         */
        inline size_t getCurrSize() const noexcept
        {
            return size_;
        }

    protected:
        /**
         * @brief Current number of tracked candidates.
         */
        size_t size_ = 0;
    };

    /**
     * @brief FIFO eviction strategy.
     */
    class EvictionStrategyFifo final : public EvictionStrategy
    {
    public:
        /**
         * @brief Constructs a FIFO strategy.
         */
        EvictionStrategyFifo() noexcept
        {
            size_ = 0;
        }

        /**
         * @brief Destructor.
         */
        ~EvictionStrategyFifo() noexcept override = default;

        /**
         * @brief Pops the next candidate from the FIFO queue.
         */
        bool nextEvictCandidate(vector_id_t* candidate_key) noexcept override
        {
            if (fifo_queue_.empty())
                return false;

            vector_id_t key = fifo_queue_.front();
            fifo_queue_.pop();

            if (candidate_key != nullptr)
                *candidate_key = key;

            if (size_ > 0)
                size_--;

            return true;
        }

        /**
         * @brief Pushes a candidate into the FIFO queue.
         */
        bool addEvictCandidate(vector_id_t candidate_key) noexcept override
        {
            /* std::queue may allocate; guard against exceptions because this
            * method is declared noexcept.
            */
            try
            {
                fifo_queue_.push(candidate_key);
            }
            catch (...)
            {
                return false;
            }

            size_++;
            return true;
        }

        /**
         * @brief FIFO strategy does not track recently accessed items.
         */
        bool recentlyAccessed(vector_id_t candidate_key) noexcept override
        {
            (void)candidate_key;
            return false;
        }

    private:
        std::queue<vector_id_t> fifo_queue_;
    };
}
