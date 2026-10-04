// 
// Author: Sukjoon Oh

#ifndef AKER_SPIN_MUTEX_HH
#define AKER_SPIN_MUTEX_HH

#include <atomic>
#include <cstdint>
#include <thread>

namespace aker
{
    /* A small spin-based mutex for low-latency critical sections.
     * This type is compatible with std::lock_guard/std::unique_lock.
     */
    class SpinMutex
    {
    private:
        std::atomic_flag lock_flag = ATOMIC_FLAG_INIT;

        static inline void cpuRelax() noexcept
        {
        #if defined(__x86_64__) || defined(__i386__)
            __asm__ __volatile__("pause");
        #elif defined(__aarch64__) || defined(__arm__)
            __asm__ __volatile__("yield");
        #else
            (void)0;
        #endif
        }

    public:
        SpinMutex() noexcept = default;
        SpinMutex(const SpinMutex&) = delete;
        SpinMutex& operator=(const SpinMutex&) = delete;

        void lock() noexcept
        {
            /* Busy-wait with a small backoff to reduce contention.
             * The backoff preserves the spin-lock semantics while being
             * friendlier to hyper-threaded CPUs.
             */
            std::uint32_t spin_count = 0;
            while (lock_flag.test_and_set(std::memory_order_acquire))
            {
                cpuRelax();
                if ((++spin_count & 0x3FFu) == 0)
                    std::this_thread::yield();
            }
        }

        bool tryLock() noexcept
        {
            return !lock_flag.test_and_set(std::memory_order_acquire);
        }

        void unlock() noexcept
        {
            lock_flag.clear(std::memory_order_release);
        }
    };
}

#endif