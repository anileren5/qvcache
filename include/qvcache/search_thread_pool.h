#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <functional>
#include <immintrin.h>
#include <mutex>
#include <thread>
#include <vector>

#if defined(__linux__)
#include <linux/futex.h>
#include <sys/syscall.h>
#include <unistd.h>
#endif

namespace qvcache {

// Long-lived workers for one parallel mini-index search at a time.
// A condition variable here costs a few hundred microseconds, which is a
// large fraction of a hit. Workers sleep on a futex and the caller searches
// the first shard itself so that wake overlaps the heavy shard.
class SearchThreadPool {
  public:
    explicit SearchThreadPool(size_t thread_count);
    ~SearchThreadPool();

    SearchThreadPool(const SearchThreadPool&) = delete;
    SearchThreadPool& operator=(const SearchThreadPool&) = delete;

    // Runs fn(0) .. fn(n-1). fn(0) runs on the caller. The rest run on the pool.
    // The first exception is rethrown after every worker has finished.
    void parallel_for(size_t n, const std::function<void(size_t)>& fn);

    size_t size() const { return workers_.size(); }

  private:
    void worker_main();
    static void futex_wait(std::atomic<uint32_t>* addr, uint32_t expected);
    static void futex_wake(std::atomic<uint32_t>* addr, int n);

    std::vector<std::thread> workers_;
    std::mutex call_mu_;
    std::mutex error_mu_;
    std::exception_ptr error_;

    std::atomic<const std::function<void(size_t)>*> job_{nullptr};
    std::atomic<size_t> job_n_{0};
    std::atomic<size_t> next_{0};
    std::atomic<size_t> arrived_{0};
    std::atomic<uint32_t> epoch_{0};
    std::atomic<bool> stop_{false};
};

inline void SearchThreadPool::futex_wait(std::atomic<uint32_t>* addr, uint32_t expected) {
#if defined(__linux__)
    syscall(SYS_futex, reinterpret_cast<uint32_t*>(addr), FUTEX_WAIT_PRIVATE, expected, nullptr, nullptr, 0);
#else
    while (addr->load(std::memory_order_acquire) == expected) {
        std::this_thread::yield();
    }
#endif
}

inline void SearchThreadPool::futex_wake(std::atomic<uint32_t>* addr, int n) {
#if defined(__linux__)
    syscall(SYS_futex, reinterpret_cast<uint32_t*>(addr), FUTEX_WAKE_PRIVATE, n, nullptr, nullptr, 0);
#else
    (void)addr;
    (void)n;
#endif
}

inline SearchThreadPool::SearchThreadPool(size_t thread_count) {
    static_assert(std::atomic<uint32_t>::is_always_lock_free, "futex word must be lock-free");
    workers_.reserve(thread_count);
    for (size_t i = 0; i < thread_count; ++i) {
        workers_.emplace_back([this] { worker_main(); });
    }
}

inline SearchThreadPool::~SearchThreadPool() {
    stop_.store(true, std::memory_order_release);
    epoch_.fetch_add(1, std::memory_order_release);
    futex_wake(&epoch_, static_cast<int>(workers_.size()));
    for (auto& worker : workers_) {
        if (worker.joinable()) {
            worker.join();
        }
    }
}

inline void SearchThreadPool::parallel_for(size_t n, const std::function<void(size_t)>& fn) {
    if (n == 0) {
        return;
    }
    if (n == 1 || workers_.empty()) {
        for (size_t i = 0; i < n; ++i) {
            fn(i);
        }
        return;
    }

    std::lock_guard<std::mutex> call(call_mu_);
    {
        std::lock_guard<std::mutex> lock(error_mu_);
        error_ = nullptr;
    }
    // Workers take ids 1..n-1. The caller keeps id 0, the fullest shard.
    next_.store(1, std::memory_order_relaxed);
    arrived_.store(workers_.size(), std::memory_order_relaxed);
    job_n_.store(n, std::memory_order_relaxed);
    job_.store(&fn, std::memory_order_release);
    epoch_.fetch_add(1, std::memory_order_release);
    futex_wake(&epoch_, static_cast<int>(workers_.size()));

    try {
        fn(0);
    } catch (...) {
        std::lock_guard<std::mutex> lock(error_mu_);
        if (!error_) {
            error_ = std::current_exception();
        }
    }

    while (arrived_.load(std::memory_order_acquire) != 0) {
        _mm_pause();
    }
    job_.store(nullptr, std::memory_order_release);

    std::exception_ptr error;
    {
        std::lock_guard<std::mutex> lock(error_mu_);
        error = error_;
        error_ = nullptr;
    }
    if (error) {
        std::rethrow_exception(error);
    }
}

inline void SearchThreadPool::worker_main() {
    uint32_t seen = 0;
    while (true) {
        uint32_t gen = epoch_.load(std::memory_order_acquire);
        if (gen == seen) {
            for (int spin = 0; spin < 64 && epoch_.load(std::memory_order_relaxed) == seen; ++spin) {
                _mm_pause();
            }
            gen = epoch_.load(std::memory_order_acquire);
            if (gen == seen) {
                if (stop_.load(std::memory_order_acquire)) {
                    return;
                }
                futex_wait(&epoch_, seen);
                continue;
            }
        }
        if (stop_.load(std::memory_order_acquire)) {
            return;
        }
        seen = gen;

        const std::function<void(size_t)>* job = job_.load(std::memory_order_acquire);
        const size_t n = job_n_.load(std::memory_order_acquire);
        if (job != nullptr) {
            while (true) {
                const size_t id = next_.fetch_add(1, std::memory_order_relaxed);
                if (id >= n) {
                    break;
                }
                try {
                    (*job)(id);
                } catch (...) {
                    std::lock_guard<std::mutex> lock(error_mu_);
                    if (!error_) {
                        error_ = std::current_exception();
                    }
                    break;
                }
            }
        }

        arrived_.fetch_sub(1, std::memory_order_acq_rel);
    }
}

}  // namespace qvcache
