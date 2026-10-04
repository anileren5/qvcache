#pragma once

#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

namespace qvcache {

// Long-lived workers for one parallel mini-index search at a time.
// Creating threads inside the query was slower than the search itself.
class SearchThreadPool {
  public:
    explicit SearchThreadPool(size_t thread_count);
    ~SearchThreadPool();

    SearchThreadPool(const SearchThreadPool&) = delete;
    SearchThreadPool& operator=(const SearchThreadPool&) = delete;

    // Runs fn(0) .. fn(n-1) on the pool. fn may run on several threads.
    // The first exception is rethrown after every worker has finished.
    void parallel_for(size_t n, const std::function<void(size_t)>& fn);

    size_t size() const { return workers_.size(); }

  private:
    void worker_main();

    std::vector<std::thread> workers_;
    std::mutex call_mu_;
    std::mutex mu_;
    std::condition_variable cv_work_;
    std::condition_variable cv_done_;
    const std::function<void(size_t)>* job_ = nullptr;
    size_t job_n_ = 0;
    size_t next_ = 0;
    size_t remaining_ = 0;
    uint64_t epoch_ = 0;
    bool stop_ = false;
    std::exception_ptr error_;
};

inline SearchThreadPool::SearchThreadPool(size_t thread_count) {
    workers_.reserve(thread_count);
    for (size_t i = 0; i < thread_count; ++i) {
        workers_.emplace_back([this] { worker_main(); });
    }
}

inline SearchThreadPool::~SearchThreadPool() {
    {
        std::lock_guard<std::mutex> lock(mu_);
        stop_ = true;
    }
    cv_work_.notify_all();
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
    if (workers_.empty()) {
        for (size_t i = 0; i < n; ++i) {
            fn(i);
        }
        return;
    }

    std::lock_guard<std::mutex> call(call_mu_);
    {
        std::lock_guard<std::mutex> lock(mu_);
        job_ = &fn;
        job_n_ = n;
        next_ = 0;
        remaining_ = workers_.size();
        error_ = nullptr;
        ++epoch_;
    }
    cv_work_.notify_all();

    std::unique_lock<std::mutex> lock(mu_);
    cv_done_.wait(lock, [&] { return remaining_ == 0; });
    job_ = nullptr;
    std::exception_ptr error = error_;
    error_ = nullptr;
    lock.unlock();
    if (error) {
        std::rethrow_exception(error);
    }
}

inline void SearchThreadPool::worker_main() {
    uint64_t seen = 0;
    while (true) {
        const std::function<void(size_t)>* job = nullptr;
        {
            std::unique_lock<std::mutex> lock(mu_);
            cv_work_.wait(lock, [&] { return stop_ || epoch_ != seen; });
            if (stop_) {
                return;
            }
            seen = epoch_;
            job = job_;
        }

        while (job != nullptr) {
            size_t id = 0;
            {
                std::lock_guard<std::mutex> lock(mu_);
                if (error_ || next_ >= job_n_) {
                    break;
                }
                id = next_++;
            }
            try {
                (*job)(id);
            } catch (...) {
                std::lock_guard<std::mutex> lock(mu_);
                if (!error_) {
                    error_ = std::current_exception();
                }
                break;
            }
        }

        {
            std::lock_guard<std::mutex> lock(mu_);
            if (--remaining_ == 0) {
                cv_done_.notify_one();
            }
        }
    }
}

}  // namespace qvcache
