#pragma once

#include <vector>
#include <queue>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <functional>
#include <atomic>
#include <memory>
#include <future>

// Forward declaration - users should include the appropriate LRU cache header
namespace qvcache {
    template<typename TagT> class LRUCache;
}

namespace qvcache {

/**
 * Asynchronous thread pool for LRU metadata updates
 * Allows non-blocking LRU access operations by queuing them for background processing
 */
template<typename TagT>
class LRUThreadPool {
public:
    explicit LRUThreadPool(size_t num_threads = 4) 
        : stop_(false), active_threads_(0) {
        start_workers(num_threads);
    }

    ~LRUThreadPool() {
        stop();
    }

    // Disable copy and assign
    LRUThreadPool(const LRUThreadPool&) = delete;
    LRUThreadPool& operator=(const LRUThreadPool&) = delete;

    /**
     * Asynchronously access a single tag in the LRU cache
     * @param lru_cache Reference to the LRU cache
     * @param tag The tag to access
     * @return Future that will be set when the operation completes
     */
    std::future<bool> async_access(LRUCache<TagT>& lru_cache, const TagT& tag) {
        auto promise = std::make_shared<std::promise<bool>>();
        auto future = promise->get_future();

        auto task = [&lru_cache, tag, promise]() {
            try {
                bool result = lru_cache.access(tag);
                promise->set_value(result);
            } catch (...) {
                promise->set_exception(std::current_exception());
            }
        };

        enqueue_task(std::move(task));
        return future;
    }

    /**
     * Asynchronously access multiple tags in the LRU cache
     * @param lru_cache Reference to the LRU cache
     * @param tags Vector of tags to access
     * @return Future that will be set when all operations complete
     */
    std::future<std::vector<bool>> async_access_batch(
        LRUCache<TagT>& lru_cache, 
        const std::vector<TagT>& tags) {
        
        auto promise = std::make_shared<std::promise<std::vector<bool>>>();
        auto future = promise->get_future();

        auto task = [&lru_cache, tags, promise]() {
            try {
                std::vector<bool> results;
                results.reserve(tags.size());
                
                for (const auto& tag : tags) {
                    results.push_back(lru_cache.access(tag));
                }
                
                promise->set_value(std::move(results));
            } catch (...) {
                promise->set_exception(std::current_exception());
            }
        };

        enqueue_task(std::move(task));
        return future;
    }

    /**
     * Asynchronously access tags from a pointer array
     * @param lru_cache Reference to the LRU cache
     * @param tags_ptr Pointer to array of tags
     * @param count Number of tags to access
     * @return Future that will be set when all operations complete
     */
    std::future<std::vector<bool>> async_access_array(
        LRUCache<TagT>& lru_cache,
        const TagT* tags_ptr,
        size_t count) {
        
        auto promise = std::make_shared<std::promise<std::vector<bool>>>();
        auto future = promise->get_future();

        auto task = [&lru_cache, tags_ptr, count, promise]() {
            try {
                std::vector<bool> results;
                results.reserve(count);
                
                for (size_t i = 0; i < count; ++i) {
                    results.push_back(lru_cache.access(tags_ptr[i]));
                }
                
                promise->set_value(std::move(results));
            } catch (...) {
                promise->set_exception(std::current_exception());
            }
        };

        enqueue_task(std::move(task));
        return future;
    }

    /**
     * Stop the thread pool and wait for all threads to finish
     */
    void stop() {
        {
            std::lock_guard<std::mutex> lock(queue_mutex_);
            stop_ = true;
        }
        condition_.notify_all();
        
        for (auto& worker : workers_) {
            if (worker.joinable()) {
                worker.join();
            }
        }
    }

    /**
     * Get the number of active threads
     */
    size_t active_threads() const {
        return active_threads_.load();
    }

    /**
     * Get the number of queued tasks
     */
    size_t queued_tasks() const {
        std::lock_guard<std::mutex> lock(queue_mutex_);
        return tasks_.size();
    }

private:
    void start_workers(size_t num_threads) {
        for (size_t i = 0; i < num_threads; ++i) {
            workers_.emplace_back([this] { worker_loop(); });
        }
    }

    void worker_loop() {
        while (true) {
            std::function<void()> task;
            
            {
                std::unique_lock<std::mutex> lock(queue_mutex_);
                condition_.wait(lock, [this] { 
                    return stop_ || !tasks_.empty(); 
                });
                
                if (stop_ && tasks_.empty()) {
                    return;
                }
                
                if (!tasks_.empty()) {
                    task = std::move(tasks_.front());
                    tasks_.pop();
                }
            }
            
            if (task) {
                active_threads_.fetch_add(1);
                task();
                active_threads_.fetch_sub(1);
            }
        }
    }

    void enqueue_task(std::function<void()> task) {
        {
            std::lock_guard<std::mutex> lock(queue_mutex_);
            if (stop_) {
                throw std::runtime_error("Thread pool is stopped");
            }
            tasks_.emplace(std::move(task));
        }
        condition_.notify_one();
    }

    std::vector<std::thread> workers_;
    std::queue<std::function<void()>> tasks_;
    mutable std::mutex queue_mutex_;
    std::condition_variable condition_;
    std::atomic<bool> stop_;
    std::atomic<size_t> active_threads_;
};

} // namespace qvcache

/*
Example usage:
#include "lru_async_pool.h"
#include "lru_cache.h"

// Create LRU cache and async pool
qvcache::LRUCache<int> lru_cache(1000);
auto async_pool = std::make_unique<qvcache::LRUAsyncPool<int>>(4);

// During search, instead of blocking LRU updates:
// OLD: for (size_t i = 0; i < K; i++) { lru_metadata.access(query_result_tags_ptr[i]); }

// NEW: Asynchronous non-blocking updates
auto future = async_pool->async_access_array(lru_cache, query_result_tags_ptr, K);

// Continue with search operations...
// The LRU updates will happen in the background

// Optionally wait for completion later:
// auto results = future.get();
*/ 