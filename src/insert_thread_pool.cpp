#include "qvcache/insert_thread_pool.h"

namespace qvcache {

template <typename T, typename TagT>
InsertThreadPool<T, TagT>::InsertThreadPool(size_t thread_count, TaskFn task_fn, ThetaUpdateFn theta_update_fn)
    : stop(false), task_function(std::move(task_fn)), theta_update_function(std::move(theta_update_fn)) {
    for (size_t i = 0; i < thread_count; ++i) {
        workers.emplace_back([this] {
            while (true) {
                std::function<void()> task;
                {
                    std::unique_lock<std::mutex> lock(mtx);
                    cv.wait(lock, [this] {
                        return stop || !tasks.empty();
                    });
                    if (stop && tasks.empty()) return;
                    task = std::move(tasks.front());
                    tasks.pop();
                }
                task();
            }
        });
    }
}

template <typename T, typename TagT>
void InsertThreadPool<T, TagT>::submit(std::unique_ptr<diskann::AbstractIndex>& index,
                                       std::vector<TagT> to_be_inserted,
                                       const std::string& data_path,
                                       size_t dim,
                                       uint32_t K,
                                       float query_distance,
                                       T* query_ptr) {
    {
        std::unique_lock<std::mutex> lock(mtx);
        tasks.emplace([=, &index]() mutable {
            // Execute the insertion task
            task_function(index, std::move(to_be_inserted), data_path, dim, K, query_distance);
            
            // After insertion is completed, perform theta update if query_ptr is provided and theta_update_function exists
            if (query_ptr != nullptr && theta_update_function) {
                // Perform theta update with the provided query pointer
                theta_update_function(query_ptr, K, query_distance);
                
                // Free the query pointer after theta update
                diskann::aligned_free(query_ptr);
            } else if (query_ptr != nullptr) {
                // If query_ptr is provided but no theta_update_function, just free the pointer
                diskann::aligned_free(query_ptr);
            }
        });
    }
    cv.notify_one();
}

template <typename T, typename TagT>
InsertThreadPool<T, TagT>::~InsertThreadPool() {
    {
        std::unique_lock<std::mutex> lock(mtx);
        stop = true;
    }
    cv.notify_all();
    for (auto& thread : workers)
        thread.join();
}

template class qvcache::InsertThreadPool<float, uint32_t>;
template class qvcache::InsertThreadPool<int8_t, uint32_t>;
template class qvcache::InsertThreadPool<uint8_t, uint32_t>;


} 
