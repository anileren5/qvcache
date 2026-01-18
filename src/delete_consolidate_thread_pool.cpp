#include "qvcache/delete_consolidate_thread_pool.h"

namespace qvcache {

template <typename T, typename TagT>
DeleteConsolidateThreadPool<T, TagT>::DeleteConsolidateThreadPool(size_t thread_count, TaskFn task_fn)
    : stop(false), task_function(std::move(task_fn)) {
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
void DeleteConsolidateThreadPool<T, TagT>::submit(std::unique_ptr<diskann::AbstractIndex>& index,
                                                  std::vector<TagT> to_be_deleted) {
    {
        std::unique_lock<std::mutex> lock(mtx);
        tasks.emplace([=, &index]() mutable {
            task_function(index, std::move(to_be_deleted));
        });
    }
    cv.notify_one();
}

template <typename T, typename TagT>
DeleteConsolidateThreadPool<T, TagT>::~DeleteConsolidateThreadPool() {
    {
        std::unique_lock<std::mutex> lock(mtx);
        stop = true;
    }
    cv.notify_all();
    for (auto& thread : workers)
        thread.join();
}

template class qvcache::DeleteConsolidateThreadPool<float, uint32_t>;
template class qvcache::DeleteConsolidateThreadPool<int8_t, uint32_t>;
template class qvcache::DeleteConsolidateThreadPool<uint8_t, uint32_t>;

} 
