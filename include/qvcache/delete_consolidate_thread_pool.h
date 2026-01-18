#pragma once

#include <vector>
#include <queue>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <functional>
#include <atomic>

#include "diskann/index_factory.h"

namespace qvcache {

template <typename T, typename TagT = uint32_t>
class DeleteConsolidateThreadPool {
public:
    using TaskFn = std::function<void(std::unique_ptr<diskann::AbstractIndex>&, std::vector<TagT>)>;

    DeleteConsolidateThreadPool(size_t thread_count, TaskFn task_fn);
    ~DeleteConsolidateThreadPool();

    DeleteConsolidateThreadPool(const DeleteConsolidateThreadPool&) = delete;
    DeleteConsolidateThreadPool& operator=(const DeleteConsolidateThreadPool&) = delete;

    void submit(std::unique_ptr<diskann::AbstractIndex>& index,
                std::vector<TagT> to_be_deleted);

private:
    std::vector<std::thread> workers;
    std::queue<std::function<void()>> tasks;
    std::mutex mtx;
    std::condition_variable cv;
    std::atomic<bool> stop;
    TaskFn task_function;
};

} 

