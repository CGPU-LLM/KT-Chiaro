#include "moe_prefetcher.h"
#include "../debug/debug.h"

// 单例获取
MOEPrefetcher& MOEPrefetcher::getInstance() {
    static MOEPrefetcher instance;
    return instance;
}

// 构造：启动 I/O 线程
MOEPrefetcher::MOEPrefetcher() {
    printf("Input prefetch_depth and num_threads: ");
    scanf("%d %d", &prefetch_depth_, &num_threads_);
    for (int i = 0; i < num_threads_; ++i) {
        io_threads_.emplace_back([this]() {
            while (true) {
                std::function<void()> task;
                {
                    std::unique_lock<std::mutex> lock(this->io_mutex_);
                    this->io_cv_.wait(lock, [this]() { return this->stop_ || !this->io_queue_.empty(); });
                    if (this->stop_ && this->io_queue_.empty()) {
                        return;
                    }
                    task = std::move(this->io_queue_.front());
                    this->io_queue_.pop();
                }
                // 执行加载任务
                task();
            }
        });
    }
}

// 析构：停止并等待线程退出
MOEPrefetcher::~MOEPrefetcher() {
    shutdown();
}

// 注册某层的 ExpertMemoryManager
void MOEPrefetcher::registerLayerManager(int layer_id, cpu_backend::ExpertMemoryManager* mgr) {
    debug_printf("[C++] registerLayerManager: layer_id = %d\n", layer_id);
    std::lock_guard<std::mutex> lock(mutex_);
    layer_mngr_[layer_id] = mgr;
}

// 在开始计算 layer_id 层时预取后续层
void MOEPrefetcher::onLayerBegin(int layer_id) {
    debug_printf("[C++] onLayerBegin: layer_id = %d\n", layer_id);
    std::lock_guard<std::mutex> lock(mutex_);
    for (int j = layer_id + 1; j <= layer_id + prefetch_depth_; ++j) {
        debug_printf("[C++] In Loop of onLayerBegin: j = %d\n", j);
        if (layer_mngr_.count(j) && scheduled_layers_.insert(j).second) {
            debug_printf("[C++] PREFETCHING layer %d\n", j);
            auto mgr = layer_mngr_[j];
            int expert_num = mgr->getExpertNum();
            // 将任务加入 I/O 队列
            {
                std::lock_guard<std::mutex> qlock(io_mutex_);
                debug_printf("[C++] Adding task to I/O queue\n");
                io_queue_.emplace([mgr, expert_num, j]() {
                    debug_printf("[C++] In task of I/O queue: expert_num = %d, layer_id = %d\n", expert_num, j);
                    for (int idx = 0; idx < expert_num; ++idx) {
                        mgr->load(idx);
                    }
                });
            }
            io_cv_.notify_one();
        }
    }
}

// 关闭 I/O 线程池
void MOEPrefetcher::shutdown() {
    {
        std::lock_guard<std::mutex> lock(io_mutex_);
        stop_ = true;
    }
    io_cv_.notify_all();
    for (auto& t : io_threads_) {
        if (t.joinable()) {
            t.join();
        }
    }
} 