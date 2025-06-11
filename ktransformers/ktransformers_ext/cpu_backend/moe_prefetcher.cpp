#include "moe_prefetcher.h"
#include "../debug/debug.h"
#include "../moe_tracker.h"
#include <iostream>

// 单例获取
MOEPrefetcher& MOEPrefetcher::getInstance() {
    static MOEPrefetcher instance;
    return instance;
}

// 构造：启动 I/O 线程池
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
                // 执行加载或卸载任务
                task();
            }
        });
    }
}

// 析构：停止线程池并等待线程退出
MOEPrefetcher::~MOEPrefetcher() {
    shutdown();
}

// 注册某层的 ExpertMemoryManager
void MOEPrefetcher::registerLayerManager(int layer_id, cpu_backend::ExpertMemoryManager* mgr) {
    debug_printf("[C++] registerLayerManager: layer_id = %d\n", layer_id);
    std::lock_guard<std::mutex> lock(mutex_);
    layer_mngr_[layer_id] = mgr;
    layer_num_++;
}

// 当模型切换到 layer_id 层时调用：卸载过期层，预取后续层
void MOEPrefetcher::onLayerChanged(int layer_id) {
    debug_printf("[C++] onLayerChanged [begin]: layer_id = %d\n", layer_id);
    std::lock_guard<std::mutex> lock(mutex_);
    
    // 1) 卸载所有小于等于 layer_id-1 的层
    // for (auto it = scheduled_layers_.begin(); it != scheduled_layers_.end();) {
    //     int ly = *it;
    //     debug_printf("[C++] onLayerChanged [unload]: ly = %d, layer_id = %d\n", ly, layer_id);
    //     if (ly <= layer_id - 1) {
    //         debug_printf("[C++] onLayerChanged unloading layer %d\n", ly);
    //         auto mgr = layer_mngr_[ly];
    //         // 异步卸载所有专家
    //         enqueueIOTask([mgr](int idx) { mgr->unload(idx); }, mgr->getExpertNum() - 1);
    //         it = scheduled_layers_.erase(it);
    //     } else {
    //         ++it;
    //     }
    // }

    int unload_layer = (layer_id - 1 + layer_num_) % layer_num_;
    debug_printf("[C++] onLayerChanged [unload]: unload_layer = %d\n", unload_layer);
    auto mgr = layer_mngr_[unload_layer];
    enqueueIOTask([mgr](int idx) { mgr->unload(idx); }, mgr->getExpertNum() - 1);
    assert(scheduled_layers_.count(unload_layer) == 1); // 这层必须要在scheduled_layers_中
    scheduled_layers_.erase(unload_layer);

    int depth = prefetch_depth_;
    int num = layer_num_;

    // 2) 预取 layer_id+1 ... layer_id+prefetch_depth_
    for (int Y = layer_id + 1; Y <= layer_id + prefetch_depth_; ++Y) {
        int y = Y % layer_num_;
        debug_printf("[C++] onLayerChanged [prefetch]: y = %d, layer_id = %d\n", y, layer_id);
        if (!layer_mngr_.count(y) || scheduled_layers_.count(y)) continue;
        scheduled_layers_.insert(y);
        auto mgr = layer_mngr_[y];
        // 异步加载所有专家
        debug_printf("[C++] onLayerChanged [Add to scheduled_layers_]: loading layer %d\n", y);
        enqueueIOTask(
            [mgr, y, depth, num](int idx) {
                int cur = moe_tracker::MoeTracker::getInstance().getCurrentLayer();
                if((y - cur + num) % num > depth || y == cur) {
                    debug_printf("[C++] in lambda : y = %d, cur = %d, depth = %d, idx = %d, return directly\n", y, cur, depth, idx);
                    return;
                }
                mgr->load(idx); 
            }, 
            mgr->getExpertNum()
        );
    }
}

// 批量派发 I/O 任务：fn(idx) 重复 count 次
void MOEPrefetcher::enqueueIOTask(std::function<void(int)> fn, int count) {
    debug_printf("[C++] enqueueIOTask: count = %d\n", count);
    std::lock_guard<std::mutex> lock(io_mutex_);
    for (int i = 0; i < count; ++i) {
        io_queue_.emplace([fn, i]() { fn(i); });
    }
    io_cv_.notify_all();
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