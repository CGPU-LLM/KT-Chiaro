#include "moe_prefetcher.h"
#include "../debug/debug.h"
#include "task_queue.h"
#include "../moe_tracker.h"
#include <algorithm>
#include <chrono>

using namespace cpu_backend;

// 单例获取
MOEPrefetcher& MOEPrefetcher::getInstance() {
    static MOEPrefetcher instance;
    return instance;
}

// 构造：创建 I/O Backend
MOEPrefetcher::MOEPrefetcher() {
    // 初始化 I/O 后端线程池及批量大小参数
    printf("Input prefetch_depth, num_threads and batch_size: ");
    scanf("%d %d %d", &prefetch_depth_, &num_threads_, &batch_size_);
    io_backend_ = std::make_unique<Backend>(num_threads_);
    // 初始化 TaskQueue，用于序列化层切换任务
    task_queue_ = std::make_unique<TaskQueue>();
}

// 析构：释放 I/O Backend
MOEPrefetcher::~MOEPrefetcher() {
    // 清理 TaskQueue 和 I/O 后端
    task_queue_.reset();
    io_backend_.reset();
}

// 注册某层的 ExpertMemoryManager
void MOEPrefetcher::registerLayerManager(int layer_id, ExpertMemoryManager* mgr) {
    debug_printf("[C++] registerLayerManager: layer_id = %d\n", layer_id);
    std::lock_guard<std::mutex> lock(mu_);
    layer_mngr_[layer_id] = mgr;
    layer_num_++;
}

// 当模型切换到 layer_id 层时调用：卸载过期层，预取后续层
void MOEPrefetcher::onLayerChanged(int layer_id) {
    debug_printf("[C++] onLayerChanged [begin]: layer_id = %d\n", layer_id);
    auto start = std::chrono::steady_clock::now();
    // 将层切换事件加入 TaskQueue，由工作线程依次处理
    task_queue_->enqueue([this, layer_id]() {
        std::lock_guard<std::mutex> lock(mu_);
        // 卸载 layer_id-1
        int unload_layer = (layer_id - 1 + layer_num_) % layer_num_;
        debug_printf("[C++] onLayerChanged [task]: unload_layer = %d\n", unload_layer);
        if (scheduled_layers_.erase(unload_layer)) {
            debug_printf("[C++] onLayerChanged [task]: unloading layer %d\n", unload_layer);
            auto mgr = layer_mngr_[unload_layer];
            io_backend_->enqueue_io_tasks(mgr->getExpertNum(), [mgr](int idx) { mgr->unload(idx); });
        }
        debug_printf("[C++] onLayerChanged [task]: unload_layer = %d\n", unload_layer);
        // 预取后续层
        for (int y = layer_id + 1; y <= layer_id + prefetch_depth_; ++y) {
            int ly = y % layer_num_;
            debug_printf("[C++] onLayerChanged [loop]: ly = %d\n", ly);
            if (!layer_mngr_.count(ly) || scheduled_layers_.count(ly)) continue;
            scheduled_layers_.insert(ly);
            auto mgr = layer_mngr_[ly];
            // 按批量大小分段调度批量加载任务
            int N = mgr->getExpertNum();
            int batch_size = this->batch_size_;
            int task_count = (N + batch_size - 1) / batch_size;
            io_backend_->enqueue_io_tasks(task_count, [mgr, ly, batch_size, N, this](int task_id) {
                int cur = moe_tracker::moe_tracker_get_current_layer();
                // 如果该层已经过时则跳过
                if (((ly - cur < 0) ? (ly - cur + layer_num_) : (ly - cur)) > prefetch_depth_ || ly == cur) {
                    debug_printf("[C++] onLayerChanged [lambda]: skip layer = %d, cur = %d, ly = %d, task_id = %d\n", ly, cur, ly, task_id);
                    return;
                }
                // 计算本批次专家索引范围
                int start = task_id * batch_size;
                int end = std::min(start + batch_size, N);
                if (start >= end) return;
                debug_printf("[C++] onLayerChanged [lambda]: batch prefetch layer = %d, task_id = %d, range [%d, %d)\n", ly, task_id, start, end);
                mgr->loadRange(start, end);
            });
        }
    });
    auto end = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    debug_printf("[C++] onLayerChanged [TIME]: %d ms\n", duration);
} 