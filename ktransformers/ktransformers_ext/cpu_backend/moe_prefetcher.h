#ifndef CPUINFER_MOE_PREFETCHER_H
#define CPUINFER_MOE_PREFETCHER_H

#include <vector>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <functional>
#include <unordered_map>
#include <unordered_set>
#include "memback.h"

class MOEPrefetcher {
public:
    // 获取单例
    static MOEPrefetcher& getInstance();
    
    // 注册每一层的内存管理器
    void registerLayerManager(int layer_id, cpu_backend::ExpertMemoryManager* mgr);
    
    // 在当前层切换时调用：卸载过期层，预取后续层权重
    void onLayerChanged(int layer_id);
    
    // 关闭预取线程池
    void shutdown();

private:
    MOEPrefetcher();
    ~MOEPrefetcher();
    MOEPrefetcher(const MOEPrefetcher&) = delete;
    MOEPrefetcher& operator=(const MOEPrefetcher&) = delete;

    int prefetch_depth_;  // 向后预取层数
    int num_threads_;     // I/O 线程数
    int layer_num_;       // 已注册的层数

    // 保护 layer_mngr_ 和 scheduled_layers_
    std::mutex mutex_;
    // 每层对应的 ExpertMemoryManager
    std::unordered_map<int, cpu_backend::ExpertMemoryManager*> layer_mngr_;
    // 已调度（已加载）的层号集合
    std::unordered_set<int> scheduled_layers_;

    // I/O 线程池
    std::vector<std::thread> io_threads_;
    std::queue<std::function<void()>> io_queue_;
    std::mutex io_mutex_;
    std::condition_variable io_cv_;
    bool stop_{false};
    // 用于批量派发 load/unload 任务
    void enqueueIOTask(std::function<void(int)> fn, int count);
};

#endif // CPUINFER_MOE_PREFETCHER_H 