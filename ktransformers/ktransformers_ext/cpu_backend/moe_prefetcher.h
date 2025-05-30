#ifndef CPUINFER_MOE_PREFETCHER_H
#define CPUINFER_MOE_PREFETCHER_H

#include <vector>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <unordered_map>
#include <unordered_set>
#include "memback.h"

class MOEPrefetcher {
public:
    // 获取单例
    static MOEPrefetcher& getInstance();
    
    // 注册每一层的内存管理器
    void registerLayerManager(int layer_id, cpu_backend::ExpertMemoryManager* mgr);
    
    // 在开始计算某一层时调用，预取后续层权重
    void onLayerBegin(int layer_id);
    
    // 关闭预取线程池
    void shutdown();

private:
    MOEPrefetcher();
    ~MOEPrefetcher();
    MOEPrefetcher(const MOEPrefetcher&) = delete;
    MOEPrefetcher& operator=(const MOEPrefetcher&) = delete;

    int prefetch_depth_;  // 向后预取层数
    int num_threads_;     // I/O 线程数

    std::mutex mutex_;  // 保护 layer_mngr_ 和 scheduled_layers_
    std::unordered_map<int, cpu_backend::ExpertMemoryManager*> layer_mngr_;
    std::unordered_set<int> scheduled_layers_;

    // I/O 线程池
    std::vector<std::thread> io_threads_;
    std::queue<std::function<void()>> io_queue_;
    std::mutex io_mutex_;
    std::condition_variable io_cv_;
    bool stop_{false};
};

#endif // CPUINFER_MOE_PREFETCHER_H 