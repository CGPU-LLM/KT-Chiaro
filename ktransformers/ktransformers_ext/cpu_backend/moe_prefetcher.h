#ifndef CPUINFER_MOE_PREFETCHER_H
#define CPUINFER_MOE_PREFETCHER_H

#include <vector>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <unordered_map>
#include <unordered_set>
#include "backend.h"
#include "memback.h"
#include <memory>
#include "task_queue.h"

class MOEPrefetcher {
public:
    // 获取单例
    static MOEPrefetcher& getInstance();
    
    // 注册每一层的内存管理器
    void registerLayerManager(int layer_id, cpu_backend::ExpertMemoryManager* mgr);
    
    // 当模型切换到某一层时调用：调度卸载过期层和预取新层
    void onLayerChanged(int layer_id);
    
    // 关闭预取线程池
    ~MOEPrefetcher();

private:
    MOEPrefetcher();
    MOEPrefetcher(const MOEPrefetcher&) = delete;
    MOEPrefetcher& operator=(const MOEPrefetcher&) = delete;

    int prefetch_depth_{2};  // 向后预取层数
    int num_threads_{8};     // I/O 线程数
    int batch_size_{3};      // 批量加载时每次的专家数量，默认3
    int layer_num_{0};       // 已注册的层数

    // 保护 layer_mngr_ 和 scheduled_layers_
    std::mutex mu_;
    // 每层对应的 ExpertMemoryManager
    std::unordered_map<int, cpu_backend::ExpertMemoryManager*> layer_mngr_;
    // 已调度（已加载）的层号集合
    std::unordered_set<int> scheduled_layers_;

    // I/O 后端线程池
    std::unique_ptr<Backend> io_backend_;

    // 序列化层切换任务的队列
    std::unique_ptr<TaskQueue> task_queue_;
};

#endif // CPUINFER_MOE_PREFETCHER_H 