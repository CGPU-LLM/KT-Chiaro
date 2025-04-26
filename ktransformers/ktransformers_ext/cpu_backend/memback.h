#ifndef CPUINFER_BACKEND_MEMBACK_H
#define CPUINFER_BACKEND_MEMBACK_H

#include <vector>
#include <atomic>
#include <cstdint>
#include <memory>
#include "../operators/llamafile/moe_config.h"

namespace cpu_backend {

// 专家内存管理器，用于动态加载/卸载 MoE 专家权重，并使用固定槽位池优化 malloc/free
class ExpertMemoryManager {
public:
    explicit ExpertMemoryManager(const MOEConfig& config);
    ~ExpertMemoryManager();

    // 禁止拷贝和赋值
    ExpertMemoryManager(const ExpertMemoryManager&) = delete;
    ExpertMemoryManager& operator=(const ExpertMemoryManager&) = delete;

    void load(int expert_id);
    void unload(int expert_id);

    void* getGate(int expert_id);
    void* getUp(int expert_id);
    void* getDown(int expert_id);

private:
    MOEConfig config_;
    int pool_size_;
    std::vector<void*> gate_pool_; 
    std::vector<void*> up_pool_;
    std::vector<void*> down_pool_;
    std::vector<std::atomic<int>> id2slot_;
    std::vector<std::atomic<int>> next_;
    std::atomic<int> free_head_;
};

} // namespace cpu_backend

#endif // CPUINFER_BACKEND_MEMBACK_H
