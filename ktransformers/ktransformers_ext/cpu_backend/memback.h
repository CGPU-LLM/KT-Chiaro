#ifndef CPUINFER_BACKEND_MEMBACK_H
#define CPUINFER_BACKEND_MEMBACK_H

#include <vector>
#include <mutex>
#include <cstdint>
#include <memory>
#include "../operators/llamafile/moe_config.h"

namespace cpu_backend {

// 专家内存管理器，用于动态加载/卸载 MoE 专家权重
class ExpertMemoryManager {
public:
    explicit ExpertMemoryManager(const MOEConfig& config);
    ~ExpertMemoryManager();

    // 禁止拷贝和赋值
    ExpertMemoryManager(const ExpertMemoryManager&) = delete;
    ExpertMemoryManager& operator=(const ExpertMemoryManager&) = delete;

    // 加载指定专家，如果已加载则跳过
    void load(int expert_id);

    // 卸载指定专家，如果未加载则跳过
    void unload(int expert_id);

    // 获取已加载专家的权重基地址
    void* getGate(int expert_id);
    void* getUp(int expert_id);
    void* getDown(int expert_id);

private:
    struct Entry {
        bool loaded;
        void* gate;
        void* up;
        void* down;
        std::mutex mtx;  // 每个专家的锁，保护该专家的状态和指针
    };

    MOEConfig config_;
    std::vector<Entry> entries_;  // 大小 = expert_num
};

} // namespace cpu_backend

#endif // CPUINFER_BACKEND_MEMBACK_H