#ifndef CPUINFER_BACKEND_MEMBACK_H
#define CPUINFER_BACKEND_MEMBACK_H

#include <vector>
#include <atomic>
#include <cstdint>
#include <memory>
#include "../operators/llamafile/moe_config.h"
#include "lru_queue.h"

namespace cpu_backend {

class ExpertMemoryManager {
public:
    explicit ExpertMemoryManager(const MOEConfig& config);
    ~ExpertMemoryManager();

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
    LRUQueue* lru_queue_;
};

} // namespace cpu_backend

#endif // CPUINFER_BACKEND_MEMBACK_H
