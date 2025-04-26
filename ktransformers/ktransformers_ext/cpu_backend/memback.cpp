#include "../operators/llamafile/moe.h"
#include "memback.h"
#include <cstdio>
#include <cstdlib>
#include <cassert>

namespace cpu_backend {

ExpertMemoryManager::ExpertMemoryManager(const MOEConfig& config)
    : config_(config), entries_(config.expert_num) {
    // printf("[C++] ExpertMemoryManager constructor: num = %d, gate_proj_file = %s, up_proj_file = %s, down_proj_file = %s\n", config.expert_num, config.gate_proj_file.c_str(), config.up_proj_file.c_str(), config.down_proj_file.c_str());
    for (auto& e : entries_) {
        e.loaded = false;
        e.gate = nullptr;
        e.up = nullptr;
        e.down = nullptr;
    }
}

ExpertMemoryManager::~ExpertMemoryManager() {
    // printf("[C++] ExpertMemoryManager destructor\n");
    // 卸载所有已加载的专家
    for (int i = 0; i < config_.expert_num; ++i) {
        if (entries_[i].loaded) {
            unload(i);
        }
    }
}

void ExpertMemoryManager::load(int expert_id) {
    // printf("[C++] ExpertMemoryManager load: expert_id = %d\n", expert_id);
    if (expert_id < 0 || expert_id >= config_.expert_num) return;
    auto& ent = entries_[expert_id];
    std::lock_guard<std::mutex> lock(ent.mtx);
    if (ent.loaded) return;
    // printf("[C++] Need to load expert %d\n", expert_id);
    // Gate 大小
    size_t gate_size = (size_t)config_.intermediate_size * config_.hidden_size * 
        ggml_type_size(config_.gate_type) / ggml_blck_size(config_.gate_type);
    // 计算专家偏移
    uint64_t gate_offset = config_.gate_proj_offset + (uint64_t)expert_id * gate_size;
    FILE* gf = fopen(config_.gate_proj_file.c_str(), "rb");
    fseek(gf, gate_offset, SEEK_SET);
    ent.gate = malloc(gate_size);
    fread(ent.gate, 1, gate_size, gf);
    fclose(gf);

    // Up 大小
    size_t up_size = gate_size; // 与 Gate 相同形状
    uint64_t up_offset = config_.up_proj_offset + (uint64_t)expert_id * up_size;
    FILE* uf = fopen(config_.up_proj_file.c_str(), "rb");
    fseek(uf, up_offset, SEEK_SET);
    ent.up = malloc(up_size);
    fread(ent.up, 1, up_size, uf);
    fclose(uf);

    // Down 大小
    size_t down_size = (size_t)config_.hidden_size * config_.intermediate_size * 
        ggml_type_size(config_.down_type) / ggml_blck_size(config_.down_type);
    uint64_t down_offset = config_.down_proj_offset + (uint64_t)expert_id * down_size;
    FILE* df = fopen(config_.down_proj_file.c_str(), "rb");
    fseek(df, down_offset, SEEK_SET);
    ent.down = malloc(down_size);
    fread(ent.down, 1, down_size, df);
    fclose(df);

    ent.loaded = true;
}

void ExpertMemoryManager::unload(int expert_id) {
    // printf("[C++] ExpertMemoryManager unload: expert_id = %d\n", expert_id);
    if (expert_id < 0 || expert_id >= config_.expert_num) return;
    auto& ent = entries_[expert_id];
    std::lock_guard<std::mutex> lock(ent.mtx);
    if (!ent.loaded) return;
    free(ent.gate);
    free(ent.up);
    free(ent.down);
    ent.gate = ent.up = ent.down = nullptr;
    ent.loaded = false;
}

void* ExpertMemoryManager::getGate(int expert_id) {
    // printf("[C++] ExpertMemoryManager getGate: expert_id = %d (config file = %s)\n", expert_id, config_.gate_proj_file.c_str());
    if (expert_id < 0 || expert_id >= config_.expert_num) return nullptr;
    auto& ent = entries_[expert_id];
    std::lock_guard<std::mutex> lock(ent.mtx);
    assert(ent.loaded);
    return ent.gate;
}

void* ExpertMemoryManager::getUp(int expert_id) {
    // printf("[C++] ExpertMemoryManager getUp: expert_id = %d (config file = %s)\n", expert_id, config_.up_proj_file.c_str());
    if (expert_id < 0 || expert_id >= config_.expert_num) return nullptr;
    auto& ent = entries_[expert_id];
    std::lock_guard<std::mutex> lock(ent.mtx);
    assert(ent.loaded);
    return ent.up;
}

void* ExpertMemoryManager::getDown(int expert_id) {
    // printf("[C++] ExpertMemoryManager getDown: expert_id = %d (config file = %s)\n", expert_id, config_.down_proj_file.c_str());
    if (expert_id < 0 || expert_id >= config_.expert_num) return nullptr;
    auto& ent = entries_[expert_id];
    std::lock_guard<std::mutex> lock(ent.mtx);
    assert(ent.loaded);
    return ent.down;
}

} // namespace cpu_backend
