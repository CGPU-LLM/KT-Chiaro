#include "../operators/llamafile/moe.h"
#include "memback.h"
#include <cstdio>
#include <cstdlib>
#include <cassert>
#include "../debug/debug.h"
#include "../moe_tracker.h"
#include <fcntl.h>
#include <unistd.h>

namespace cpu_backend {

ExpertMemoryManager::ExpertMemoryManager(const MOEConfig& config)
    : config_(config), entries_(config.expert_num) {
    // printf("[C++] ExpertMemoryManager constructor: num = %d, gate_proj_file = %s, up_proj_file = %s, down_proj_file = %s\n", config.expert_num, config.gate_proj_file.c_str(), config.up_proj_file.c_str(), config.down_proj_file.c_str());
    if (config_.use_external_proj) {
        gate_fd = open(config_.gate_proj_file.c_str(), O_RDONLY | O_CLOEXEC);
        up_fd   = open(config_.up_proj_file.c_str(),   O_RDONLY | O_CLOEXEC);
        down_fd = open(config_.down_proj_file.c_str(), O_RDONLY | O_CLOEXEC);
    } else {
        gate_fd = up_fd = down_fd = -1;
    }
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
    if (gate_fd >= 0) close(gate_fd);
    if (up_fd   >= 0) close(up_fd);
    if (down_fd >= 0) close(down_fd);
}

void ExpertMemoryManager::load(int expert_id) {
    debug_printf("[C++] ExpertMemoryManager load: expert_id = %d, layer_id = %d\n", expert_id, config_.layer_id);
    debug_printf("[C++] Now calculating layer: %d\n", moe_tracker::moe_tracker_get_current_layer());
    if (expert_id < 0 || expert_id >= config_.expert_num) return;
    auto& ent = entries_[expert_id];
    std::lock_guard<std::mutex> lock(ent.mtx);
    if (ent.loaded) {
        debug_printf("[C++] Expert %d ALREADY loaded\n\n", expert_id);
        return;
    }
    debug_printf("[C++] NEED to load expert %d\n\n", expert_id);
    if (config_.use_external_proj) {
        size_t gate_size = (size_t)config_.intermediate_size * config_.hidden_size * ggml_type_size(config_.gate_type) / ggml_blck_size(config_.gate_type);
        uint64_t gate_offset = config_.gate_proj_offset + (uint64_t)expert_id * gate_size;
        ent.gate = malloc(gate_size);
        pread(gate_fd, ent.gate, gate_size, gate_offset);

        size_t up_size = gate_size;
        uint64_t up_offset = config_.up_proj_offset + (uint64_t)expert_id * up_size;
        ent.up = malloc(up_size);
        pread(up_fd, ent.up, up_size, up_offset);

        size_t down_size = (size_t)config_.hidden_size * config_.intermediate_size * ggml_type_size(config_.down_type) / ggml_blck_size(config_.down_type);
        uint64_t down_offset = config_.down_proj_offset + (uint64_t)expert_id * down_size;
        ent.down = malloc(down_size);
        pread(down_fd, ent.down, down_size, down_offset);
    } else {
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
    }
    ent.loaded = true;
}

void ExpertMemoryManager::unload(int expert_id) {
    debug_printf("[C++] ExpertMemoryManager unload: expert_id = %d, layer_id = %d\n", expert_id, config_.layer_id);
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

int ExpertMemoryManager::getExpertNum() const {
    return config_.expert_num;
}

} // namespace cpu_backend