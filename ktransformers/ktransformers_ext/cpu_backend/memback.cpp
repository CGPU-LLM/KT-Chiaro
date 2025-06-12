#include "../operators/llamafile/moe.h"
#include "memback.h"
#include <cstdio>
#include <cstdlib>
#include <cassert>
#include "../debug/debug.h"
#include "../moe_tracker.h"
#include <fcntl.h>
#include <unistd.h>
#include <ctime>
#include <cstring>  // for memcpy

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
    srand(time(nullptr));
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
    // if(rand() % 5 == 0) print_info();
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
    // if(rand() % 5 == 0) print_info();
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

void ExpertMemoryManager::print_info() {
    std::lock_guard<std::mutex> lock(mtx_);
    // printf("[C++] ExpertMemoryManager print_info: expert_num = %d\n", config_.expert_num);
    debug_printf("load info: gate = %d, up = %d, down = %d\n", gate_fd, up_fd, down_fd);
    for (int i = 0; i < config_.expert_num; ++i) {
        std::lock_guard<std::mutex> lock(entries_[i].mtx);
        debug_printf("[%d: %d], ", i, entries_[i].loaded);
    }
    debug_printf("\n");
}

// 添加批量加载连续专家的实现
void ExpertMemoryManager::loadRange(int start_expert_id, int end_expert_id_exclusive) {
    debug_printf("[C++] ExpertMemoryManager loadRange: [start, end) = [%d, %d)\n", start_expert_id, end_expert_id_exclusive);
    debug_printf("[C++] Now calculating layer: %d\n", moe_tracker::moe_tracker_get_current_layer());
    // 边界检查
    if (start_expert_id < 0) start_expert_id = 0;
    if (end_expert_id_exclusive > config_.expert_num) end_expert_id_exclusive = config_.expert_num;
    int count = end_expert_id_exclusive - start_expert_id;
    if (count <= 0) return;
    // 计算单个专家的大小
    size_t gate_size = (size_t)config_.intermediate_size * config_.hidden_size * ggml_type_size(config_.gate_type) / ggml_blck_size(config_.gate_type);
    size_t up_size = gate_size;
    size_t down_size = (size_t)config_.hidden_size * config_.intermediate_size * ggml_type_size(config_.down_type) / ggml_blck_size(config_.down_type);
    // 总大小
    size_t total_gate = gate_size * count;
    size_t total_up = up_size * count;
    size_t total_down = down_size * count;
    if (config_.use_external_proj) {
        // 批量 pread
        void* gate_buf = malloc(total_gate);
        pread(gate_fd, gate_buf, total_gate, config_.gate_proj_offset + (uint64_t)start_expert_id * gate_size);
        void* up_buf = malloc(total_up);
        pread(up_fd, up_buf, total_up, config_.up_proj_offset + (uint64_t)start_expert_id * up_size);
        void* down_buf = malloc(total_down);
        pread(down_fd, down_buf, total_down, config_.down_proj_offset + (uint64_t)start_expert_id * down_size);
        // 分发到各专家
        for (int i = 0; i < count; ++i) {
            debug_printf("[C++] loadRange [loop]: i = %d, id = %d\n", i, start_expert_id + i);
            int id = start_expert_id + i;
            auto& ent = entries_[id];
            std::lock_guard<std::mutex> lock(ent.mtx);
            if (ent.loaded) {
                debug_printf("[C++] loadRange [loop]: id = %d, ALREADY loaded\n", id);
                continue;
            }
            debug_printf("[C++] loadRange [loop]: id = %d, LOADING\n", id);
            ent.gate = malloc(gate_size);
            memcpy(ent.gate, (char*)gate_buf + (size_t)i * gate_size, gate_size);
            ent.up = malloc(up_size);
            memcpy(ent.up, (char*)up_buf + (size_t)i * up_size, up_size);
            ent.down = malloc(down_size);
            memcpy(ent.down, (char*)down_buf + (size_t)i * down_size, down_size);
            ent.loaded = true;
        }
        free(gate_buf);
        free(up_buf);
        free(down_buf);
    } else {
        // 批量 fread
        FILE* gf = fopen(config_.gate_proj_file.c_str(), "rb");
        fseek(gf, config_.gate_proj_offset + (uint64_t)start_expert_id * gate_size, SEEK_SET);
        void* gate_buf = malloc(total_gate);
        fread(gate_buf, 1, total_gate, gf);
        fclose(gf);
        FILE* uf = fopen(config_.up_proj_file.c_str(), "rb");
        fseek(uf, config_.up_proj_offset + (uint64_t)start_expert_id * up_size, SEEK_SET);
        void* up_buf = malloc(total_up);
        fread(up_buf, 1, total_up, uf);
        fclose(uf);
        FILE* df = fopen(config_.down_proj_file.c_str(), "rb");
        fseek(df, config_.down_proj_offset + (uint64_t)start_expert_id * down_size, SEEK_SET);
        void* down_buf = malloc(total_down);
        fread(down_buf, 1, total_down, df);
        fclose(df);
        // 分发到各专家
        for (int i = 0; i < count; ++i) {
            int id = start_expert_id + i;
            auto& ent = entries_[id];
            std::lock_guard<std::mutex> lock(ent.mtx);
            if (ent.loaded) continue;
            ent.gate = malloc(gate_size);
            memcpy(ent.gate, (char*)gate_buf + (size_t)i * gate_size, gate_size);
            ent.up = malloc(up_size);
            memcpy(ent.up, (char*)up_buf + (size_t)i * up_size, up_size);
            ent.down = malloc(down_size);
            memcpy(ent.down, (char*)down_buf + (size_t)i * down_size, down_size);
            ent.loaded = true;
        }
        free(gate_buf);
        free(up_buf);
        free(down_buf);
    }
}

} // namespace cpu_backend