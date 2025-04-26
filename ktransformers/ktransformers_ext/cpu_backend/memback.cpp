#include "../operators/llamafile/moe.h"
#include "memback.h"
#include <cstdio>
#include <cstdlib>
#include <cassert>

namespace cpu_backend {

ExpertMemoryManager::ExpertMemoryManager(const MOEConfig& config)
    : config_(config),
      pool_size_(config.routed_expert_num),
      gate_pool_(pool_size_, nullptr),
      up_pool_(pool_size_, nullptr),
      down_pool_(pool_size_, nullptr),
      id2slot_(config.expert_num),
      next_(pool_size_),
      free_head_(-1) {
    // 初始化专家ID映射为未加载状态 (-1)
    for (int i = 0; i < config_.expert_num; ++i) {
        id2slot_[i].store(-1);
    }
    // 预分配各槽位缓冲区
    size_t gate_size = (size_t)config_.intermediate_size * config_.hidden_size *
        ggml_type_size(config_.gate_type) / ggml_blck_size(config_.gate_type);
    size_t up_size = gate_size;
    size_t down_size = (size_t)config_.hidden_size * config_.intermediate_size *
        ggml_type_size(config_.down_type) / ggml_blck_size(config_.down_type);
    for (int i = 0; i < pool_size_; ++i) {
        gate_pool_[i] = malloc(gate_size);
        up_pool_[i]   = malloc(up_size);
        down_pool_[i] = malloc(down_size);
        // 初始化链表 next 指针
        next_[i].store(i + 1 < pool_size_ ? i + 1 : -1);
    }
    free_head_.store(pool_size_ > 0 ? 0 : -1);
}

ExpertMemoryManager::~ExpertMemoryManager() {
    for (int i = 0; i < pool_size_; ++i) {
        free(gate_pool_[i]);
        free(up_pool_[i]);
        free(down_pool_[i]);
    }
}

void ExpertMemoryManager::load(int expert_id) {
    if (expert_id < 0 || expert_id >= config_.expert_num) return;
    // 原子标记为"正在加载"(-2)，若不是 -1 则已经加载或加载中
    int expect = -1;
    if (!id2slot_[expert_id].compare_exchange_strong(expect, -2)) {
        return;
    }
    // 弹出一个空闲槽位 (无锁栈)
    int slot;
    while (true) {
        int head = free_head_.load();
        assert(head >= 0);
        int next = next_[head].load();
        if (free_head_.compare_exchange_weak(head, next)) {
            slot = head;
            break;
        }
    }
    // 从文件读取到预分配缓冲区
    size_t gate_size = (size_t)config_.intermediate_size * config_.hidden_size *
        ggml_type_size(config_.gate_type) / ggml_blck_size(config_.gate_type);
    uint64_t gate_offset = config_.gate_proj_offset + (uint64_t)expert_id * gate_size;
    FILE* gf = fopen(config_.gate_proj_file.c_str(), "rb"); 
    fseek(gf, gate_offset, SEEK_SET);
    fread(gate_pool_[slot], 1, gate_size, gf); 
    fclose(gf);
    size_t up_size = gate_size;
    uint64_t up_offset = config_.up_proj_offset + (uint64_t)expert_id * up_size;
    FILE* uf = fopen(config_.up_proj_file.c_str(), "rb"); 
    fseek(uf, up_offset, SEEK_SET);
    fread(up_pool_[slot], 1, up_size, uf); 
    fclose(uf);
    size_t down_size = (size_t)config_.hidden_size * config_.intermediate_size *
        ggml_type_size(config_.down_type) / ggml_blck_size(config_.down_type);
    uint64_t down_offset = config_.down_proj_offset + (uint64_t)expert_id * down_size;
    FILE* df = fopen(config_.down_proj_file.c_str(), "rb"); 
    fseek(df, down_offset, SEEK_SET);
    fread(down_pool_[slot], 1, down_size, df); 
    fclose(df);
    // 完成加载，更新 id2slot_
    id2slot_[expert_id].store(slot);
}

void ExpertMemoryManager::unload(int expert_id) {
    if (expert_id < 0 || expert_id >= config_.expert_num) return;
    // 原子交换到未加载状态，获取旧槽位
    int slot = id2slot_[expert_id].exchange(-1);
    if (slot < 0) return;
    // 无锁压栈回收槽位
    int head;
    do {
        head = free_head_.load();
        next_[slot].store(head);
    } while (!free_head_.compare_exchange_weak(head, slot));
}

void* ExpertMemoryManager::getGate(int expert_id) {
    if (expert_id < 0 || expert_id >= config_.expert_num) return nullptr;
    int slot = id2slot_[expert_id].load();
    assert(slot >= 0);
    return gate_pool_[slot];
}

void* ExpertMemoryManager::getUp(int expert_id) {
    if (expert_id < 0 || expert_id >= config_.expert_num) return nullptr;
    int slot = id2slot_[expert_id].load();
    assert(slot >= 0);
    return up_pool_[slot];
}

void* ExpertMemoryManager::getDown(int expert_id) {
    if (expert_id < 0 || expert_id >= config_.expert_num) return nullptr;
    int slot = id2slot_[expert_id].load();
    assert(slot >= 0);
    return down_pool_[slot];
}

} // namespace cpu_backend
