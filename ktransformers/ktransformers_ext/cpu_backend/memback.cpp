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
      down_pool_(pool_size_, nullptr) {
    printf("[C++]: input the size of memory pool: ");
    scanf("%d", &pool_size_);
    gate_pool_.resize(pool_size_);
    up_pool_.resize(pool_size_);
    down_pool_.resize(pool_size_);
    
    size_t gate_size = (size_t)config_.intermediate_size * config_.hidden_size *
        ggml_type_size(config_.gate_type) / ggml_blck_size(config_.gate_type);
    size_t up_size = gate_size;
    size_t down_size = (size_t)config_.hidden_size * config_.intermediate_size *
        ggml_type_size(config_.down_type) / ggml_blck_size(config_.down_type);
    for (int i = 0; i < pool_size_; ++i) {
        gate_pool_[i] = malloc(gate_size);
        up_pool_[i]   = malloc(up_size);
        down_pool_[i] = malloc(down_size);
    }
    lru_queue_ = new LRUQueue(config_.expert_num, pool_size_, gate_pool_, up_pool_, down_pool_);
}

ExpertMemoryManager::~ExpertMemoryManager() {
    for (int i = 0; i < pool_size_; ++i) {
        free(gate_pool_[i]);
        free(up_pool_[i]);
        free(down_pool_[i]);
    }
    delete lru_queue_;
}

void ExpertMemoryManager::load(int expert_id) {
    assert(expert_id >= 0 && expert_id < config_.expert_num);
    if(lru_queue_->update(expert_id))
        return;
    auto memPointer = lru_queue_->find(expert_id);
    size_t gate_size = (size_t)config_.intermediate_size * config_.hidden_size *
        ggml_type_size(config_.gate_type) / ggml_blck_size(config_.gate_type);
    uint64_t gate_offset = config_.gate_proj_offset + (uint64_t)expert_id * gate_size;
    FILE* gf = fopen(config_.gate_proj_file.c_str(), "rb"); 
    fseek(gf, gate_offset, SEEK_SET);
    fread(memPointer[0], 1, gate_size, gf); 
    fclose(gf);
    size_t up_size = gate_size;
    uint64_t up_offset = config_.up_proj_offset + (uint64_t)expert_id * up_size;
    FILE* uf = fopen(config_.up_proj_file.c_str(), "rb"); 
    fseek(uf, up_offset, SEEK_SET);
    fread(memPointer[1], 1, up_size, uf); 
    fclose(uf);
    size_t down_size = (size_t)config_.hidden_size * config_.intermediate_size *
        ggml_type_size(config_.down_type) / ggml_blck_size(config_.down_type);
    uint64_t down_offset = config_.down_proj_offset + (uint64_t)expert_id * down_size;
    FILE* df = fopen(config_.down_proj_file.c_str(), "rb"); 
    fseek(df, down_offset, SEEK_SET);
    fread(memPointer[2], 1, down_size, df); 
    fclose(df);
}

void ExpertMemoryManager::unload(int expert_id) {
    assert(expert_id >= 0 && expert_id < config_.expert_num);
}

void* ExpertMemoryManager::getGate(int expert_id) {
    assert(expert_id >= 0 && expert_id < config_.expert_num);
    return lru_queue_->find(expert_id)[0];
}

void* ExpertMemoryManager::getUp(int expert_id) {
    assert(expert_id >= 0 && expert_id < config_.expert_num);
    return lru_queue_->find(expert_id)[1];
}

void* ExpertMemoryManager::getDown(int expert_id) {
    assert(expert_id >= 0 && expert_id < config_.expert_num);
    return lru_queue_->find(expert_id)[2];
}

} // namespace cpu_backend
