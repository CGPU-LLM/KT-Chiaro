#ifndef CPUINFER_MOE_CONFIG_H
#define CPUINFER_MOE_CONFIG_H

#include "../../cpu_backend/backend.h"
#include "conversion.h"
#include "llama.cpp/ggml-impl.h"
#include "llama.cpp/ggml-quants.h"
#include "llama.cpp/ggml.h"
#include "llamafile/sgemm.h"
#include "shared_mem_buffer.h"

struct MOEConfig {
    int expert_num;
    int routed_expert_num;
    int hidden_size;
    int intermediate_size;
    int stride;
    int group_min_len;
    int group_max_len;
    void* gate_proj;
    void* up_proj;
    void* down_proj;
    ggml_type gate_type;
    ggml_type up_type;
    ggml_type down_type;
    ggml_type hidden_type;
    bool use_external_proj = false;
    std::string gate_proj_file;
    uint64_t gate_proj_offset = 0;
    std::string up_proj_file;
    uint64_t up_proj_offset = 0;
    std::string down_proj_file;
    uint64_t down_proj_offset = 0;
    int layer_id = -1;

    MOEConfig() {}

    MOEConfig(int expert_num, int routed_expert_num, int hidden_size, int intermediate_size, int stride, int group_min_len, int group_max_len, void* gate_ptr, void* up_ptr, void* down_ptr, ggml_type gate_type, ggml_type up_type, ggml_type down_type, ggml_type hidden_type, int layer_id = -1)
        : expert_num(expert_num),
          routed_expert_num(routed_expert_num),
          hidden_size(hidden_size),
          intermediate_size(intermediate_size),
          stride(stride),
          group_min_len(group_min_len),
          group_max_len(group_max_len),
          gate_proj(gate_ptr),
          up_proj(up_ptr),
          down_proj(down_ptr),
          gate_type(gate_type),
          up_type(up_type),
          down_type(down_type),
          hidden_type(hidden_type),
          use_external_proj(false),
          layer_id(layer_id) {}

    MOEConfig(int expert_num, int routed_expert_num, int hidden_size, int intermediate_size, int stride, int group_min_len, int group_max_len, const std::string& gate_file, uint64_t gate_offset, const std::string& up_file, uint64_t up_offset, const std::string& down_file, uint64_t down_offset, ggml_type gate_type, ggml_type up_type, ggml_type down_type, ggml_type hidden_type, int layer_id = -1)
        : expert_num(expert_num),
          routed_expert_num(routed_expert_num),
          hidden_size(hidden_size),
          intermediate_size(intermediate_size),
          stride(stride),
          group_min_len(group_min_len),
          group_max_len(group_max_len),
          gate_proj(nullptr),
          up_proj(nullptr),
          down_proj(nullptr),
          gate_type(gate_type),
          up_type(up_type),
          down_type(down_type),
          hidden_type(hidden_type),
          use_external_proj(true),
          gate_proj_file(gate_file),
          gate_proj_offset(gate_offset),
          up_proj_file(up_file),
          up_proj_offset(up_offset),
          down_proj_file(down_file),
          down_proj_offset(down_offset),
          layer_id(layer_id) {}
};

#endif
