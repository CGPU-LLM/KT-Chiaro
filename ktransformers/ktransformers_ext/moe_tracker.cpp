#include "moe_tracker.h"
#include <cstring>
#include <thread>
#include "debug/debug.h"

namespace moe_tracker {

MoeTracker& MoeTracker::getInstance() {
    static MoeTracker instance;
    return instance;
}

void MoeTracker::initialize() {
    std::lock_guard<std::mutex> lock(mutex_);
    layers_.clear();
    layer_name_to_id_.clear();
    current_layer_id_ = -1;
}

int MoeTracker::registerLayer(const std::string& layer_name) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    debug_printf("registerLayer: %s\n", layer_name.c_str());
    // 检查是否已经注册过
    auto it = layer_name_to_id_.find(layer_name);
    if (it != layer_name_to_id_.end()) {
        debug_printf("layer already registered: %s\n", layer_name.c_str());
        return it->second;
    }
    
    // 注册新层
    int layer_id = static_cast<int>(layers_.size());
    layers_.push_back(layer_name);
    layer_name_to_id_[layer_name] = layer_id;
    debug_printf("registered layer: %s, id: %d\n", layer_name.c_str(), layer_id);
    return layer_id;
}

int MoeTracker::getLayerCount() {
    std::lock_guard<std::mutex> lock(mutex_);
    return static_cast<int>(layers_.size());
}

void MoeTracker::setCurrentLayer(int layer_id) {
    std::lock_guard<std::mutex> lock(mutex_);
    current_layer_id_ = layer_id;
    debug_printf("setCurrentLayer: %d\n", layer_id);
    // TODO
}

int MoeTracker::getCurrentLayer() {
    std::lock_guard<std::mutex> lock(mutex_);
    return current_layer_id_;
}

std::string MoeTracker::getLayerName(int layer_id) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (layer_id >= 0 && layer_id < static_cast<int>(layers_.size())) {
        return layers_[layer_id];
    }
    return "";
}

int MoeTracker::getLayerIdByName(const std::string& layer_name) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = layer_name_to_id_.find(layer_name);
    if (it != layer_name_to_id_.end()) {
        return it->second;
    }
    return -1;
}

// C风格API
extern "C" {

void moe_tracker_initialize() {
    MoeTracker::getInstance().initialize();
}

int moe_tracker_register_layer(const char* layer_name) {
    return MoeTracker::getInstance().registerLayer(std::string(layer_name));
}

int moe_tracker_get_layer_count() {
    return MoeTracker::getInstance().getLayerCount();
}

void moe_tracker_set_current_layer(int layer_id) {
    MoeTracker::getInstance().setCurrentLayer(layer_id);
}

int moe_tracker_get_current_layer() {
    return MoeTracker::getInstance().getCurrentLayer();
}

const char* moe_tracker_get_layer_name(int layer_id) {
    static thread_local std::string layer_name;
    layer_name = MoeTracker::getInstance().getLayerName(layer_id);
    return layer_name.c_str();
}

int moe_tracker_get_layer_id_by_name(const char* layer_name) {
    return MoeTracker::getInstance().getLayerIdByName(std::string(layer_name));
}

} // extern "C"

} // namespace moe_tracker 