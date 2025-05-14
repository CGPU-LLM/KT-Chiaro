#ifndef KTRANSFORMERS_MOE_TRACKER_H
#define KTRANSFORMERS_MOE_TRACKER_H

#include <mutex>
#include <vector>
#include <unordered_map>
#include <string>
#include <iostream>

namespace moe_tracker {

/**
 * @brief MoE层跟踪器类，用于跟踪模型中MoE层的状态
 */
class MoeTracker {
public:
    /**
     * @brief 获取MoeTracker的单例实例
     * @return MoeTracker的单例实例引用
     */
    static MoeTracker& getInstance();

    /**
     * @brief 初始化跟踪器，清除所有已注册的层
     */
    void initialize();

    /**
     * @brief 注册一个新的MoE层
     * @param layer_name 层的名称或标识符
     * @return 注册的层ID
     */
    int registerLayer(const std::string& layer_name);

    /**
     * @brief 获取已注册的MoE层数量
     * @return 已注册的层数量
     */
    int getLayerCount();

    /**
     * @brief 设置当前正在计算的MoE层
     * @param layer_id 层ID
     */
    void setCurrentLayer(int layer_id);

    /**
     * @brief 获取当前正在计算的MoE层
     * @return 当前层ID，如果没有层在计算则返回-1
     */
    int getCurrentLayer();

    /**
     * @brief 获取指定层ID对应的层名称
     * @param layer_id 层ID
     * @return 层名称，如果不存在则返回空字符串
     */
    std::string getLayerName(int layer_id);

    /**
     * @brief 获取指定层名对应的层ID
     * @param layer_name 层名称
     * @return 层ID，如果不存在则返回-1
     */
    int getLayerIdByName(const std::string& layer_name);

private:
    MoeTracker() : current_layer_id_(-1) {}
    
    MoeTracker(const MoeTracker&) = delete;
    MoeTracker& operator=(const MoeTracker&) = delete;

    std::mutex mutex_;
    
    // 已注册的MoE层
    std::vector<std::string> layers_; 
    // 层名到ID的映射
    std::unordered_map<std::string, int> layer_name_to_id_;
    // 当前正在计算的层ID
    int current_layer_id_;
};

// C风格的API
extern "C" {
    void moe_tracker_initialize();
    int moe_tracker_register_layer(const char* layer_name);
    int moe_tracker_get_layer_count();
    void moe_tracker_set_current_layer(int layer_id);
    int moe_tracker_get_current_layer();
    const char* moe_tracker_get_layer_name(int layer_id);
    int moe_tracker_get_layer_id_by_name(const char* layer_name);
}

} // namespace moe_tracker

#endif // KTRANSFORMERS_MOE_TRACKER_H 