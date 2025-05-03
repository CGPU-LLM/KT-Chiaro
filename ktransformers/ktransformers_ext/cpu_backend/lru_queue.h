#pragma once

#include <deque>
#include <mutex>
#include <vector>
#include <array>

typedef std::array <int, 2> Pair;
typedef std::array <void*, 3> expertsPointer;

class LRUQueue {
    private:
    std::deque <Pair> Q; // [experts_id, idx_to_memoryPool]
    std::vector <int> vis;
    std::vector <expertsPointer> memoryPool;
    std::vector <int> availableIdx;
    int experts_total, max_load;
    std::mutex mtx;

    void push(int expert_id); // 直接往里加
    void pop(); // 淘汰
    void refresh(int expert_id); // 刷新位置

    public:
    LRUQueue(int experts_total, int max_load, std::vector <void*> gate_pool, std::vector <void*> up_pool, std::vector <void*> down_pool);

    expertsPointer find(int expert_id);
    bool update(int expert_id);
    void __debug_print();
};