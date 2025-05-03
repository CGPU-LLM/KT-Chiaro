#include "lru_queue.h"
#include <cmath>
#include <cstdio>
#include <functional>
#include <mutex>
#include <vector>
#include <string>
#include <cassert>
#include "debug.h"

LRUQueue::LRUQueue(int experts_total, int max_load, std::vector <void*> gate_pool, std::vector <void*> up_pool, std::vector <void*> down_pool) : 
    experts_total(experts_total), 
    max_load(max_load) {
    vis.resize(experts_total, 0);
    for (int i = 0; i < max_load; i++) {
        memoryPool.push_back({gate_pool[i], up_pool[i], down_pool[i]});
        availableIdx.push_back(i);
    }
}

void LRUQueue::push(int expert_id) {
    assert(vis[expert_id] == 0);
    assert(experts_total > expert_id);
    if (Q.size() == max_load)
        pop();
    assert(availableIdx.size() > 0);
    int idx = availableIdx.back();
    availableIdx.pop_back();
    Q.push_front({expert_id, idx});
    vis[expert_id] = 1;
}

void LRUQueue::pop() {
    assert(Q.size() > 0);
    vis[Q.back()[0]] = 0;
    availableIdx.push_back(Q.back()[1]);
    Q.pop_back();
}

void LRUQueue::refresh(int expert_id) {
    assert(vis[expert_id] == 1);
    for (int i = 0; i < Q.size(); i++) {
        if (Q[i][0] == expert_id) {
            auto tmp = Q[i];
            Q.erase(Q.begin() + i);
            Q.push_front(tmp);
            return;
        }
    }
    assert(false);
}

expertsPointer LRUQueue::find(int expert_id) {
    std::lock_guard <std::mutex> lock(mtx);
    assert(experts_total > expert_id);
    assert(vis[expert_id] == 1);
    for (int i = 0; i < Q.size(); i++) {
        if (Q[i][0] == expert_id) {
            return memoryPool[Q[i][1]];
        }
    }
    assert(false);
}

bool LRUQueue::update(int expert_id) {
    std::lock_guard <std::mutex> lock(mtx);
    assert(experts_total > expert_id);
    if (vis[expert_id] == 1) {
        refresh(expert_id);
        return true;
    } else {
        push(expert_id);
        return false;
    }
}

void LRUQueue::__debug_print() {
    debug_printf("[C++]: LRUQueue: ");
    for (int i = 0; i < Q.size(); i++) {
        debug_printf("%d ", Q[i][0]);
    }
    debug_printf("\n");
}