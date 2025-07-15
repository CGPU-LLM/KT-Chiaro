/**
 * @Description  :
 * @Author       : chenht2022
 * @Date         : 2024-07-22 02:03:05
 * @Version      : 1.0.0
 * @LastEditors  : chenht2022
 * @LastEditTime : 2024-07-25 10:33:34
 * @Copyright (c) 2024 by KVCache.AI, All Rights Reserved.
 **/

#include "backend.h"
#include <cassert>

#ifdef USE_NUMA
#include <numa.h>
#include <numaif.h>

thread_local int Backend::numa_node = -1;
#endif

thread_local int Backend::thread_local_id = -1;

Backend::Backend(int max_thread_num) {
    max_thread_num_ = max_thread_num;
    thread_state_.resize(max_thread_num_);
    for (int i = 0; i < max_thread_num_; i++) {
        thread_state_[i].curr = std::make_unique<std::atomic<int>>();
        thread_state_[i].status =
            std::make_unique<std::atomic<ThreadStatus>>(ThreadStatus::WAITING);
    }
    workers_.resize(max_thread_num_);
    for (int i = 1; i < max_thread_num_; i++) {
        workers_[i] = std::thread(&Backend::worker_thread, this, i);
    }

    // 启动异步 I/O 工作线程 (max_thread_num_-1 个)
    int async_thread_num = std::max(1, max_thread_num_ - 1);
    async_workers_.reserve(async_thread_num);
    for (int i = 0; i < async_thread_num; ++i) {
        async_workers_.emplace_back(&Backend::async_worker_loop, this, i);
    }
}

Backend::~Backend() {
    for (int i = 0; i < max_thread_num_; i++) {
        thread_state_[i].status->store(ThreadStatus::EXIT,
                                       std::memory_order_release);
    }
    for (int i = 1; i < max_thread_num_; i++) {
        if (workers_[i].joinable()) {
            workers_[i].join();
        }
    }

    // 关闭异步工作线程
    async_exit_.store(true, std::memory_order_release);
    async_cv_.notify_all();
    for (auto &t : async_workers_) {
        if (t.joinable()) t.join();
    }
}

int Backend::get_thread_num() { return max_thread_num_; }

void Backend::do_work_stealing_job(int task_num,
                                   std::function<void(int)> init_func,
                                   std::function<void(int)> compute_func,
                                   std::function<void(int)> finalize_func) {
    init_func_ = init_func;
    compute_func_ = compute_func;
    finalize_func_ = finalize_func;
#ifdef USE_NUMA
    // numa node location will be calculated based on the number of threads
    thread_num_ = max_thread_num_;
#else
    thread_num_ = std::min(max_thread_num_, task_num);
#endif
    int base = task_num / thread_num_;
    int remain = task_num % thread_num_;
    thread_state_[0].end = base + (0 < remain);

    // 为主线程设置 thread_local_id
    thread_local_id = 0;

    for (int i = 1; i < thread_num_; i++) {
        thread_state_[i].curr->store(thread_state_[i - 1].end,
                                     std::memory_order_relaxed);
        thread_state_[i].end = thread_state_[i - 1].end + base + (i < remain);
        thread_state_[i].status->store(ThreadStatus::WORKING,
                                       std::memory_order_release);
    }
    thread_state_[0].curr->store(0, std::memory_order_relaxed);
    thread_state_[0].status->store(ThreadStatus::WORKING,
                                   std::memory_order_release);
    process_tasks(0);
    for (int i = 1; i < thread_num_; i++) {
        while (thread_state_[i].status->load(std::memory_order_acquire) ==
               ThreadStatus::WORKING) {
        }
    }
}

void Backend::process_tasks(int thread_id) {
    
    #ifdef USE_NUMA
    if(numa_node == -1){
        numa_node = thread_id * numa_num_configured_nodes() / thread_num_;
        struct bitmask* mask = numa_bitmask_alloc(numa_num_configured_nodes());
        numa_bitmask_setbit(mask, numa_node);
        numa_bind(mask);
    }
    #endif

    if (init_func_ != nullptr) {
        init_func_(thread_id);
    }
    while (true) {
        int task_id = thread_state_[thread_id].curr->fetch_add(
            1, std::memory_order_acq_rel);
        if (task_id >= thread_state_[thread_id].end) {
            break;
        }
        compute_func_(task_id);
    }
    for (int t_offset = 1; t_offset < thread_num_; t_offset++) {
        int t_i = (thread_id + t_offset) % thread_num_;
        if (thread_state_[t_i].status->load(std::memory_order_acquire) !=
            ThreadStatus::WORKING) {
            continue;
        }
        while (true) {
            int task_id = thread_state_[t_i].curr->fetch_add(
                1, std::memory_order_acq_rel);
            if (task_id >= thread_state_[t_i].end) {
                break;
            }
            compute_func_(task_id);
        }
    }
    if (finalize_func_ != nullptr) {
        finalize_func_(thread_id);
    }
    thread_state_[thread_id].status->store(ThreadStatus::WAITING,
                                           std::memory_order_release);
}

void Backend::worker_thread(int thread_id) {
    auto start = std::chrono::steady_clock::now();
    thread_local_id = thread_id; // 设置线程本地变量
    while (true) {
        ThreadStatus status =
            thread_state_[thread_id].status->load(std::memory_order_acquire);
        if (status == ThreadStatus::WORKING) {
            process_tasks(thread_id);
            start = std::chrono::steady_clock::now();
        } else if (status == ThreadStatus::WAITING) {
            auto now = std::chrono::steady_clock::now();
            auto duration =
                std::chrono::duration_cast<std::chrono::milliseconds>(now -
                                                                      start)
                    .count();
            if (duration > 50) {
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
        } else if (status == ThreadStatus::EXIT) {
            return;
        }
    }
}

// Execute I/O tasks (load/unload) using the existing thread pool and block until completion
void Backend::do_io_tasks(int task_num, std::function<void(int)> io_func) {
    do_work_stealing_job(task_num, nullptr, io_func, nullptr);
}

// 异步接口：仅把任务分配给后台 worker，caller 立即返回
void Backend::dispatch_io_tasks(int task_num, std::function<void(int)> io_func) {
    // 如果后台线程数不足，退化为同步执行
    if (max_thread_num_ <= 1) {
        do_io_tasks(task_num, io_func);
        return;
    }

    init_func_ = nullptr;
    compute_func_ = io_func;
    finalize_func_ = nullptr;

    // 只使用后台线程 1..max_thread_num_-1
    int worker_threads = std::min(max_thread_num_ - 1, task_num);
    if (worker_threads <= 0) {
        // 不应该走到这里，但以防万一
        do_io_tasks(task_num, io_func);
        assert(false);
        return;
    }

    thread_num_ = worker_threads + 1; // 包含 caller 线程0

    int base = task_num / worker_threads;
    int remain = task_num % worker_threads;

    // thread 0 不工作
    thread_state_[0].curr->store(0, std::memory_order_relaxed);
    thread_state_[0].end = 0;
    thread_state_[0].status->store(ThreadStatus::WAITING, std::memory_order_release);

    int start = 0;
    for (int i = 1; i <= worker_threads; ++i) {
        int end = start + base + ((i - 1) < remain);
        thread_state_[i].curr->store(start, std::memory_order_relaxed);
        thread_state_[i].end = end;
        thread_state_[i].status->store(ThreadStatus::WORKING, std::memory_order_release);
        start = end;
    }

    // 多余的线程（若 max_thread_num_ > worker_threads+1）保持 WAITING
    for (int i = worker_threads + 1; i < max_thread_num_; ++i) {
        thread_state_[i].status->store(ThreadStatus::WAITING, std::memory_order_release);
    }
    // caller 立即返回，由后台 worker 执行
}

// 异步工作线程主循环
void Backend::async_worker_loop(int /*worker_id*/) {
    while (true) {
        AsyncJob job;
        {
            std::unique_lock<std::mutex> lk(async_mu_);
            async_cv_.wait(lk, [&]() { return async_exit_.load() || !async_queue_.empty(); });
            if (async_exit_.load() && async_queue_.empty()) {
                return;
            }
            job = async_queue_.front();
            async_queue_.pop();
        }
        for (int idx = job.begin; idx < job.end; ++idx) {
            job.fn(idx);
        }
    }
}

void Backend::enqueue_io_tasks(int task_num, std::function<void(int)> io_func) {
    if (task_num <= 0) return;

    int worker_threads = std::max(1, static_cast<int>(async_workers_.size()));
    int base = task_num / worker_threads;
    int remain = task_num % worker_threads;

    int start = 0;
    for (int w = 0; w < worker_threads; ++w) {
        int end = start + base + (w < remain ? 1 : 0);
        if (start >= end) break;
        {
            std::lock_guard<std::mutex> lk(async_mu_);
            async_queue_.push({start, end, io_func});
        }
        async_cv_.notify_one();
        start = end;
    }
}
