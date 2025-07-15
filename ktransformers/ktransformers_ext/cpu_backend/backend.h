/**
 * @Description  :
 * @Author       : chenht2022
 * @Date         : 2024-07-22 02:03:05
 * @Version      : 1.0.0
 * @LastEditors  : chenht2022
 * @LastEditTime : 2024-07-25 10:33:38
 * @Copyright (c) 2024 by KVCache.AI, All Rights Reserved.
 **/
#ifndef CPUINFER_BACKEND_H
#define CPUINFER_BACKEND_H

#include <atomic>
#include <condition_variable>
#include <cstdio>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>
#include <queue>

enum ThreadStatus {
    WORKING,
    WAITING,
    EXIT,
};

struct ThreadState {
    std::unique_ptr<std::atomic<ThreadStatus>> status;
    std::unique_ptr<std::atomic<int>> curr;
    int end;
};

class Backend {
  public:
    Backend(int);
    ~Backend();
    int get_thread_num();
    void do_work_stealing_job(int, std::function<void(int)>,
                              std::function<void(int)>,
                              std::function<void(int)>);
    void do_io_tasks(int task_num, std::function<void(int)> io_func);
    // 异步版本：仅调度任务，立即返回，由后台线程执行
    void dispatch_io_tasks(int task_num, std::function<void(int)> io_func);
    #ifdef USE_NUMA
    static thread_local int numa_node;
    #endif
    static thread_local int thread_local_id;

    struct AsyncJob {
        int begin;
        int end;
        std::function<void(int)> fn;
    };

    // 异步 I/O 工作线程相关成员
    std::queue<AsyncJob> async_queue_;
    std::mutex async_mu_;
    std::condition_variable async_cv_;
    std::vector<std::thread> async_workers_;
    std::atomic<bool> async_exit_{false};

    void async_worker_loop(int worker_id);

    // 异步入队接口：将 task_num 个任务拆分并推入队列，立刻返回
    void enqueue_io_tasks(int task_num, std::function<void(int)> io_func);

  private:
    int thread_num_;
    int max_thread_num_;
    std::vector<ThreadState> thread_state_; // [thread_num]
    std::function<void(int)> init_func_;
    std::function<void(int)> compute_func_;
    std::function<void(int)> finalize_func_;
    std::vector<std::thread> workers_;

    void process_tasks(int);
    void worker_thread(int);
};
#endif