#include "debug.h"
#include <cstdio>
#include <cstdarg>
#include <mutex>

// 互斥锁，保证多线程安全
static std::mutex debug_mutex;

// 静态对象，在程序启动阶段构造，用于清空旧的日志文件
struct DebugLogInitializer {
    DebugLogInitializer() {
        FILE* f = std::fopen(DEBUG_LOG_FILENAME, "w");
        if (f) {
            std::fclose(f);
        }
    }
};
// 定义全局静态实例
static DebugLogInitializer debugLogInitializer;

void debug_printf(const char* format, ...) {
    std::lock_guard<std::mutex> lock(debug_mutex);
    FILE* f = std::fopen(DEBUG_LOG_FILENAME, "a");
    if (!f) {
        return;
    }
    va_list args;
    va_start(args, format);
    std::vfprintf(f, format, args);
    va_end(args);
    std::fclose(f);
} 