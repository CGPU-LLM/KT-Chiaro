#ifndef DEBUG_H
#define DEBUG_H

#include <cstdarg>

// 日志文件名
static constexpr char DEBUG_LOG_FILENAME[] = "/home/chiarolrg/work/ktransformers/ktransformers/ktransformers_ext/debug/debug_log.txt";

// 将格式化字符串写入日志文件，每次程序启动时日志会被清空
void debug_printf(const char* format, ...);

#endif // DEBUG_H 