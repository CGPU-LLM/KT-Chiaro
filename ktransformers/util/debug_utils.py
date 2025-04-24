import datetime
import os

# 日志文件名
DEBUG_LOG_FILENAME = "/home/chiarolrg/work/ktransformers/python_debug_log.txt"

# 初始化器：首次导入时清空日志文件
_log_initialized = False
def _initialize_log():
    global _log_initialized
    if not _log_initialized:
        try:
            # 确保目录存在
            log_dir = os.path.dirname(DEBUG_LOG_FILENAME)
            if log_dir and not os.path.exists(log_dir):
                 os.makedirs(log_dir, exist_ok=True)
            # 清空文件
            with open(DEBUG_LOG_FILENAME, "w") as f:
                f.write(f"Log initialized at {datetime.datetime.now()}\n")
                f.write("-" * 30 + "\n")
            _log_initialized = True
        except Exception as e:
            print(f"Error initializing debug log file {DEBUG_LOG_FILENAME}: {e}")

_initialize_log()

def debug_log(format_string: str, *args):
    """
    将格式化字符串写入日志文件。
    """
    try:
        message = format_string % args
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        log_entry = f"[{timestamp}] {message}\n"

        # 直接写入文件，不再使用锁
        with open(DEBUG_LOG_FILENAME, "a") as f:
            f.write(log_entry)
    except Exception as e:
        # 避免日志函数本身出错导致程序崩溃
        print(f"Error writing to debug log: {e}")

if __name__ == '__main__':
    # 简单测试
    debug_log("Test message 1: %s", "hello")
    debug_log("Test message 2: %d + %d = %d", 5, 3, 8)
    print(f"Debug log written to {DEBUG_LOG_FILENAME}") 