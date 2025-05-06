import os
try:
    open(os.path.join(os.getcwd(), 'call.log'), 'w').close()
except Exception:
    pass
import functools
import torch

def log_function_call(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        class_name = func.__qualname__.split('.<locals>', 1)[0].rsplit('.', 1)[0]
        with open('./call.log', 'a') as f:
            f.write('\n')
            if class_name:
                f.write(f"{class_name}.{func.__name__}\n")
            else:
                f.write(f"{func.__name__}\n")
            # 安全地记录参数信息，避免打印完整张量
            arg_info = []
            for arg in args:
                if isinstance(arg, torch.Tensor):
                    arg_info.append(f"Tensor(shape={arg.shape}, dtype={arg.dtype}, device={arg.device})")
                else:
                    try:
                        arg_info.append(str(arg))
                    except Exception:
                        # fallback to type name if repr fails
                        arg_info.append(f"{type(arg).__name__}")
            
            # 类似地处理关键字参数
            kwarg_info = {}
            for k, v in kwargs.items():
                if isinstance(v, torch.Tensor):
                    kwarg_info[k] = f"Tensor(shape={v.shape}, dtype={v.dtype}, device={v.device})"
                else:
                    try:
                        kwarg_info[k] = str(v)
                    except Exception:
                        # fallback to type name if repr fails
                        kwarg_info[k] = f"{type(v).__name__}"
            
            f.write(f"Args: {arg_info}\n")
            f.write(f"Kwargs: {kwarg_info}\n")
        return func(*args, **kwargs)
    return wrapper

