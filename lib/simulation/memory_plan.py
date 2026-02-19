import os
import sys
def get_available_memory_linux():
    """
    获取 Linux 上的可用内存 (bytes) 通过解析 /proc/meminfo.
    """
    if sys.platform != 'Linux':
        raise EnvironmentError("This function is only supported on Linux systems.")
    try:
        with open('/proc/meminfo', 'r') as f:
            for line in f:
                if line.startswith('MemAvailable:'):
                    parts = line.split()
                    available_kb = int(parts[1])
                    return available_kb / 1024 / 1024  # Convert to GB
        return None  # 如果找不到 MemAvailable
    except FileNotFoundError:
        return None  # 如果 /proc/meminfo 不存在

if __name__ == '__main__':
    print(get_available_memory_linux())