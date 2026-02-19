import shutil
import os

def check_command_availability(command):
    """
    检查一个命令是否存在于系统的PATH中并且是可执行的。

    参数:
        command (str): 需要检查的命令名称。

    返回:
        bool: 如果命令可用，则返回 True，否则返回 False。
        str: 命令的完整路径，如果找不到则为 None。
    """
    path = shutil.which(command)
    if path:
        print(f"✅ Find command '{command}' in PATH.")
        print(f"   Path: {path}")
        # 在非 Windows 系统上，额外检查其可执行权限
        if os.name != 'nt' and not os.access(path, os.X_OK):
            print(f"   ⚠️ 警告: 文件 '{path}' 存在但没有执行权限。")
            return False, path
        return True, path
    else:
        print(f"❌ Cannot find command'{command}' in PATH.")
        return False, ""