#!/bin/bash

# --- 脚本说明 ---
echo "----------------------------------------------------------------"
echo "提示：正在准备同步数据"
echo "路径：从 [超算] /fsb/home/miaofeng/mf_zhaoyichen/hpc_xyce_results "
echo "目标：到 [云盘] hpc_box:hpc_xyce_results"
echo "----------------------------------------------------------------"
echo "！！请确认此脚本是在【超算终端】中执行，而非本地电脑！！"
echo ""

# --- 确认逻辑 ---
read -p "是否确认开始同步？(y/n): " confirm

if [[ "$confirm" == [yY] || "$confirm" == [yY][eE][sS] ]]; then
    echo "确认成功，开始同步..."
    # 执行同步命令
    rclone sync /fsb/home/miaofeng/mf_zhaoyichen/hpc_xyce_results hpc_box:hpc_xyce_results -P --update
    
    if [ $? -eq 0 ]; then
        echo "------------------------------------------------"
        echo "✅ 同步完成！"
    else
        echo "------------------------------------------------"
        echo "❌ 同步过程中出现错误，请检查网络或路径。"
    fi
else
    echo "操作已取消。"
    exit 1
fi