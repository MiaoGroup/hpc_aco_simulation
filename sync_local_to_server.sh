#!/usr/bin/env bash

set -e

# ================== 配置区 ==================
LOCAL_DIR="."
REMOTE="hpc_box:hpc_xyce_box"
FILTER_FILE="rclone-filter.txt"
PROGRESS_OPTS="-P"
# ===========================================

echo "=========================================="
echo "⚠️  rclone 同步确认"
echo "------------------------------------------"
echo "运行位置 : 本地计算机"
echo "同步方向 : 本地  -->  远端"
echo "本地目录 : $(pwd)"
echo "远端目录 : ${REMOTE}"
echo "过滤规则 : ${FILTER_FILE}"
echo
echo "⚠️ 注意：rclone sync 会删除远端中本地不存在的文件！"
echo "=========================================="
echo

read -rp "是否确认继续？请输入 yes 继续，其他任意键取消: " CONFIRM

if [[ "$CONFIRM" != "yes" ]]; then
    echo "❌ 已取消同步。"
    exit 0
fi

echo
echo "🚀 开始执行同步命令："
echo "rclone sync ${LOCAL_DIR} ${REMOTE} ${PROGRESS_OPTS} --filter-from ${FILTER_FILE}" --update
echo

rclone sync "${LOCAL_DIR}" "${REMOTE}" ${PROGRESS_OPTS} --filter-from "${FILTER_FILE}" --update

echo
echo "✅ 同步完成。"
