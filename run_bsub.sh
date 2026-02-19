#!/bin/bash
# 获取传入的测试文件路径
# 如果没有传入参数，或者传入了空字符串，则设置一个默认值（可以根据需要修改或报错）
TEST_FILE=${1:-"tests/test_vmm.py"} 

# 检查测试文件路径是否为空，如果为空则退出
if [ -z "$TEST_FILE" ]; then
    echo "Usage: $0 <test_file_path>"
    echo "Example: $0 tests/test_vmm.py"
    exit 1
fi

# 定义 Apptainer 镜像路径
APPTAINER_IMAGE="xyce.sif"

# 定义 PYTHONPATH 变量
# 注意：当在 shell 脚本中引用 $PYTHONPATH 时，如果它可能为空，最好用双引号包围
# 或者使用 -z 检查并省略冒号。这里我们假设 PYTHONPATH 在执行前可能为空，
# 所以我们用一个条件表达式来处理。
# PYTHONPATH_VAL="$HOME/hpc_xyce"
PYTHONPATH_VAL="$PWD"

# 如果环境变量 PYTHONPATH 已经设置了，就追加；否则只用 ~/hpc_xyce
if [ -n "$PYTHONPATH" ]; then
    PYTHONPATH_VAL="${PYTHONPATH_VAL}:$PYTHONPATH"
fi

echo "Running test: ${TEST_FILE}"
echo "Using Apptainer image: ${APPTAINER_IMAGE}"
echo "Setting PYTHONPATH: ${PYTHONPATH_VAL}"
mkdir -p logs
# 执行 Apptainer 命令
# bsub -q e5v3ib -n 24 -o logs/output%J.log -R "mem256g" apptainer exec --env PYTHONPATH="${PYTHONPATH_VAL}" "${APPTAINER_IMAGE}" python3 "${TEST_FILE}" "${@:2}"
# bsub -q e7v4ib -n 64 -o logs/output%J.log apptainer exec --env PYTHONPATH="${PYTHONPATH_VAL}" "${APPTAINER_IMAGE}" python3 "${TEST_FILE}" "${@:2}"
bsub -q 6330ib -n 56 -o logs/output%J.log apptainer exec --env PYTHONPATH="${PYTHONPATH_VAL}" "${APPTAINER_IMAGE}" python3 "${TEST_FILE}" "${@:2}"
# 检查上一个命令的退出状态码
if [ $? -eq 0 ]; then
    echo "Run completed successfully."
else
    echo "Run failed or encountered an error."
    exit 1 # 返回非零状态码表示失败
fi
