#!/bin/bash
# DAML-RAG Framework - 快速开始脚本

set -e

echo "🚀 DAML-RAG Framework - 快速开始"
echo "================================"

# 检查 Python
if ! command -v python &> /dev/null; then
    echo "❌ 错误: 未找到 Python"
    exit 1
fi

echo ""
echo "Python 版本:"
python --version

# 步骤1: 构建
echo ""
echo "📦 第1步: 构建框架..."
./scripts/build.sh

# 步骤2: 本地测试
echo ""
echo "🧪 第2步: 本地测试..."
./scripts/test-install.sh

# 完成
echo ""
echo "✨ 快速开始完成！"
echo ""
echo "下一步："
echo "  1. 查看发布指南: cat PUBLISHING.md"
echo "  2. 发布到 TestPyPI: ./scripts/publish.sh test"
echo "  3. 发布到 PyPI: ./scripts/publish.sh prod"



