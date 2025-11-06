#!/bin/bash
# DAML-RAG Framework - 构建脚本
# 用于构建发布包

set -e

echo "🔨 DAML-RAG Framework - 构建脚本"
echo "================================"

# 检查必要的工具
if ! command -v python &> /dev/null; then
    echo "❌ 错误: 未找到 Python"
    exit 1
fi

# 清理旧的构建产物
echo ""
echo "🧹 清理旧的构建产物..."
rm -rf build/
rm -rf dist/
rm -rf *.egg-info
rm -rf daml_rag.egg-info

# 安装构建依赖
echo ""
echo "📦 安装构建依赖..."
pip install --upgrade build twine wheel setuptools

# 构建发布包
echo ""
echo "🏗️  构建发布包..."
python -m build

# 检查构建结果
echo ""
echo "✅ 构建完成！"
echo ""
echo "📦 构建产物:"
ls -lh dist/

# 检查包完整性
echo ""
echo "🔍 检查包完整性..."
twine check dist/*

echo ""
echo "✨ 构建成功！"
echo ""
echo "下一步："
echo "  1. 本地测试: ./scripts/test-install.sh"
echo "  2. 发布到 TestPyPI: ./scripts/publish.sh test"
echo "  3. 发布到 PyPI: ./scripts/publish.sh prod"


