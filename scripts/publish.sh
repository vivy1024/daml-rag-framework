#!/bin/bash
# DAML-RAG Framework - PyPI 发布脚本

set -e

echo "🚀 DAML-RAG Framework - PyPI 发布脚本"
echo "====================================="

# 检查参数
TARGET=${1:-"test"}

if [ "$TARGET" != "test" ] && [ "$TARGET" != "prod" ]; then
    echo "❌ 错误: 无效的目标环境"
    echo ""
    echo "用法: $0 [test|prod]"
    echo "  test - 发布到 TestPyPI (测试环境)"
    echo "  prod - 发布到 PyPI (生产环境)"
    exit 1
fi

# 检查是否已构建
if [ ! -d "dist" ] || [ -z "$(ls -A dist)" ]; then
    echo "❌ 错误: 未找到构建产物，请先运行 ./scripts/build.sh"
    exit 1
fi

# 检查 Twine
if ! command -v twine &> /dev/null; then
    echo "📦 安装 twine..."
    pip install --upgrade twine
fi

echo ""
echo "📦 准备发布..."
echo "  环境: $TARGET"
echo "  产物:"
ls -lh dist/

# 确认发布
if [ "$TARGET" == "prod" ]; then
    echo ""
    echo "⚠️  警告: 即将发布到 PyPI 生产环境！"
    read -p "确认发布? (yes/no): " CONFIRM
    if [ "$CONFIRM" != "yes" ]; then
        echo "❌ 取消发布"
        exit 0
    fi
fi

# 执行发布
echo ""
if [ "$TARGET" == "test" ]; then
    echo "📤 发布到 TestPyPI..."
    twine upload --repository testpypi dist/*
    echo ""
    echo "✅ 发布成功！"
    echo ""
    echo "测试安装:"
    echo "  pip install --index-url https://test.pypi.org/simple/ daml-rag-framework"
else
    echo "📤 发布到 PyPI..."
    twine upload dist/*
    echo ""
    echo "✅ 发布成功！"
    echo ""
    echo "安装:"
    echo "  pip install daml-rag-framework"
fi

echo ""
echo "🎉 发布完成！"

