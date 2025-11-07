#!/bin/bash
# DAML-RAG Framework - 本地安装测试脚本

set -e

echo "🧪 DAML-RAG Framework - 本地安装测试"
echo "===================================="

# 创建临时虚拟环境
echo ""
echo "📦 创建测试虚拟环境..."
python -m venv /tmp/daml-rag-test-env
source /tmp/daml-rag-test-env/bin/activate

# 升级 pip
echo ""
echo "⬆️  升级 pip..."
pip install --upgrade pip

# 从构建的包安装
echo ""
echo "📥 从本地构建包安装..."
if [ -d "dist" ]; then
    pip install dist/*.whl
else
    echo "❌ 错误: 未找到构建产物，请先运行 ./scripts/build.sh"
    exit 1
fi

# 测试导入
echo ""
echo "🔍 测试导入..."
python -c "from daml_rag import DAMLRAGFramework; print('✅ 核心框架导入成功')"
python -c "from daml_rag.retrieval import VectorRetriever; print('✅ 检索模块导入成功')"
python -c "from daml_rag.learning import ModelProvider; print('✅ 学习模块导入成功')"
python -c "from daml_rag.adapters import FitnessDomainAdapter; print('✅ 适配器导入成功')"

# 测试 CLI
echo ""
echo "🔍 测试命令行工具..."
daml-rag --help > /dev/null && echo "✅ CLI 工具正常"

# 清理
echo ""
echo "🧹 清理测试环境..."
deactivate
rm -rf /tmp/daml-rag-test-env

echo ""
echo "✨ 本地测试通过！"
echo ""
echo "包可以正常安装和使用。"



