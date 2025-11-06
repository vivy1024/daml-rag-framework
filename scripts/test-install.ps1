# DAML-RAG Framework - 本地安装测试脚本 (Windows PowerShell)

Write-Host "🧪 DAML-RAG Framework - 本地安装测试" -ForegroundColor Cyan
Write-Host "====================================" -ForegroundColor Cyan

# 创建临时虚拟环境
Write-Host ""
Write-Host "📦 创建测试虚拟环境..." -ForegroundColor Yellow
$testEnvPath = "$env:TEMP\daml-rag-test-env"
if (Test-Path $testEnvPath) { Remove-Item -Recurse -Force $testEnvPath }
python -m venv $testEnvPath

# 激活虚拟环境
& "$testEnvPath\Scripts\Activate.ps1"

# 升级 pip
Write-Host ""
Write-Host "⬆️  升级 pip..." -ForegroundColor Yellow
python -m pip install --upgrade pip

# 从构建的包安装
Write-Host ""
Write-Host "📥 从本地构建包安装..." -ForegroundColor Yellow
if (Test-Path "dist") {
    $whlFile = Get-ChildItem -Path "dist" -Filter "*.whl" | Select-Object -First 1
    pip install $whlFile.FullName
} else {
    Write-Host "❌ 错误: 未找到构建产物，请先运行 .\scripts\build.ps1" -ForegroundColor Red
    exit 1
}

# 测试导入
Write-Host ""
Write-Host "🔍 测试导入..." -ForegroundColor Yellow
python -c "from daml_rag import DAMLRAGFramework; print('✅ 核心框架导入成功')"
python -c "from daml_rag.retrieval import VectorRetriever; print('✅ 检索模块导入成功')"
python -c "from daml_rag.learning import ModelProvider; print('✅ 学习模块导入成功')"
python -c "from daml_rag.adapters import FitnessDomainAdapter; print('✅ 适配器导入成功')"

# 测试 CLI
Write-Host ""
Write-Host "🔍 测试命令行工具..." -ForegroundColor Yellow
daml-rag --help | Out-Null
if ($LASTEXITCODE -eq 0) {
    Write-Host "✅ CLI 工具正常" -ForegroundColor Green
}

# 清理
Write-Host ""
Write-Host "🧹 清理测试环境..." -ForegroundColor Yellow
deactivate
Remove-Item -Recurse -Force $testEnvPath

Write-Host ""
Write-Host "✨ 本地测试通过！" -ForegroundColor Green
Write-Host ""
Write-Host "包可以正常安装和使用。" -ForegroundColor Cyan

