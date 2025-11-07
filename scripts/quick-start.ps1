# DAML-RAG Framework - 快速开始脚本 (Windows PowerShell)

Write-Host "🚀 DAML-RAG Framework - 快速开始" -ForegroundColor Cyan
Write-Host "================================" -ForegroundColor Cyan

# 检查 Python
if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
    Write-Host "❌ 错误: 未找到 Python" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "Python 版本:" -ForegroundColor Yellow
python --version

# 步骤1: 构建
Write-Host ""
Write-Host "📦 第1步: 构建框架..." -ForegroundColor Yellow
& ".\scripts\build.ps1"

# 步骤2: 本地测试
Write-Host ""
Write-Host "🧪 第2步: 本地测试..." -ForegroundColor Yellow
& ".\scripts\test-install.ps1"

# 完成
Write-Host ""
Write-Host "✨ 快速开始完成！" -ForegroundColor Green
Write-Host ""
Write-Host "下一步：" -ForegroundColor Cyan
Write-Host "  1. 查看发布指南: Get-Content PUBLISHING.md"
Write-Host "  2. 发布到 TestPyPI: .\scripts\publish.ps1 test"
Write-Host "  3. 发布到 PyPI: .\scripts\publish.ps1 prod"



