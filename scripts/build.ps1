# DAML-RAG Framework - 构建脚本 (Windows PowerShell)
# 用于构建发布包

Write-Host "🔨 DAML-RAG Framework - 构建脚本" -ForegroundColor Cyan
Write-Host "================================" -ForegroundColor Cyan

# 检查 Python
if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
    Write-Host "❌ 错误: 未找到 Python" -ForegroundColor Red
    exit 1
}

# 清理旧的构建产物
Write-Host ""
Write-Host "🧹 清理旧的构建产物..." -ForegroundColor Yellow
if (Test-Path "build") { Remove-Item -Recurse -Force "build" }
if (Test-Path "dist") { Remove-Item -Recurse -Force "dist" }
Get-ChildItem -Filter "*.egg-info" -Recurse | Remove-Item -Recurse -Force

# 安装构建依赖
Write-Host ""
Write-Host "📦 安装构建依赖..." -ForegroundColor Yellow
python -m pip install --upgrade build twine wheel setuptools

# 构建发布包
Write-Host ""
Write-Host "🏗️  构建发布包..." -ForegroundColor Yellow
python -m build

# 检查构建结果
Write-Host ""
Write-Host "✅ 构建完成！" -ForegroundColor Green
Write-Host ""
Write-Host "📦 构建产物:" -ForegroundColor Cyan
Get-ChildItem -Path "dist" | Format-Table Name, Length, LastWriteTime

# 检查包完整性
Write-Host ""
Write-Host "🔍 检查包完整性..." -ForegroundColor Yellow
python -m twine check dist/*

Write-Host ""
Write-Host "✨ 构建成功！" -ForegroundColor Green
Write-Host ""
Write-Host "下一步：" -ForegroundColor Cyan
Write-Host "  1. 本地测试: .\scripts\test-install.ps1"
Write-Host "  2. 发布到 TestPyPI: .\scripts\publish.ps1 test"
Write-Host "  3. 发布到 PyPI: .\scripts\publish.ps1 prod"


