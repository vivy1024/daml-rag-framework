# DAML-RAG Framework - PyPI 发布脚本 (Windows PowerShell)

param(
    [string]$Target = "test"
)

Write-Host "🚀 DAML-RAG Framework - PyPI 发布脚本" -ForegroundColor Cyan
Write-Host "=====================================" -ForegroundColor Cyan

# 检查参数
if ($Target -ne "test" -and $Target -ne "prod") {
    Write-Host "❌ 错误: 无效的目标环境" -ForegroundColor Red
    Write-Host ""
    Write-Host "用法: .\scripts\publish.ps1 [test|prod]"
    Write-Host "  test - 发布到 TestPyPI (测试环境)"
    Write-Host "  prod - 发布到 PyPI (生产环境)"
    exit 1
}

# 检查是否已构建
if (-not (Test-Path "dist") -or -not (Get-ChildItem "dist")) {
    Write-Host "❌ 错误: 未找到构建产物，请先运行 .\scripts\build.ps1" -ForegroundColor Red
    exit 1
}

# 检查 Twine
if (-not (Get-Command twine -ErrorAction SilentlyContinue)) {
    Write-Host "📦 安装 twine..." -ForegroundColor Yellow
    python -m pip install --upgrade twine
}

Write-Host ""
Write-Host "📦 准备发布..." -ForegroundColor Yellow
Write-Host "  环境: $Target" -ForegroundColor Cyan
Write-Host "  产物:" -ForegroundColor Cyan
Get-ChildItem -Path "dist" | Format-Table Name, Length

# 确认发布
if ($Target -eq "prod") {
    Write-Host ""
    Write-Host "⚠️  警告: 即将发布到 PyPI 生产环境！" -ForegroundColor Yellow
    $confirm = Read-Host "确认发布? (yes/no)"
    if ($confirm -ne "yes") {
        Write-Host "❌ 取消发布" -ForegroundColor Red
        exit 0
    }
}

# 执行发布
Write-Host ""
if ($Target -eq "test") {
    Write-Host "📤 发布到 TestPyPI..." -ForegroundColor Yellow
    python -m twine upload --repository testpypi dist/*
    Write-Host ""
    Write-Host "✅ 发布成功！" -ForegroundColor Green
    Write-Host ""
    Write-Host "测试安装:" -ForegroundColor Cyan
    Write-Host "  pip install --index-url https://test.pypi.org/simple/ daml-rag-framework"
} else {
    Write-Host "📤 发布到 PyPI..." -ForegroundColor Yellow
    python -m twine upload dist/*
    Write-Host ""
    Write-Host "✅ 发布成功！" -ForegroundColor Green
    Write-Host ""
    Write-Host "安装:" -ForegroundColor Cyan
    Write-Host "  pip install daml-rag-framework"
}

Write-Host ""
Write-Host "🎉 发布完成！" -ForegroundColor Green

