# Windows 安装依赖脚本

Write-Host "=" * 50
Write-Host "AI病例分析工作流 - 依赖安装"
Write-Host "=" * 50
Write-Host ""

# 检查pip是否可用
try {
    Write-Host "检查pip..."
    python -m pip --version
    Write-Host "✓ pip可用"
    Write-Host ""

    # 安装依赖
    Write-Host "安装依赖包..."
    Write-Host "正在安装 openai..."
    python -m pip install openai

    Write-Host ""
    Write-Host "=" * 50
    Write-Host "安装完成！"
    Write-Host "=" * 50
    Write-Host ""
    Write-Host "现在可以运行程序了："
    Write-Host "  python run.py"
    Write-Host ""

} catch {
    Write-Host "❌ 错误: $_"
    Write-Host ""
    Write-Host "请确保："
    Write-Host "1. 已安装Python"
    Write-Host "2. Python已添加到PATH环境变量"
    Write-Host ""
    Write-Host "可以手动运行: pip install openai"
}
