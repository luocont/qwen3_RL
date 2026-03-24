# Windows PowerShell 环境变量设置脚本
# 使用方法：在PowerShell中运行 .\setup_windows.ps1

# 临时设置当前会话的环境变量
$env:DASHSCOPE_API_KEY="sk-40fb3997d3ed485ba390a9c4ae3bd2d2"

Write-Host "=" * 50
Write-Host "API密钥已设置"
Write-Host "=" * 50
Write-Host ""
Write-Host "现在可以运行程序了："
Write-Host "python run.py"
Write-Host ""
Write-Host "注意：此设置仅在当前PowerShell会话中有效"
Write-Host "关闭PowerShell窗口后需要重新设置"
