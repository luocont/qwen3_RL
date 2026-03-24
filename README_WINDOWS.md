# Windows 使用指南

## 方法1：直接在config.py中设置API密钥（推荐）

我已经将你的API密钥添加到 `config.py` 文件中，现在可以直接运行程序：

```powershell
python run.py
```

## 方法2：在PowerShell中设置环境变量

### 临时设置（当前会话有效）
```powershell
$env:DASHSCOPE_API_KEY="sk-ee3f451e73ef44a892add1bf733e9c0b"
python run.py
```

### 使用设置脚本
```powershell
.\setup_windows.ps1
python run.py
```

### 永久设置（系统级）
在PowerShell（管理员权限）中运行：
```powershell
[System.Environment]::SetEnvironmentVariable('DASHSCOPE_API_KEY', 'sk-ee3f451e73ef44a892add1bf733e9c0b', 'User')
```

## 方法3：在命令提示符(cmd)中设置

### 临时设置
```cmd
set DASHSCOPE_API_KEY=sk-ee3f451e73ef44a892add1bf733e9c0b
python run.py
```

## 快速开始

由于我已经在 `config.py` 中设置了API密钥，你现在可以直接运行：

```powershell
# 快速启动模式
python run.py

# 命令行模式
python main.py --mode batch --max-rounds 3

# 交互式模式
python main.py --mode interactive
```

## 常见问题

### PowerShell脚本执行限制
如果遇到"无法运行脚本"错误，运行：
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### 查看当前环境变量
```powershell
$env:DASHSCOPE_API_KEY
```

### 删除环境变量
```powershell
$env:DASHSCOPE_API_KEY=""
```

## 推荐方式

由于你已经在 `config.py` 中设置了API密钥，建议直接使用 **方法1**，无需任何额外配置，直接运行 `python run.py` 即可。
