# Chat with AI Workflow

## 说明

这个脚本使用我们构建的AI工作流替代原来的本地三联动模型，作为咨询师（Counselor Agent）与来访者（Client Agent）进行对话。

## 主要变化

1. **Counselor Agent**: 从本地三联动模型 → AI工作流（5Ps病例分析 + 思考回复）
2. **API调用**: 使用阿里云百炼API，无需本地GPU
3. **简化部署**: 不需要加载大型模型文件

## 使用方法

### 1. 基础使用

```bash
# 使用环境变量中的API密钥
python chat_with_workflow.py

# 直接指定API密钥
python chat_with_workflow.py --api_key your-api-key

# 限制处理样本数量
python chat_with_workflow.py --num_samples 3
```

### 2. 配置参数

```bash
python chat_with_workflow.py \
  --api_key your-api-key \
  --output_dir ../output-workflow \
  --max_turns 20 \
  --num_samples 5
```

### 3. 参数说明

- `--api_key`: 阿里云百炼API密钥（必需）
- `--output_dir`: 输出目录（默认：`../output-workflow`）
- `--max_turns`: 最大对话轮数（默认：20）
- `--num_samples`: 处理样本数量（默认：全部）

## 工作流程

```
来访者输入
    ↓
Client Agent (保持原有逻辑)
    ↓
历史记录
    ↓
Counselor Agent (AI工作流)
  ├─ 5Ps病例分析
  └─ 思考与回复
    ↓
咨询师回复
```

## 与原版本的区别

| 特性 | 原版本 | 新版本 |
|------|--------|--------|
| Counselor Agent | 本地三联动模型 | AI工作流 |
| 硬件要求 | 需要GPU | 无需GPU |
| 模型加载 | 需要下载模型文件 | 使用API |
| 部署复杂度 | 较高 | 较低 |
| 响应速度 | 较快（本地） | 取决于网络 |

## 输出格式

输出格式与原版本保持一致，生成 `session_*.json` 文件：

```json
{
  "example": {...},
  "cbt_technique": "AI Workflow (5Ps Case Analysis + Think & Reply)",
  "cbt_plan": "AI工作流: 基于阿里云百炼API...",
  "cost": 0,
  "history": [...]
}
```

## 注意事项

1. **API密钥安全**: 不要在代码中硬编码API密钥，使用环境变量或配置文件
2. **网络连接**: 需要稳定的网络连接调用API
3. **费用控制**: 注意API调用量和费用
4. **错误处理**: 网络问题可能导致调用失败，脚本会保存错误信息

## 故障排除

### API密钥错误
```
错误: 请设置API密钥
```
解决方案：检查API密钥是否正确设置

### 网络连接问题
```
[错误] 工作流处理失败: connection timeout
```
解决方案：检查网络连接和API服务状态

### Client Agent无法工作
```
警告: 无法导入 llm_client
```
解决方案：确保 `.env` 文件配置正确，且 `llm_client.py` 文件存在

## 示例

```bash
# 处理3个样本
python chat_with_workflow.py --num_samples 3

# 使用自定义输出目录
python chat_with_workflow.py --output_dir my_results
```
