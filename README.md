# AI病例分析工作流

基于5Ps个案概念化框架和理情行为疗法（REBT）的AI心理工作流系统。

## 工作流程

```
患者输入
    ↓
Step 1: 5Ps模型分析病例
    ↓
Step 2: 思考与回复模型处理
    ↓
Step 3: 更新历史记录
    ↓
下一轮对话
```

## 功能特性

- **5Ps病例分析**：使用5Ps框架结构化分析患者信息
- **思考与回复**：基于REBT疗法提供专业的心理咨询
- **历史记录管理**：自动维护病例和对话历史
- **多种运行模式**：支持批处理、交互式和单次处理

## 安装依赖

```bash
pip install openai
```

## 配置

### 方法1：环境变量（推荐）
```bash
export DASHSCOPE_API_KEY="your-api-key"
```

### 方法2：配置文件
编辑 `config.py`，填入你的API密钥：
```python
DASHSCOPE_API_KEY = "sk-your-actual-api-key"
```

## 使用方法

### 1. 批处理模式（处理RL.json中的用户输入）

```bash
# 处理所有用户输入
python main.py --mode batch

# 只处理前3轮对话
python main.py --mode batch --max-rounds 3

# 指定输入输出文件
python main.py --mode batch --input data.json --output results.json
```

### 2. 交互式模式（实时对话）

```bash
python main.py --mode interactive
```

### 3. 单次处理模式

```bash
python main.py --mode single --message "我最近感到很焦虑"
```

## 项目结构

```
智能体/
├── main.py                        # 主程序入口
├── workflow.py                    # 工作流核心逻辑
├── config.py                      # 配置文件
├── prompts/                       # 提示词
│   ├── five_ps_teacher.py        # 5Ps病例分析提示词
│   └── think_and_reply_teacher.py # 思考回复提示词
├── RL.json                        # 输入数据
└── workflow_results.json          # 输出结果
```

## 提示词说明

### 5Ps教师模型
负责使用5Ps框架（Presenting、Predisposing、Precipitating、Perpetuating、Protective）对患者的病例进行结构化分析。

### 思考与回复教师模型
基于理情行为疗法（REBT），根据5Ps病例内容进行思考并给出专业的心理咨询回复。

## 输出格式

### 5Ps病例输出
```
<5Ps>
P1 主诉：（患者当前主诉）

P2 易感因素：（患者的易感因素）

P3 诱发因素：（诱发事件）

P4 维持因素：（维持因素）

P5 保护因素：（保护因素/资源）
</5Ps>
```

### 思考与回复输出
```
思考阶段:前/后期

思考内容

回复内容
```

## 技术栈

- **语言**: Python 3.7+
- **API**: 阿里云百炼（DashScope）
- **模型**: Qwen-Max, Qwen3-Max
- **框架**: OpenAI Compatible API

## 注意事项

1. 确保API密钥已正确配置
2. RL.json文件格式需符合要求
3. 每个messages之间的历史记录不共通
4. 建议使用环境变量管理敏感信息

## 故障排除

### API密钥错误
```
错误：请设置API密钥
```
解决方案：检查环境变量或配置文件中的API密钥是否正确

### 文件路径错误
```
FileNotFoundError: [Errno 2] No such file or directory
```
解决方案：检查RL.json文件路径是否正确

### 网络连接问题
```
ConnectionError
```
解决方案：检查网络连接和API服务状态
