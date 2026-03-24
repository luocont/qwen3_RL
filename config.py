"""
配置文件
"""

# 阿里云百炼API密钥
# 可以通过环境变量 DASHSCOPE_API_KEY 设置，或在此处直接填写
DASHSCOPE_API_KEY = "sk-40fb3997d3ed485ba390a9c4ae3bd2d2"  # 请替换为你的API密钥

# 模型配置
MODELS = {
    "case_analysis": "qwen3-max",      # 5Ps病例分析模型
    "think_reply": "qwen3-max"        # 思考回复模型
}

# 文件路径配置
PATHS = {
    "rl_json": "e:/GitLoadWareHouse/work/智能体/RL.json",
    "output": "e:/GitLoadWareHouse/work/智能体/workflow_results.json",
    "five_ps_prompt": "prompts/five_ps_teacher.py",
    "think_reply_prompt": "prompts/think_and_reply_teacher.py"
}
