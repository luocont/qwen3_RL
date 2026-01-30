"""
训练配置文件
包含所有GRPO训练的配置参数
"""
import os
from dataclasses import dataclass, field


@dataclass
class TrainingConfig:
    """GRPO训练配置"""

    # ==================== 模型配置 ====================
    model_name: str = field(
        default_factory=lambda: os.environ.get(
            "MODEL_NAME",
            "Qwen/Qwen2.5-7B-Instruct"
        )
    )
    """模型名称或本地路径"""

    max_prompt_length: int = 2048
    """prompt最大长度"""

    max_response_length: int = 1024
    """生成回复最大长度"""

    # ==================== GRPO配置 ====================
    group_size: int = 4
    """每次生成的候选回复数量"""

    learning_rate: float = 1e-5
    """学习率"""

    num_train_epochs: int = 3
    """训练轮数"""

    batch_size: int = 1
    """批次大小（根据显存调整）"""

    gradient_accumulation_steps: int = 4
    """梯度累积步数"""

    warmup_ratio: float = 0.1
    """预热比例"""

    # ==================== 奖励权重 ====================
    reward_format_weight: float = 1.0
    """格式奖励权重"""

    reward_continuity_weight: float = 1.0
    """渐进性奖励权重"""

    reward_item_nums_weight: float = 1.0
    """条目数奖励权重"""

    # ==================== 5Ps检查配置 ====================
    five_ps_check_threshold: int = 10
    """开始检查条目数的对话轮次阈值"""

    five_ps_min_items: int = 3
    """每个P的最少条目数"""

    five_ps_max_items: int = 4
    """每个P的最多条目数"""

    # ==================== WandB配置 ====================
    wandb_project: str = "qwen3-5ps-rl"
    """WandB项目名称"""

    wandb_entity: str = field(
        default_factory=lambda: os.environ.get("WANDB_ENTITY", None)
    )
    """WandB实体（从环境变量读取）"""

    wandb_api_key: str = field(
        default_factory=lambda: os.environ.get("WANDB_API_KEY", "")
    )
    """WandB API密钥（从环境变量读取）"""

    # ==================== 训练输出配置 ====================
    output_dir: str = "./output"
    """模型输出目录"""

    save_steps: int = 100
    """每多少步保存一次模型"""

    logging_steps: int = 10
    """每多少步记录一次日志"""

    # ==================== 数据配置 ====================
    data_path: str = "RL.json"
    """训练数据路径"""

    prompt_path: str = "prompts.txt"
    """系统提示词路径"""

    # ==================== 生成配置 ====================
    generation_temperature: float = 0.7
    """生成温度"""

    generation_top_p: float = 0.9
    """生成top_p"""

    generation_do_sample: bool = True
    """是否采样生成"""

    # ==================== 其他配置 ====================
    seed: int = 42
    """随机种子"""

    use_fp16: bool = False
    """是否使用fp16"""

    use_bf16: bool = True
    """是否使用bf16"""

    gradient_checkpointing: bool = False
    """是否使用梯度检查点"""

    def __post_init__(self):
        """配置后处理"""
        # 确保输出目录存在
        os.makedirs(self.output_dir, exist_ok=True)

        # 验证奖励权重
        if self.reward_format_weight < 0:
            raise ValueError("reward_format_weight must be non-negative")
        if self.reward_continuity_weight < 0:
            raise ValueError("reward_continuity_weight must be non-negative")
        if self.reward_item_nums_weight < 0:
            raise ValueError("reward_item_nums_weight must be non-negative")

        # 验证5Ps配置
        if self.five_ps_min_items > self.five_ps_max_items:
            raise ValueError("five_ps_min_items must be <= five_ps_max_items")


def get_config() -> TrainingConfig:
    """
    获取训练配置

    Returns:
        TrainingConfig实例
    """
    return TrainingConfig()


def load_system_prompt(prompt_path: str = "prompts.txt") -> str:
    """
    加载系统提示词

    Args:
        prompt_path: 提示词文件路径

    Returns:
        系统提示词字符串
    """
    with open(prompt_path, 'r', encoding='utf-8') as f:
        return f.read()
