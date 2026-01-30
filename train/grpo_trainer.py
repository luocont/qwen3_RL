"""
GRPO训练模块
使用trl库实现GRPO算法训练
"""
import os
import torch
import numpy as np
from typing import List, Dict, Optional, Callable
from dataclasses import dataclass
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import PreTrainedModel, PreTrainedTokenizer

try:
    from trl import GRPOTrainer as TrlGRPOTrainer, GRPOConfig
    TRL_AVAILABLE = True
except ImportError:
    TRL_AVAILABLE = False
    GRPOConfig = None

from reward.rewards import RewardFunction, compute_rewards


@dataclass
class GRPOTrainingConfig:
    """GRPO训练配置"""
    # 模型配置
    model_name: str = "Qwen/Qwen2.5-7B-Instruct"
    max_prompt_length: int = 2048
    max_response_length: int = 1024

    # GRPO配置
    group_size: int = 4
    learning_rate: float = 1e-5
    num_train_epochs: int = 3
    batch_size: int = 1

    # 奖励权重
    reward_format_weight: float = 1.0
    reward_continuity_weight: float = 1.0
    reward_item_nums_weight: float = 1.0

    # 5Ps检查阈值
    five_ps_check_threshold: int = 10
    five_ps_min_items: int = 3
    five_ps_max_items: int = 4

    # WandB配置
    wandb_project: str = "qwen3-5ps-rl"
    wandb_entity: Optional[str] = None

    # 训练输出
    output_dir: str = "./output"
    save_steps: int = 100
    logging_steps: int = 10

    # 其他
    gradient_accumulation_steps: int = 4
    warmup_ratio: float = 0.1


class GRPOTrainer:
    """
    GRPO训练器
    封装GRPO训练逻辑
    """

    def __init__(
        self,
        config: GRPOTrainingConfig,
        reward_function: RewardFunction,
        tokenizer: PreTrainedTokenizer,
        model: PreTrainedModel,
        ref_model: Optional[PreTrainedModel] = None
    ):
        """
        初始化训练器

        Args:
            config: 训练配置
            reward_function: 奖励函数
            tokenizer: 分词器
            model: 策略模型
            ref_model: 参考模型（用于KL散度计算）
        """
        self.config = config
        self.reward_function = reward_function
        self.tokenizer = tokenizer
        self.model = model
        self.ref_model = ref_model

        # 初始化WandB
        self._init_wandb()

    def _init_wandb(self):
        """初始化WandB"""
        try:
            import wandb

            # 从环境变量读取API key和entity
            wandb_api_key = os.environ.get("WANDB_API_KEY")
            wandb_entity = os.environ.get("WANDB_ENTITY") or self.config.wandb_entity

            if wandb_api_key:
                wandb.login(key=wandb_api_key)

            wandb.init(
                project=self.config.wandb_project,
                entity=wandb_entity,
                config={
                    "model_name": self.config.model_name,
                    "group_size": self.config.group_size,
                    "learning_rate": self.config.learning_rate,
                    "max_prompt_length": self.config.max_prompt_length,
                    "max_response_length": self.config.max_response_length,
                }
            )
            self.wandb = wandb
        except ImportError:
            print("WandB not installed, skipping logging")
            self.wandb = None
        except Exception as e:
            print(f"Failed to initialize WandB: {e}")
            self.wandb = None

    def generate_responses(
        self,
        prompts: List[str],
        group_size: int
    ) -> List[List[str]]:
        """
        生成多个候选回复

        Args:
            prompts: prompt列表
            group_size: 每个prompt生成的候选数量

        Returns:
            生成的回复列表，形状为 [len(prompts), group_size]
        """
        all_responses = []

        self.model.eval()
        with torch.no_grad():
            for prompt in prompts:
                group_responses = []
                for _ in range(group_size):
                    # Tokenize
                    inputs = self.tokenizer(
                        prompt,
                        return_tensors="pt",
                        max_length=self.config.max_prompt_length,
                        truncation=True
                    ).to(self.model.device)

                    # Generate
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=self.config.max_response_length,
                        do_sample=True,
                        temperature=0.7,
                        top_p=0.9,
                        pad_token_id=self.tokenizer.pad_token_id
                    )

                    # Decode
                    response = self.tokenizer.decode(
                        outputs[0][inputs['input_ids'].shape[1]:],
                        skip_special_tokens=True
                    )
                    group_responses.append(response)

                all_responses.append(group_responses)

        return all_responses

    def compute_advantages(
        self,
        rewards: List[List[float]],
        group_size: int
    ) -> List[List[float]]:
        """
        计算优势函数

        Args:
            rewards: 奖励列表，形状为 [batch_size, group_size]
            group_size: 组大小

        Returns:
            优势值列表
        """
        advantages = []

        for group_rewards in rewards:
            # 计算组内平均奖励
            mean_reward = np.mean(group_rewards)

            # 计算优势
            group_advantages = [r - mean_reward for r in group_rewards]
            advantages.append(group_advantages)

        return advantages

    def train_step(
        self,
        prompts: List[str],
        previous_5ps_cases: List[str],
        turn_counts: List[int]
    ) -> Dict[str, float]:
        """
        执行一步训练

        Args:
            prompts: prompt列表
            previous_5ps_cases: 对应的上轮5Ps病例
            turn_counts: 对应的对话轮次

        Returns:
            训练指标字典
        """
        # 1. 生成候选回复
        responses = self.generate_responses(prompts, self.config.group_size)

        # 2. 计算奖励
        all_rewards = []
        for i, prompt in enumerate(prompts):
            group_rewards = []
            for response in responses[i]:
                result = self.reward_function.compute(
                    response,
                    previous_5ps_cases[i],
                    turn_counts[i]
                )
                group_rewards.append(result.total_reward)
            all_rewards.append(group_rewards)

        # 3. 计算优势
        advantages = self.compute_advantages(all_rewards, self.config.group_size)

        # 4. 更新策略（使用TRL的GRPOTrainer）
        # 这里需要构造适合TRL的数据格式
        if TRL_AVAILABLE:
            # 使用TRL的GRPOTrainer
            pass
        else:
            # 手动实现PPO更新
            self._ppo_update(prompts, responses, advantages)

        # 5. 收集指标
        metrics = self._compute_metrics(all_rewards, advantages)

        # 6. 记录到WandB
        if self.wandb:
            self.wandb.log(metrics)

        return metrics

    def _ppo_update(
        self,
        prompts: List[str],
        responses: List[List[str]],
        advantages: List[List[float]]
    ):
        """
        手动实现PPO更新

        Args:
            prompts: prompt列表
            responses: 生成的回复列表
            advantages: 优势值列表
        """
        # 构造训练数据
        self.model.train()

        for i, prompt in enumerate(prompts):
            for j, response in enumerate(responses[i]):
                advantage = advantages[i][j]

                # Tokenize prompt和response
                prompt_ids = self.tokenizer.encode(
                    prompt,
                    max_length=self.config.max_prompt_length,
                    truncation=True
                )
                response_ids = self.tokenizer.encode(
                    response,
                    max_length=self.config.max_response_length,
                    truncation=True
                )

                # 拼接
                full_ids = prompt_ids + response_ids
                input_ids = torch.tensor([full_ids]).to(self.model.device)

                # 计算log prob
                outputs = self.model(input_ids, labels=input_ids)
                loss = outputs.loss

                # PPO损失（简化版）
                policy_loss = -advantage * loss

                # 反向传播
                policy_loss.backward()

        # 更新参数
        # （实际实现需要更复杂的梯度处理和优化器更新）

    def _compute_metrics(
        self,
        rewards: List[List[float]],
        advantages: List[List[float]]
    ) -> Dict[str, float]:
        """
        计算训练指标

        Args:
            rewards: 奖励列表
            advantages: 优势值列表

        Returns:
            指标字典
        """
        flat_rewards = [r for group in rewards for r in group]
        flat_advantages = [a for group in advantages for a in group]

        return {
            "reward/mean": np.mean(flat_rewards),
            "reward/std": np.std(flat_rewards),
            "reward/max": np.max(flat_rewards),
            "reward/min": np.min(flat_rewards),
            "advantage/mean": np.mean(flat_advantages),
            "advantage/std": np.std(flat_advantages),
        }

    def save_model(self, path: str):
        """
        保存模型

        Args:
            path: 保存路径
        """
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)

    def finish(self):
        """结束训练"""
        if self.wandb:
            self.wandb.finish()


def create_trainer(
    config: GRPOTrainingConfig,
    system_prompt: str
) -> GRPOTrainer:
    """
    创建训练器

    Args:
        config: 训练配置
        system_prompt: 系统提示词

    Returns:
        GRPOTrainer实例
    """
    # 初始化奖励函数
    reward_function = RewardFunction(
        format_weight=config.reward_format_weight,
        continuity_weight=config.reward_continuity_weight,
        item_nums_weight=config.reward_item_nums_weight,
        check_threshold=config.five_ps_check_threshold,
        min_items=config.five_ps_min_items,
        max_items=config.five_ps_max_items
    )

    # 加载模型和tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name,
        trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )

    # 创建训练器
    trainer = GRPOTrainer(
        config=config,
        reward_function=reward_function,
        tokenizer=tokenizer,
        model=model
    )

    return trainer
