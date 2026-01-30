#!/usr/bin/env python
"""
Qwen3 5Ps强化学习训练入口脚本
使用GRPO算法训练模型生成规范的5Ps心理咨询格式
"""
import os
import sys
import argparse
import random
import numpy as np
import torch
from tqdm import tqdm
from typing import List

from config import TrainingConfig, load_system_prompt
from data.dataset import QwenRLDataset
from train.grpo_trainer import create_trainer, GRPOTrainingConfig


def set_seed(seed: int):
    """设置随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_epoch(
    dataset: QwenRLDataset,
    trainer,
    epoch: int,
    steps_per_epoch: int,
    config: TrainingConfig
):
    """
    训练一个epoch

    Args:
        dataset: 数据集
        trainer: 训练器
        epoch: 当前epoch
        steps_per_epoch: 每个epoch的步数
        config: 训练配置
    """
    print(f"\n{'='*50}")
    print(f"Epoch {epoch + 1}/{config.num_train_epochs}")
    print(f"{'='*50}")

    # 重置数据集
    dataset.current_message_idx = 0
    dataset.current_turn_idx = 1

    progress_bar = tqdm(
        range(steps_per_epoch),
        desc=f"Epoch {epoch + 1}",
        unit="step"
    )

    cumulative_metrics = {
        "reward/mean": [],
        "reward/std": [],
        "reward/max": [],
        "reward/min": [],
        "advantage/mean": [],
        "advantage/std": [],
    }

    for step in progress_bar:
        # 收集一个batch的样本
        prompts = []
        previous_5ps_cases = []
        turn_counts = []

        batch_size = config.batch_size
        for _ in range(batch_size):
            sample = dataset.get_next_sample()
            if sample is None:
                # 数据集遍历完成，重新开始
                sample = dataset.get_random_sample()

            prompts.append(sample.prompt)
            previous_5ps_cases.append(sample.previous_5ps_case)
            turn_counts.append(sample.message_state.turn_count)

        # 执行训练步骤
        try:
            metrics = trainer.train_step(
                prompts=prompts,
                previous_5ps_cases=previous_5ps_cases,
                turn_counts=turn_counts
            )

            # 更新累积指标
            for key, value in metrics.items():
                if key in cumulative_metrics:
                    cumulative_metrics[key].append(value)

            # 更新进度条
            progress_bar.set_postfix({
                "reward": f"{metrics['reward/mean']:.4f}",
                "adv": f"{metrics['advantage/mean']:.4f}"
            })

            # 定期保存模型
            if (step + 1) % config.save_steps == 0:
                save_path = os.path.join(
                    config.output_dir,
                    f"checkpoint-epoch-{epoch + 1}-step-{step + 1}"
                )
                trainer.save_model(save_path)
                print(f"\n模型已保存到: {save_path}")

        except Exception as e:
            print(f"\n训练步骤出错 (step {step + 1}): {e}")
            import traceback
            traceback.print_exc()
            continue

    # 打印epoch统计
    print(f"\nEpoch {epoch + 1} 完成，平均指标:")
    for key, values in cumulative_metrics.items():
        if values:
            avg_value = np.mean(values)
            print(f"  {key}: {avg_value:.4f}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="Qwen3 5Ps强化学习训练"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default=None,
        help="模型名称或本地路径（默认从环境变量读取）"
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="RL.json",
        help="训练数据路径"
    )
    parser.add_argument(
        "--prompt_path",
        type=str,
        default="prompts.txt",
        help="系统提示词路径"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./output",
        help="模型输出目录"
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=None,
        help="训练轮数"
    )
    parser.add_argument(
        "--group_size",
        type=int,
        default=None,
        help="GRPO group size"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=None,
        help="学习率"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="批次大小"
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default=None,
        help="WandB项目名称"
    )
    parser.add_argument(
        "--steps_per_epoch",
        type=int,
        default=100,
        help="每个epoch的训练步数"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子"
    )

    args = parser.parse_args()

    # 设置随机种子
    set_seed(args.seed)

    # 加载配置
    config = TrainingConfig()

    # 命令行参数覆盖配置
    if args.model_name:
        config.model_name = args.model_name
    if args.data_path:
        config.data_path = args.data_path
    if args.prompt_path:
        config.prompt_path = args.prompt_path
    if args.output_dir:
        config.output_dir = args.output_dir
    if args.num_epochs:
        config.num_train_epochs = args.num_epochs
    if args.group_size:
        config.group_size = args.group_size
    if args.learning_rate:
        config.learning_rate = args.learning_rate
    if args.batch_size:
        config.batch_size = args.batch_size
    if args.wandb_project:
        config.wandb_project = args.wandb_project

    # 打印配置
    print("\n" + "="*50)
    print("Qwen3 5Ps强化学习训练")
    print("="*50)
    print(f"模型: {config.model_name}")
    print(f"数据: {config.data_path}")
    print(f"提示词: {config.prompt_path}")
    print(f"输出目录: {config.output_dir}")
    print(f"训练轮数: {config.num_train_epochs}")
    print(f"Group Size: {config.group_size}")
    print(f"学习率: {config.learning_rate}")
    print(f"批次大小: {config.batch_size}")
    print(f"WandB项目: {config.wandb_project}")
    print(f"每轮步数: {args.steps_per_epoch}")
    print("="*50 + "\n")

    # 加载系统提示词
    print(f"加载系统提示词: {config.prompt_path}")
    system_prompt = load_system_prompt(config.prompt_path)

    # 加载数据集
    print(f"加载数据集: {config.data_path}")
    dataset = QwenRLDataset(config.data_path, system_prompt)
    print(f"数据集大小: {len(dataset)} 个message\n")

    # 创建训练器
    print("创建训练器...")
    grpo_config = GRPOTrainingConfig(
        model_name=config.model_name,
        max_prompt_length=config.max_prompt_length,
        max_response_length=config.max_response_length,
        group_size=config.group_size,
        learning_rate=config.learning_rate,
        num_train_epochs=config.num_train_epochs,
        batch_size=config.batch_size,
        reward_format_weight=config.reward_format_weight,
        reward_continuity_weight=config.reward_continuity_weight,
        reward_item_nums_weight=config.reward_item_nums_weight,
        five_ps_check_threshold=config.five_ps_check_threshold,
        five_ps_min_items=config.five_ps_min_items,
        five_ps_max_items=config.five_ps_max_items,
        wandb_project=config.wandb_project,
        wandb_entity=config.wandb_entity,
        output_dir=config.output_dir,
        save_steps=config.save_steps,
        logging_steps=config.logging_steps,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        warmup_ratio=config.warmup_ratio,
    )
    trainer = create_trainer(grpo_config, system_prompt)
    print("训练器创建完成\n")

    # 训练
    try:
        for epoch in range(config.num_train_epochs):
            train_epoch(
                dataset=dataset,
                trainer=trainer,
                epoch=epoch,
                steps_per_epoch=args.steps_per_epoch,
                config=config
            )

        # 保存最终模型
        final_model_path = os.path.join(config.output_dir, "final_model")
        trainer.save_model(final_model_path)
        print(f"\n最终模型已保存到: {final_model_path}")

    except KeyboardInterrupt:
        print("\n训练被用户中断")
    except Exception as e:
        print(f"\n训练出错: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # 清理
        trainer.finish()
        print("\n训练结束")


if __name__ == "__main__":
    main()
