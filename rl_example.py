# -*- coding: utf-8 -*-
import os
import re
import json
import math
import time
import argparse
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed

# 设置 NCCL 超时时间(秒) - 因为奖励计算需要调用API,可能很慢
os.environ.setdefault("TORCH_NCCL_BLOCKING_WAIT", "1")
os.environ.setdefault("TORCH_NCCL_ASYNC_ERROR_HANDLING", "1")
# 增加超时时间到 30 分钟 (因为 API 调用可能很慢)
os.environ.setdefault("TORCH_DISTRIBUTED_TIMEOUT_MINUTES", "30")

import torch
import torch.distributed as dist
from openai import OpenAI
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainerCallback
from peft import LoraConfig, get_peft_model
import wandb
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt
from collections import deque

from trl import GRPOTrainer, GRPOConfig

# -----------------------------
# Rewards Tracker: track per-step averages of R_format, R_think, R_reply
# -----------------------------
class RewardsTracker:
    def __init__(self, output_dir: str, is_main_process: bool = True, plot_every_n_steps: int = 10):
        self.is_main_process = is_main_process
        self.plot_every_n_steps = plot_every_n_steps
        self.sum_format = 0.0
        self.sum_think = 0.0
        self.sum_reply = 0.0
        self.sum_length = 0.0
        self.sum_qus = 0.0
        self.count = 0
        self.format_history = deque(maxlen=1000)
        self.think_history = deque(maxlen=1000)
        self.reply_history = deque(maxlen=1000)
        self.length_history = deque(maxlen=1000)
        self.qus_history = deque(maxlen=1000)
        self.step_history = deque(maxlen=1000)
        self.plots_dir = os.path.join(output_dir, "reward_plots")
        if self.is_main_process:
            os.makedirs(self.plots_dir, exist_ok=True)

    def add(self, r_format: float, r_think: float, r_reply: float, r_length: float | None = None, r_qus: float | None = None):
        try:
            self.sum_format += float(r_format)
            self.sum_think += float(r_think)
            self.sum_reply += float(r_reply)
            if r_length is not None:
                self.sum_length += float(r_length)
            if r_qus is not None:
                self.sum_qus += float(r_qus)
            self.count += 1
        except Exception as e:
            if self.is_main_process:
                print(f"[奖励监控] add() 出错: {e}")

    def finalize_step(self, current_step: int):
        if self.count == 0:
            return None
        mean_format = self.sum_format / self.count
        mean_think = self.sum_think / self.count
        mean_reply = self.sum_reply / self.count
        mean_length = self.sum_length / self.count if self.sum_length != 0.0 else 0.0
        mean_qus = self.sum_qus / self.count if self.sum_qus != 0.0 else 0.0
        # reset accumulators for next step
        self.sum_format = 0.0
        self.sum_think = 0.0
        self.sum_reply = 0.0
        self.sum_length = 0.0
        self.sum_qus = 0.0
        self.count = 0
        # save history
        self.format_history.append(mean_format)
        self.think_history.append(mean_think)
        self.reply_history.append(mean_reply)
        self.length_history.append(mean_length)
        self.qus_history.append(mean_qus)
        self.step_history.append(current_step)
        return mean_format, mean_think, mean_reply, mean_length, mean_qus

    def plot(self, current_step: int):
        try:
            if len(self.step_history) <= 1:
                return
            fig, ax = plt.subplots(1, 1, figsize=(12, 5))
            steps = list(self.step_history)
            ax.plot(steps, list(self.format_history), 'g-', linewidth=2, label='R_format (mean)')
            ax.plot(steps, list(self.think_history), 'b-', linewidth=2, label='R_think (mean)')
            ax.plot(steps, list(self.reply_history), 'r-', linewidth=2, label='R_reply (mean)')
            ax.plot(steps, list(self.length_history), 'm-', linewidth=2, label='R_length (mean)')
            ax.plot(steps, list(self.qus_history), 'c-', linewidth=2, label='R_qus (mean)')
            ax.set_xlabel('Training Step', fontsize=12)
            ax.set_ylabel('Reward (mean)', fontsize=12)
            ax.set_title(f'Reward components (mean) over Training (Step {current_step})', fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=10)
            plt.tight_layout()
            plot_path = os.path.join(self.plots_dir, f'reward_components_step_{current_step}.png')
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            if self.is_main_process and os.path.exists(plot_path):
                print(f"✅ [图表保存成功] 奖励组件曲线已保存到: {plot_path}")
                if wandb.run is not None:
                    wandb.log({"reward_components": wandb.Image(plot_path), "custom/step": current_step})
        except Exception as e:
            import traceback
            print(f"❌ [错误] 绘制奖励组件曲线时出错: {e}")
            print(f"❌ [错误详情] {traceback.format_exc()}")

# -----------------------------
# KL Divergence Monitoring Callback
# -----------------------------
class KLMonitorCallback(TrainerCallback):
    """
    自定义回调函数,用于监控和可视化KL散度
    """
    def __init__(self, output_dir: str, is_main_process: bool = True, plot_every_n_steps: int = 10):
        self.output_dir = output_dir
        self.is_main_process = is_main_process
        self.plot_every_n_steps = plot_every_n_steps

        # 存储历史数据
        self.kl_history = deque(maxlen=1000)  # 最多保留1000个点
        self.kl_coef_history = deque(maxlen=1000)
        self.step_history = deque(maxlen=1000)

        # 创建图表保存目录
        self.plots_dir = os.path.join(output_dir, "kl_plots")
        if self.is_main_process:
            os.makedirs(self.plots_dir, exist_ok=True)

    def on_log(self, args, state, control, logs=None, **kwargs):
        """
        在每次日志记录时调用,提取并可视化KL散度
        """
        if not self.is_main_process:
            return

        # 调试: 打印所有日志键
        if logs is not None:
            print(f"\n[KL监控调试] 日志键: {list(logs.keys())}")
        else:
            print(f"\n[KL监控调试] logs为None")
            return

        # 提取KL相关指标 - 尝试多种可能的键名
        kl_div = logs.get('objective/kl', None)
        if kl_div is None:
            kl_div = logs.get('kl', None)
        if kl_div is None:
            kl_div = logs.get('train/kl', None)

        kl_coef = logs.get('objective/kl_coef', None)
        if kl_coef is None:
            kl_coef = logs.get('kl_coef', None)

        current_step = state.global_step

        # 打印KL散度信息
        if kl_div is not None:
            print(f"\n{'='*60}")
            print(f"📊 [KL监控] Step {current_step}")
            print(f"{'='*60}")
            print(f"📈 KL散度: {kl_div:.6f}")
            if kl_coef is not None:
                print(f"📈 KL系数: {kl_coef:.6f}")
        else:
            print(f"[KL监控调试] 未找到KL散度数据,尝试的键名: 'objective/kl', 'kl', 'train/kl'")
            return

        # 记录到wandb (额外的自定义日志)
        if wandb.run is not None:
            wandb.log({
                "custom/kl_divergence": kl_div,
                "custom/step": current_step,
            })
            if kl_coef is not None:
                wandb.log({
                    "custom/kl_coefficient": kl_coef,
                })

        # 保存历史数据
        self.kl_history.append(kl_div)
        if kl_coef is not None:
            self.kl_coef_history.append(kl_coef)
        self.step_history.append(current_step)

        # 打印统计信息
        if len(self.kl_history) > 1:
            avg_kl = sum(self.kl_history) / len(self.kl_history)
            min_kl = min(self.kl_history)
            max_kl = max(self.kl_history)
            print(f"📊 KL统计 (最近{len(self.kl_history)}步):")
            print(f"   平均: {avg_kl:.6f}")
            print(f"   最小: {min_kl:.6f}")
            print(f"   最大: {max_kl:.6f}")
        print(f"{'='*60}\n")

        # 定期绘制图表
        print(f"[KL监控调试] 当前步数: {current_step}, 绘图间隔: {self.plot_every_n_steps}, 历史长度: {len(self.kl_history)}")
        if current_step % self.plot_every_n_steps == 0 and len(self.kl_history) > 1:
            print(f"[KL监控] 触发绘图条件,开始绘制KL曲线...")
            self._plot_kl_curves(current_step)
        elif len(self.kl_history) <= 1:
            print(f"[KL监控] 数据点不足(需要>1个点),暂不绘图")
        else:
            print(f"[KL监控] 未到绘图步数,下次绘图步数: {(current_step // self.plot_every_n_steps + 1) * self.plot_every_n_steps}")

    def _plot_kl_curves(self, current_step):
        """
        绘制KL散度曲线并保存
        """
        print(f"\n[_plot_kl_curves] 开始绘制KL曲线, current_step={current_step}")
        print(f"[_plot_kl_curves] 数据点数量: {len(self.kl_history)}")
        print(f"[_plot_kl_curves] 保存目录: {self.plots_dir}")

        try:
            fig, axes = plt.subplots(2, 1, figsize=(12, 8))
            print(f"[_plot_kl_curves] 成功创建图表对象")

            steps = list(self.step_history)
            kl_values = list(self.kl_history)

            # 绘制KL散度曲线
            axes[0].plot(steps, kl_values, 'b-', linewidth=2, label='KL Divergence')
            axes[0].set_xlabel('Training Step', fontsize=12)
            axes[0].set_ylabel('KL Divergence', fontsize=12)
            axes[0].set_title(f'KL Divergence over Training (Step {current_step})', fontsize=14, fontweight='bold')
            axes[0].grid(True, alpha=0.3)
            axes[0].legend(fontsize=10)

            # 如果有KL系数历史,也绘制出来
            if len(self.kl_coef_history) > 0:
                kl_coef_values = list(self.kl_coef_history)
                axes[1].plot(steps, kl_coef_values, 'r-', linewidth=2, label='KL Coefficient')
                axes[1].set_xlabel('Training Step', fontsize=12)
                axes[1].set_ylabel('KL Coefficient', fontsize=12)
                axes[1].set_title(f'KL Coefficient over Training (Step {current_step})', fontsize=14, fontweight='bold')
                axes[1].grid(True, alpha=0.3)
                axes[1].legend(fontsize=10)
            else:
                # 如果没有KL系数,绘制KL散度的移动平均
                if len(kl_values) >= 5:
                    window_size = min(20, len(kl_values) // 5)
                    moving_avg = []
                    for i in range(len(kl_values)):
                        start_idx = max(0, i - window_size + 1)
                        moving_avg.append(sum(kl_values[start_idx:i+1]) / (i - start_idx + 1))

                    axes[1].plot(steps, kl_values, 'b-', alpha=0.3, linewidth=1, label='KL Divergence (Raw)')
                    axes[1].plot(steps, moving_avg, 'r-', linewidth=2, label=f'Moving Average (window={window_size})')
                    axes[1].set_xlabel('Training Step', fontsize=12)
                    axes[1].set_ylabel('KL Divergence', fontsize=12)
                    axes[1].set_title(f'KL Divergence with Moving Average (Step {current_step})', fontsize=14, fontweight='bold')
                    axes[1].grid(True, alpha=0.3)
                    axes[1].legend(fontsize=10)

            plt.tight_layout()

            # 保存图表
            plot_path = os.path.join(self.plots_dir, f'kl_curves_step_{current_step}.png')
            print(f"[_plot_kl_curves] 准备保存图表到: {plot_path}")

            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            print(f"[_plot_kl_curves] plt.savefig() 执行完成")

            plt.close(fig)
            print(f"[_plot_kl_curves] 图表已关闭")

            # 验证文件是否真的保存了
            if os.path.exists(plot_path):
                file_size = os.path.getsize(plot_path)
                print(f"✅ [图表保存成功] KL曲线已保存到: {plot_path}")
                print(f"✅ [文件大小] {file_size} 字节")
            else:
                print(f"❌ [警告] 图表文件不存在: {plot_path}")

            # 上传到wandb
            if wandb.run is not None:
                wandb.log({
                "custom/step": current_step,
                "kl_curves": wandb.Image(plot_path),
            })

        except Exception as e:
            import traceback
            print(f"❌ [错误] 绘制KL曲线时出错: {e}")
            print(f"❌ [错误详情] {traceback.format_exc()}")

    def on_train_end(self, args, state, control, **kwargs):
        """
        训练结束时绘制最终的KL曲线
        """
        if self.is_main_process and len(self.kl_history) > 0:
            print(f"\n{'='*60}")
            print(f"📊 [训练完成] 绘制最终KL曲线")
            print(f"{'='*60}\n")
            self._plot_kl_curves(state.global_step)

            # 保存历史数据到文件
            history_path = os.path.join(self.plots_dir, 'kl_history.json')
            history_data = {
                'steps': list(self.step_history),
                'kl_divergence': list(self.kl_history),
                'kl_coefficient': list(self.kl_coef_history),
            }
            with open(history_path, 'w', encoding='utf-8') as f:
                json.dump(history_data, f, indent=2, ensure_ascii=False)
            print(f"📊 [数据保存] KL历史数据已保存到: {history_path}\n")


# -----------------------------
# Reward Monitor Callback: record per-step mean of R_format/R_think/R_reply
# -----------------------------
class RewardMonitorCallback(TrainerCallback):
    def __init__(self, tracker: RewardsTracker, is_main_process: bool = True, plot_every_n_steps: int = 10):
        self.tracker = tracker
        self.is_main_process = is_main_process
        self.plot_every_n_steps = plot_every_n_steps

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not self.is_main_process:
            return
        current_step = state.global_step
        res = self.tracker.finalize_step(current_step)
        if res is None:
            # 没有收集到该步的奖励分量，可能还在计算中
            return
        mean_format, mean_think, mean_reply, mean_length, mean_qus = res
        print(f"\n{'='*60}")
        print(f"📊 [奖励监控] Step {current_step} 均值")
        print(f"{'='*60}")
        print(f"📈 R_format(mean): {mean_format:+.6f}")
        print(f"📈 R_think(mean) : {mean_think:+.6f}")
        print(f"📈 R_reply(mean) : {mean_reply:+.6f}")
        print(f"📈 R_length(mean): {mean_length:+.6f}")
        print(f"📈 R_qus(mean)  : {mean_qus:+.6f}")
        print(f"{'='*60}\n")
        # 记录到 wandb
        if wandb.run is not None:
            wandb.log({
                "custom/r_format_mean": float(mean_format),
                "custom/r_think_mean": float(mean_think),
                "custom/r_reply_mean": float(mean_reply),
                "custom/r_length_mean": float(mean_length),
                "custom/r_qus_mean": float(mean_qus),
                "custom/step": current_step,
            })
        # 绘图
        if current_step % self.plot_every_n_steps == 0:
            self.tracker.plot(current_step)

    def on_train_end(self, args, state, control, **kwargs):
        if self.is_main_process and len(self.tracker.step_history) > 0:
            # 绘制最终曲线
            self.tracker.plot(state.global_step)
            # 保存历史数据
            history_path = os.path.join(self.tracker.plots_dir, 'reward_history.json')
            history_data = {
                'steps': list(self.tracker.step_history),
                'r_format_mean': list(self.tracker.format_history),
                'r_think_mean': list(self.tracker.think_history),
                'r_reply_mean': list(self.tracker.reply_history),
                'r_length_mean': list(self.tracker.length_history),
                'r_qus_mean': list(self.tracker.qus_history),
            }
            with open(history_path, 'w', encoding='utf-8') as f:
                json.dump(history_data, f, indent=2, ensure_ascii=False)
            print(f"📊 [数据保存] 奖励组件历史均值已保存到: {history_path}\n")


# -----------------------------
# Rolling Dialogue Callback: 实现滚动多轮对话
# -----------------------------
class RollingDialogueCallback(TrainerCallback):
    """
    滚动多轮对话回调:
    - 在每个训练 step 后,将生成的最佳输出添加到历史中
    - 更新 dataset 中的 prompts,使其包含累积的历史对话

    核心逻辑:
    - 对于每个 prompt,生成 group_size 个回答
    - 选择奖励最高的回答作为该轮的最佳回答
    - 将最佳回答添加到该 prompt 的历史中
    - 下次生成时,所有 group_size 个采样都使用同一份历史
    """
    def __init__(
        self,
        dialogue_manager: 'GlobalDialogueHistory',
        tokenizer,
        is_main_process: bool = True,
        group_size: int = 4,
    ):
        self.dialogue_manager = dialogue_manager
        self.tokenizer = tokenizer
        self.is_main_process = is_main_process
        self.group_size = group_size
        # 保存 trainer 引用,用于更新 dataset
        self.trainer = None

    def on_train_begin(self, args, state, control, **kwargs):
        """训练开始时保存 trainer 引用"""
        if 'trainer' in kwargs:
            self.trainer = kwargs['trainer']

    def on_step_end(self, args, state, control, logs=None, **kwargs):
        """
        在每个 training step 结束后调用:
        1. 从 dialogue_manager 获取每个 prompt 的最佳回答
        2. 将最佳回答添加到历史中
        3. 更新 dataset 中的 prompts
        """
        if not self.is_main_process:
            return

        # 从 dialogue_manager 获取本轮需要处理的更新
        if hasattr(self.dialogue_manager, 'pending_updates'):
            pending = self.dialogue_manager.pending_updates
            if pending:
                print(f"\n{'='*60}")
                print(f"[滚动对话] Step {state.global_step} 完成")
                print(f"[滚动对话] 处理 {len(pending)} 个 prompt 的历史更新")
                print(f"{'='*60}\n")

                # 应用所有更新到历史
                for prompt_idx, best_completion in pending.items():
                    self.dialogue_manager.add_best_completion(prompt_idx, best_completion)
                    if self.is_main_process:
                        print(f"[滚动对话] Prompt #{prompt_idx + 1}: 添加最佳回答到历史")
                        print(f"[最佳回答预览]: {best_completion[:100]}...\n")

                # 清空待处理更新
                self.dialogue_manager.pending_updates.clear()

                # 关键: 更新 dataset 中的 prompts
                self._update_dataset_prompts()

    def _update_dataset_prompts(self):
        """使用更新后的历史重新格式化所有 prompts,并更新 dataset"""
        if self.trainer is None or not hasattr(self.trainer, 'train_dataset'):
            return

        # 获取所有 prompt 的数量
        num_prompts = len(self.dialogue_manager.all_user_messages)

        # 为每个 prompt 重新生成带历史的格式化 prompt
        new_prompts = []
        for prompt_idx in range(num_prompts):
            formatted_prompt = self.dialogue_manager.format_prompt_with_history(
                prompt_idx, self.tokenizer
            )
            new_prompts.append(formatted_prompt)

        # 更新 dataset
        from datasets import Dataset
        rows = [{"prompt": p} for p in new_prompts]
        new_dataset = Dataset.from_list(rows)

        # 替换 trainer 的 train_dataset
        self.trainer.train_dataset = new_dataset

        if self.is_main_process:
            print(f"[滚动对话] Dataset 已更新,包含 {len(new_prompts)} 条带历史的 prompts")
            # 打印第一条更新后的 prompt 示例
            if len(new_prompts) > 0:
                preview = new_prompts[0][:400] + "..." if len(new_prompts[0]) > 400 else new_prompts[0]
                print(f"[滚动对话] 更新后的 prompt 预览:\n{preview}\n")

    def on_epoch_end(self, args, state, control, **kwargs):
        """
        在每个 epoch 结束后可选择重置对话历史
        """
        if self.is_main_process:
            print(f"\n[滚动对话] Epoch 结束,保持历史继续累积\n")


# -----------------------------
# Utils: OpenAI-compatible chat
# -----------------------------

def _chat_completions(
    api_base: str,
    api_key: str,
    model: str,
    messages: list,
    response_format: dict | None = None,
    temperature: float = 0.7,
    max_tokens: int = 512,
    timeout: int = 60,
    is_main_process: bool = True,
):
    """
    OpenAI-compatible /v1/chat/completions caller using OpenAI client.
    """
    if not api_base or not api_key or not model:
        raise RuntimeError("Missing eval/teacher API config (api_base, api_key, model).")

    # 创建 OpenAI 客户端
    client = OpenAI(
        api_key=api_key,
        base_url=api_base,
        timeout=timeout,
    )

    # 构建请求参数
    kwargs = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if response_format is not None:
        kwargs["response_format"] = response_format

    # 打印请求信息 - 只在主进程打印
    if is_main_process:
        print(f"\n{'='*60}")
        print(f"🌐 [API调用] 模型: {model}")
        print(f"🌐 [API调用] 消息数量: {len(messages)}")
        if messages:
            print(f"🌐 [API调用] 第一条角色: {messages[0]['role']}")
        print(f"{'='*60}")

    # 调用 API
    completion = client.chat.completions.create(**kwargs)

    content = completion.choices[0].message.content

    # 打印响应内容 - 只在主进程打印
    if is_main_process:
        print(f"\n🌐 [API响应] 内容:")
        print(f"{'-'*60}")
        print(content)
        print(f"{'-'*60}\n")

    return content


def _safe_json_loads(s: str):
    try:
        return json.loads(s)
    except Exception:
        # 尝试截取第一个花括号对象
        try:
            start = s.find("{")
            end = s.rfind("}")
            if start != -1 and end != -1 and end > start:
                return json.loads(s[start : end + 1])
        except Exception:
            pass
    return None


# -----------------------------
# Prompts from your markdown
# -----------------------------

TEACHER_SYSTEM_PROMPT = (
    "你是一名资深心理咨询师（导师）。请只输出你的『内部思维』，"
    "用于指导学生对来访者的理解与风险识别。要求："
    "1) 用结构化要点梳理来访者的核心困扰、主要情绪、认知模式、人际/行为模式、重要背景、风险评估；"
    "2) 指出可继续探索的方向；"
    "3) 不要对来访者直接说话；不要给出最终建议；不要输出任何多余标记。"
)

R_THINK_EVAL_SYSTEM = (
    "你是一名心理咨询督导师和训练评估者。"
    "请比较老师咨询师的内部思维与学生咨询师的内部思维，从『是否在点上 + 分析深度』两个方面进行评分。"
    "你只负责评估，不需要与来访者互动。请严格只输出合法 JSON，包含整数 score(0–5) 与对应的 label。"
)

R_THINK_EVAL_USER_TEMPLATE = """你会看到三部分内容：

1. 来访者的描述；
2. 老师咨询师的内部思维（参照标准）；
3. 学生咨询师的内部思维（被评估对象）。

评估维度与细则（请在内部严格逐条核对，不要输出过程）：
A. 是否在点上（On-point）
- 准确识别来访者的核心问题/诉求与主要情绪；
- 涵盖关键背景与情境因素（时间线、因果、关系角色）；
- 与老师思维的关键点匹配度（覆盖/遗漏/误解的具体情况）；
- 事实一致性与逻辑连贯性（避免自相矛盾/凭空推断）；
- 结构化表达（要点分层、重点突出）。

B. 分析深度（Depth）
- 多维度分析：情绪、认知/信念、行为模式、人际互动、家庭/成长史、资源与保护因素；
- 风险评估：自伤/他伤/脆弱性风险的识别与证据（若适用），风险严重性与紧迫性；
- 证据与推理：基于来访者叙述与老师思维的具体证据、合理推论，而非空泛判断；
- 具体性与可操作性：提出明确的探索方向或工作重点（避免口号式表述）；
- 层次与条理：从表层现象到深层机制的层级化分析与连结。

评分锚点（请选择最符合的一档，不做加权平均）：
- 0（缺失）：几乎无分析；严重误解或与事实相反；未识别核心点；完全缺乏结构；未触及风险；
- 1（太差）：多为浅层改写；大量关键点缺失；存在明显遗漏/误解；风险触及极少或不当；
- 2（及格）：方向基本正确；覆盖部分关键点；深度与完整性不足；风险仅笼统提及；结构一般；
- 3（良）：覆盖关键点较充分；有一定深入与合理推理；能识别潜在风险并提出初步方向；结构清晰；
- 4（优）：深入、全面、几乎无误解；补充有价值视角；风险识别明确并与后续工作连结；分析具体可执行；
- 5（完美）：全面、严谨、无明显遗漏；推理充分且高质量补充；与老师思维高度一致并有所拓展。

输出要求（只输出合法 JSON，不要任何额外文字/代码块；label 必须为下列之一）：
{{
  "score": 0,
  "label": "缺失 | 太差 | 及格 | 良 | 优 | 完美"
}}

请先在内部比较老师与学生思维（不输出比较过程），再给出评分与标签。

### 待评估内容

【来访者的描述】
{user_message}

【老师咨询师的内部思维】
{teacher_think}

【学生咨询师的内部思维】
{student_think}

请直接输出 JSON，不要添加任何解释性文字或代码块标记。
"""
R_REPLY_EVAL_SYSTEM = (
    "你是一名心理咨询督导，只负责评估文本质量，不直接与来访者互动。"
    "请严格按照给定的维度和分值范围，对「咨询师回复」进行打分，并只输出 JSON。"
)

R_REPLY_EVAL_USER_TEMPLATE = """现在给你一段来访者的描述，以及咨询师给出的文字回复。
请根据下述 5 个维度，对咨询师回复进行评分。
每个维度都有自己的分值上限，总分为 13 分。
最终请给出：

- 每个维度的分数（整数）；
- 每个维度 1–2 句简短点评；
- 总分（0–13）；
- 总分换算后的 0–100 分整数。

### 维度与打分规则

#### 1. 全面性（0–2 分）

回答对来访者情况及心理问题的反映程度，包括但不限于：

- 是否反映了来访者的基本信息、主要处境；
- 是否抓住了来访者表达的核心心理困扰。

#### 2. 专业性（0–4 分）

回答中体现的心理咨询师专业性，包括但不限于：

- 是否展现出对来访者心理问题的专业理解和判断；
- 是否合理运用了专业心理咨询技术（如澄清、反馈、共情、探询等）；
- 用语是否专业、是否存在诱导性或评价性对话；
- 是否能切实帮助来访者理解自己、看到可能的方向；
- 是否体现了基本的咨询流程意识（如初步评估、澄清问题、探索资源等）；
- 是否有具体可执行的咨询技术实施或专业建议。

#### 3. 真实性（0–3 分）

回答中体现的真诚、贴近感，包括但不限于：

- 表达是否与来访者的叙述历史和语境相符，避免生硬套话；
- 是否避免对来访者经历的误读或明显不真实的描述；
- 语言是否自然、不做作，符合真实心理咨询场景。

#### 4. 共情性（0–3 分）

回答中传递的对来访者情绪与处境的理解、感受程度，包括但不限于：

- 能否识别并回应来访者明确表达的情绪；
- 能否敏感觉察并回应来访者潜在、未明说的情绪；
- 共情表达是否真诚自然，而不是机械模板化；
- 是否能让来访者感到被理解、被接纳。

#### 5. 安全性（0–1 分）

对来访者隐私和风险的保护程度，包括但不限于：

- 是否遵循心理咨询隐私保护准则，未泄露或索要无关敏感信息（姓名、工作单位、联系方式、家庭住址等）；
- 是否尊重来访者的想法与情绪，避免鼓励自伤、自杀或伤害他人，遇到严重风险时是否建议寻求线下专业帮助或紧急热线。

### 打分要求

- 各维度必须在各自范围内打整数分（例如专业性必须 0–4 之间的整数）；
- 总分 = 各维度分数之和，范围 0–13；
- 将总分按公式换算为 0–100 分的整数：
   `total_100 = round(total_13 / 13 * 100)`
- 打分要综合考虑内容质量和安全性，只要出现明显违反安全性的内容（鼓励自伤/他伤、给出危险行为建议等），安全性应打 0 分，并适当拉低其他维度评分。

### 输出格式

请严格输出符合 JSON 语法的对象，不要添加任何额外文字：

{{
  "dimensions": {{
    "comprehensiveness": {{ "score": 0, "comment": "..." }},
    "professionalism":   {{ "score": 0, "comment": "..." }},
    "authenticity":      {{ "score": 0, "comment": "..." }},
    "empathy":           {{ "score": 0, "comment": "..." }},
    "safety":            {{ "score": 0, "comment": "..." }}
  }},
  "total_13": 0,
  "total_100": 0
}}

### 待评估内容

- 来访者的描述：
   {user_message}
- 咨询师给出的回复：
   {assistant_reply}
"""


# -----------------------------
# Reward computer (R_format, R_think, R_reply)
# -----------------------------
class RewardComputer:
    def __init__(
        self,
        teacher_api_base: str,
        teacher_api_key: str,
        teacher_model: str,
        eval_api_base: str,
        eval_api_key: str,
        eval_model: str,
        teacher_temp: float = 0.7,
        eval_temp: float = 0.0,
        timeout: int = 60,
        max_teacher_tokens: int = 512,
        max_eval_tokens: int = 512,
        is_main_process: bool = True,
    ):
        self.teacher_api_base = teacher_api_base
        self.teacher_api_key = teacher_api_key
        self.teacher_model = teacher_model
        self.eval_api_base = eval_api_base
        self.eval_api_key = eval_api_key
        self.eval_model = eval_model
        self.teacher_temp = teacher_temp
        self.eval_temp = eval_temp
        self.timeout = timeout
        self.max_teacher_tokens = max_teacher_tokens
        self.max_eval_tokens = max_eval_tokens
        self.is_main_process = is_main_process

        # 最近一次奖励分量缓存（用于外部读取/统计）
        self.last_r_format = 0.0
        self.last_r_think = 0.0
        self.last_r_reply = 0.0

        # cache teacher_think per user_message
        self._teacher_cache = {}

        # regex for <think>...</think>
        self._think_pat = re.compile(r"<think>([\s\S]*?)</think>([\s\S]*)$", re.MULTILINE)

    def get_teacher_think(self, teacher_input: str) -> str:
        if teacher_input in self._teacher_cache:
            return self._teacher_cache[teacher_input]

        messages = [
            {"role": "system", "content": TEACHER_SYSTEM_PROMPT},
            {"role": "user", "content": f"{teacher_input}\n请仅输出你的内部思维。"},
        ]
        content = _chat_completions(
            api_base=self.teacher_api_base,
            api_key=self.teacher_api_key,
            model=self.teacher_model,
            messages=messages,
            temperature=self.teacher_temp,
            max_tokens=self.max_teacher_tokens,
            timeout=self.timeout,
            is_main_process=self.is_main_process,
        )
        teacher_think = content.strip()
        self._teacher_cache[teacher_input] = teacher_think
        return teacher_think

    def pregenerate_teacher_thinks(self, user_messages: list[str]):
        """
        批量预生成所有导师思维并缓存,避免训练时重复调用API
        """
        if self.is_main_process:
            print(f"\n{'='*60}")
            print(f"[预生成] 开始预生成 {len(user_messages)} 条导师思维...")
            print(f"{'='*60}\n")

        start_time = time.time()
        for idx, user_message in enumerate(user_messages, 1):
            if user_message not in self._teacher_cache:
                if self.is_main_process:
                    print(f"\n[预生成进度] {idx}/{len(user_messages)}")
                self.get_teacher_think(user_message)
            else:
                if self.is_main_process:
                    print(f"\n[预生成进度] {idx}/{len(user_messages)} (已缓存,跳过)")

        elapsed = time.time() - start_time
        if self.is_main_process:
            print(f"\n{'='*60}")
            print(f"[预生成完成] 共生成 {len(self._teacher_cache)} 条导师思维")
            print(f"[预生成完成] 耗时: {elapsed:.2f} 秒")
            print(f"{'='*60}\n")

    def _compute_R_format(self, sample: str):
        """
        满足：有且仅有一段 <think>...</think>，并在 think 外有纯回复正文；且 think 外部不残留任何标签
        则 R_format = 0.05；否则 0.0
        """
        m = self._think_pat.search(sample)
        if not m:
            return 0.0, None, None
        student_think = m.group(1).strip()
        outside = m.group(2).strip()

        # 不允许 outside 再含有任何标签(包括 think, thought 等)
        if "<" in outside and ">" in outside:
            # 检查是否有类似标签的模式
            if re.search(r'<\w+>', outside) or re.search(r'</\w+>', outside):
                return 0.0, None, None
        # outside 需非空（必须有正式回复）
        if len(outside) == 0:
            return 0.0, None, None
        return 0.05, student_think, outside

    def _compute_R_think(self, user_message: str, teacher_think: str, student_think: str, silent: bool = False) -> float:
        user_content = R_THINK_EVAL_USER_TEMPLATE.format(
            user_message=user_message,
            teacher_think=teacher_think,
            student_think=student_think,
        )
        messages = [
            {"role": "system", "content": R_THINK_EVAL_SYSTEM},
            {"role": "user", "content": user_content},
        ]
        content = _chat_completions(
            api_base=self.eval_api_base,
            api_key=self.eval_api_key,
            model=self.eval_model,
            messages=messages,
            response_format={"type": "json_object"},
            temperature=self.eval_temp,
            max_tokens=self.max_eval_tokens,
            timeout=self.timeout,
            is_main_process=(self.is_main_process and not silent),
        )
        data = _safe_json_loads(content) or {}
        # 优先读取整数 score 并映射到奖励
        score = data.get("score", None)
        label = str(data.get("label", "")).strip()
        try:
            score = int(score) if score is not None else None
        except Exception:
            score = None
        mapping = {0: -0.15, 1: -0.05, 2: 0.0, 3: 0.05, 4: 0.10, 5: 0.15}
        if isinstance(score, int) and score in mapping:
            return float(mapping[score])
        # 兼容旧字段 level 或仅有 label 的情况
        level = str(data.get("level", "")).strip()
        text = label or level
        if text:
            if ("缺失" in text) or ("空" in text) or ("未" in text and "给出" in text):
                return -0.15
            if "太差" in text:
                return -0.05
            if ("及格" in text) or ("一般" in text):
                return 0.0
            if "良" in text:
                return 0.05
            if "优" in text:
                return 0.10
            if ("完美" in text) or ("详细" in text):
                return 0.15
        # 旧逻辑兜底
        if "缺乏" in level:
            return -0.15
        if "详细" in level:
            return +0.15
        return 0.0

    def _compute_R_reply(self, user_message: str, student_reply: str, silent: bool = False) -> float:
        user_content = R_REPLY_EVAL_USER_TEMPLATE.format(
            user_message=user_message,
            assistant_reply=student_reply,
        )
        messages = [
            {"role": "system", "content": R_REPLY_EVAL_SYSTEM},
            {"role": "user", "content": user_content},
        ]
        content = _chat_completions(
            api_base=self.eval_api_base,
            api_key=self.eval_api_key,
            model=self.eval_model,
            messages=messages,
            response_format={"type": "json_object"},
            temperature=self.eval_temp,
            max_tokens=self.max_eval_tokens,
            timeout=self.timeout,
            is_main_process=(self.is_main_process and not silent),
        )
        data = _safe_json_loads(content) or {}
        try:
            total_13 = int(data.get("total_13", 0))
        except Exception:
            total_13 = 0
        total_13 = max(0, min(13, total_13))
        total_100 = int(round(total_13 / 13 * 100))
        r_reply = total_100 / 100.0 - 0.5
        r_reply = max(-0.5, min(0.5, r_reply))
        return float(r_reply)

    def _compute_R_length(self, student_reply: str) -> float:
        """
        硬编码长度奖励：仅统计 <think> 外 reply 的字符数（去首尾空白）。
        线性规则，范围 [-0.2, 0.2]：
        - 窗口 [45,75] 内线性递减：中心 60 得 +0.2，45/75 得 0。
          公式：d = |n-60|/15，score = 0.2*(1 - d)
        - 窗口外线性惩罚：越远越惩罚，至多 -0.2。
          公式：d = |n-60|/15，score = max(-0.2, -0.2*(d - 1))
        """
        n = len(str(student_reply).strip())
        d = abs(n - 60.0) / 15.0
        if d <= 1.0:
            return float(0.2 * (1.0 - d))
        val = -0.2 * (d - 1.0)
        if val < -0.2:
            val = -0.2
        return float(val)

    def compute_total_reward(self, user_message: str, sample: str, should_pause: bool = False) -> float:
        # R_format
        r_format, student_think, student_reply = self._compute_R_format(sample)

        if self.is_main_process:
            print(f"\n{'='*60}")
            print(f"📊 [奖励计算] 开始计算奖励分数")
            print(f"{'='*60}")
            print(f"🤖 [本地模型输出] 原始生成内容:")
            print(f"{'-'*60}")
            print(sample)
            print(f"{'-'*60}")
            print(f"📊 [R_format] 格式分数: {r_format:.4f}")

        if r_format <= 0.0:
            # 该样本不满足格式,总奖励为 0.0；同时将最近一次奖励分量缓存为 (0.0, 0.0, 0.0)
            self.last_r_format = 0.0
            self.last_r_think = 0.0
            self.last_r_reply = 0.0
            if self.is_main_process:
                print(f"⚠️  [警告] 格式不符合要求,总分为 0.0")
                print(f"⚠️  [原因] 可能的问题:")
                print(f"  - 缺少 <think>...</think> 标签")
                print(f"  - <think> 标签外残留了标签痕迹")
                print(f"  - <think> 标签外没有正式回复内容")
                print(f"{'='*60}\n")
            return 0.0  # 提前返回
        # teacher_think
        teacher_think = self.get_teacher_think(user_message)

        # R_think + R_reply + R_length + R_qus（可并行）
        with ThreadPoolExecutor(max_workers=4) as ex:
            f_think = ex.submit(self._compute_R_think, user_message, teacher_think, student_think)
            f_reply = ex.submit(self._compute_R_reply, user_message, student_reply)
            f_len   = ex.submit(self._compute_R_length, student_reply)
            f_qus   = ex.submit(self._compute_R_qus, student_reply)
            r_think = f_think.result()
            r_reply = f_reply.result()
            r_length = f_len.result()
            r_qus   = f_qus.result()

        total = r_format + r_think + r_reply + r_length + r_qus  # 五项系数均为1

        # 缓存最近一次奖励分量，供外部跟踪每步均值
        self.last_r_format = float(r_format)
        self.last_r_think = float(r_think)
        self.last_r_reply = float(r_reply)
        # self.last_r_length = float(r_length)
        # self.last_r_qus = float(r_qus)

        if self.is_main_process:
            print(f"📊 [R_think]   思维分数: {r_think:+.4f}")
            print(f"📊 [R_reply]   回复分数: {r_reply:+.4f}")
            print(f"📊 [R_length]  长度分数: {r_length:+.4f}")
            print(f"📊 [R_qus]     多问句惩罚: {r_qus:+.4f}")
            print(f"{'-'*60}")
            print(f"🎯 [总分]      R_format + R_think + R_reply + R_length + R_qus = {total:+.4f}")
            print(f"{'='*60}\n")
            # 记录到 wandb，便于外部可视化
            if wandb.run is not None:
                wandb.log({
                    "objective/r_format": float(r_format),
                    "objective/r_think": float(r_think),
                    "objective/r_reply": float(r_reply),
                    "objective/r_length": float(r_length),
                    "objective/r_qus": float(r_qus),
                })
        return float(total)

    def _compute_R_qus(self, student_reply: str) -> float:
        """
        多问句惩罚：统计回复中的问句数量，超过1个问号给予阶梯式惩罚。
        规则：
        - 问句数 <= 1: 无惩罚 (0.0)
        - 问句数 = 2: 扣 0.1 分
        - 问句数 = 3: 扣 0.2 分
        - 问句数 >= 4: 扣 0.3 分（上限）
        """
        # 统计问号数量（包括中文问号和英文问号）
        question_count = student_reply.count('?') + student_reply.count('？')
        if question_count <= 1:
            return 0.0
        if question_count == 2:
            return -0.1
        if question_count == 3:
            return -0.2
        # 4个及以上问句，扣0.3分（上限）
        return -0.3

    def compute_total_reward_components(self, user_message: str, sample: str, should_pause: bool = False, verbose: bool = True):
        """
        并行评估同一样本内的 r_think/r_reply，返回 (total, r_format, r_think, r_reply)。
        verbose=False 时不在内部打印，以便外部控制输出顺序。
        """
        # R_format
        r_format, student_think, student_reply = self._compute_R_format(sample)

        if verbose and self.is_main_process:
            print(f"\n{'='*60}")
            print(f"📊 [奖励计算] 开始计算奖励分数")
            print(f"{'='*60}")
            print(f"🤖 [本地模型输出] 原始生成内容:")
            print(f"{'-'*60}")
            print(sample)
            print(f"{'-'*60}")
            print(f"📊 [R_format] 格式分数: {r_format:.4f}")

        if r_format <= 0.0:
            self.last_r_format = 0.0
            self.last_r_think = 0.0
            self.last_r_reply = 0.0
            if verbose and self.is_main_process:
                print(f"⚠️  [警告] 格式不符合要求,总分为 0.0")
                print(f"⚠️  [原因] 可能的问题:")
                print(f"  - 缺少 <think>...</think> 标签")
                print(f"  - <think> 标签外残留了标签痕迹")
                print(f"  - <think> 标签外没有正式回复内容")
                print(f"{'='*60}\n")
            # 早退同样返回6元组，保持调用处解包一致
            return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

        # teacher_think
        teacher_think = self.get_teacher_think(user_message)

        # R_think + R_reply（可并行）
        with ThreadPoolExecutor(max_workers=3) as ex:
            f_think = ex.submit(self._compute_R_think, user_message, teacher_think, student_think, not verbose)
            f_reply = ex.submit(self._compute_R_reply, user_message, student_reply, not verbose)
            f_len = ex.submit(self._compute_R_length, student_reply)
            r_think = f_think.result()
            r_reply = f_reply.result()
            r_length = f_len.result()

        # R_qus（本地快速计算，统计问句数量）
        r_qus = self._compute_R_qus(student_reply)

        total = r_format + r_think + r_reply + r_length + r_qus  # 五项系数均为1

        # 缓存最近一次奖励分量，供外部跟踪每步均值
        self.last_r_format = float(r_format)
        self.last_r_think = float(r_think)
        self.last_r_reply = float(r_reply)
        # 可按需暴露 last_r_length
        # self.last_r_length = float(r_length)

        # 记录到 wandb（与原逻辑一致，新增 r_length）
        if wandb.run is not None:
            wandb.log({
                "objective/r_format": float(r_format),
                "objective/r_think": float(r_think),
                "objective/r_reply": float(r_reply),
                "objective/r_length": float(r_length),
            })

        if verbose and self.is_main_process:
            print(f"📊 [R_think]  思维分数: {r_think:+.4f}")
            print(f"📊 [R_reply]  回复分数: {r_reply:+.4f}")
            print(f"📊 [R_length] 长度分数: {r_length:+.4f}")
            print(f"📊 [R_qus]    问句惩罚: {r_qus:+.4f}")
            print(f"{'-'*60}")
            print(f"🎯 [总分]     R_format + R_think + R_reply + R_length + R_qus = {total:+.4f}")
            print(f"{'='*60}\n")

        return float(total), float(r_format), float(r_think), float(r_reply), float(r_length), float(r_qus)



# -----------------------------
# Data loading: RL.json -> multi-turn dialogues
# -----------------------------
def load_multi_turn_dialogues_from_rl_json(path: str) -> list[list[dict]]:
    """
    从RL.json加载数据,提取完整的多轮对话历史
    用于滚动多轮对话: 加载所有用户消息,按轮次依次使用

    返回格式: list of list of dict
    [
        [{"role": "user", "content": "第1轮用户消息"}],
        [{"role": "user", "content": "第1轮用户消息"}, {"role": "user", "content": "第2轮用户消息"}],
        ...
    ]
    每个列表包含该对话的所有用户消息(按轮次顺序)
    """
    print(f"\n[加载数据] 正在从 {path} 加载数据...")
    print(f"[加载数据] 使用滚动多轮对话模式: 加载完整多轮对话历史\n")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise RuntimeError(f"RL.json格式错误: 期望列表,得到 {type(data).__name__}")

    print(f"[加载数据] 找到 {len(data)} 个对话")

    dialogues = []
    for idx, item in enumerate(data):
        if not isinstance(item, dict):
            print(f"[警告] 对话 {idx} 不是字典类型,跳过")
            continue

        user_messages = []

        # 支持两种格式: messages 和 conversations
        if "messages" in item:
            # 格式: [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}, ...]
            messages = item.get("messages", [])
            # 提取所有用户消息
            for m in messages:
                if m.get("role") == "user" and isinstance(m.get("content"), str):
                    user_msg = m["content"].strip()
                    if user_msg:
                        user_messages.append({"role": "user", "content": user_msg})
        elif "conversations" in item:
            # 格式: [{"user": "...", "assistant": "..."}, ...]
            conversations = item.get("conversations", [])
            # 提取所有用户消息
            for conv in conversations:
                if isinstance(conv, dict) and "user" in conv:
                    user_msg = conv["user"].strip()
                    if user_msg:
                        user_messages.append({"role": "user", "content": user_msg})
        else:
            print(f"[警告] 对话 {idx} 既没有messages也没有conversations字段,跳过")

        if user_messages:
            dialogues.append(user_messages)
            print(f"[加载数据] 对话 {idx}: 提取到 {len(user_messages)} 轮用户消息")

    print(f"\n[加载数据] 共加载 {len(dialogues)} 个对话")
    total_rounds = sum(len(d) for d in dialogues)
    print(f"[加载数据] 总计 {total_rounds} 轮用户消息(平均每对话 {total_rounds/len(dialogues):.1f} 轮)\n")

    if not dialogues:
        raise RuntimeError("No user messages found in RL.json")

    return dialogues


# -----------------------------
# Global Dialogue History: 管理所有 prompt 的多轮对话历史
# -----------------------------
class GlobalDialogueHistory:
    """
    全局对话历史管理器 - 多轮对话模式:
    - 为每个 prompt_idx 维护一份对话历史
    - 历史包含: 所有轮次的 [用户消息, 最佳回答] 对
    - 每轮训练后,将最佳回答添加到历史,并添加下一轮的用户消息
    - 下次生成时,使用完整的交替对话历史重新格式化 prompt

    历史结构:
    [
        {"role": "user", "content": "第1轮用户消息"},
        {"role": "assistant", "content": "第1轮最佳回答"},
        {"role": "user", "content": "第2轮用户消息"},
        {"role": "assistant", "content": "第2轮最佳回答"},
        ...
    ]
    """
    def __init__(self, all_user_messages: list[list[dict]], is_main_process: bool = True):
        """
        Args:
            all_user_messages: list of list, 每个元素是一个对话的所有用户消息(按轮次)
                例如: [[{"role": "user", "content": "msg1"}, {"role": "user", "content": "msg2"}], ...]
        """
        self.is_main_process = is_main_process
        # 每个 prompt_idx 的历史: {prompt_idx: [{"role": "user", "content": "..."}, ...]}
        self.dialogue_history: dict[int, list[dict]] = {}
        # 每个对话的所有用户消息(按轮次): {prompt_idx: [{"role": "user", "content": "..."}, ...]}
        self.all_user_messages: dict[int, list[dict]] = {}
        # 每个对话当前轮次: {prompt_idx: current_round}
        self.current_rounds: dict[int, int] = {}
        # 待处理的更新: {prompt_idx: best_completion}
        self.pending_updates: dict[int, str] = {}
        # 初始化所有 prompt 的历史
        self._initialize_history(all_user_messages)

    def _initialize_history(self, all_user_messages: list[list[dict]]):
        """初始化所有 prompt 的历史,从第1轮用户消息开始"""
        for idx, user_msgs in enumerate(all_user_messages):
            # 存储该对话的所有用户消息
            self.all_user_messages[idx] = user_msgs
            # 当前轮次从0开始
            self.current_rounds[idx] = 0
            # 历史从第一个用户消息开始
            if user_msgs:
                self.dialogue_history[idx] = [user_msgs[0]]
            else:
                self.dialogue_history[idx] = []

    def get_history(self, prompt_idx: int) -> list[dict]:
        """获取指定 prompt 的对话历史"""
        if prompt_idx not in self.dialogue_history:
            return []
        return self.dialogue_history[prompt_idx].copy()

    def add_best_completion(self, prompt_idx: int, best_completion: str):
        """
        将最佳回答添加到指定 prompt 的历史中
        然后添加下一轮的用户消息(如果有的话)
        """
        if prompt_idx not in self.dialogue_history:
            if self.is_main_process:
                print(f"[警告] Prompt #{prompt_idx} 不在历史中,跳过")
            return

        # 1. 添加 assistant 的最佳回答
        self.dialogue_history[prompt_idx].append({"role": "assistant", "content": best_completion})

        # 2. 获取下一轮的用户消息
        self.current_rounds[prompt_idx] += 1
        next_round = self.current_rounds[prompt_idx]

        # 3. 如果还有下一轮用户消息,添加到历史中
        if prompt_idx in self.all_user_messages and next_round < len(self.all_user_messages[prompt_idx]):
            next_user_msg = self.all_user_messages[prompt_idx][next_round]
            self.dialogue_history[prompt_idx].append(next_user_msg)
            if self.is_main_process:
                print(f"[滚动对话] Prompt #{prompt_idx + 1}: 添加第{next_round + 1}轮用户消息到历史")
        else:
            # 没有更多用户消息了,保持历史不变
            if self.is_main_process:
                print(f"[滚动对话] Prompt #{prompt_idx + 1}: 已到达最后一轮用户消息")

    def has_more_rounds(self, prompt_idx: int) -> bool:
        """检查是否还有更多轮次的用户消息"""
        if prompt_idx not in self.all_user_messages or prompt_idx not in self.current_rounds:
            return False
        current_round = self.current_rounds[prompt_idx]
        return current_round < len(self.all_user_messages[prompt_idx])

    def format_prompt_with_history(self, prompt_idx: int, tokenizer) -> str:
        """
        获取带历史的格式化 prompt 用于生成
        """
        history = self.get_history(prompt_idx)
        if not history:
            return ""

        # 使用 ChatML 格式化
        return format_messages_for_student(history, tokenizer)

    def queue_best_completion(self, prompt_idx: int, best_completion: str):
        """
        将最佳回答加入待处理队列
        (由 reward_fn 调用,在 step 结束时由 callback 批量处理)
        """
        self.pending_updates[prompt_idx] = best_completion

    def apply_pending_updates(self):
        """
        应用所有待处理的更新
        (由 callback 在 step_end 时调用)
        """
        for prompt_idx, best_completion in self.pending_updates.items():
            self.add_best_completion(prompt_idx, best_completion)
        self.pending_updates.clear()

    def reset_all(self):
        """重置所有历史记录"""
        all_user_msgs_list = [self.all_user_messages[idx] for idx in sorted(self.all_user_messages.keys())]
        self._initialize_history(all_user_msgs_list)
        self.pending_updates.clear()


# -----------------------------
# Build GRPO trainer
# -----------------------------
def build_dataset(prompts: list[str]) -> Dataset:
    # TRL GRPO 默认期望一个包含 'prompt' 字段的 dataset
    rows = [{"prompt": p} for p in prompts]
    return Dataset.from_list(rows)


def make_student_system_prompt() -> str:
    # 约束学生输出格式：先<think> 内部思维，再在外给出正式中文回复
    return (
        """"""
    )


def format_prompt_for_student(user_message: str, tokenizer=None) -> str:
    """
    使用ChatML格式化prompt,确保符合Qwen2.5的ChatML格式
    """
    messages = [{"role": "user", "content": f"{user_message}\n"}]

    # 如果提供了tokenizer且有chat_template,使用它
    if tokenizer is not None and hasattr(tokenizer, 'chat_template') and tokenizer.chat_template is not None:
        try:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,  # 返回字符串而不是token ids
                add_generation_prompt=True
            )
        except Exception as e:
            print(f"[警告] apply_chat_template失败: {e}, 使用手动ChatML格式")

    # 手动构建ChatML格式(兼容没有tokenizer的情况)
    prompt = ""
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        prompt += f"<|im_start|>{role}\n{content}<|im_end|>\n"
    prompt += "<|im_start|>assistant\n"  # 添加assistant前缀用于生成
    return prompt


def format_messages_for_student(messages: list[dict], tokenizer=None) -> str:
    """
    将 messages 列表格式化为 ChatML 字符串
    添加空的 system 消息框架
    """
    # 在 messages 开头添加一个空的 system 消息
    full: list[dict] = [{"role": "system", "content": ""}]
    full.extend(messages)

    if tokenizer is not None and hasattr(tokenizer, 'chat_template') and tokenizer.chat_template is not None:
        try:
            return tokenizer.apply_chat_template(full, tokenize=False, add_generation_prompt=True)
        except Exception as e:
            print(f"[警告] apply_chat_template失败: {e}, 使用手动ChatML")

    # 手动构建ChatML格式
    prompt = ""
    for m in full:
        prompt += f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n"
    prompt += "<|im_start|>assistant\n"
    return prompt


# -----------------------------
# Multi-turn helpers (用于教师/评估模型输入)
# -----------------------------

def _make_teacher_input_from_chatml(prompt_text: str, history_rounds: int = 3) -> str:
    """从 ChatML prompt 解析出最近 N 个问答对 + 当前用户消息，供教师/评估模型阅读"""
    blocks = re.findall(r"<\|im_start\|\>(\w+)\n([\s\S]*?)<\|im_end\|\>\n", prompt_text)
    messages = [{"role": r, "content": c.strip()} for r, c in blocks if r in ("user", "assistant")]
    if not messages:
        return str(prompt_text)
    # 找到最后一个 user 作为当前用户
    cur_user_idx = None
    if messages[-1]["role"] == "user":
        cur_user_idx = len(messages) - 1
    else:
        for k in range(len(messages) - 1, -1, -1):
            if messages[k]["role"] == "user":
                cur_user_idx = k
                break
    if cur_user_idx is None:
        return str(prompt_text)

    # 抽取最近 N 个 (user,assistant) 对
    pairs: list[tuple[str, str]] = []
    j = cur_user_idx - 1
    while j > 0 and len(pairs) < history_rounds:
        if messages[j-1]["role"] == "user" and messages[j]["role"] == "assistant":
            pairs.insert(0, (messages[j-1]["content"], messages[j]["content"]))
            j -= 2
        else:
            j -= 1

    lines: list[str] = []
    if pairs:
        lines.append(f"上下文（最近{len(pairs)}轮）：")
        for idx, (u, a) in enumerate(pairs, 1):
            lines.append(f"{idx}. 用户：{u}")
            lines.append(f"   助手：{a}")
        lines.append("")
    lines.append("当前用户消息：")
    lines.append(messages[cur_user_idx]["content"])
    return "\n".join(lines)

# -----------------------------
# Reward fn wrapper for TRL
# -----------------------------
def build_reward_fn(reward_computer: RewardComputer, group_size: int, is_main_process: bool = True, tracker=None, dialogue_manager=None, tokenizer=None):
    """
    TRL GRPO reward_fn - 适配新版本 GRPO 的调用方式
    接受关键字参数: inputs, prompts, completions, completion_ids_list

    滚动多轮对话模式:
    - 使用 dialogue_manager 管理每个prompt的对话历史
    - 每次生成后,将最佳模型输出添加到历史,并添加下一轮用户消息
    - 下次生成时,使用包含完整交替历史(user-assistant-user...)的prompt

    分布式训练优化: 只在 rank0 计算奖励(调用外部API),然后广播给其他 rank
    避免因 API 调用时延不一致导致 NCCL 超时

    同时,如果提供了 tracker(RewardsTracker),将把每个样本的 r_format/r_think/r_reply 加到累计器,
    以便在每个 step 结束时统计 batch 均值。
    """
    def reward_fn(*args, **kwargs):
        # 每步开始时tracker为当前步累计,不在此处重置,由 RewardMonitorCallback.finalize_step 完成统计与重置。

        # 滚动对话模式: 在每个新 step 开始时重置历史
        # (在 TRL 中,可以通过检测 prompts 是否变化来判断是否是新 step)

        # 优先从 kwargs 中获取参数
        if kwargs:
            inputs = kwargs.get('inputs', None)
            prompts = kwargs.get('prompts', [])
            completions = kwargs.get('completions', [])
            completion_ids_list = kwargs.get('completion_ids_list', None)
        elif len(args) >= 3:
            # 位置参数方式
            inputs, prompts, completions = args[0], args[1], args[2]
        elif len(args) == 2:
            # 旧版本: reward_fn(samples, prompts)
            prompts, completions = args[0], args[1]
        else:
            # 调试信息
            if is_main_process:
                print(f"DEBUG: args={args}, kwargs={kwargs}")
            raise ValueError(f"Unexpected arguments: args={len(args)}, kwargs={list(kwargs.keys())}")

        # 检查是否在分布式环境
        is_distributed = dist.is_available() and dist.is_initialized()
        current_rank = dist.get_rank() if is_distributed else 0

        # completions 可能是字符串列表或其他格式
        if not isinstance(completions, list):
            completions = list(completions)
        if not isinstance(prompts, list):
            prompts = list(prompts)

        # 只在 rank0 计算奖励
        if current_rank == 0:
            # 预分配奖励列表，保持与 completions 对齐
            rewards = [0.0 for _ in range(len(completions))]

            # 按组并行：每个 prompt 的 group_size 个 completion 并行评估
            total_comps = len(completions)
            idx = 0
            while idx < total_comps:
                group_indices = list(range(idx, min(idx + group_size, total_comps)))

                # 找到该组对应的 prompt
                first_idx = group_indices[0]
                prompt_idx = first_idx // group_size if len(prompts) < len(completions) else first_idx
                if prompt_idx < len(prompts):
                    prompt = prompts[prompt_idx]
                else:
                    prompt = prompts[0] if prompts else ""

                # 多轮对话模式: prompt 已包含完整的历史,直接作为教师输入
                teacher_input = str(prompt)

                # 首次打印该 prompt 的头部
                if is_main_process:
                    print(f"\n{'#'*60}")
                    print(f"📝 [新Prompt] Prompt #{prompt_idx + 1}")
                    print(f"{'#'*60}")
                    print(f"📄 [Prompt内容]:")
                    print(f"{'-'*60}")
                    print(teacher_input)
                    print(f"{'-'*60}\n")

                # 并行提交该组所有样本的奖励计算（内部不打印，保证日志不乱）
                futures = {}
                with ThreadPoolExecutor(max_workers=min(group_size, len(group_indices))) as ex:
                    for j in group_indices:
                        sample_text = str(completions[j])
                        futures[j] = ex.submit(
                            reward_computer.compute_total_reward_components,
                            teacher_input,
                            sample_text,
                            (j % group_size + 1 == group_size),  # should_pause 保持兼容
                            False,  # verbose=False：内部不打印
                        )

                # 统一按顺序打印每个样本的结果，保持原有输出顺序
                group_rewards = []  # 存储该组的 (completion_idx, reward, completion)
                for j in group_indices:
                    within_group_idx = j % group_size + 1
                    try:
                        total, r_format, r_think, r_reply, r_length, r_qus = futures[j].result()
                    except Exception as e:
                        if is_main_process:
                            warnings.warn(f"reward error for idx {j}: {type(e).__name__}: {e}")
                        total, r_format, r_think, r_reply, r_length, r_qus = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

                    if is_main_process:
                        print(f"\n🔄 [进度] Prompt {prompt_idx + 1}, 生成 {within_group_idx}/{group_size}")
                        print(f"\n{'='*60}")
                        print(f"📊 [奖励计算] 开始计算奖励分数")
                        print(f"{'='*60}")
                        print(f"🤖 [本地模型输出] 原始生成内容:")
                        print(f"{'-'*60}")
                        print(str(completions[j]))
                        print(f"{'-'*60}")
                        print(f"📊 [R_format] 格式分数: {r_format:.4f}")
                        print(f"📊 [R_think]  思维分数: {r_think:+.4f}")
                        print(f"📊 [R_reply]  回复分数: {r_reply:+.4f}")
                        print(f"📊 [R_length] 长度分数: {r_length:+.4f}")
                        print(f"📊 [R_qus]    问句惩罚: {r_qus:+.4f}")
                        print(f"{'-'*60}")
                        print(f"🎯 [总分]     R_format + R_think + R_reply + R_length + R_qus = {total:+.4f}")
                        print(f"{'='*60}\n")

                    # 累计该样本的奖励分量，用于统计每步均值
                    if tracker is not None:
                        try:
                            tracker.add(r_format, r_think, r_reply, r_length, r_qus)
                        except Exception as te:
                            if is_main_process:
                                warnings.warn(f"tracker.add() 失败: {te}")

                    rewards[j] = float(total)
                    group_rewards.append((j, total, str(completions[j])))

                # 滚动对话: 选择该组最佳回答并加入待处理队列
                if dialogue_manager is not None and group_rewards:
                    # 找到奖励最高的回答
                    best_idx, best_reward, best_completion = max(group_rewards, key=lambda x: x[1])
                    if is_main_process:
                        print(f"\n[滚动对话] Prompt #{prompt_idx + 1}: 最佳回答奖励 = {best_reward:+.4f}")
                        print(f"[最佳回答预览]: {best_completion[:150]}...\n")
                    # 将最佳回答加入待处理队列（由 callback 在 step_end 时统一应用）
                    dialogue_manager.queue_best_completion(prompt_idx, best_completion)

                # 下一组
                idx = group_indices[-1] + 1
        else:
            # 其他 rank 创建占位符(与 completions 相同长度)
            rewards = [0.0 for _ in range(len(completions))]

        # 如果是分布式训练,将 rank0 的奖励广播给所有 rank
        if is_distributed:
            # 使用 broadcast_object_list 广播奖励列表
            rewards_obj = [rewards]  # 包装成列表以便广播
            dist.broadcast_object_list(rewards_obj, src=0)
            rewards = rewards_obj[0]

            if current_rank != 0 and is_main_process:
                print(f"\n[Rank {current_rank}] 从 rank0 接收到 {len(rewards)} 个奖励值")

        return rewards

    return reward_fn


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rl_json", default=r"/root/autodl-tmp/RL.json", type=str)
    parser.add_argument("--model_name", default="/root/autodl-tmp/qwen2.5_SFT", type=str)
    parser.add_argument("--output_dir", default="/root/autodl-tmp/GRPO_RL_v3.2", type=str)

    # generation / GRPO
    parser.add_argument("--group_size", type=int, default=4, help="num_generations per prompt")
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)

    # train
    parser.add_argument("--per_device_train_batch_size", type=int, default=32)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--max_steps", type=int, default=1000)
    parser.add_argument("--warmup_steps", type=int, default=10)
    parser.add_argument("--logging_steps", type=int, default=1)
    parser.add_argument("--save_steps", type=int, default=20)
    parser.add_argument("--save_total_limit", type=int, default=1000)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--fp16", action="store_true")

    # LoRA optional
    parser.add_argument("--use_lora", action="store_true")
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--target_modules", type=str, default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj")

    # teacher/eval API config (OpenAI-compatible)
    parser.add_argument("--teacher_api_base", type=str, default="https://dashscope.aliyuncs.com/compatible-mode/v1")
    parser.add_argument("--teacher_api_key", type=str, default="sk-40fb3997d3ed485ba390a9c4ae3bd2d2")
    parser.add_argument("--teacher_model", type=str, default="deepseek-v3.2")

    parser.add_argument("--eval_api_base", type=str, default="https://dashscope.aliyuncs.com/compatible-mode/v1")
    parser.add_argument("--eval_api_key", type=str, default="sk-40fb3997d3ed485ba390a9c4ae3bd2d2")
    parser.add_argument("--eval_model", type=str, default="deepseek-v3.2")

    # 分布式训练参数
    parser.add_argument("--local_rank", type=int, default=-1, help="本地进程rank,由torch.distributed.launch自动设置")
    parser.add_argument("--deepspeed", type=str, default="/root/autodl-tmp/ds_config.json", help="DeepSpeed配置文件路径")
    parser.add_argument("--fsdp", type=str, default=None, help="FSDP配置,例如: 'full_shard auto_wrap'")
    parser.add_argument("--ddp_find_unused_parameters", action="store_true", help="DDP查找未使用的参数")

    # KL监控参数
    parser.add_argument("--kl_plot_every_n_steps", type=int, default=1, help="每N步绘制一次KL曲线图")
    parser.add_argument("--history_rounds", type=int, default=3, help="最近 N 个问答对作为上下文")

    args = parser.parse_args()

    # 初始化分布式环境
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if local_rank != -1:
        torch.cuda.set_device(local_rank)
        # 初始化进程组(如果还未初始化)
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl")
        args.local_rank = local_rank
        is_main_process = local_rank == 0
    else:
        is_main_process = True

    # 只在主进程打印信息
    def print_rank0(*msg):
        if is_main_process:
            print(*msg)

    # 1) Load tokenizer first (需要用于格式化prompts)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True, trust_remote_code=True)

    # 确保tokenizer有pad_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print_rank0(f"[Tokenizer] 设置 pad_token = eos_token: {tokenizer.eos_token}")

    # 显式设置ChatML模板 (保留空system消息框架)
    print_rank0("[Tokenizer] 覆盖chat_template，保留空system消息框架")
    tokenizer.chat_template = (
        "{% for message in messages %}"
        "{% if message['role'] == 'system' %}"
        "<|im_start|>system\n{{ message['content'] }}<|im_end|>\n"
        "{% elif message['role'] == 'user' %}"
        "<|im_start|>user\n{{ message['content'] }}<|im_end|>\n"
        "{% elif message['role'] == 'assistant' %}"
        "<|im_start|>assistant\n{{ message['content'] }}<|im_end|>\n"
        "{% endif %}"
        "{% endfor %}"
        "{% if add_generation_prompt %}"
        "<|im_start|>assistant\n"
        "{% endif %}"
    )

    # 确保ChatML特殊token存在
    special_tokens_to_add = []
    if "<|im_start|>" not in tokenizer.get_vocab():
        special_tokens_to_add.extend(["<|im_start|>", "<|im_end|>"])

    if special_tokens_to_add:
        tokenizer.add_special_tokens({"additional_special_tokens": special_tokens_to_add})
        print_rank0(f"[Tokenizer] 添加ChatML特殊token: {special_tokens_to_add}")

    print_rank0(f"[Tokenizer] 词表大小: {len(tokenizer)}")
    print_rank0(f"[Tokenizer] EOS token: {tokenizer.eos_token} (id: {tokenizer.eos_token_id})")
    print_rank0(f"[Tokenizer] PAD token: {tokenizer.pad_token} (id: {tokenizer.pad_token_id})")

    # 2) Load multi-turn dialogues from RL.json
    # 加载完整的多轮对话历史,每个对话包含所有轮次的用户消息
    all_user_messages = load_multi_turn_dialogues_from_rl_json(args.rl_json)

    # 创建全局对话历史管理器
    dialogue_manager = GlobalDialogueHistory(
        all_user_messages=all_user_messages,
        is_main_process=is_main_process
    )

    # 格式化为ChatML格式(初始仅包含第一个用户消息)
    formatted_prompts = []
    for idx in range(len(all_user_messages)):
        formatted_prompt = dialogue_manager.format_prompt_with_history(idx, tokenizer)
        formatted_prompts.append(formatted_prompt)

    ds = build_dataset(formatted_prompts)

    # 打印示例格式化结果(仅主进程)
    if is_main_process and len(formatted_prompts) > 0:
        print_rank0("\n" + "="*60)
        print_rank0("[示例] 格式化后的第一个prompt(多轮对话模式):")
        print_rank0("="*60)
        print_rank0(formatted_prompts[0][:500] + "..." if len(formatted_prompts[0]) > 500 else formatted_prompts[0])
        print_rank0("="*60 + "\n")

    # 3) Build model

    # 根据分布式策略选择不同的加载方式
    if args.deepspeed or args.fsdp:
        # DeepSpeed/FSDP会自动处理模型分布,不需要device_map
        print_rank0("[分布式] 使用DeepSpeed/FSDP,模型将由训练器自动分布")
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            torch_dtype=torch.bfloat16 if args.bf16 else (torch.float16 if args.fp16 else torch.float32),
            trust_remote_code=True,
        )
    else:
        # DDP模式或单GPU: 使用device_map="auto"进行张量并行
        print_rank0("[分布式] 使用标准DDP或单GPU模式")
        if local_rank != -1:
            # 多GPU DDP: 每个进程加载完整模型到各自的GPU
            model = AutoModelForCausalLM.from_pretrained(
                args.model_name,
                torch_dtype=torch.bfloat16 if args.bf16 else (torch.float16 if args.fp16 else torch.float32),
                trust_remote_code=True,
            )
        else:
            # 单GPU: 使用device_map="auto"自动分配
            model = AutoModelForCausalLM.from_pretrained(
                args.model_name,
                device_map="auto",
                torch_dtype="auto",
                trust_remote_code=True,
            )

    # 如果添加了新token,需要调整embedding大小
    if len(special_tokens_to_add) > 0:
        model.resize_token_embeddings(len(tokenizer))
        print_rank0(f"[Model] 调整embedding大小到: {len(tokenizer)}")

    # 启用梯度检查点
    if hasattr(model, 'gradient_checkpointing_enable'):
        model.gradient_checkpointing_enable()
    # 关闭缓存(训练时不需要)
    model.config.use_cache = False

    if args.use_lora:
        target_modules = [s.strip() for s in args.target_modules.split(",") if s.strip()]
        peft_cfg = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=target_modules,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, peft_cfg)

    # 3) Reward computer
    reward_computer = RewardComputer(
        teacher_api_base=args.teacher_api_base,
        teacher_api_key=args.teacher_api_key,
        teacher_model=args.teacher_model,
        eval_api_base=args.eval_api_base,
        eval_api_key=args.eval_api_key,
        eval_model=args.eval_model,
        teacher_temp=0.7,
        eval_temp=0.0,
        timeout=90,
        max_teacher_tokens=512,
        max_eval_tokens=512,
        is_main_process=is_main_process,
    )

    # 导师思维将在训练过程中按需生成(通过缓存机制避免重复调用)
    # 注意: 由于数据集较大且不会在一次训练中用完,不进行全量预生成
    if is_main_process:
        print_rank0("[训练模式] 导师思维将按需生成并缓存,避免浪费API调用")

    # 初始化 wandb (只在主进程)
    if is_main_process:
        wandb.init(
            project="grpo-psychology-counselor",  # 项目名称,可以修改
            name=f"grpo-{args.model_name.split('/')[-1]}",  # 运行名称
            config={
                "model_name": args.model_name,
                "learning_rate": args.learning_rate,
                "batch_size": args.per_device_train_batch_size,
                "gradient_accumulation_steps": args.gradient_accumulation_steps,
                "max_steps": args.max_steps,
                "group_size": args.group_size,
                "temperature": args.temperature,
                "use_lora": args.use_lora,
            }
        )

    # 4) GRPO config
    print_rank0(f"[History] 使用最近 {args.history_rounds} 个问答对作为上下文")
    grpo_cfg = GRPOConfig(
        output_dir=args.output_dir,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        warmup_steps=args.warmup_steps,
        max_steps=args.max_steps,
        bf16=args.bf16,
        fp16=args.fp16,
        beta=0.03,
        report_to="wandb",  # 启用TensorBoard记录奖励和KL曲线
        logging_dir="/root/autodl-tmp/logs",  # TensorBoard日志目录
        # 分布式训练配置
        local_rank=args.local_rank,
        ddp_find_unused_parameters=args.ddp_find_unused_parameters,
        deepspeed=args.deepspeed,
        fsdp=args.fsdp,
        # 多GPU时禁用device_map
        ddp_backend="nccl" if local_rank != -1 else None,
    )

    # 生成参数配置
    generation_config = {
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "do_sample": True,
    }

    # 设置模型的 tokenizer (某些版本的 GRPO 从 model 自动获取)
    if not hasattr(model, 'config'):
        model.config.pad_token_id = tokenizer.pad_token_id

    # 构建 reward function (滚动多轮对话模式: 传入 dialogue_manager 和 tokenizer)
    # 奖励均值跟踪器 + 回调
    rewards_tracker = RewardsTracker(output_dir=args.output_dir, is_main_process=is_main_process, plot_every_n_steps=args.kl_plot_every_n_steps)
    reward_fn = build_reward_fn(
        reward_computer,
        group_size=args.group_size,
        is_main_process=is_main_process,
        tracker=rewards_tracker,
        dialogue_manager=dialogue_manager,
        tokenizer=tokenizer,
    )

    # 创建KL监控回调 + 奖励监控回调
    kl_monitor = KLMonitorCallback(
        output_dir=args.output_dir,
        is_main_process=is_main_process,
        plot_every_n_steps=args.kl_plot_every_n_steps
    )
    reward_monitor = RewardMonitorCallback(
        tracker=rewards_tracker,
        is_main_process=is_main_process,
        plot_every_n_steps=args.kl_plot_every_n_steps
    )

    # 滚动多轮对话回调
    rolling_dialogue_callback = RollingDialogueCallback(
        dialogue_manager=dialogue_manager,
        tokenizer=tokenizer,
        is_main_process=is_main_process,
        group_size=args.group_size,
    )

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,  # 使用 processing_class 而不是 tokenizer
        args=grpo_cfg,
        train_dataset=ds,
        reward_funcs=[reward_fn],  # 注意是 reward_funcs (复数),传入列表
        callbacks=[kl_monitor, reward_monitor, rolling_dialogue_callback],  # 添加滚动对话回调
    )

    # 设置生成参数 - 逐个设置属性而不是用 update
    if hasattr(trainer, 'generation_config'):
        for key, value in generation_config.items():
            setattr(trainer.generation_config, key, value)

    # 设置其他参数
    if hasattr(trainer, 'num_generations'):
        trainer.num_generations = args.group_size

    # 5) Train
    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    # 关闭 wandb
    if is_main_process:
        wandb.finish()

    # 分布式清理
    if dist.is_available() and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()

    print("Training complete. Model saved to:", args.output_dir)

    # 注意: 在分布式训练下,只有rank0会看到完整的日志输出与保存提示


if __name__ == "__main__":
    main()