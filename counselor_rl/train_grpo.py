# -*- coding: utf-8 -*-
"""
counselor_rl/train_grpo.py

在线交互式 GRPO 训练脚本。

训练逻辑（每轮）：
  1. 来访者模型读取 intake_form，根据对话历史发言
  2. 固定智能体群（5Ps + 参考咨询师）生成参考输出
     输出格式：<thinking>思考</thinking><5Ps>病例</5Ps>回复内容
  3. 将参考输出嵌入 prompt，被训练模型（Qwen3）采样 G=4 个回答
  4. 每个回答独立计算 PANAS 改善奖励 + 格式奖励
  5. GRPO 组内归一化，更新模型参数
  6. 分数最高的回答追加到共享对话历史（同时作为来访者下一轮的上下文）
  7. 来访者模型读取最优回答后发言，进入下一轮
  8. 下一轮固定智能体群的上下文 = 共享对话历史（与被训练模型对齐）
"""

import os
import re
import sys
import json
import logging
import argparse
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agents.five_ps_agent import FivePsAgent, DEFAULT_5PS
from agents.counselor_agent import CounselorAgent
from agents.client_agent import ClientAgent
from agents.panas_evaluator import PanasEvaluator
from agents.prompts import TRAINED_MODEL_TRAINING_SYSTEM_PROMPT

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


@dataclass
class ScriptArguments:
    model_name_or_path: str = field(
        default="Qwen/Qwen3-8B",
        metadata={"help": "被训练的 Qwen3 模型路径或 HuggingFace Hub 名称"},
    )
    dataset_path: str = field(
        default="data/counseling_dataset.json",
        metadata={"help": "训练数据集路径（JSON 格式）"},
    )
    output_dir: str = field(
        default="outputs/grpo_counselor",
        metadata={"help": "模型输出目录"},
    )
    num_episodes_per_case: int = field(
        default=3,
        metadata={"help": "每个来访者人设运行的 episode 数量"},
    )
    num_generations: int = field(
        default=4,
        metadata={"help": "GRPO 每步采样数量（G）"},
    )
    max_prompt_length: int = field(default=2048)
    max_completion_length: int = field(default=1024)
    max_dialogue_turns: int = field(
        default=30,
        metadata={"help": "每个 episode 最大对话轮数"},
    )
    learning_rate: float = field(default=1e-6)
    grpo_epsilon: float = field(
        default=0.2,
        metadata={"help": "GRPO clip epsilon"},
    )
    panas_weight: float = field(default=0.7)
    format_weight: float = field(default=0.3)
    panas_delta_weight: float = field(
        default=0.5,
        metadata={"help": "PANAS 奖励中改善量的权重（0~1），剩余部分为绝对状态权重"},
    )
    agent_api_base: str = field(default="https://dashscope.aliyuncs.com/compatible-mode/v1")
    agent_api_key: str = field(default="")
    five_ps_model: str = field(default="qwen3-max")
    counselor_model: str = field(default="qwen3-max")
    client_model: str = field(default="qwen3-max")
    panas_model: str = field(default="qwen3-max")
    gradient_accumulation_steps: int = field(
        default=20,
        metadata={"help": "梯度累积步数，每累积 N 轮对话才更新一次参数"},
    )
    save_steps: int = field(default=40)
    plot_steps: int = field(default=10)
    best_samples_path: str = field(default="outputs/best_samples.jsonl")
    temperature: float = field(default=0.7)
    top_p: float = field(default=0.9)
    repetition_penalty: float = field(default=1.05)


def parse_args() -> ScriptArguments:
    parser = argparse.ArgumentParser()
    for f in ScriptArguments.__dataclass_fields__.values():
        parser.add_argument(
            f"--{f.name}",
            type=type(f.default) if not isinstance(f.default, bool) else lambda x: x.lower() == "true",
            default=f.default,
            help=f.metadata.get("help", ""),
        )
    args = parser.parse_args()
    return ScriptArguments(**vars(args))


def extract_5ps_from_output(text: str) -> str:
    m = re.search(r"(<5Ps>[\s\S]*?</5Ps>)", text)
    if m:
        return m.group(1).strip()
    return ""


def extract_reply_from_output(text: str) -> str:
    m = re.search(r"</5Ps>([\s\S]*)$", text)
    if m:
        return m.group(1).strip()
    m = re.search(r"</thinking>([\s\S]*)$", text)
    if m:
        reply = re.sub(r"<5Ps>[\s\S]*?</5Ps>", "", m.group(1)).strip()
        if reply:
            return reply
    return text.strip()


def compute_format_reward(model_output: str, agent_stage: str = "") -> float:
    think_blocks = re.findall(r"<thinking>[\s\S]*?</thinking>", model_output)
    five_ps_blocks = re.findall(r"<5Ps>[\s\S]*?</5Ps>", model_output)

    if len(think_blocks) != 1 or len(five_ps_blocks) != 1:
        return 0.0

    m_think = re.search(r"<thinking>([\s\S]*?)</thinking>", model_output)
    m_5ps = re.search(r"<5Ps>([\s\S]*?)</5Ps>", model_output)

    think_content = m_think.group(1).strip()
    has_think = len(think_content) > 0

    has_5ps = len(m_5ps.group(1).strip()) > 0

    end_5ps = m_5ps.end()
    reply_part = model_output[end_5ps:].strip()
    has_reply = len(reply_part) > 10

    extra_tags = re.findall(r"<(?!thinking>|/thinking>|5Ps>|/5Ps>)[^>]+>", model_output)
    has_extra_tags = len(extra_tags) > 0

    prefix = model_output[:m_think.start()].strip()
    has_prefix_noise = len(prefix) > 0

    if has_extra_tags or has_prefix_noise:
        return 0.0

    stage_match = re.search(r"会话阶段[:：]\s*(前期|后期)", think_content)
    has_stage = stage_match is not None

    stage_aligned = False
    if has_stage and agent_stage:
        stage_aligned = stage_match.group(1) == agent_stage

    score = sum([has_5ps, has_think, has_reply, has_stage, stage_aligned])

    order_correct = False
    if has_think and has_5ps and has_reply:
        end_think = m_think.end()
        start_5ps = m_5ps.start()
        reply_start = model_output.find(reply_part[:20], end_5ps)
        order_correct = (end_think <= start_5ps) and (end_5ps <= reply_start)

    if score == 5 and order_correct:
        return 1.0
    elif score == 5:
        return 0.9
    elif score == 4 and order_correct:
        return 0.8
    elif score == 4:
        return 0.7
    elif score == 3 and order_correct:
        return 0.6
    elif score == 3:
        return 0.4
    elif score == 2:
        return 0.2
    elif score == 1:
        return 0.1
    return 0.0


def _pa(panas: dict | None) -> float:
    if panas is None:
        return 2.5
    return float(panas.get("positive_affect", {}).get("average", 2.5))


def _na(panas: dict | None) -> float:
    if panas is None:
        return 2.5
    return float(panas.get("negative_affect", {}).get("average", 2.5))


class GRPOTrainLoop:
    """
    在线交互式 GRPO 训练循环。

    每轮对话流程：
      固定智能体群生成参考输出
        → 被训练模型采样 G=4
        → 每个采样独立计算奖励
        → GRPO 更新参数
        → 最优采样追加到共享对话历史
        → 来访者模型读取最优回复后发言
        → 下一轮（固定智能体群上下文与共享历史对齐）
    """

    def __init__(self, args: ScriptArguments, model, tokenizer, device):
        self.args = args
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

        agent_api_key = args.agent_api_key or os.getenv("DASHSCOPE_API_KEY") or ""

        self.five_ps_agent = FivePsAgent(
            api_base=args.agent_api_base,
            api_key=agent_api_key,
            model=args.five_ps_model,
        )
        self.counselor_agent = CounselorAgent(
            api_base=args.agent_api_base,
            api_key=agent_api_key,
            model=args.counselor_model,
        )
        self.client_agent = ClientAgent(
            api_base=args.agent_api_base,
            api_key=agent_api_key,
            model=args.client_model,
        )
        self.panas_evaluator = PanasEvaluator(
            api_base=args.agent_api_base,
            api_key=agent_api_key,
            model=args.panas_model,
        )

        self.optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
        self.global_step = 0
        self._total_turn = 0
        self._accum_turn = 0
        self._accum_loss = 0.0
        self._accum_kl = 0.0
        self._accum_reward_mean = 0.0
        self._accum_format_mean = 0.0
        self._accum_panas_mean = 0.0
        self.optimizer.zero_grad()

        self._metric_steps: list[int] = []
        self._metric_reward_mean: list[float] = []
        self._metric_format_mean: list[float] = []
        self._metric_panas_mean: list[float] = []
        self._metric_loss: list[float] = []
        self._metric_kl: list[float] = []
        self._metric_lr: list[float] = []

        self._plots_dir = os.path.join(args.output_dir, "thinkingand5psplots")
        os.makedirs(self._plots_dir, exist_ok=True)
        os.makedirs(os.path.dirname(args.best_samples_path) or ".", exist_ok=True)
        os.makedirs(args.output_dir, exist_ok=True)

    def build_prompt(
        self,
        prev_5ps: str,
        current_5ps: str,
        agent_full_output: str,
        shared_dialogue_history: list[dict],
    ) -> str:
        agent_think_match = re.search(r"<thinking>[\s\S]*?</thinking>", agent_full_output)
        agent_think_block = agent_think_match.group(0).strip() if agent_think_match else ""
        agent_reply = extract_reply_from_output(agent_full_output)
        combined_agent_output = "\n".join(filter(None, [agent_think_block, current_5ps.strip(), agent_reply]))
        system_content = TRAINED_MODEL_TRAINING_SYSTEM_PROMPT.format(
            **{"5ps_case": prev_5ps, "智能体输出": combined_agent_output}
        )
        messages = [{"role": "system", "content": system_content}]
        for msg in shared_dialogue_history[-2:]:
            if msg["role"] == "assistant":
                clean_content = re.sub(r"<thinking>[\s\S]*?</thinking>\s*", "", msg["content"])
                clean_content = re.sub(r"<5Ps>[\s\S]*?</5Ps>\s*", "", clean_content).strip()
                messages.append({"role": "assistant", "content": clean_content})
            else:
                messages.append(msg)
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        return prompt

    @torch.no_grad()
    def sample_completions(self, prompt: str) -> tuple[list[str], torch.Tensor]:
        """
        对同一个 prompt 采样 G 个 completion，返回文本列表和对应的 log_probs。
        """
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=False,
        ).to(self.device)

        if inputs["input_ids"].shape[1] > self.args.max_prompt_length:
            inputs["input_ids"] = inputs["input_ids"][:, -self.args.max_prompt_length:]
            inputs["attention_mask"] = inputs["attention_mask"][:, -self.args.max_prompt_length:]

        prompt_len = inputs["input_ids"].shape[1]

        self.model.gradient_checkpointing_disable()
        self.model.config.use_cache = True
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=self.args.max_completion_length,
            do_sample=True,
            temperature=self.args.temperature,
            top_p=self.args.top_p,
            repetition_penalty=self.args.repetition_penalty,
            num_return_sequences=self.args.num_generations,
            pad_token_id=self.tokenizer.pad_token_id,
            return_dict_in_generate=True,
            output_scores=True,
        )
        self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        self.model.config.use_cache = False

        sequences = outputs.sequences
        completion_ids = sequences[:, prompt_len:]

        completions = self.tokenizer.batch_decode(
            completion_ids, skip_special_tokens=True
        )

        log_probs_list = []
        for i in range(self.args.num_generations):
            scores = torch.stack(outputs.scores, dim=1)[i].to(torch.bfloat16)
            token_ids = completion_ids[i]
            valid_len = (token_ids != self.tokenizer.pad_token_id).sum().item()
            lp = F.log_softmax(scores[:valid_len], dim=-1)
            token_lp = lp[range(valid_len), token_ids[:valid_len]]
            log_probs_list.append(token_lp.sum())

        log_probs = torch.stack(log_probs_list)
        return completions, log_probs, completion_ids, prompt_len

    def compute_rewards(
        self,
        completions: list[str],
        panas_prev_pa: float,
        panas_prev_na: float,
        client_response: str,
        agent_stage: str = "",
    ) -> list[float]:
        """
        对 G 个 completion 各自独立计算奖励。
        PANAS 只评测本轮：来访者发言 + 被训练咨询师回复。
        """
        rewards = []
        format_rewards = []
        panas_rewards = []
        for i, completion in enumerate(completions):
            format_reward = compute_format_reward(completion, agent_stage=agent_stage)

            reply = extract_reply_from_output(completion)
            panas_reward = 0.0
            if reply.strip():
                eval_text = f"来访者: {client_response}\n咨询师: {reply}"
                panas_curr = self.panas_evaluator.evaluate(eval_text)
                curr_pa = _pa(panas_curr)
                curr_na = _na(panas_curr)
                delta_score = (curr_pa - panas_prev_pa - curr_na + panas_prev_na) / 4.0
                abs_score = (curr_pa - curr_na) / 4.0
                dw = self.args.panas_delta_weight
                panas_reward = dw * delta_score + (1.0 - dw) * abs_score
                logger.info(
                    f"  [G{i+1}] PA: {panas_prev_pa:.2f}→{curr_pa:.2f} "
                    f"NA: {panas_prev_na:.2f}→{curr_na:.2f} "
                    f"delta={delta_score:+.4f} abs={abs_score:+.4f} "
                    f"PANAS: {panas_reward:+.4f} 格式: {format_reward:.1f}"
                )

            total = self.args.panas_weight * panas_reward + self.args.format_weight * format_reward
            rewards.append(total)
            format_rewards.append(format_reward)
            panas_rewards.append(panas_reward)

        format_mean = sum(format_rewards) / len(format_rewards)
        panas_mean = sum(panas_rewards) / len(panas_rewards)
        return rewards, format_mean, panas_mean

    def grpo_update(
        self,
        prompt: str,
        completions: list[str],
        completion_ids: torch.Tensor,
        prompt_len: int,
        old_log_probs: torch.Tensor,
        rewards: list[float],
    ) -> tuple[float, float]:
        """
        GRPO 前向+反向传播（不含 optimizer.step）。
        支持梯度累积：loss 已除以 gradient_accumulation_steps。
        返回 (loss_val, kl_divergence)。
        """
        reward_tensor = torch.tensor(rewards, dtype=torch.float32, device=self.device)

        mean_r = reward_tensor.mean()
        std_r = reward_tensor.std() + 1e-8
        advantages = (reward_tensor - mean_r) / std_r

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=False,
        ).to(self.device)

        if inputs["input_ids"].shape[1] > self.args.max_prompt_length:
            inputs["input_ids"] = inputs["input_ids"][:, -self.args.max_prompt_length:]
            inputs["attention_mask"] = inputs["attention_mask"][:, -self.args.max_prompt_length:]

        total_kl = 0.0
        valid_count = 0
        accumulated_loss = 0.0

        scale = self.args.num_generations * self.args.gradient_accumulation_steps

        for i in range(self.args.num_generations):
            token_ids = completion_ids[i]
            valid_len = (token_ids != self.tokenizer.pad_token_id).sum().item()
            if valid_len == 0:
                continue

            full_ids = torch.cat([inputs["input_ids"][0], token_ids[:valid_len]], dim=0).unsqueeze(0)
            full_mask = torch.ones_like(full_ids)

            logits = self.model(input_ids=full_ids, attention_mask=full_mask).logits
            completion_logits = logits[0, prompt_len - 1: prompt_len - 1 + valid_len]
            log_probs = F.log_softmax(completion_logits, dim=-1)
            new_token_lp = log_probs[range(valid_len), token_ids[:valid_len]]
            new_log_prob = new_token_lp.sum()

            kl_i = (old_log_probs[i].detach() - new_log_prob).item()
            total_kl += kl_i
            valid_count += 1

            ratio = torch.exp(new_log_prob - old_log_probs[i].detach())
            adv = advantages[i]

            loss_unclipped = ratio * adv
            loss_clipped = torch.clamp(ratio, 1 - self.args.grpo_epsilon, 1 + self.args.grpo_epsilon) * adv
            loss_i = -torch.min(loss_unclipped, loss_clipped) / scale

            loss_i.backward()
            accumulated_loss += loss_i.item()

            del logits, completion_logits, log_probs, new_token_lp, new_log_prob, ratio, loss_i
            torch.cuda.empty_cache()

        mean_kl = total_kl / max(valid_count, 1)
        return accumulated_loss, mean_kl

    def run_episode(self, intake_form: str, attitude: str):
        """
        运行一个完整的在线交互式 episode。

        共享对话历史（shared_dialogue_history）同时作为：
          - 被训练模型 prompt 的对话上下文（取最近1轮）
          - 固定智能体群（5Ps/参考咨询师）的对话上下文
          - 来访者模型的对话历史
        每轮追加的是被训练模型的最优采样，而非参考咨询师的回复。
        """
        current_5ps = DEFAULT_5PS
        prev_5ps = DEFAULT_5PS
        shared_dialogue_history: list[dict] = []
        shared_text_history = ""

        panas_prev = self.panas_evaluator.evaluate(
            f"来访者初始状态：{intake_form}\n态度：{attitude}"
        )
        logger.info(f"[Episode 开始] PANAS 基线: PA={_pa(panas_prev):.2f} NA={_na(panas_prev):.2f}")

        for turn in range(self.args.max_dialogue_turns):
            logger.info(f"\n{'='*60}")
            logger.info(f"[Turn {turn+1}/{self.args.max_dialogue_turns}]  global_step={self.global_step}")
            logger.info(f"{'='*60}")

            client_response = self.client_agent.respond(
                intake_form=intake_form,
                attitude=attitude,
                dialogue_history=shared_text_history,
            )
            logger.info(f"\n>>> [来访者发言]\n{client_response}\n")

            shared_dialogue_history.append({"role": "user", "content": client_response})
            shared_text_history += f"\nClient: {client_response}"

            current_5ps = self.five_ps_agent.update(
                dialogue_context=shared_text_history,
                current_5ps=current_5ps,
            )
            logger.info(f"\n>>> [5Ps 更新]\n{current_5ps}\n")

            agent_full_output = self.counselor_agent.respond(
                dialogue_history=shared_dialogue_history,
                current_5ps=current_5ps,
            )
            logger.info(f"\n>>> [参考咨询师完整输出]\n{agent_full_output}\n")
            agent_full_output_clean = re.sub(r"<5Ps>[\s\S]*?</5Ps>\s*", "", agent_full_output).strip()

            prompt = self.build_prompt(
                prev_5ps=prev_5ps,
                current_5ps=current_5ps,
                agent_full_output=agent_full_output_clean,
                shared_dialogue_history=shared_dialogue_history,
            )
            logger.info(f"\n{'─'*60}")
            logger.info(f">>> [被训练模型 Prompt]\n{prompt}")
            logger.info(f"{'─'*60}\n")

            completions, old_log_probs, completion_ids, prompt_len = self.sample_completions(prompt)
            logger.info(f">>> [Qwen3 采样 G={self.args.num_generations} 个回答]")
            for i, c in enumerate(completions):
                logger.info(f"\n  ── G{i+1} 生成内容 ──\n{c}\n")

            agent_stage_match = re.search(r"会话阶段[:：]\s*(前期|后期)", agent_full_output_clean)
            agent_stage = agent_stage_match.group(1) if agent_stage_match else ""

            rewards, format_mean, panas_mean = self.compute_rewards(
                completions=completions,
                panas_prev_pa=_pa(panas_prev),
                panas_prev_na=_na(panas_prev),
                client_response=client_response,
                agent_stage=agent_stage,
            )
            logger.info(f"\n>>> [奖励汇总] {[f'G{i+1}={r:.4f}' for i, r in enumerate(rewards)]}")

            loss, kl = self.grpo_update(
                prompt=prompt,
                completions=completions,
                completion_ids=completion_ids,
                prompt_len=prompt_len,
                old_log_probs=old_log_probs,
                rewards=rewards,
            )
            reward_mean = float(sum(rewards) / len(rewards))

            self._total_turn += 1
            self._accum_turn += 1
            self._accum_loss += loss
            self._accum_kl += kl
            self._accum_reward_mean += reward_mean
            self._accum_format_mean += format_mean
            self._accum_panas_mean += panas_mean

            if self._accum_turn >= self.args.gradient_accumulation_steps:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                self.optimizer.zero_grad()
                self.global_step += 1

                avg_loss = self._accum_loss / self._accum_turn
                avg_kl = self._accum_kl / self._accum_turn
                avg_reward = self._accum_reward_mean / self._accum_turn
                avg_format = self._accum_format_mean / self._accum_turn
                avg_panas = self._accum_panas_mean / self._accum_turn
                current_lr = self.optimizer.param_groups[0]["lr"]
                logger.info(
                    f">>> [GRPO 更新] step={self.global_step}  loss={avg_loss:.6f}  "
                    f"kl={avg_kl:.6f}  reward_mean={avg_reward:.4f}  "
                    f"format={avg_format:.4f}  panas={avg_panas:.4f}  lr={current_lr:.2e}"
                )

                self._metric_steps.append(self.global_step)
                self._metric_loss.append(avg_loss)
                self._metric_kl.append(avg_kl)
                self._metric_reward_mean.append(avg_reward)
                self._metric_format_mean.append(avg_format)
                self._metric_panas_mean.append(avg_panas)
                self._metric_lr.append(current_lr)

                self._accum_turn = 0
                self._accum_loss = 0.0
                self._accum_kl = 0.0
                self._accum_reward_mean = 0.0
                self._accum_format_mean = 0.0
                self._accum_panas_mean = 0.0

            if self._total_turn % self.args.plot_steps == 0:
                self._save_metrics_plot()

            if self._total_turn % self.args.save_steps == 0:
                self._save_checkpoint()

            best_idx = int(torch.tensor(rewards).argmax().item())
            best_completion = completions[best_idx]
            best_reply = extract_reply_from_output(best_completion)
            logger.info(
                f"\n>>> [最优采样 G{best_idx+1}]  奖励={rewards[best_idx]:.4f}\n"
                f"  完整输出:\n{best_completion}\n"
                f"  提取回复:\n{best_reply}\n"
            )

            self._save_best_sample(prompt, best_completion, rewards[best_idx], rewards)

            agent_reply = extract_reply_from_output(agent_full_output_clean)
            hist_think_match = re.search(r"<thinking>[\s\S]*?</thinking>", agent_full_output_clean)
            hist_think_block = hist_think_match.group(0).strip() if hist_think_match else ""
            combined_agent_output = "\n".join(filter(None, [hist_think_block, current_5ps.strip(), agent_reply]))
            shared_dialogue_history.append({"role": "assistant", "content": combined_agent_output})
            shared_text_history += f"\nCounselor: {agent_reply}"

            prev_5ps = current_5ps
            logger.info(f">>> [prev_5ps 更新] 使用当前 current_5ps:\n{current_5ps}\n")

            panas_curr = self.panas_evaluator.evaluate(
                f"来访者: {client_response}\n咨询师: {agent_reply}"
            )
            logger.info(
                f">>> [本轮 PANAS]  PA: {_pa(panas_prev):.2f} → {_pa(panas_curr):.2f}  "
                f"NA: {_na(panas_prev):.2f} → {_na(panas_curr):.2f}"
            )
            panas_prev = panas_curr

            if self.client_agent.is_session_ended(client_response):
                logger.info(f"\n>>> [Turn {turn+1}] 来访者结束咨询。")
                break

    def _save_metrics_plot(self):
        steps = self._metric_steps
        if len(steps) < 2:
            return

        fig, axes = plt.subplots(2, 3, figsize=(18, 8))
        fig.suptitle(f"Training Metrics (step {self.global_step})", fontsize=14)

        axes[0, 0].plot(steps, self._metric_reward_mean, color="steelblue", linewidth=1.5)
        axes[0, 0].set_title("Reward Mean")
        axes[0, 0].set_xlabel("Step")
        axes[0, 0].set_ylabel("Reward")
        axes[0, 0].grid(True, alpha=0.3)

        axes[0, 1].plot(steps, self._metric_format_mean, color="mediumpurple", linewidth=1.5)
        axes[0, 1].set_title("Format Reward Mean")
        axes[0, 1].set_xlabel("Step")
        axes[0, 1].set_ylabel("Format Reward")
        axes[0, 1].grid(True, alpha=0.3)

        axes[0, 2].plot(steps, self._metric_panas_mean, color="cornflowerblue", linewidth=1.5)
        axes[0, 2].set_title("Quality Reward Mean (PANAS)")
        axes[0, 2].set_xlabel("Step")
        axes[0, 2].set_ylabel("PANAS Reward")
        axes[0, 2].grid(True, alpha=0.3)

        axes[1, 0].plot(steps, self._metric_loss, color="tomato", linewidth=1.5)
        axes[1, 0].set_title("Loss")
        axes[1, 0].set_xlabel("Step")
        axes[1, 0].set_ylabel("Loss")
        axes[1, 0].grid(True, alpha=0.3)

        axes[1, 1].plot(steps, self._metric_kl, color="darkorange", linewidth=1.5)
        axes[1, 1].set_title("KL Divergence")
        axes[1, 1].set_xlabel("Step")
        axes[1, 1].set_ylabel("KL")
        axes[1, 1].grid(True, alpha=0.3)

        axes[1, 2].plot(steps, self._metric_lr, color="mediumseagreen", linewidth=1.5)
        axes[1, 2].set_title("Learning Rate")
        axes[1, 2].set_xlabel("Step")
        axes[1, 2].set_ylabel("LR")
        axes[1, 2].ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
        axes[1, 2].grid(True, alpha=0.3)

        plt.tight_layout()
        save_path = os.path.join(self._plots_dir, f"metrics_step{self.global_step:06d}.png")
        plt.savefig(save_path, dpi=120)
        plt.close(fig)
        logger.info(f"[图像] 指标图保存到 {save_path}")

    def _save_best_sample(self, prompt: str, completion: str, best_reward: float, all_rewards: list[float]):
        sample = {
            "step": self.global_step,
            "prompt": prompt[:300],
            "completion": completion,
            "best_reward": best_reward,
            "all_rewards": all_rewards,
        }
        with open(self.args.best_samples_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    def _save_checkpoint(self):
        ckpt_dir = os.path.join(self.args.output_dir, f"checkpoint-{self.global_step}")
        self.model.save_pretrained(ckpt_dir)
        self.tokenizer.save_pretrained(ckpt_dir)
        logger.info(f"[Checkpoint] 保存到 {ckpt_dir}")


def main():
    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    log_file = os.path.join(args.output_dir, "thinkingand5pstrain.log")
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logging.getLogger().addHandler(file_handler)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"使用设备: {device}")

    logger.info(f"加载 tokenizer: {args.model_name_or_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    logger.info(f"加载模型: {args.model_name_or_path}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.train()

    train_loop = GRPOTrainLoop(args, model, tokenizer, device)

    logger.info(f"加载数据集: {args.dataset_path}")
    with open(args.dataset_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    logger.info("开始在线交互式 GRPO 训练...")
    for episode_idx, item in enumerate(raw_data):
        ai_client = item.get("AI_client", item)
        intake_form = ai_client.get("intake_form", "")
        attitude = ai_client.get("attitude_instruction", ai_client.get("attitude", "来访者愿意配合咨询，但情绪较为低落。"))

        for ep in range(args.num_episodes_per_case):
            logger.info(f"\n{'#'*60}")
            logger.info(f"[Case {episode_idx+1}/{len(raw_data)}] Episode {ep+1}/{args.num_episodes_per_case}")
            logger.info(f"{'#'*60}")
            try:
                train_loop.run_episode(intake_form=intake_form, attitude=attitude)
            except Exception as e:
                logger.warning(f"Episode 失败: {e}", exc_info=True)
                continue

    logger.info(f"训练完成，保存模型到 {args.output_dir}")
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    logger.info("完成！")


if __name__ == "__main__":
    main()
