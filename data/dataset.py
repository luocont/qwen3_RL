"""
数据加载和预处理模块
负责加载RL.json数据，并维护每个message的对话历史和5Ps病例历史
"""
import json
import re
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field


@dataclass
class MessageState:
    """单个message的状态"""
    messages: List[Dict]  # 原始messages列表
    conversation_history: str = ""  # 对话历史
    current_5ps_case: str = ""  # 当前5Ps病例
    5ps_history: List[str] = field(default_factory=list)  # 5Ps病例历史
    turn_count: int = 0  # 对话轮次计数


@dataclass
class TrainingSample:
    """单个训练样本"""
    prompt: str  # 完整的prompt
    previous_5ps_case: str  # 上轮的5Ps病例
    message_state: MessageState  # 关联的message状态
    turn_index: int  # 当前轮次索引


def load_rl_data(data_path: str) -> List[MessageState]:
    """
    加载RL.json数据

    Args:
        data_path: RL.json文件路径

    Returns:
        MessageState列表，每个元素代表一个独立的message
    """
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    message_states = []

    for item in data:
        messages = item.get('messages', [])
        if not messages:
            continue

        # 创建新的message状态
        state = MessageState(messages=messages)
        message_states.append(state)

    return message_states


def extract_5ps_from_text(text: str) -> Optional[str]:
    """
    从文本中提取5Ps病例内容

    Args:
        text: 包含5Ps标签的文本

    Returns:
        提取的5Ps内容，如果没有找到则返回None
    """
    pattern = r'<5Ps>(.*?)</5Ps>'
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None


def format_conversation_history(messages: List[Dict], end_index: int) -> str:
    """
    格式化对话历史
    只包含user的内容，不包含assistant的回复

    Args:
        messages: 消息列表
        end_index: 截止索引（不包含）

    Returns:
        格式化的对话历史字符串（仅患者发言）
    """
    history_parts = []
    for msg in messages[:end_index]:
        if msg['role'] == 'user':
            history_parts.append(f"患者：{msg['content']}")

    return '\n'.join(history_parts)


def build_prompt(
    system_prompt: str,
    conversation_history: str,
    previous_5ps_case: str,
    current_user_input: str
) -> str:
    """
    构造完整的训练prompt

    Args:
        system_prompt: 系统提示词（从prompts.txt读取）
        conversation_history: 对话历史
        previous_5ps_case: 上轮的5Ps病例
        current_user_input: 当前用户输入

    Returns:
        完整的prompt字符串
    """
    prompt_parts = []

    # 系统提示
    prompt_parts.append(system_prompt.strip())

    # 5Ps病例历史
    if previous_5ps_case:
        prompt_parts.append(f"\n这是患者过往的5Ps病例:\n{previous_5ps_case}")
    else:
        prompt_parts.append("\n这是患者过往的5Ps病例:\n（暂无）")

    # 对话历史
    if conversation_history:
        prompt_parts.append(f"\n以下是你和患者的对话历史：\n{conversation_history}")

    # 当前用户输入
    prompt_parts.append(f"\n当前患者说：\n{current_user_input}")

    return '\n'.join(prompt_parts)


class QwenRLDataset:
    """
    Qwen强化学习数据集
    管理message采样和prompt构造
    """

    def __init__(self, data_path: str, system_prompt: str):
        """
        初始化数据集

        Args:
            data_path: RL.json文件路径
            system_prompt: 系统提示词
        """
        self.message_states = load_rl_data(data_path)
        self.system_prompt = system_prompt
        self.current_message_idx = 0
        self.current_turn_idx = 1  # 从第一轮用户输入开始（跳过system）

    def get_next_sample(self) -> Optional[TrainingSample]:
        """
        获取下一个训练样本
        按顺序遍历message，每个message从第一轮开始连续采样

        Returns:
            TrainingSample，如果没有更多样本则返回None
        """
        if self.current_message_idx >= len(self.message_states):
            # 重置到第一个message
            self.current_message_idx = 0
            self.current_turn_idx = 1
            return None

        state = self.message_states[self.current_message_idx]
        messages = state.messages

        # 找到下一个用户轮次
        while self.current_turn_idx < len(messages):
            msg = messages[self.current_turn_idx]

            if msg['role'] == 'user':
                # 构造训练样本
                conversation_history = format_conversation_history(messages, self.current_turn_idx)
                previous_5ps_case = state.current_5ps_case or ""

                prompt = build_prompt(
                    self.system_prompt,
                    conversation_history,
                    previous_5ps_case,
                    msg['content']
                )

                sample = TrainingSample(
                    prompt=prompt,
                    previous_5ps_case=previous_5ps_case,
                    message_state=state,
                    turn_index=self.current_turn_idx
                )

                # 移动到下一轮
                self.current_turn_idx += 1
                state.turn_count += 1

                return sample

            self.current_turn_idx += 1

        # 当前message处理完成，移动到下一个message
        self.current_message_idx += 1
        self.current_turn_idx = 1
        return self.get_next_sample()

    def update_5ps_case(self, state: MessageState, new_5ps_text: str):
        """
        更新message的5Ps病例

        Args:
            state: MessageState对象
            new_5ps_text: 新生成的5Ps文本
        """
        extracted_5ps = extract_5ps_from_text(new_5ps_text)
        if extracted_5ps:
            state.current_5ps_case = extracted_5ps
            state.5ps_history.append(extracted_5ps)

    def get_random_sample(self) -> TrainingSample:
        """
        随机获取一个message，从其第一轮开始

        Returns:
            TrainingSample
        """
        import random

        idx = random.randint(0, len(self.message_states) - 1)
        state = self.message_states[idx]

        # 重置该message的状态
        state.conversation_history = ""
        state.current_5ps_case = ""
        state.5ps_history = []
        state.turn_count = 0

        # 从第一轮用户输入开始
        messages = state.messages
        for i, msg in enumerate(messages):
            if msg['role'] == 'user':
                conversation_history = format_conversation_history(messages, i)

                prompt = build_prompt(
                    self.system_prompt,
                    conversation_history,
                    "",  # 第一轮没有历史病例
                    msg['content']
                )

                return TrainingSample(
                    prompt=prompt,
                    previous_5ps_case="",
                    message_state=state,
                    turn_index=i
                )

        # 如果没找到用户消息，返回第一个可用的
        return self.get_random_sample()

    def __len__(self) -> int:
        return len(self.message_states)

    def __iter__(self):
        self.current_message_idx = 0
        self.current_turn_idx = 1
        return self
