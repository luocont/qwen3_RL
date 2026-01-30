"""
奖励函数模块
实现GRPO训练的奖励计算
"""
import re
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass


@dataclass
class RewardResult:
    """奖励计算结果"""
    total_reward: float
    r_format: float
    r_5ps_continuity: float
    r_5ps_item_nums: float
    details: Dict = None


class RewardFunction:
    """
    奖励函数类
    """

    # P标签的正则模式
    P_LABELS = [
        r'P1\s*主诉\s*[:：]',
        r'P2\s*易感因素\s*[:：]',
        r'P3\s*诱发因素\s*[:：]',
        r'P4\s*维持因素\s*[:：]',
        r'P5\s*保护因素\s*[:：]'
    ]

    def __init__(
        self,
        format_weight: float = 1.0,
        continuity_weight: float = 1.0,
        item_nums_weight: float = 1.0,
        check_threshold: int = 10,
        min_items: int = 3,
        max_items: int = 4
    ):
        """
        初始化奖励函数

        Args:
            format_weight: 格式奖励权重
            continuity_weight: 渐进性奖励权重
            item_nums_weight: 条目数奖励权重
            check_threshold: 开始检查条目数的对话轮次阈值
            min_items: 每个P的最少条目数
            max_items: 每个P的最多条目数
        """
        self.format_weight = format_weight
        self.continuity_weight = continuity_weight
        self.item_nums_weight = item_nums_weight
        self.check_threshold = check_threshold
        self.min_items = min_items
        self.max_items = max_items

    def compute_format_reward(self, response: str) -> float:
        """
        计算格式奖励 r_format
        检查：```代码块 → <5Ps>标签 → P1-P5标签 → 回复内容，且顺序正确

        Args:
            response: 模型生成的回复

        Returns:
            1.0 或 0.0
        """
        if not response:
            return 0.0

        # 1. 检查 ``` 代码块
        code_block_pattern = r'```[\s\S]*?```'
        code_block_match = re.search(code_block_pattern, response)
        if not code_block_match:
            return 0.0
        code_block_end = code_block_match.end()

        # 2. 检查 <5Ps> 标签（必须在代码块之后）
        five_ps_start_pattern = r'<5Ps>'
        five_ps_start_match = re.search(five_ps_start_pattern, response[code_block_end:])
        if not five_ps_start_match:
            return 0.0
        five_ps_start_pos = code_block_end + five_ps_start_match.start()

        # 3. 检查 </5Ps> 标签
        five_ps_end_pattern = r'</5Ps>'
        five_ps_end_match = re.search(five_ps_end_pattern, response)
        if not five_ps_end_match:
            return 0.0
        five_ps_end_pos = five_ps_end_match.end()

        # 确保 </5Ps> 在 <5Ps> 之后
        if five_ps_end_pos <= five_ps_start_pos:
            return 0.0

        # 4. 提取 <5Ps> 和 </5Ps> 之间的内容
        five_ps_content = response[five_ps_start_pos + 5:five_ps_end_pos - 6]

        # 5. 检查P1-P5是否都存在且顺序正确
        last_pos = 0
        for p_pattern in self.P_LABELS:
            match = re.search(p_pattern, five_ps_content[last_pos:])
            if not match:
                return 0.0
            last_pos += match.end()

        # 6. 检查 </5Ps> 后是否有回复内容（非空）
        after_five_ps = response[five_ps_end_pos:].strip()
        if not after_five_ps:
            return 0.0

        return 1.0

    def compute_continuity_reward(
        self,
        current_5ps_text: str,
        previous_5ps_text: str
    ) -> float:
        """
        计算渐进性奖励 r_5ps_continuity
        如果一次性填写了≥3个原本为"待完善"的P，惩罚0.2分

        Args:
            current_5ps_text: 当前轮次的5Ps文本
            previous_5ps_text: 上一轮的5Ps文本

        Returns:
            0.0 或 -0.2
        """
        if not previous_5ps_text:
            # 第一轮，不惩罚
            return 0.0

        # 解析上一轮5Ps，找出哪些P已有内容（非"待完善"）
        previous_filled = self._get_filled_ps(previous_5ps_text)

        # 解析当前5Ps，找出哪些P有内容
        current_filled = self._get_filled_ps(current_5ps_text)

        # 计算新增的有内容的P数量
        new_filled = current_filled - previous_filled
        if len(new_filled) >= 3:
            return -0.2

        return 0.0

    def _get_filled_ps(self, five_ps_text: str) -> set:
        """
        获取已填写的P标签集合

        Args:
            five_ps_text: 5Ps文本

        Returns:
            已填写的P标签索引集合 {0, 1, 2, 3, 4}
        """
        filled = set()
        for i, p_pattern in enumerate(self.P_LABELS):
            match = re.search(p_pattern, five_ps_text)
            if match:
                # 获取该P标签后的内容
                content_start = match.end()

                # 找到下一个P标签或结尾
                next_p_pos = len(five_ps_text)
                for j in range(i + 1, len(self.P_LABELS)):
                    next_match = re.search(self.P_LABELS[j], five_ps_text[content_start:])
                    if next_match:
                        next_p_pos = content_start + next_match.start()
                        break

                content = five_ps_text[content_start:next_p_pos].strip()

                # 检查是否有实际内容（不是"待完善"）
                if content and "待完善" not in content:
                    filled.add(i)

        return filled

    def compute_item_nums_reward(
        self,
        five_ps_text: str,
        turn_count: int
    ) -> float:
        """
        计算条目数奖励 r_5ps_item_nums
        当对话轮次>=10时，检查每个P的条目数是否在3-4之间

        Args:
            five_ps_text: 5Ps文本
            turn_count: 当前对话轮次

        Returns:
            0.0 或 -0.2
        """
        if turn_count < self.check_threshold:
            return 0.0

        # 解析每个P的条目数
        for i, p_pattern in enumerate(self.P_LABELS):
            match = re.search(p_pattern, five_ps_text)
            if not match:
                # 找不到P标签，肯定不达标
                return -0.2

            # 获取该P标签后的内容
            content_start = match.end()

            # 找到下一个P标签或结尾
            next_p_pos = len(five_ps_text)
            for j in range(i + 1, len(self.P_LABELS)):
                next_match = re.search(self.P_LABELS[j], five_ps_text[content_start:])
                if next_match:
                    next_p_pos = content_start + next_match.start()
                    break

            content = five_ps_text[content_start:next_p_pos].strip()

            # 计算条目数（按行分割，过滤空行）
            if not content or "待完善" in content:
                # 该P未填写，不达标
                return -0.2

            lines = [line.strip() for line in content.split('\n') if line.strip()]
            item_count = len(lines)

            if item_count < self.min_items or item_count > self.max_items:
                return -0.2

        return 0.0

    def compute(
        self,
        response: str,
        previous_5ps_text: str,
        turn_count: int
    ) -> RewardResult:
        """
        计算总奖励

        Args:
            response: 模型生成的回复
            previous_5ps_text: 上一轮的5Ps文本
            turn_count: 当前对话轮次

        Returns:
            RewardResult
        """
        # 提取当前5Ps文本
        current_5ps_text = ""
        five_ps_match = re.search(r'<5Ps>(.*?)</5Ps>', response, re.DOTALL)
        if five_ps_match:
            current_5ps_text = five_ps_match.group(1).strip()

        # 计算各项奖励
        r_format = self.compute_format_reward(response)
        r_continuity = self.compute_continuity_reward(current_5ps_text, previous_5ps_text)
        r_item_nums = self.compute_item_nums_reward(current_5ps_text, turn_count)

        # 计算总奖励
        total_reward = (
            self.format_weight * r_format +
            self.continuity_weight * r_continuity +
            self.item_nums_weight * r_item_nums
        )

        return RewardResult(
            total_reward=total_reward,
            r_format=r_format,
            r_5ps_continuity=r_continuity,
            r_5ps_item_nums=r_item_nums,
            details={
                "format": r_format,
                "continuity": r_continuity,
                "item_nums": r_item_nums
            }
        )


def compute_rewards(
    responses: List[str],
    previous_5ps_texts: List[str],
    turn_counts: List[int],
    reward_fn: RewardFunction
) -> List[float]:
    """
    批量计算奖励

    Args:
        responses: 模型生成的回复列表
        previous_5ps_texts: 对应的上一轮5Ps文本列表
        turn_counts: 对应的对话轮次列表
        reward_fn: 奖励函数实例

    Returns:
        奖励值列表
    """
    rewards = []
    for response, prev_5ps, turn_count in zip(responses, previous_5ps_texts, turn_counts):
        result = reward_fn.compute(response, prev_5ps, turn_count)
        rewards.append(result.total_reward)
    return rewards
