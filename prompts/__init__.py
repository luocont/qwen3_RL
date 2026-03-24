"""
提示词模块初始化文件
"""

from .five_ps_teacher import (
    FIVE_PS_TEACHER_SYSTEM_PROMPT,
)
from .think_and_reply_teacher import (
    REPLY_TEACHER_SYSTEM_PROMPT,
)

__all__ = [
    "FIVE_PS_TEACHER_SYSTEM_PROMPT",
    "REPLY_TEACHER_SYSTEM_PROMPT"
]
