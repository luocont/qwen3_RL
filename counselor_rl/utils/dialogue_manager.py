# -*- coding: utf-8 -*-
"""
counselor_rl/utils/dialogue_manager.py

多轮对话上下文管理器。
维护对话历史、5Ps 病例状态、PANAS 评分历史。
"""

from dataclasses import dataclass, field


@dataclass
class DialogueTurn:
    role: str
    content: str
    turn_index: int


@dataclass
class EpisodeState:
    intake_form: str
    attitude: str
    current_5ps: str = ""
    panas_history: list[dict] = field(default_factory=list)
    dialogue_turns: list[DialogueTurn] = field(default_factory=list)
    is_ended: bool = False

    def add_turn(self, role: str, content: str):
        turn = DialogueTurn(
            role=role,
            content=content,
            turn_index=len(self.dialogue_turns),
        )
        self.dialogue_turns.append(turn)

    def get_dialogue_history_as_messages(self) -> list[dict]:
        messages = []
        for turn in self.dialogue_turns:
            if turn.role == "client":
                messages.append({"role": "user", "content": turn.content})
            elif turn.role == "counselor":
                messages.append({"role": "assistant", "content": turn.content})
        return messages

    def get_dialogue_text(self) -> str:
        lines = []
        for turn in self.dialogue_turns:
            if turn.role == "client":
                lines.append(f"Client: {turn.content}")
            elif turn.role == "counselor":
                lines.append(f"Counselor: {turn.content}")
        return "\n".join(lines)

    def get_turn_count(self) -> int:
        return len([t for t in self.dialogue_turns if t.role == "client"])

    def add_panas_score(self, panas: dict):
        self.panas_history.append(panas)

    def get_latest_panas(self) -> dict | None:
        if self.panas_history:
            return self.panas_history[-1]
        return None

    def get_first_panas(self) -> dict | None:
        if self.panas_history:
            return self.panas_history[0]
        return None
