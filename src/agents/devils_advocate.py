"""
🔍 深度质疑 Agent (Devil's Advocate) — 刻意找盲点和反向解读。
"""

from src.agents.base import BaseAgent
from src.llm.client import LLMClient


class DevilsAdvocateAgent(BaseAgent):
    """魔鬼代言人——挑战其他 Agent 的结论，找出可能的盲点。

    职责：不是来同意的，是来挑战的。假设其他分析可能遗漏了什么。
    输出：潜在盲点列表 + 信息缺失提示 + 反向解读。
    """

    @property
    def name(self) -> str:
        return "🔍 深度质疑 Agent"

    @property
    def prompt_file(self) -> str:
        return "devils_advocate"
