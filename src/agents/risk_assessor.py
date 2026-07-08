"""
⚠️ 风险评估 Agent — 从四个维度评估风险，给出等级和依据。
"""

from src.agents.base import BaseAgent
from src.llm.client import LLMClient


class RiskAssessorAgent(BaseAgent):
    """从市场、信用、流动、操作四个维度逐项评估风险。

    职责：基于文档数据做专业风险判断，每项判断必须引用原文证据。
    输出：四维风险矩阵 + 综合评分 + 关键风险点。
    """

    @property
    def name(self) -> str:
        return "⚠️ 风险评估 Agent"

    @property
    def prompt_file(self) -> str:
        return "risk_assessor"
