"""
📋 合规审查 Agent — 对照监管要求，逐条检查合规风险。
"""

from src.agents.base import BaseAgent
from src.llm.client import LLMClient


class ComplianceCheckerAgent(BaseAgent):
    """对照金融监管框架，逐条审查文档中的合规风险。

    职责：以监管者视角扫描文档，发现潜在的合规问题。
    输出：合规检查清单（通过/警示/违规），每条标注涉及的具体条款依据。
    """

    @property
    def name(self) -> str:
        return "📋 合规审查 Agent"

    @property
    def prompt_file(self) -> str:
        return "compliance_checker"
