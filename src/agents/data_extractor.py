"""
📊 数据提取 Agent — 从文档中提取结构化数据，不发表意见。
"""

from src.agents.base import BaseAgent
from src.llm.client import LLMClient


class DataExtractorAgent(BaseAgent):
    """从金融文档中提取关键财务指标和结构化数据。

    职责：只提取数字和事实，不做任何分析或推断。
    输出：结构化表格，每项数据标注来源页码。
    """

    @property
    def name(self) -> str:
        return "📊 数据提取 Agent"

    @property
    def prompt_file(self) -> str:
        return "data_extractor"
