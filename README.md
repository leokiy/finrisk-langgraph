# FinRisk MultiAgent (LangGraph 版)

基于 LangGraph 重构的多 Agent 协作金融风险智能分析系统。

原版架构完整保留：Coordinator ReAct 决策循环 + 4 个专业 Agent + 混合 RAG + 双路联网搜索。
唯一改动：手写 for 循环 → LangGraph StateGraph 管理状态和路由。

## 快速开始

```bash
pip install -r requirements.txt
python -m streamlit run app.py
```

## 架构

LangGraph 图结构精确映射原版 OrchestratorV2：

- **coordinator** 节点 = 原版 `_think()`：LLM 产出决策 JSON
- **actions** 节点 = 原版 `_execute_actions()` + `_format_observations()`：并行执行搜索和 Agent 分发
- **synthesizer** 节点 = 原版 `_synthesize()`：综合报告生成
- **router** = 原版的 `need_more` 判断：继续循环还是进入综合

## 项目结构

```
finrisk-langchain/
├── app.py                  # Streamlit 界面
├── src/
│   ├── graph.py            # LangGraph 编排层（核心）
│   ├── agents/             # 4 个专业 Agent（复用原版）
│   ├── rag/engine.py       # RAG 引擎（复用原版）
│   ├── llm/client.py       # LLM 客户端（复用原版）
│   └── search/             # 联网搜索（复用原版）
└── prompts/zh/             # 中文 Prompt 模板
```

## 与原版对比

| 维度 | 原版 | LangGraph 版 |
|------|------|-------------|
| 编排引擎 | 手写 for 循环 | StateGraph + conditional edges |
| 状态管理 | 散落在局部变量中 | TypedDict 统一管理 |
| 流程控制 | if/else | router + add_conditional_edges |
| 编排代码量 | 1355 行 | graph.py ~400 行 |
| Agent 实现 | 不变 | 直接复用 |
| RAG 引擎 | 不变 | 直接复用 |
