"""
LangGraph 编排层 — 一比一复制原版 OrchestratorV2 的 ReAct 决策循环。

原版流程:
  run() → _think() → _execute_actions() → _format_observations()
       → 循环(最多5轮) → _synthesize()

LangGraph 映射:
  START → coordinator → [need_more?] → actions → coordinator (loop)
                      → [!need_more] → synthesizer → END

架构不变、Prompt 不变、Agent 不变、RAG 不变——只把手写的 for 循环
换成 LangGraph 的状态管理和条件路由。
"""

import json
import concurrent.futures
from typing import Literal
from langgraph.graph import StateGraph, END

from src.llm.client import LLMClient, LLMConfig
from src.rag.engine import VectorStore
from src.agents.base import _load_prompt
from src.agents.data_extractor import DataExtractorAgent
from src.agents.risk_assessor import RiskAssessorAgent
from src.agents.compliance_checker import ComplianceCheckerAgent
from src.agents.devils_advocate import DevilsAdvocateAgent


# ═══════════════════════════════════════════════════════
# Coordinator System Prompt（直接从原版复用）
# ═══════════════════════════════════════════════════════

COORDINATOR_PROMPT_ZH = """# 你是金融分析团队的 Coordinator

## 你的团队

4 个专家。每人有明确的输入契约和输出契约。见下表：

| 成员 | 输入 | 产出 | 触发 |
|------|------|------|------|
| data_extractor | 搜索文本 + 聚焦指令 | `{status, data: [{metric, value, source_type, source}]}` | 需要提取数字时 |
| risk_assessor | 数据提取员 JSON + 搜索补充 | `{status, findings: [{dimension, fact, judgment, risk_level, evidence}]}` | 需要风险判断时 |
| compliance_checker | 数据提取员 JSON + 搜索补充 | `{status, findings: [{area, finding, verdict, evidence}]}` | 涉及合规时 |
| devils_advocate | 另外三人完整 JSON | `{status, challenges: [{target, target_conclusion, challenge, severity, evidence}]}` | 另外三人完成后，需要质疑时 |

## 你的职责

1. 判断问题类型 → 选策略
2. 写清晰的派发指令（用户问什么 + 已知什么 + 聚焦什么）
3. 审查 Agent 返回的 JSON：COMPLETE 真的够了吗？不同 Agent 结论有矛盾吗？
4. 维护任务账本：谁被派了 → 回没回来 → 结论是什么
5. 决定何时信息足够，结束循环

## 你的工具

- search_document(query) — 搜上传的 PDF。用关键词。
- search_web(query) — AI 联网搜索。用自然语言问题。不要用 site: 语法。
- run_analyst(analyst, instruction, context) — 派专家。instruction 按"用户问X。已掌握Y。聚焦Z。"格式写。

## 决策流程

### 问题类型 → 策略

**事实查询**（"...是多少""...什么时候"）→ search_web → 搜到就结束。不派专家。
**分析判断**（"怎么样""是否合理""有什么风险"）→ search_web → data_extractor → risk_assessor/compliance_checker → devils_advocate
**全面评估**（"全面""综合"）→ search_web → data_extractor → risk_assessor + compliance_checker 并行 → devils_advocate
**文档分析**（用户上传了文档且针对文档内容）→ search_document 优先 → search_web 仅补充 → 文档数据 vs 网络数据冲突时以文档为准

## JSON 决策格式

每轮一个 JSON:

```json
{
  "thought": "当前理解和进展（含任务账本摘要）",
  "plan": "下一步做什么，为什么",
  "actions": [
    {"tool": "search_web", "query": "自然语言搜索"},
    {"tool": "run_analyst", "analyst": "data_extractor", "instruction": "用户问XX。已掌握：YY。聚焦ZZ。", "context": "关键发现摘要"}
  ],
  "need_more": true
}
```

信息足够时:
```json
{"thought": "信息足够，无矛盾", "need_more": false}
```

## 铁律

1. 事实查询搜到就答，不派专家
2. devils_advocate 最后派，给三人完整输出
3. 指令按"用户问X。已掌握Y。聚焦Z。"格式
4. Agent NEED_MORE → 补搜 → 重派（最多 2 次）
5. 搜索无结果 → 换词重试一次 → 仍无 → 说明未找到，继续
6. need_more: false = 确认信息足够 + 无未解决矛盾"""

COORDINATOR_PROMPT_EN = """# You are the Chief Coordinator of a financial document analysis system.

## Your Toolbox

### search_document(query)
Search the uploaded PDF. Returns most relevant passages.

### search_web(query)
Search the web for latest information. Returns web results.

### run_analyst(analyst_name, instruction)
Dispatch an analyst: "data_extractor", "risk_assessor", "compliance_checker", "devils_advocate".
Give each a clear instruction. Multiple analysts run in parallel.

## Output Format

Each round, output ONE JSON object:
```json
{
  "thought": "my understanding and reasoning",
  "plan": "what I plan to do and why",
  "actions": [{"tool": "...", ...}],
  "need_more": true
}
```

When done: `{"thought": "...", "need_more": false}`"""


# ═══════════════════════════════════════════════════════
# JSON 解析（直接从原版复用）
# ═══════════════════════════════════════════════════════

def _parse_json(text: str) -> dict | None:
    """从 LLM 回复中提取 JSON。"""
    import re
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:]) if len(lines) > 1 else text
        if text.endswith("```"):
            text = text[:-3]
    text = re.sub(r'^```(?:json)?\s*', '', text)
    text = re.sub(r'\s*```$', '', text)

    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        try:
            return json.loads(text[start:end + 1])
        except json.JSONDecodeError:
            try:
                fixed = re.sub(r',\s*}', '}', text[start:end + 1])
                fixed = re.sub(r',\s*]', ']', fixed)
                return json.loads(fixed)
            except json.JSONDecodeError:
                pass
    return None


# ═══════════════════════════════════════════════════════
# 快速通道：简单查数据（直接从原版复用）
# ═══════════════════════════════════════════════════════

def _is_simple_lookup(query: str) -> bool:
    """判断是否为简单事实查询。"""
    q = query.strip()
    fact_keywords = ['多少', '什么时候', '代码', '股价', '市值', '市盈率',
                    '地址', '电话', '利率', '汇率', '几点', '在哪',
                    'what is', 'how many', 'how much', 'when did']
    analysis_keywords = ['分析', '评估', '风险', '怎么样', '为什么', '是否',
                       '全面', '综合', '完整', '对比', '比较', '有哪些',
                       '哪些', '名单', '列表', '排名', '走势', '趋势',
                       '如何', '怎样', '增长', '下降', '变化', '影响',
                       'analyze', 'assess', 'risk', 'comprehensive', 'compare']

    has_fact = any(kw in q.lower() for kw in fact_keywords)
    has_analysis = any(kw in q.lower() for kw in analysis_keywords)

    if has_analysis:
        return False
    if has_fact and len(q) < 30:
        return True
    return False


# ═══════════════════════════════════════════════════════
# LangGraph State（直接用 dict，和原版的局部变量对应）
# ═══════════════════════════════════════════════════════

def make_initial_state(
    user_query: str, vector_store, api_key: str,
    language: str = "zh", web_search_enabled: bool = True,
    doc_type: str = "", doc_company: str = "",
    max_rounds: int = 5, chat_history: list | None = None,
) -> dict:
    """构造初始状态，对应原版 run() 方法开头的变量初始化。"""

    from datetime import datetime
    today = datetime.now().strftime("%Y年%m月%d日" if language == "zh" else "%B %d, %Y")

    # 对话历史上下文（原版 lines 267-278）
    history_context = ""
    if chat_history and len(chat_history) > 0:
        recent = chat_history[-12:]
        history_lines = []
        for msg in recent:
            role_label = "用户" if msg["role"] == "user" else "系统"
            if language != "zh":
                role_label = "User" if msg["role"] == "user" else "Assistant"
            content = msg["content"][:500]
            history_lines.append(f"{role_label}: {content}")
        history_context = "\n".join(history_lines)

    # 文档信息（原版 lines 289-297）
    doc_info = f"用户上传了一份金融文档（{doc_type or '未知类型'}"
    if doc_company:
        doc_info += f"，公司：{doc_company}"
    doc_info += "）。"
    if doc_company:
        doc_info += ("\n文档是关于该公司的。所有 search_web 查询必须包含公司名，"
                    "不要搜到其他公司的数据。"
                    if language == "zh" else
                    "\nDocument is about this company. ALL search_web queries MUST include the company name.")

    # Coordinator system prompt（原版 lines 280-286）
    coordinator_prompt = COORDINATOR_PROMPT_ZH if language == "zh" else COORDINATOR_PROMPT_EN
    system_with_date = coordinator_prompt + (
        f"\n\n**当前日期: {today}**。你是在{today}进行分析。不要依赖训练数据中的日期——"
        f"今天是真实日期，所有财务数据可能已经发布。"
        if language == "zh" else
        f"\n\n**Current date: {today}**. You are analyzing on {today}. "
        f"Don't rely on training data dates - use the real current date."
    )

    # 用户消息（原版 lines 299-307）
    user_msg = (
        f"{doc_info}\n\n"
        + (f"## 之前的对话\n{history_context}\n\n" if history_context else "")
        + f"## 当前问题\n{user_query}\n\n"
        + ("如果当前问题是追问（如'那…呢？''同比呢？'），请结合之前的对话理解指代。"
           if language == "zh" else
           "If this is a follow-up question, use the conversation history to resolve references.")
        + "\n请开始分析。先思考你需要什么信息。"
    )

    return {
        # ── 配置 ──
        "user_query": user_query,
        "api_key": api_key,
        "language": language,
        "web_search_enabled": web_search_enabled,
        "doc_type": doc_type,
        "doc_company": doc_company,
        "max_rounds": max_rounds,

        # ── VectorStore ──
        "vector_store": vector_store,

        # ── Coordinator 对话（原版 lines 309-315）──
        "conversation": [
            {"role": "system", "content": system_with_date},
            {"role": "user", "content": user_msg},
        ],

        # ── 累积数据（对应原版 lines 319-321）──
        "all_findings": [],
        "structured_findings": [],
        "accumulated_web": [],
        "searched_queries": [],

        # ── 循环控制 ──
        "round_count": 0,
        "current_decision": None,
        "need_more": True,

        # ── 输出 ──
        "final_report": "",
        "followup_questions": [],
        "execution_log": [],

        # ── Agent 详情（用于 UI 展示）──
        "data_extraction": "",
        "risk_assessment": "",
        "compliance_check": "",
        "devils_advocate": "",

        # ── 错误 ──
        "error": "",
    }


# ═══════════════════════════════════════════════════════
# Node 1: Coordinator = 原版 _think()
# ═══════════════════════════════════════════════════════

def coordinator_node(state: GraphState) -> dict:
    """对应原版 _think() + 决策逻辑。LLM 产出下一步行动的 JSON。"""

    conversation = list(state["conversation"])
    api_key = state["api_key"]
    lang = state["language"]
    round_count = state["round_count"] + 1
    max_rounds = state["max_rounds"]

    log = list(state.get("execution_log", []))

    # 调用 LLM（原版 _think 方法 lines 419-432）
    config = LLMConfig(api_key=api_key, model="qwen-plus", temperature=0.2, max_tokens=1200)
    client = LLMClient(config)

    try:
        resp = client.chat(conversation)
    except Exception as e:
        log.append({"agent": "Coordinator", "status": "error", "content": f"Think failed: {e}"})
        return {
            "round_count": round_count,
            "need_more": False,
            "execution_log": log,
            "error": str(e),
        }

    decision = _parse_json(resp)

    # JSON 解析失败 → 重试一次（原版 lines 333-348）
    if decision is None:
        conversation.append({
            "role": "user",
            "content": ("上一轮输出不是有效 JSON。请只输出一行 JSON，不要任何解释。"
                       '格式: {"thought":"...","plan":"...","actions":[],"need_more":true}')
            if lang == "zh" else
            "Last output was not valid JSON. Output ONLY one line of JSON."
        })
        try:
            resp2 = client.chat(conversation)
            decision = _parse_json(resp2)
        except Exception:
            pass

        if decision is None:
            # 两次都失败 → 基于已有信息强制综合
            decision = {"thought": "JSON解析失败，基于已收集信息综合", "need_more": False}

    log.append({"agent": "Coordinator", "status": "running",
                "content": f"第{round_count}轮 | {decision.get('thought', '')[:150]}"})

    return {
        "conversation": conversation,
        "current_decision": decision,
        "round_count": round_count,
        "need_more": decision.get("need_more", False) if round_count < max_rounds else False,
        "execution_log": log,
    }


# ═══════════════════════════════════════════════════════
# Node 2: Actions = 原版 _execute_actions() + _format_observations()
# ═══════════════════════════════════════════════════════

def _run_analyst(name: str, instruction: str, context: str,
                 vector_store, api_key: str, lang: str,
                 web_results: list | None, accumulated_web: list) -> dict:
    """对应原版 _run_analyst() lines 596-627。"""
    from src.agents.base import AgentResult

    # 创建 Agent 实例
    cfg = LLMConfig(api_key=api_key, model="qwen-plus", temperature=0.3, max_tokens=3000)
    llm = LLMClient(cfg)

    agents = {
        "data_extractor": DataExtractorAgent(llm, lang),
        "risk_assessor": RiskAssessorAgent(llm, lang),
        "compliance_checker": ComplianceCheckerAgent(llm, lang),
        "devils_advocate": DevilsAdvocateAgent(llm, lang),
    }
    agent = agents.get(name)
    if not agent:
        return {"type": "error", "text": f"未知分析师: {name}", "analyst": name}

    # 组装任务指令（原版 lines 512-519）
    full_instr = instruction
    if context:
        full_instr = f"""## 你的任务
{instruction}

## 团队已掌握的信息
{context}"""

    # 格式化 web 结果（原版 lines 165-177）
    web_for_agent = None
    if accumulated_web and len(accumulated_web) > 0:
        from src.search.web_search import WebResult
        web_for_agent = []
        for item in accumulated_web[-10:]:
            web_for_agent.append(WebResult(
                title=item.get("title", ""),
                url=item.get("url", ""),
                snippet=item.get("snippet", ""),
            ))

    result = agent.run(full_instr, vector_store, api_key=api_key,
                      web_search_results=web_for_agent)
    raw = result.content[:3000] if result.success else result.error

    return {
        "type": "analyst_output",
        "analyst": name,
        "success": result.success,
        "content": raw,
    }


def actions_node(state: GraphState) -> dict:
    """对应原版 _execute_actions() + _format_observations()。"""

    decision = state.get("current_decision", {})
    actions = decision.get("actions", [])
    user_query = state["user_query"]
    vector_store = state["vector_store"]
    api_key = state["api_key"]
    lang = state["language"]
    web_search_enabled = state["web_search_enabled"]
    searched_queries = list(state.get("searched_queries", []))
    accumulated_web = list(state.get("accumulated_web", []))
    all_findings = list(state.get("all_findings", []))
    structured_findings = list(state.get("structured_findings", []))
    conversation = list(state["conversation"])
    log = list(state.get("execution_log", []))

    results: dict[str, list] = {}
    agent_name_map = {
        "data_extractor": "数据提取 Agent",
        "risk_assessor": "风险评估 Agent",
        "compliance_checker": "合规审查 Agent",
        "devils_advocate": "深度质疑 Agent",
    }

    def _do(action: dict):
        tool = action.get("tool", "")
        key = tool
        try:
            if tool == "search_document":
                q = action.get("query", user_query)
                key = f"search_doc:{q[:60]}"
                if q in searched_queries:
                    return key, [{"type": "cached", "text": "(重复查询，跳过)"}]
                searched_queries.append(q)
                if vector_store is None or vector_store.is_empty:
                    return key, [{"type": "empty", "text": "文档为空"}]
                try:
                    from src.rag.engine import rewrite_query_for_rag
                    extra = rewrite_query_for_rag(q, api_key, lang)
                    results_list = vector_store.search(q, top_k=5, extra_queries=extra, api_key=api_key)
                    return key, [{
                        "type": "doc_chunk",
                        "page": r.chunk.page,
                        "score": round(r.score, 3),
                        "text": r.chunk.text[:500]
                    } for r in results_list]
                except Exception as e:
                    return key, [{"type": "error", "text": str(e)}]

            elif tool == "search_web":
                if not web_search_enabled:
                    return key, [{"type": "disabled", "text": "联网搜索未启用"}]
                q = action.get("query", user_query)
                key = f"search_web:{q[:60]}"
                if q in searched_queries:
                    return key, [{"type": "cached", "text": "(重复查询，跳过)"}]
                searched_queries.append(q)

                # enable_search（原版 lines 566-593）
                try:
                    from src.llm.client import LLMConfig
                    prompt = f"搜索以下内容，返回具体数据和来源：{q}"
                    cfg = LLMConfig(api_key=api_key, model="qwen-turbo", temperature=0.1, max_tokens=1200)
                    client = LLMClient(cfg)
                    resp = client.chat([{"role": "user", "content": prompt}], enable_search=True)
                    if resp and len(resp.strip()) > 20:
                        web_result = {
                            "type": "ai_search",
                            "title": f"AI搜索: {q[:60]}",
                            "url": "",
                            "snippet": resp.strip()[:1000],
                        }
                        accumulated_web.append(web_result)
                        return key, [web_result]
                except Exception:
                    pass
                return key, [{"type": "empty", "text": "搜索无结果"}]

            elif tool == "run_analyst":
                name = action.get("analyst", "data_extractor")
                instr = action.get("instruction", user_query)
                ctx = action.get("context", "") or ""
                if not ctx and len(all_findings) > 0:
                    ctx = "\n".join(all_findings[-3:])[:2000]
                key = f"analyst:{name}"
                agent_result = _run_analyst(name, instr, ctx, vector_store, api_key, lang,
                                           accumulated_web, accumulated_web)
                return key, [agent_result]

            else:
                return key, [{"type": "error", "text": f"未知工具: {tool}"}]
        except Exception as e:
            return key, [{"type": "error", "text": str(e)}]

    # 并行执行（原版 lines 530-535）
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        futures = {pool.submit(_do, a): a for a in actions}
        for future in concurrent.futures.as_completed(futures):
            key, items = future.result()
            results.setdefault(key, []).extend(items)

    # ── 格式化观察（原版 _format_observations lines 633-682）──
    parts = []
    agent_statuses = []

    for key, items in results.items():
        parts.append(f"\n### {key}")
        for item in items:
            t = item.get("type", "?")
            if t == "doc_chunk":
                parts.append(f"[文档 p{item['page']}, score={item['score']}] {item['text'][:300]}")
            elif t in ("web_result", "ai_search"):
                url = f" ({item.get('url', '')})" if item.get('url') else ""
                parts.append(f"[{item.get('title', '')}{url}] {item['snippet'][:500]}")
            elif t == "analyst_output":
                status = "✓" if item.get("success") else "✗"
                name = item.get("analyst", "?")
                content = item.get("content", "")
                cn_name = agent_name_map.get(name, name)

                # 检测完整性声明
                if "[COMPLETE]" in content:
                    agent_statuses.append(f"[{cn_name}] COMPLETE")
                elif "[NEED_MORE]" in content:
                    need_idx = content.find("[NEED_MORE]")
                    need_text = content[need_idx:need_idx+300]
                    agent_statuses.append(f"[{cn_name}] NEED_MORE: {need_text}")
                else:
                    agent_statuses.append(f"[{cn_name}] 未声明")

                parts.append(f"[{cn_name} {status}] {content[:500]}")
                log.append({"agent": cn_name, "status": "done", "content": content[:150]})

                # 收集结构化输出
                if item.get("parsed"):
                    structured_findings.append({
                        "analyst": name,
                        **item["parsed"]
                    })
            else:
                parts.append(f"[{t}] {item.get('text', str(item))[:200]}")

    # 汇总 Agent 状态
    if agent_statuses:
        parts.insert(0, "\n## Agent 完整性汇总")
        all_complete = all("COMPLETE" in s and "NEED_MORE" not in s for s in agent_statuses)
        if all_complete:
            parts.insert(1, "**所有 Agent 确认完成 → 你下一轮必须 need_more: false**")
        else:
            incomplete = [s for s in agent_statuses if "NEED_MORE" in s]
            parts.insert(1, f"**{len(incomplete)}个Agent需要更多信息 → 你必须继续搜索**")
            for s in agent_statuses:
                parts.insert(2, f"- {s}")

    observation = "\n".join(parts)
    all_findings.append(observation)

    # 更新对话（原版 lines 370-377）
    conversation.append({
        "role": "assistant",
        "content": json.dumps(decision, ensure_ascii=False)
    })
    conversation.append({
        "role": "user",
        "content": f"<observation>\n{observation}\n</observation>\n\n请基于以上观察继续思考。信息够了吗？不够还需要什么？"
    })

    return {
        "conversation": conversation,
        "all_findings": all_findings,
        "structured_findings": structured_findings,
        "accumulated_web": accumulated_web,
        "searched_queries": searched_queries,
        "execution_log": log,
    }


# ═══════════════════════════════════════════════════════
# Node 3: Synthesizer = 原版 _synthesize()
# ═══════════════════════════════════════════════════════

def synthesizer_node(state: GraphState) -> dict:
    """对应原版 _synthesize() lines 688-736。"""

    user_query = state["user_query"]
    all_findings = state.get("all_findings", [])
    vector_store = state["vector_store"]
    api_key = state["api_key"]
    lang = state["language"]
    accumulated_web = state.get("accumulated_web", [])
    structured_findings = state.get("structured_findings", [])
    log = list(state.get("execution_log", []))

    synthesis_prompt = _load_prompt("orchestrator", lang)

    # 最终文档检索（原版 lines 701-709）
    final_doc = ""
    if vector_store is not None and not vector_store.is_empty:
        try:
            results = vector_store.search(user_query, top_k=3, api_key=api_key)
            if results:
                final_doc = "\n".join(
                    f"[p{r.chunk.page}] {r.chunk.text[:300]}" for r in results[:3]
                )
        except Exception:
            pass

    # ── 构建简报 ──
    if structured_findings:
        # 路径 A：Agent 结构化数据
        parts = [f"## 用户问题\n{user_query}\n"]
        parts.append("## 团队分析结果（唯一数据来源）")
        for s in structured_findings:
            aname = s.get("analyst", "?")
            if "data" in s:
                parts.append(f"### {aname}\n| 指标 | 数值 | 类型 | 来源 |\n|------|------|------|------|")
                for d in s["data"]:
                    parts.append(f"| {d.get('metric','?')} | {d.get('value','?')} | {d.get('source_type','?')} | {d.get('source','?')} |")
                parts.append("")
            if "findings" in s:
                parts.append(f"### {aname}")
                for f in s["findings"]:
                    parts.append(f"- **{f.get('dimension', f.get('area', '?'))}** [{f.get('risk_level', f.get('verdict', ''))}]: {f.get('judgment', f.get('finding','?'))} (证据: {f.get('evidence','?')})")
                parts.append("")
            if "challenges" in s:
                parts.append(f"### {aname}")
                for c in s["challenges"]:
                    parts.append(f"- [{c.get('severity','?')}] **{c.get('target','?')}**: {c.get('challenge','?')}")
                parts.append("")
        parts.append("")
        if final_doc.strip():
            parts.append(f"## 文档关键段落\n{final_doc}\n")
        parts.append("---\n## 输出要求\n1. 上面「团队分析结果」是你唯一允许使用的数据来源。禁止使用表中没有的数字。\n2. 第一句直接给答案。\n3. 标注来源。\n4. 不同 Agent 结论有矛盾时必须指出并裁决。")
        brief = "\n".join(parts)
    else:
        # 路径 B：原始观察文本
        raw_text = ""
        for w in accumulated_web[-10:]:
            raw_text += f"[{w.get('title','')}] {w.get('snippet','')[:500]}\n"
        findings_text = "\n---\n".join(all_findings[-4:])[:4000]
        raw_text += findings_text

        brief_parts = [f"## 用户问题\n{user_query}\n"]
        brief_parts.append(f"## 搜索结果与分析过程\n{raw_text}\n")
        if final_doc.strip():
            brief_parts.append(f"## 文档关键段落\n{final_doc}\n")
        brief_parts.append("---\n## 输出要求\n1. 第一句直接给答案。\n2. 标注来源。\n3. 不确定就说'不确定'。")
        brief = "\n".join(brief_parts)

    # LLM 写报告（原版 lines 722-735）
    config = LLMConfig(api_key=api_key, model="qwen-max", temperature=0.3, max_tokens=4000)
    client = LLMClient(config)

    messages = [
        {"role": "system", "content": synthesis_prompt},
        {"role": "user", "content": brief},
    ]

    try:
        report = client.chat(messages)
    except Exception as e:
        log.append({"agent": "Synthesizer", "status": "error", "content": str(e)})
        report = f"报告生成失败: {e}"

    # 生成追问（原版 lines 1311-1344）
    questions = []
    try:
        cfg_fu = LLMConfig(api_key=api_key, model="qwen-turbo", temperature=0.4, max_tokens=300)
        cl_fu = LLMClient(cfg_fu)
        fu_prompt = f"""基于以下对话上下文，生成4个值得继续追问的问题。
用户刚才问了：{user_query}
系统的分析结论（摘要）：{report[:1500]}
要求：紧跟分析中的关键发现、覆盖不同角度、每个不超过20字、直接输出列表每行一个以"- "开头"""
        fu_resp = cl_fu.chat([{"role": "user", "content": fu_prompt}])
        for line in fu_resp.strip().split("\n"):
            q = line.strip().lstrip("- ").lstrip("0123456789. ").strip()
            if q and len(q) > 3:
                questions.append(q)
    except Exception:
        pass

    log.append({"agent": "Synthesizer", "status": "done", "content": "报告生成完成"})

    # 收集 Agent 详情
    agent_outputs = {}
    for sf in structured_findings:
        aname = sf.get("analyst", "")
        content = json.dumps(sf, ensure_ascii=False, indent=2)
        if aname == "data_extractor":
            agent_outputs["data_extraction"] = content
        elif aname == "risk_assessor":
            agent_outputs["risk_assessment"] = content
        elif aname == "compliance_checker":
            agent_outputs["compliance_check"] = content
        elif aname == "devils_advocate":
            agent_outputs["devils_advocate"] = content

    return {
        "final_report": report,
        "followup_questions": questions[:4],
        "execution_log": log,
        **agent_outputs,
    }


# ═══════════════════════════════════════════════════════
# Router（原版 lines 380-389 的 need_more 判断）
# ═══════════════════════════════════════════════════════

def router(state: GraphState) -> Literal["actions", "synthesizer"]:
    """对应原版: if need_more → loop / else → synthesize"""
    if state.get("need_more", False) and state.get("current_decision", {}).get("actions"):
        return "actions"
    return "synthesizer"


# ═══════════════════════════════════════════════════════
# Build Graph
# ═══════════════════════════════════════════════════════

from typing import TypedDict, Any

class GraphState(TypedDict, total=False):
    user_query: str
    api_key: str
    language: str
    web_search_enabled: bool
    doc_type: str
    doc_company: str
    max_rounds: int
    vector_store: Any
    conversation: list
    all_findings: list
    structured_findings: list
    accumulated_web: list
    searched_queries: list
    round_count: int
    current_decision: Any
    need_more: bool
    final_report: str
    followup_questions: list
    execution_log: list
    data_extraction: str
    risk_assessment: str
    compliance_check: str
    devils_advocate: str
    error: str

_graph = None

def build_graph() -> StateGraph:
    global _graph
    if _graph is not None:
        return _graph

    workflow = StateGraph(GraphState)

    workflow.add_node("coordinator", coordinator_node)
    workflow.add_node("actions", actions_node)
    workflow.add_node("synthesizer", synthesizer_node)

    workflow.set_entry_point("coordinator")

    workflow.add_conditional_edges("coordinator", router, {
        "actions": "actions",
        "synthesizer": "synthesizer",
    })

    workflow.add_edge("actions", "coordinator")
    workflow.add_edge("synthesizer", END)

    _graph = workflow.compile()
    return _graph


# ═══════════════════════════════════════════════════════
# Public API
# ═══════════════════════════════════════════════════════

def run_analysis(
    user_query: str,
    vector_store,
    api_key: str,
    language: str = "zh",
    web_search_enabled: bool = True,
    doc_type: str = "",
    doc_company: str = "",
    max_rounds: int = 5,
    chat_history: list | None = None,
) -> dict:
    """运行一次完整的金融风险分析。和原版 OrchestratorV2.run() 接口一致。"""

    # 快速通道（原版 lines 249-261）
    if _is_simple_lookup(user_query):
        from datetime import datetime
        today = datetime.now().strftime("%Y年%m月%d日")

        doc_context = ""
        if vector_store is not None and not vector_store.is_empty:
            try:
                results = vector_store.search(user_query, top_k=2, api_key=api_key)
                if results:
                    doc_context = "\n".join(
                        f"[第{r.chunk.page}页] {r.chunk.text[:300]}" for r in results[:2]
                    )
            except Exception:
                pass

        company_hint = f"\n注意：你正在分析的公司是{doc_company}。只回答关于该公司的数据，忽略其他公司的信息。\n" if doc_company else ""

        prompt = f"""今天是{today}。请联网搜索最新信息后回答。

{'## 用户上传的文档（仅供参考）\n' + doc_context[:1500] if doc_context else ''}
{company_hint}
## 用户问题
{user_query}

请联网搜索后直接回答。有具体数据列出并标注来源。不确定说"不确定"。"""

        config = LLMConfig(api_key=api_key, model="qwen-max", temperature=0.3, max_tokens=2000)
        client = LLMClient(config)
        answer = client.chat([{"role": "user", "content": prompt}], enable_search=web_search_enabled)

        return {
            "final_report": answer,
            "data_extraction": "", "risk_assessment": "", "compliance_check": "", "devils_advocate": "",
            "followup_questions": [], "execution_log": [], "rounds": 0,
        }

    # 主流程
    initial_state = make_initial_state(
        user_query=user_query, vector_store=vector_store, api_key=api_key,
        language=language, web_search_enabled=web_search_enabled,
        doc_type=doc_type, doc_company=doc_company, max_rounds=max_rounds,
        chat_history=chat_history,
    )

    graph = build_graph()

    try:
        final_state = graph.invoke(initial_state, {"recursion_limit": 50})
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {
            "final_report": f"分析出错: {e}",
            "data_extraction": "", "risk_assessment": "", "compliance_check": "", "devils_advocate": "",
            "followup_questions": [], "execution_log": [], "rounds": 0,
        }

    return {
        "final_report": final_state.get("final_report", ""),
        "data_extraction": final_state.get("data_extraction", ""),
        "risk_assessment": final_state.get("risk_assessment", ""),
        "compliance_check": final_state.get("compliance_check", ""),
        "devils_advocate": final_state.get("devils_advocate", ""),
        "followup_questions": final_state.get("followup_questions", []),
        "execution_log": final_state.get("execution_log", []),
        "rounds": final_state.get("round_count", 0),
    }
