"""
FinRisk MultiAgent (LangGraph 版)

原版架构不变：Coordinator ReAct 决策循环 + 4 Agent + 混合 RAG + 双路搜索。
唯一改动：手写 for 循环 → LangGraph StateGraph 管理状态和路由。

启动: /c/Python314/python -m streamlit run app.py
"""

import os, sys, time, tempfile
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st

st.set_page_config(page_title="FinRisk MultiAgent (LangGraph)", page_icon="🏦", layout="wide", initial_sidebar_state="expanded")

# ── 样式 ──
st.markdown("""<style>
.main-title{font-size:2rem;font-weight:700;text-align:center;color:#1a3c5e}
.subtitle{font-size:.95rem;color:#888;text-align:center;margin-bottom:1rem}
.disclaimer{color:#aaa;font-size:.75rem;margin-top:2rem;text-align:center}
</style>""", unsafe_allow_html=True)

# ── Session ──
for k, v in [("vector_store",None),("file_processed",False),("chat_history",[]),("last_analysis",None)]:
    if k not in st.session_state: st.session_state[k] = v

# ── 侧边栏 ──
with st.sidebar:
    st.markdown("## ⚙️ 配置")
    api_key = st.text_input("DashScope API Key", type="password", value=os.getenv("DASHSCOPE_API_KEY",""), help="dashscope.console.aliyun.com/apiKey")
    web_search_enabled = st.checkbox("🌐 联网搜索", value=True)
    st.divider()
    st.markdown("### 📖 关于")
    st.markdown("LangGraph 版 FinRisk MultiAgent\n- 🧠 Coordinator ReAct\n- 📊 数据提取\n- ⚠️ 风险评估\n- 📋 合规审查\n- 🔍 深度质疑")

# ── 主页 ──
st.markdown('<div class="main-title">🏦 FinRisk MultiAgent</div>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">LangGraph 驱动 · Coordinator 自动决策 · 4 Agent 协作</p>', unsafe_allow_html=True)

uploaded_file = st.file_uploader("📄 上传金融文档 (PDF/TXT/MD)", type=["pdf","txt","md"])

if uploaded_file and not st.session_state.file_processed:
    if not api_key:
        st.error("⚠️ 请先输入 API Key")
        st.stop()
    with st.status("处理文档...", expanded=True) as status:
        try:
            suffix = Path(uploaded_file.name).suffix or ".pdf"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(uploaded_file.getvalue())
                tmp_path = tmp.name
            from src.rag.engine import build_rag_from_file
            st.session_state.vector_store = build_rag_from_file(tmp_path, api_key=api_key)
            st.session_state.file_processed = True
            st.session_state.chat_history = []
            os.unlink(tmp_path)
            status.update(label=f"✅ 完成: {st.session_state.vector_store.chunk_count} 块", state="complete")
        except Exception as e:
            status.update(label=f"❌ 失败: {e}", state="error")
            st.stop()

if st.session_state.file_processed:
    st.success(f"📄 已就绪 ({st.session_state.vector_store.chunk_count} 块)")

# ── 对话 ──
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]): st.markdown(msg["content"])

quick_qs = ["对这份文件做个全面的风险评估", "这家公司最大的风险是什么？", "有哪些合规风险需要关注？", "偿债能力如何？"]
if not st.session_state.chat_history:
    cols = st.columns(len(quick_qs))
    for c, q in zip(cols, quick_qs):
        with c:
            if st.button(q, key=f"q_{q[:8]}", use_container_width=True):
                st.session_state.pending_query = q

user_query = st.chat_input("输入风险分析问题...")
if "pending_query" in st.session_state:
    user_query = st.session_state.pop("pending_query")

if user_query:
    if not api_key:
        st.error("⚠️ 请先输入 API Key")
        st.stop()

    with st.chat_message("user"): st.markdown(user_query)
    st.session_state.chat_history.append({"role":"user","content":user_query})

    from src.graph import run_analysis
    from src.rag.engine import VectorStore

    vs = st.session_state.vector_store if st.session_state.file_processed else VectorStore()

    # 提取文档类型和公司名
    doc_type, doc_company = "", ""
    if st.session_state.file_processed and vs and not vs.is_empty:
        try:
            from src.llm.client import LLMClient, LLMConfig
            front = vs.search("年报 公司 股份 有限", top_k=3, api_key=api_key)
            ft = " ".join(r.chunk.text[:200] for r in front)[:1000]
            if ft.strip():
                cfg = LLMConfig(api_key=api_key, model="qwen-turbo", temperature=0.1, max_tokens=80)
                resp = LLMClient(cfg).chat([{"role":"user","content":f"输出两行：第一行=文档类型，第二行=公司全称。只输出这两行：\n{ft}"}])
                lines = [l.strip() for l in resp.strip().split("\n") if l.strip()]
                if len(lines)>=1: doc_type = lines[0]
                if len(lines)>=2: doc_company = lines[1]
        except Exception: pass

    with st.chat_message("assistant"):
        with st.spinner("🤖 LangGraph 多 Agent 分析中..."):
            result = run_analysis(
                user_query=user_query, vector_store=vs, api_key=api_key,
                language="zh", web_search_enabled=web_search_enabled,
                doc_type=doc_type, doc_company=doc_company, max_rounds=5,
                chat_history=st.session_state.get("chat_history", []),
            )
        report = result.get("final_report", "分析失败")
        st.markdown(report)
        st.download_button("📥 下载报告", data=report, file_name=f"finrisk_{time.strftime('%Y%m%d_%H%M%S')}.md", mime="text/markdown")

    st.session_state.chat_history.append({"role":"assistant","content":report})
    st.session_state.last_analysis = result

# ── Agent 详情 ──
if st.session_state.get("last_analysis"):
    r = st.session_state.last_analysis
    with st.expander("🔍 Agent 分析详情", expanded=False):
        tabs = st.tabs(["📊 数据提取","⚠️ 风险评估","📋 合规审查","🔍 深度质疑","📋 执行日志"])
        for tab, key in zip(tabs[:4], ["data_extraction","risk_assessment","compliance_check","devils_advocate"]):
            with tab:
                content = r.get(key, "")
                st.markdown(content if content else "（未调用）")
        with tabs[4]:
            for log in r.get("execution_log", []):
                icon = {"running":"⏳","done":"✅","error":"❌"}.get(log["status"],"•")
                st.markdown(f"{icon} **{log['agent']}**: {log.get('content','')[:200]}")

st.divider()
st.markdown('<p class="disclaimer">⚠️ AI 驱动，仅供参考，不构成投资建议。</p>', unsafe_allow_html=True)
