"""
RAG 模块 — 长文档优化 + 全量表提取 + 多路检索。

核心能力:
  1. 表格全量提取: pdfplumber 提取所有页面表格 → 关键词匹配注入 Agent 上下文
  2. 大块分块: 1200字/块, 300字重叠
  3. 首页锚定: 文档前3页始终注入检索结果
  4. LLM 查询改写: 用户问题 → 2-3 路精准检索查询
  5. 多路检索 + 去重: 20 chunks/次, Jaccard 去重
"""

import os
import re
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import faiss
from langchain_text_splitters import RecursiveCharacterTextSplitter


@dataclass
class DocumentChunk:
    text: str
    source: str
    page: int
    chunk_index: int

    def citation(self) -> str:
        return f"[{self.source}, 第{self.page}页]"


@dataclass
class DocumentTable:
    """从PDF提取的表格。"""
    rows: list[list[str]]     # 二维数组 (含表头)
    page: int
    source: str
    caption: str = ""         # 表格上方/下方的标题文字

    def to_markdown(self) -> str:
        if not self.rows:
            return ""
        lines = []
        # 表头
        header = self.rows[0]
        lines.append("| " + " | ".join(str(c) for c in header) + " |")
        lines.append("|" + "|".join("---" for _ in header) + "|")
        # 数据行 (最多20行，避免太长)
        for row in self.rows[1:21]:
            lines.append("| " + " | ".join(str(c)[:100] for c in row) + " |")
        if len(self.rows) > 21:
            lines.append(f"... (共{len(self.rows)-1}行数据，仅展示前20行)")
        return "\n".join(lines)

    def citation(self) -> str:
        return f"[{self.source}, 第{self.page}页, 表格]"

    @property
    def header_text(self) -> str:
        """表头文字，用于关键词匹配。"""
        return " ".join(str(c) for c in self.rows[0]) if self.rows else ""


@dataclass
class RetrievalResult:
    chunk: DocumentChunk
    score: float


class DocumentLoader:
    @staticmethod
    def load_pdf(file_path: str) -> tuple[list[dict], list[DocumentTable]]:
        """返回 (pages, tables)。"""
        ext = os.path.splitext(file_path)[1].lower()
        if ext == ".pdf":
            return DocumentLoader._load_pdfplumber(file_path)
        if ext in (".txt", ".md", ".csv"):
            text = open(file_path, encoding="utf-8").read()
            return [{"page": 1, "text": text}], []
        raise ValueError(f"不支持的文件格式: {ext}")

    @staticmethod
    def _load_pdfplumber(file_path: str) -> tuple[list[dict], list[DocumentTable]]:
        import pdfplumber
        source = os.path.basename(file_path)
        pages = []
        tables: list[DocumentTable] = []

        with pdfplumber.open(file_path) as pdf:
            for i, page in enumerate(pdf.pages):
                text = page.extract_text()
                if text and text.strip():
                    pages.append({"page": i + 1, "text": text.strip()})

                # 提取本页所有表格
                raw_tables = page.extract_tables()
                for rt in raw_tables:
                    if rt and len(rt) >= 2:  # 至少表头+一行数据
                        # 清洗：去掉全是None的行，None→""
                        cleaned = []
                        for row in rt:
                            if row and any(c is not None for c in row):
                                cleaned.append([str(c) if c is not None else "" for c in row])
                        if len(cleaned) >= 2:
                            tables.append(DocumentTable(
                                rows=cleaned,
                                page=i + 1,
                                source=source,
                            ))

        if not pages:
            raise RuntimeError(f"PDF 中未提取到文本: {file_path}")
        return pages, tables


class TextChunker:
    """大块分块: 1200字/块, 300字重叠, 自动过滤低质量文本。"""

    # 低质量文本模式（目录页、页眉页脚、纯数字表格等）
    _LOW_QUALITY_PATTERNS = [
        # 目录特征
        r'^[\.\s\d]*目\s*录[\.\s\d]*$',
        r'^[IVX]+[\.\s、].*$',  # 罗马数字章节
        r'^第[一二三四五六七八九十\d]+[章节篇]',  # 第X章/第X节
        # 纯页码/页眉
        r'^[\s\d]+$',  # 纯空白和数字
        r'^\d{1,3}\s*$',  # 1-3位数字（页码）
        # 纯分隔符
        r'^[-—\.·•]{3,}\s*$',
        # 证券代码/公告编号（不是内容）
        r'^证券代码[：:].*$',
        r'^公告编号[：:].*$',
    ]

    def __init__(self, chunk_size: int = 1200, chunk_overlap: int = 300):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=[
                "\n\n\n", "\n\n", "\n",
                "。", "；", ".", ";",
                "，", ",", " ", ""
            ],
            length_function=len,
            is_separator_regex=False,
        )

    def _is_low_quality(self, text: str, page: int) -> bool:
        """判断文本块是否低质量（目录页、页眉页脚、纯数字行等）。"""
        t = text.strip()

        # 太短
        if len(t) < 50:
            return True

        # 前5页的文本，如果以章节标题开头（目录页特征），降为低质量
        if page <= 5:
            # 检查是否主要是章节列表
            chapter_lines = sum(1 for line in t.split("\n")
                              if line.strip() and
                              any(kw in line for kw in ["目录", "节", "章", "部分", "释义", "声明"]))
            if chapter_lines >= 3 and len(t) < 400:
                return True

        # 正则匹配低质量模式
        import re as _re
        for pattern in self._LOW_QUALITY_PATTERNS:
            if _re.match(pattern, t):
                return True

        # 非中英文内容占比过高
        content_chars = sum(1 for c in t if '一' <= c <= '鿿' or c.isalpha())
        if content_chars / max(len(t), 1) < 0.3:
            return True

        return False

    def chunk_document(self, pages: list[dict], source_name: str) -> list[DocumentChunk]:
        chunks = []
        for page_info in pages:
            texts = self.splitter.split_text(page_info["text"])
            for idx, text in enumerate(texts):
                if self._is_low_quality(text, page_info["page"]):
                    continue
                chunks.append(DocumentChunk(
                    text=text.strip(),
                    source=source_name,
                    page=page_info["page"],
                    chunk_index=idx,
                ))
        return chunks


class VectorStore:
    """FAISS 向量存储 + 表格存储 + 首页锚定 + 多路融合检索。"""

    # 表格关键词 → 用于匹配用户问题是否涉及表格数据
    _TABLE_KEYWORDS = [
        "资产", "负债", "收入", "利润", "现金流", "毛利率", "净利率",
        "应收", "应付", "存货", "借款", "费用", "营收", "净利润",
        "损益", "资产负债", "现金流量", "股东权益", "每股收益",
        "revenue", "profit", "asset", "liability", "cash flow",
        "income statement", "balance sheet",
    ]

    def __init__(self):
        self._chunks: list[DocumentChunk] = []
        self._tables: list[DocumentTable] = []
        self._index: Optional[faiss.IndexFlatIP] = None
        self._front_chunks: list[DocumentChunk] = []

    @property
    def chunk_count(self) -> int:
        return len(self._chunks)

    @property
    def table_count(self) -> int:
        return len(self._tables)

    @property
    def is_empty(self) -> bool:
        return len(self._chunks) == 0
    @property
    def has_semantic_search(self) -> bool:
        return self._index is not None

    def build_index(self, chunks: list[DocumentChunk], tables: list[DocumentTable],
                    api_key: str = ""):
        self._chunks = chunks
        self._tables = tables
        if api_key:
            import dashscope
            dashscope.api_key = api_key
        try:
            vectors = self._embed_batch([c.text for c in chunks])
            self._index = faiss.IndexFlatIP(vectors.shape[1])
            self._index.add(vectors.astype(np.float32))
        except Exception as e:
            # embedding 失败（如 DNS/网络问题）→ 退化为纯关键词搜索
            import sys
            print(f"[RAG] Embedding failed, falling back to keyword-only search: {e}", file=sys.stderr)
            self._index = None  # 标记 FAISS 不可用
        self._front_chunks = [c for c in chunks if c.page <= 3]

    def search(self, query: str, top_k: int = 20,
               extra_queries: list[str] | None = None,
               **kwargs) -> list[RetrievalResult]:
        """主检索: 关键词精确搜索 + FAISS语义检索混合。

        关键词命中 → 高分锚定
        FAISS语义检索 → 补盲区（概念性问题如"盈利质量"）
        两路结果去重合并，关键词结果优先。
        """
        if self.is_empty:
            return []

        # ── 第一路：关键词精确搜索 ──
        kw_results = self.keyword_search(query, top_k=top_k)

        # ── 第二路：FAISS 语义检索（如果 embedding 可用）──
        faiss_results = []
        if self._index is not None:
            queries = [query]
            if extra_queries:
                queries.extend(extra_queries)
            try:
                for q in queries:
                    vec = self._embed_batch([q])[0].reshape(1, -1).astype(np.float32)
                    k = min(top_k * 2, len(self._chunks))
                    scores, indices = self._index.search(vec, k)
                    for score, idx in zip(scores[0], indices[0]):
                        if 0 <= idx < len(self._chunks):
                            chunk = self._chunks[idx]
                            adjusted_score = float(score)
                            if chunk.page <= 5:
                                adjusted_score *= 0.4
                            faiss_results.append(
                                RetrievalResult(chunk=chunk, score=adjusted_score)
                            )
            except Exception:
                pass  # FAISS 搜索失败，退回关键词结果

        # ── 合并去重：关键词结果放前面，FAISS结果补后面 ──
        seen_sigs = set()
        merged = []

        # 先放关键词结果（高分、精确）
        for r in kw_results:
            sig = r.chunk.text[:80].strip()
            if sig not in seen_sigs:
                seen_sigs.add(sig)
                merged.append(r)

        # 再补 FAISS 结果（去重）
        for r in self._dedup_rank(faiss_results, top_k * 2):
            sig = r.chunk.text[:80].strip()
            if sig not in seen_sigs:
                seen_sigs.add(sig)
                merged.append(r)
                if len(merged) >= top_k:
                    break

        return merged[:top_k]

    def keyword_search(self, query: str, top_k: int = 10) -> list[RetrievalResult]:
        """关键词精确搜索：对 chunk 文本做关键词 + 正则匹配。

        比 FAISS 向量检索更精准——搜"净利润"就只返回含"净利润"的段落，
        不会因为语义相似度返回目录页。

        支持多关键词（空格分隔），匹配越多得分越高。
        """
        if self.is_empty:
            return []

        # 提取关键词
        keywords = [kw.strip() for kw in query.replace("？", " ").replace("?", " ").split()
                    if len(kw.strip()) >= 2]
        if not keywords:
            return []

        scored = []
        for idx, chunk in enumerate(self._chunks):
            text = chunk.text
            score = 0
            for kw in keywords:
                count = text.count(kw)
                if count > 0:
                    # 匹配次数越多得分越高，但做对数衰减防止长文本刷分
                    score += min(count, 5) * 0.2
            if score > 0:
                # 前置页降权（同FAISS规则）
                if chunk.page <= 5:
                    score *= 0.4
                scored.append((score, idx, chunk))

        scored.sort(key=lambda x: x[0], reverse=True)

        # 去重
        seen_sigs = set()
        results = []
        for score, idx, chunk in scored[:top_k * 2]:
            sig = chunk.text[:80].strip()
            if sig not in seen_sigs:
                seen_sigs.add(sig)
                results.append(RetrievalResult(chunk=chunk, score=min(score, 0.99)))
                if len(results) >= top_k:
                    break

        return results

    def search_tables(self, query: str, top_k: int = 5) -> list[DocumentTable]:
        """关键词匹配检索相关表格。"""
        if not self._tables:
            return []

        query_lower = query.lower()
        scored = []
        for t in self._tables:
            header = t.header_text.lower()
            caption = t.caption.lower()
            score = 0
            for kw in self._TABLE_KEYWORDS:
                if kw in query_lower and kw in header:
                    score += 2
                if kw in query_lower and kw in caption:
                    score += 1
            if score > 0:
                scored.append((score, t))

        scored.sort(key=lambda x: x[0], reverse=True)
        # 去重：相同表头的表只保留一个
        seen_headers = set()
        result = []
        for _, t in scored:
            sig = t.header_text[:80]
            if sig not in seen_headers:
                seen_headers.add(sig)
                result.append(t)
                if len(result) >= top_k:
                    break
        return result

    def clear(self):
        self._chunks = []
        self._tables = []
        self._index = None
        self._front_chunks = []

    @staticmethod
    def _dedup_rank(results: list[RetrievalResult], top_k: int) -> list[RetrievalResult]:
        ranked = sorted(results, key=lambda r: r.score, reverse=True)
        deduped = []
        seen_signatures = set()
        for r in ranked:
            # 用前 60 字做去重标记
            sig = r.chunk.text[:60].strip()
            if sig in seen_signatures:
                continue
            seen_signatures.add(sig)
            deduped.append(r)
            if len(deduped) >= top_k:
                break
        return deduped

    @staticmethod
    def _embed_batch(texts: list[str]) -> np.ndarray:
        import dashscope
        all_vecs = []
        for i in range(0, len(texts), 10):
            batch = texts[i:i + 10]
            last_err = None
            for attempt in range(3):
                try:
                    resp = dashscope.TextEmbedding.call(
                        model="text-embedding-v3",
                        input=batch,
                    )
                    if resp.status_code == 200:
                        for emb in resp.output["embeddings"]:
                            all_vecs.append(np.array(emb["embedding"], dtype=np.float32))
                        break
                    last_err = RuntimeError(f"Embedding API error (status={resp.status_code}): {resp.message}")
                except Exception as e:
                    last_err = e
                if attempt < 2:
                    import time
                    time.sleep(2.0 * (attempt + 1))
            else:
                raise RuntimeError(f"Embedding API 调用失败（重试3次后）。请检查网络连接。如果用VPN，试试关掉VPN。错误: {last_err}")
        return np.stack(all_vecs)


def build_rag_from_file(file_path: str, api_key: str = "") -> VectorStore:
    loader = DocumentLoader()
    chunker = TextChunker()
    source_name = os.path.basename(file_path)
    pages, tables = loader.load_pdf(file_path)
    chunks = chunker.chunk_document(pages, source_name)
    store = VectorStore()
    store.build_index(chunks, tables, api_key=api_key)
    return store


def rewrite_query_for_rag(query: str, api_key: str, language: str = "zh") -> list[str]:
    """用 LLM 将用户问题改写为 2-3 个 RAG 检索关键词。

    与 rewrite_query_for_search 不同：这个改写目标是文档内部检索，
    提取问题中的关键实体和指标，生成更精准的搜索词。

    例："毛利率为什么下降？"
      → ["毛利率 变动", "营业成本 营业收入 趋势", "成本上升"]
    """
    if not api_key or len(query) < 5:
        return []

    prompt_zh = f"""你是文档检索专家。把用户问题改写为2-3个关键词组合，用于在PDF文档中搜索最相关的段落。

规则:
- 只输出关键词组合，每行一个，不要编号和解释
- 提取问题中的核心实体和指标，拆分不同角度
- 每个关键词组合不超过15字

用户问题: {query}

检索关键词:"""

    prompt_en = f"""You are a document retrieval expert. Rewrite the user question into 2-3 keyword combinations for searching a PDF document.

Rules:
- Output only keyword combos, one per line, no numbering
- Extract core entities and metrics, split different angles
- Each combo under 10 words

Question: {query}

Keywords:"""

    prompt = prompt_zh if language == "zh" else prompt_en

    from src.llm.client import LLMClient, LLMConfig
    config = LLMConfig(api_key=api_key, model="qwen-turbo", temperature=0.0, max_tokens=150)
    client = LLMClient(config)
    try:
        resp = client.chat([{"role": "user", "content": prompt}])
    except Exception:
        return []

    queries = []
    for line in resp.strip().split("\n"):
        q = line.strip().lstrip("- ").lstrip("0123456789. ").strip()
        if q and len(q) > 2 and q != query:
            queries.append(q)

    return queries[:3] if queries else []


def rewrite_query_for_search(query: str, api_key: str, language: str = "zh") -> list[str]:
    """用 LLM 将用户问题改写为 2-3 个更精准的搜索查询（用于外部联网搜索）。"""
    prompt_zh = f"""你是一个搜索引擎查询优化器。把用户的问题改写为2-3个不同的搜索查询，每个查询从不同角度检索最相关的文档内容。

规则:
- 只输出查询，每行一个，不要编号，不要解释
- 查询要覆盖问题的不同方面（如：财务数据角度、风险因素角度、合规角度）
- 查询要具体，包含问题中的关键实体和指标

用户问题: {query}

搜索查询:"""

    prompt_en = f"""You are a search query optimizer. Rewrite the user's question into 2-3 different search queries, each targeting different aspects of the document.

Rules:
- Output only the queries, one per line, no numbering, no explanation
- Cover different aspects of the question (e.g., financial data, risk factors, compliance)
- Be specific, include key entities and metrics from the question

User question: {query}

Search queries:"""

    prompt = prompt_zh if language == "zh" else prompt_en

    from src.llm.client import LLMClient, LLMConfig
    config = LLMConfig(api_key=api_key, model="qwen-turbo", temperature=0.1, max_tokens=200)
    client = LLMClient(config)
    resp = client.chat([{"role": "user", "content": prompt}])

    queries = []
    for line in resp.strip().split("\n"):
        q = line.strip().lstrip("- ").lstrip("0123456789. ").strip()
        if q and len(q) > 5 and q != query:
            queries.append(q)

    return queries[:3] if queries else []
