"""
LLM 客户端 — 统一封装对 DashScope (Qwen) 和 OpenAI 兼容接口的调用。

用户可在 Streamlit 侧边栏输入自己的 API Key，不读取任何硬编码密钥。
"""

import os
from dataclasses import dataclass, field
from typing import Optional

import dashscope
from openai import OpenAI


@dataclass
class LLMConfig:
    """LLM 配置，所有敏感信息由用户在前端输入。"""

    api_key: str = ""
    api_base: str = ""  # 为空则使用 DashScope 默认端点
    model: str = "qwen-plus"
    temperature: float = 0.3
    max_tokens: int = 4096
    enable_search: bool = False

    @property
    def is_dashscope(self) -> bool:
        return not self.api_base


class LLMClient:
    """统一的 LLM 调用客户端。

    支持两种模式:
    - DashScope 原生协议（默认，用于 Qwen 系列）
    - OpenAI 兼容协议（用于任何兼容 /v1/chat/completions 的服务）
    """

    def __init__(self, config: LLMConfig):
        self.config = config
        self._openai_client: Optional[OpenAI] = None
        # 设置全局 api_key + 国内备用端点
        if config.is_dashscope and config.api_key:
            import dashscope
            clean_key = config.api_key.strip().encode('ascii', errors='ignore').decode('ascii')
            dashscope.api_key = clean_key
            # 尝试使用国内直连端点，避免 VPN DNS 解析问题
            try:
                dashscope.base_http_api_url = 'https://dashscope.aliyuncs.com/api/v1'
            except Exception:
                pass

    # ------------------------------------------------------------------
    # 公开接口
    # ------------------------------------------------------------------

    def chat(self, messages: list[dict], **kwargs) -> str:
        """发送消息并返回模型回复文本。自动重试，DNS 错误等更长时间。"""
        last_error = None
        for attempt in range(3):
            try:
                resp = self._call_api(messages, **kwargs)
                return self._extract_content(resp)
            except Exception as e:
                last_error = e
                if attempt < 2:
                    import time
                    # DNS/网络错误等更久，其他错误等短一些
                    err_msg = str(e).lower()
                    wait = 3.0 * (attempt + 1) if any(kw in err_msg for kw in
                        ['dns', 'resolve', 'connection', 'timeout', 'refused']) else 1.5 * (attempt + 1)
                    time.sleep(wait)
        # 给出更明确的错误信息
        err_type = "网络连接" if any(kw in str(last_error).lower() for kw in
            ['dns', 'resolve', 'connection', 'timeout']) else "API调用"
        raise RuntimeError(f"{err_type}失败（重试3次后）。请检查网络连接。如果用VPN，试试关掉VPN后重试。错误: {last_error}")

    def chat_stream(self, messages: list[dict], on_token=None, **kwargs) -> str:
        """流式调用 LLM，每收到一个 token 调用 on_token(token)，返回完整文本。"""
        model = kwargs.get("model", self.config.model)
        temperature = kwargs.get("temperature", self.config.temperature)
        max_tokens = kwargs.get("max_tokens", self.config.max_tokens)

        full_text = []
        if self.config.is_dashscope:
            for token in self._stream_dashscope(messages, model, temperature, max_tokens):
                full_text.append(token)
                if on_token:
                    on_token(token)
        else:
            for token in self._stream_openai(messages, model, temperature, max_tokens):
                full_text.append(token)
                if on_token:
                    on_token(token)
        return "".join(full_text)

    def stream_chat(self, messages: list[dict], **kwargs):
        """流式调用，yield 每个增量文本片段。"""
        model = kwargs.get("model", self.config.model)
        temperature = kwargs.get("temperature", self.config.temperature)
        max_tokens = kwargs.get("max_tokens", self.config.max_tokens)
        enable_search = kwargs.get("enable_search", self.config.enable_search)

        if self.config.is_dashscope:
            for chunk in self._stream_dashscope(messages, model, temperature,
                                                max_tokens, enable_search):
                yield chunk
        else:
            for chunk in self._stream_openai(messages, model, temperature, max_tokens):
                yield chunk

    # ------------------------------------------------------------------
    # 内部实现
    # ------------------------------------------------------------------

    def _call_api(self, messages: list[dict], **kwargs) -> object:
        model = kwargs.get("model", self.config.model)
        temperature = kwargs.get("temperature", self.config.temperature)
        max_tokens = kwargs.get("max_tokens", self.config.max_tokens)
        enable_search = kwargs.get("enable_search", self.config.enable_search)

        if self.config.is_dashscope:
            return self._call_dashscope(messages, model, temperature, max_tokens,
                                        enable_search)
        return self._call_openai(messages, model, temperature, max_tokens)

    # ---- DashScope ----

    def _call_dashscope(self, messages: list[dict], model: str, temperature: float,
                        max_tokens: int, enable_search: bool = False) -> object:
        kwargs = dict(
            model=model, messages=messages,
            temperature=temperature, max_tokens=max_tokens,
            result_format="message",
        )
        if enable_search:
            kwargs["enable_search"] = True
        resp = dashscope.Generation.call(**kwargs)
        if resp.status_code != 200:
            raise RuntimeError(
                f"DashScope API 错误 (code={resp.status_code}): {resp.message}"
            )
        return resp

    def _stream_dashscope(self, messages: list[dict], model: str, temperature: float,
                          max_tokens: int, enable_search: bool = False):
        kwargs = dict(
            model=model, messages=messages,
            temperature=temperature, max_tokens=max_tokens,
            result_format="message", stream=True, incremental_output=True,
        )
        if enable_search:
            kwargs["enable_search"] = True
        resp = dashscope.Generation.call(**kwargs)
        for chunk in resp:
            if chunk.status_code == 200:
                try:
                    text = chunk.output.choices[0].message.content
                    if isinstance(text, str):
                        yield text
                except (AttributeError, IndexError):
                    continue

    # ---- OpenAI-compatible ----

    def _get_openai_client(self) -> OpenAI:
        if self._openai_client is None:
            self._openai_client = OpenAI(
                api_key=self.config.api_key,
                base_url=self.config.api_base,
            )
        return self._openai_client

    def _call_openai(self, messages: list[dict], model: str, temperature: float,
                     max_tokens: int) -> object:
        client = self._get_openai_client()
        return client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    def _stream_openai(self, messages: list[dict], model: str, temperature: float,
                       max_tokens: int):
        client = self._get_openai_client()
        stream = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
        )
        for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    # ---- 工具 ----

    @staticmethod
    def _extract_content(resp: object) -> str:
        # DashScope
        if hasattr(resp, "output"):
            try:
                return resp.output.choices[0].message.content
            except (AttributeError, IndexError):
                pass
        # OpenAI
        if hasattr(resp, "choices"):
            try:
                return resp.choices[0].message.content
            except (AttributeError, IndexError):
                pass
        raise RuntimeError("无法从 API 响应中提取内容")
