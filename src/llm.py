from __future__ import annotations
from openai import OpenAI


class _DummyLLM:
    """Returns empty JSON — use for testing without an API key."""

    def chat(self, system: str, user: str) -> str:
        return "[]"


class DeepSeekLLM:
    """
    DeepSeek chat model via OpenAI-compatible API.
    Accessible from China without proxy.

    Usage:
        llm = DeepSeekLLM(api_key="sk-...")
        llm.chat(system="...", user="...")
    """

    def __init__(self, api_key: str, model: str = "deepseek-chat"):
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://api.deepseek.com",
        )
        self.model = model

    def chat(self, system: str, user: str) -> str:
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user",   "content": user},
            ],
            temperature=0,
        )
        return resp.choices[0].message.content


class OpenAILLM:
    def __init__(self, api_key: str, model: str = "gpt-4o-mini"):
        self.client = OpenAI(api_key=api_key)
        self.model  = model

    def chat(self, system: str, user: str) -> str:
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user",   "content": user},
            ],
            temperature=0,
        )
        return resp.choices[0].message.content

    def search_chat(self, query: str) -> str:
        """
        Use OpenAI Responses API with web_search_preview tool.
        Returns the grounded text response directly.
        """
        resp = self.client.responses.create(
            model=self.model,
            tools=[{"type": "web_search_preview"}],
            input=query,
        )
        return resp.output_text

class QwenLLM:
    """
    Alibaba Qwen via DashScope — accessible in China without proxy.

    Usage:
        llm = QwenLLM(api_key="sk-...")
        llm.chat(system="...", user="...")
    """

    def __init__(self, api_key: str, model: str = "qwen-max"):
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )
        self.model = model

    def chat(self, system: str, user: str) -> str:
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user",   "content": user},
            ],
            temperature=0,
        )
        return resp.choices[0].message.content


class GLMLLM:
    """
    Zhipu GLM via BigModel — accessible in China without proxy.

    Usage:
        llm = GLMLLM(api_key="...")
        llm.chat(system="...", user="...")
    """

    def __init__(self, api_key: str, model: str = "glm-4"):
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://open.bigmodel.cn/api/paas/v4/",
        )
        self.model = model

    def chat(self, system: str, user: str) -> str:
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user",   "content": user},
            ],
            temperature=0,
        )
        return resp.choices[0].message.content