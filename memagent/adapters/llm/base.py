"""LLM 适配抽象：任何聊天/嵌入后端实现此接口即可接入。"""
from __future__ import annotations

from abc import ABC, abstractmethod


class LLMClient(ABC):
    """最小 LLM 合同：chat 失败返回 None（调用方自行降级），embed 必须有本地兜底。"""

    @abstractmethod
    def available(self) -> bool:
        """后端是否可用。"""

    @abstractmethod
    def chat(self, prompt: str, system: str = "你是严谨的记忆整理助手。",
             temperature: float = 0.2) -> str | None:
        """返回文本；失败返回 None。"""

    @abstractmethod
    def embed(self, text: str) -> list[float]:
        """返回嵌入向量；后端不可用时必须返回本地兜底嵌入。"""

    def chat_stream(self, prompt: str, system: str = "你是严谨的记忆整理助手。",
                    temperature: float = 0.2, on_delta=None) -> str | None:
        """流式聊天默认实现（非抽象）：不覆写的后端获得「一次性增量」降级。

        拿到 chat() 的完整结果后作为单个 "answer" 增量发出——流式消费方
        无需特判非流式后端；chat 失败返回 None 且不发出任何增量。
        实现真流式的后端（CloudClient / LocalStudioClient）覆写本方法。
        """
        out = self.chat(prompt, system=system, temperature=temperature)
        if out and on_delta:
            on_delta("answer", out)
        return out
