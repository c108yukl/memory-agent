"""本地 LM Studio 客户端（OpenAI 兼容端点）。

无模型时由适配层选择器自动降级到云端或本地哈希嵌入。
E8：本地请求一律绕过环境代理（ProxyHandler({})）——本机 clash 的 HTTPS_PROXY
会把 localhost 请求劫去代理层空转，LM Studio 没开时「瞬时拒绝连接」被拖成
数秒超时（实测 4s+），绿色通道的本地快车道探测每进程都要付一次这个代价。
"""
from __future__ import annotations

import json
import socket
import urllib.error
import urllib.request
from urllib.parse import urlsplit

from memagent import settings
from memagent.adapters.llm.base import LLMClient
# 同包内部复用云端适配的 SSE 解析助手：LM Studio 的流式响应与 OpenAI 兼容网关同格式
from memagent.adapters.llm.cloud_client import _iter_sse, _openai_answer_delta
from memagent.core.vectors import hash_embed

# 类级共享 opener：无代理、直连 localhost
_NO_PROXY_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))

# 探测预检（socket 层）超时：只判断「端口上有没有东西在听」。LM Studio 在听时
# accept 是毫秒级的，0.3s 绰绰有余。不能放长：本机实测「端口没开」不是拒绝连接
# 而是静默丢包（Windows 防火墙行为），一次 connect 尝试要烧 ~2s，而 localhost
# 会 ::1 与 127.0.0.1 各试一轮 = 4s/次——这正是 E8 修掉代理后残留的探测开销。
# 误判代价可控：LM Studio 真在听却 0.3s 内没 accept 的场景不存在（除非进程假死，
# 那种状态降级到云端反而更对）；预检通过后仍走完整 HTTP /models 探测兜底。
_PROBE_CONNECT_TIMEOUT = 0.3


class LocalStudioClient(LLMClient):
    def _post(self, path: str, payload: dict) -> dict | None:
        url = settings.LM_STUDIO_URL.rstrip("/") + path
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        try:
            with _NO_PROXY_OPENER.open(req, timeout=settings.LM_STUDIO_TIMEOUT) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            e.close()
            return None
        except Exception:
            return None

    def available(self) -> bool:
        # 毫秒级端口预检先行：LM Studio 没开时不发完整 HTTP 探测——后者在本机
        # 实测 ~4s/次（丢包式「不可达」+ localhost 双栈各试一轮），而探测每进程
        # 要在 chat/maint/embed/local 四个独立槽各付一次，冷启动十几秒全是这笔。
        # 预检只连 127.0.0.1（跳过 localhost 的 ::1 第一轮）：LM Studio 默认绑
        # 127.0.0.1，且在听时 accept 毫秒级、不在听时预算到点即弃。预检通过后
        # 仍走 HTTP /models 探测（真实验证服务能答）。
        parts = urlsplit(settings.LM_STUDIO_URL)
        host = parts.hostname or "localhost"
        port = parts.port or (443 if parts.scheme == "https" else 80)
        probe_host = "127.0.0.1" if host in ("localhost",) else host
        try:
            with socket.create_connection((probe_host, port),
                                          timeout=_PROBE_CONNECT_TIMEOUT):
                pass
        except OSError:
            return False
        return self._post("/models", {"model": ""}) is not None

    def chat(self, prompt: str, system: str = "你是严谨的记忆整理助手。",
             temperature: float = 0.2) -> str | None:
        data = self._post("/chat/completions", {
            "model": settings.LLM_MODEL,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            "temperature": temperature,
        })
        if not data or not data.get("choices"):
            return None
        return data["choices"][0]["message"]["content"].strip()

    def chat_stream(self, prompt: str, system: str = "你是严谨的记忆整理助手。",
                    temperature: float = 0.2, on_delta=None) -> str | None:
        """LM Studio 流式聊天：OpenAI 兼容 SSE，增量经 on_delta(kind, text) 推送。

        kind："thinking"（模型自发推理增量）/ "answer"（回答增量）；"reset" 由
        适配层发出，客户端层不产生。同包内部复用云端的 SSE 解析助手（_iter_sse +
        _openai_answer_delta）——LM Studio 流式响应与 OpenAI 兼容网关同格式。
        on_delta 异常不捕获（调用方 UI，崩了让它崩）；HTTP/连接失败、中途断流、
        answer 全空 → None，对齐 chat() 语义。
        """
        req = urllib.request.Request(
            settings.LM_STUDIO_URL.rstrip("/") + "/chat/completions",
            data=json.dumps({
                "model": settings.LLM_MODEL,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ],
                "temperature": temperature,
                "stream": True,
            }).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        try:
            resp = _NO_PROXY_OPENER.open(req, timeout=settings.LM_STUDIO_TIMEOUT)
        except urllib.error.HTTPError as e:
            e.close()
            return None
        except Exception:
            return None
        answer: list[str] = []
        with resp:
            events = _iter_sse(resp, "openai")
            while True:
                try:
                    event = next(events)
                except StopIteration:
                    break
                except Exception:
                    return None  # 流中途断开：对齐 chat() 的失败语义
                _openai_answer_delta(event, answer, on_delta)
        text = "".join(answer).strip()
        return text if text else None

    def embed(self, text: str) -> list[float]:
        data = self._post("/embeddings", {"model": settings.LLM_MODEL, "input": [text]})
        if data and data.get("data"):
            return data["data"][0]["embedding"]
        return hash_embed(text, settings.EMBED_FALLBACK_DIM)
