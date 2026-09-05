"""云端 LLM 客户端（任意 OpenAI 兼容网关：SiliconFlow / tokenra(ox-alpha) 等）。

密钥从 settings（.env / 环境变量）读取，不落代码库。
聊天与嵌入可指向不同服务商：kind="chat" 只做聊天，kind="embed" 只做嵌入。
部分网关不提供 /models 端点，聊天实例会用最小 chat 请求探测可用性。
protocol 选厂商协议："openai"（默认，/chat/completions + Bearer）或
"anthropic"（Anthropic Messages API：/messages + x-api-key + anthropic-version
头，system 为顶层字段，max_tokens 必填；官方无嵌入端点，embed 一律本地哈希）。
协议仅作用于聊天通道；嵌入通道固定 OpenAI 兼容（见适配层 _embed_candidates）。
HTTPS 强制 IPv4：Cloudflare 系网关解析出 v6 优先，而本机代理 TUN 层的 v6 路由
间歇性黑洞，urllib 按序逐地址尝试会先烧满超时（实测探测耗时 20~145s 甚至失败）。
chat_stream 提供 SSE 流式聊天（thinking/answer 增量，_iter_sse 解析），认证头
与直连 opener 与 chat 同款；本地 LM Studio 客户端同包复用该解析助手。
"""
from __future__ import annotations

import http.client
import json
import socket
import urllib.error
import urllib.request
from collections.abc import Iterator

from memagent import settings
from memagent.adapters.llm.base import LLMClient
from memagent.core.vectors import hash_embed


class _IPv4HTTPSConnection(http.client.HTTPSConnection):
    """只连 AF_INET 地址，行为对齐 curl 的 happy-eyeballs 结果。"""

    def connect(self):
        infos = socket.getaddrinfo(self.host, self.port, socket.AF_INET, socket.SOCK_STREAM)
        af, socktype, proto, _canonname, sa = infos[0]
        sock = socket.socket(af, socktype, proto)
        sock.settimeout(self.timeout)
        try:
            sock.connect(sa)
            # NODELAY 必须在 TLS 包装前设置：wrap 后原 socket 在 Windows 上不可再操作
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        except OSError:
            sock.close()
            raise
        self.sock = self._context.wrap_socket(sock, server_hostname=self.host)


class _IPv4HTTPSHandler(urllib.request.HTTPSHandler):
    def https_open(self, req):
        return self.do_open(_IPv4HTTPSConnection, req, context=self._context)


def _iter_sse(line_iter, protocol: str = "openai") -> Iterator[dict]:
    """解析 SSE 响应行流，逐条产出 JSON 事件 dict（openai / anthropic 流共用）。

    行形如 "data: {...}"；空行与 event:/id:/retry:/注释行等非 data 行跳过；
    "data: [DONE]" 是 openai 语义的流终止哨兵（anthropic 流无此哨兵，遇中也只
    会被跳过——anthropic 以 message_stop 事件收尾，由连接关闭自然终止）。
    JSON 解析失败或非对象的行静默跳过（免费网关偶发脏行，不能让整条流炸掉）。
    urllib 响应按字节行迭代（文件对象语义），str 行同样接受（便于测试桩）。
    """
    for raw in line_iter:
        line = raw.decode("utf-8", errors="replace") if isinstance(raw, bytes) else raw
        line = line.strip()
        if not line.startswith("data:"):
            continue
        payload = line[len("data:"):].strip()
        if payload == "[DONE]":
            if protocol != "anthropic":
                break
            continue
        try:
            event = json.loads(payload)
        except ValueError:
            continue  # 脏行：静默跳过
        if isinstance(event, dict):
            yield event


def _openai_answer_delta(event: dict, answer: list, on_delta) -> None:
    """分派 openai 兼容流的一个 chunk：reasoning_content→thinking，content→answer。

    思考增量不进入 answer（对齐 chat() 只取 content 的语义）；None/空串跳过。
    """
    choices = event.get("choices") or []
    delta = choices[0].get("delta") if choices and isinstance(choices[0], dict) else None
    if not isinstance(delta, dict):
        return
    thinking = delta.get("reasoning_content")
    if thinking and on_delta is not None:
        on_delta("thinking", thinking)
    content = delta.get("content")
    if content:
        if on_delta is not None:
            on_delta("answer", content)
        answer.append(content)


class CloudClient(LLMClient):
    def __init__(self, url: str = "", api_key: str = "", model: str = "",
                 embed_model: str = "", timeout: int = 0, kind: str = "chat",
                 protocol: str = "openai"):
        self.url = (url or settings.CLOUD_URL).rstrip("/")
        self.api_key = api_key or settings.CLOUD_API_KEY
        self.model = model or settings.CLOUD_MODEL
        self.embed_model = embed_model or settings.CLOUD_EMBED_MODEL
        self.timeout = timeout or settings.CLOUD_TIMEOUT
        self.kind = kind  # "chat" | "embed"
        # "openai" | "anthropic"；未知值回落 openai（容错：拼错的 env 不至于全挂）
        self.protocol = protocol if protocol in ("openai", "anthropic") else "openai"
        # ProxyHandler({}) = 无视环境变量代理：本机 Windows 用户变量固化了 HTTPS_PROXY=127.0.0.1:7890，
        # 走该代理时 OpenRouter 的 POST 会被 Clash 节点掐断（GET 正常，探测通过但聊天全挂）；
        # 两家网关直连均验证可用（Clash TUN 负责国际路由），故云端请求一律直连
        self._opener = urllib.request.build_opener(urllib.request.ProxyHandler({}),
                                                   _IPv4HTTPSHandler)

    @property
    def configured(self) -> bool:
        return bool(self.api_key)

    def _request(self, path: str, payload: dict | None = None) -> dict | None:
        url = self.url + path
        data = json.dumps(payload).encode("utf-8") if payload is not None else None
        if self.protocol == "anthropic":
            # Anthropic Messages API 认证：x-api-key + anthropic-version 头（非 Bearer）
            headers = {"Content-Type": "application/json", "x-api-key": self.api_key,
                       "anthropic-version": "2023-06-01"}
        else:
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            }
        req = urllib.request.Request(url, data=data, headers=headers)
        try:
            with self._opener.open(req, timeout=self.timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            e.close()
            return None
        except Exception:
            return None

    def _open_stream(self, path: str, payload: dict):
        """打开流式 POST，返回响应对象（调用方负责 with 关闭）；失败返回 None。

        认证头与直连 opener 与 _request 同款，但不读全量 body——SSE 流由调用方
        经 _iter_sse 逐行消费。HTTPError/连接异常一律 None（对齐 chat() 语义）。
        """
        if self.protocol == "anthropic":
            headers = {"Content-Type": "application/json", "x-api-key": self.api_key,
                       "anthropic-version": "2023-06-01"}
        else:
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            }
        req = urllib.request.Request(self.url + path,
                                     data=json.dumps(payload).encode("utf-8"),
                                     headers=headers)
        try:
            return self._opener.open(req, timeout=self.timeout)
        except urllib.error.HTTPError as e:
            e.close()
            return None
        except Exception:
            return None

    def available(self) -> bool:
        if not self.configured:
            return False
        if self.protocol == "anthropic":
            # Anthropic 无嵌入端点：嵌入实例永不报告可用（embed 也绝不发网络）
            if self.kind == "embed":
                return False
            if self._request("/models") is not None:
                return True
            # /models 不可达时用最小 messages 请求探测（同 openai 路径的兜底思路）
            data = self._request("/messages", {
                "model": self.model,
                "messages": [{"role": "user", "content": "ping"}],
                "max_tokens": 1,
            })
            return bool(data and data.get("content"))
        if self._request("/models") is not None:
            return True
        if self.kind == "chat":
            # 网关无 /models 时用最小聊天请求探测可用性
            data = self._request("/chat/completions", {
                "model": self.model,
                "messages": [{"role": "user", "content": "ping"}],
                "max_tokens": 1,
            })
            return bool(data and data.get("choices"))
        # 嵌入实例：/models 探测失败也视为可用，embed() 自带哈希兜底
        return True

    def chat(self, prompt: str, system: str = "你是严谨的记忆整理助手。",
             temperature: float = 0.2) -> str | None:
        if self.protocol == "anthropic":
            data = self._request("/messages", {
                "model": self.model,
                "max_tokens": 2048,  # Anthropic Messages API 必填项
                "temperature": temperature,
                "system": system,  # 系统提示是顶层独立字段（Messages API 无 system role）
                "messages": [{"role": "user", "content": prompt}],
            })
            if not data:
                return None
            # content 是块数组：只拼接 type=="text" 块的 text（thinking/tool_use 等跳过）
            text = "".join(block.get("text", "") for block in data.get("content") or []
                           if isinstance(block, dict) and block.get("type") == "text")
            # 拼接为空（推理块耗尽 max_tokens 等）→ 按失败走降级链，
            # 对齐 openai 路径「content=null 按失败」的语义
            return text.strip() if text.strip() else None
        data = self._request("/chat/completions", {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            "temperature": temperature,
            "max_tokens": 2048,
        })
        if not data or not data.get("choices"):
            return None
        content = data["choices"][0]["message"].get("content")
        # 推理型模型思考耗尽 max_tokens 时 content 为 null（只剩 reasoning），按失败走降级链
        return content.strip() if isinstance(content, str) and content.strip() else None

    def chat_stream(self, prompt: str, system: str = "你是严谨的记忆整理助手。",
                    temperature: float = 0.2, on_delta=None) -> str | None:
        """流式聊天：SSE 增量经 on_delta(kind, text) 实时推送，返回拼接后的完整 answer。

        kind 三种："thinking"（模型自发产生的推理增量：openai 网关的
        delta.reasoning_content / anthropic 的 thinking_delta）、"answer"（正式
        回答增量）、"reset"（text 恒为 ""，由适配层在 validate 失败换备用模型
        重流前发出，客户端层不产生）。返回值与 chat() 同语义：HTTP/连接失败、
        中途断流、answer 全空（含只有思考没有内容）→ None。
        注意：on_delta 是调用方的 UI 回调，其异常故意不捕获——崩了让它崩。
        anthropic 协议只解析模型自发产生的思考增量，不主动加 thinking 开关参数
        （extended thinking 会引入 max_tokens>budget、temperature 限制等约束）。
        """
        if self.protocol == "anthropic":
            path, payload = "/messages", {
                "model": self.model,
                "max_tokens": 2048,  # Anthropic Messages API 必填项
                "temperature": temperature,
                "system": system,  # 系统提示是顶层独立字段（Messages API 无 system role）
                "messages": [{"role": "user", "content": prompt}],
                "stream": True,
            }
        else:
            path, payload = "/chat/completions", {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ],
                "temperature": temperature,
                "max_tokens": 2048,
                "stream": True,
            }
        resp = self._open_stream(path, payload)
        if resp is None:
            return None
        answer: list[str] = []
        with resp:
            events = _iter_sse(resp, self.protocol)
            while True:
                try:
                    event = next(events)
                except StopIteration:
                    break
                except Exception:
                    # 流中途断开（连接重置/读超时等）：对齐 chat() 的失败语义；
                    # on_delta 的调用在 next() 之外，回调异常不受此捕获影响
                    return None
                if self.protocol == "anthropic":
                    if event.get("type") != "content_block_delta":
                        continue  # message_start / ping / message_stop 等事件跳过
                    delta = event.get("delta") or {}
                    dtype = delta.get("type")
                    if dtype == "thinking_delta":
                        text = delta.get("thinking")
                        if text and on_delta is not None:
                            on_delta("thinking", text)
                    elif dtype == "text_delta":
                        text = delta.get("text")
                        if text:
                            if on_delta is not None:
                                on_delta("answer", text)
                            answer.append(text)
                else:
                    _openai_answer_delta(event, answer, on_delta)
        text = "".join(answer).strip()
        return text if text else None

    def embed(self, text: str) -> list[float]:
        if self.protocol == "anthropic":
            # Anthropic 官方无嵌入端点：绝不发网络，直接本地哈希兜底
            return hash_embed(text, settings.EMBED_FALLBACK_DIM)
        data = self._request("/embeddings", {
            "model": self.embed_model, "input": [text]})
        if data and data.get("data"):
            return data["data"][0]["embedding"]
        return hash_embed(text, settings.EMBED_FALLBACK_DIM)
