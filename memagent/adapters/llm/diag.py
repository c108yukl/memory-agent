"""通道连通性诊断（CLI `python main.py test` 与 TUI 设置屏 t 键共用）。

五项检查：本地 LM Studio / 主聊天 /models / 主聊天对话 / 维护通道对话 / 云端嵌入。
全部复用 CloudClient/LocalStudioClient 实例的 _opener 直连（无代理、强制 IPv4），
不另写 HTTP；模块级 _raw 是全项目唯一允许裸碰 opener 的「诊断特权」——
适配层 _request 吞掉 HTTPError 拿不到状态码，诊断必须区分 200/400/401/403 并读取错误体。

密钥只进请求头：detail/异常消息一律经 _scrub 清洗（含网关错误体回显密钥的最坏情况），
展示配置时 key 用 _mask 掩码（k[:5]+"***"+k[-4:]，过短只给 *** 防前后段重叠泄露）。
所有请求超时 20s；单项异常一律转结果项，run_diag 永不抛。

结果项 shape：{"name", "ok", "level", "detail", "latency_ms"}
level: "info" 完全正常 | "warn" 降级可用（本地没开/维护失败/哈希兜底，均不算错）
       | "error" 该通道失败（仅主聊天对话会令 CLI 退出码为 1）。
"""
from __future__ import annotations

import json
import time
import urllib.error
import urllib.request

from memagent import settings
from memagent.adapters.llm.cloud_client import CloudClient
from memagent.adapters.llm.local_client import LocalStudioClient

_DIAG_TIMEOUT = 20  # 秒：所有诊断请求统一超时（适配层各通道超时不同，诊断要对齐体验）
_PING_PROMPT = "ping（只回复：pong）"


def _mask(key: str) -> str:
    """key 掩码：k[:5]+"***"+k[-4:]；空=未配置，过短（前后段会重叠）只给 ***。"""
    if not key:
        return "(未配置)"
    if len(key) < 10:
        return "***"
    return key[:5] + "***" + key[-4:]


def _scrub(text: object, *secrets: str) -> str:
    """输出清洗：压平换行（诊断行是单行渲染）并抹掉任何密钥串。"""
    out = " ".join(str(text).split())
    for s in secrets:
        if s:
            out = out.replace(s, "***")
    return out


def _item(name: str, ok: bool, level: str, detail: str, latency_ms: float) -> dict:
    return {"name": name, "ok": ok, "level": level, "detail": detail,
            "latency_ms": round(latency_ms, 1)}


def _raw(client, path: str, payload: dict | None,
         timeout: int = _DIAG_TIMEOUT) -> tuple[int | None, float, str]:
    """诊断特权：用 client._opener 裸发一个请求 -> (状态码|None, 耗时ms, 响应体)。

    认证头与 CloudClient._request 完全同款（openai=Bearer / anthropic=x-api-key）；
    HTTPError 读出状态码与错误体；其余异常（超时/拒连/黑洞）统一 (None, 耗时, "")。
    仅 diag.py 允许调用（其余代码必须走 _request/chat/embed，保持熔断语义）。
    """
    url = client.url + path
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    if getattr(client, "protocol", "openai") == "anthropic":
        headers = {"Content-Type": "application/json", "x-api-key": client.api_key,
                   "anthropic-version": "2023-06-01"}
    else:
        headers = {"Content-Type": "application/json",
                   "Authorization": f"Bearer {client.api_key}"}
    req = urllib.request.Request(url, data=data, headers=headers)
    t0 = time.monotonic()
    try:
        with client._opener.open(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8", "replace")
            return getattr(resp, "status", None), (time.monotonic() - t0) * 1000.0, body
    except urllib.error.HTTPError as e:
        try:
            body = e.read().decode("utf-8", "replace")
        except Exception:
            body = ""
        e.close()
        return e.code, (time.monotonic() - t0) * 1000.0, body
    except Exception:
        return None, (time.monotonic() - t0) * 1000.0, ""


def _host(client) -> str:
    return str(client.url).split("//")[-1]


def _chat_path_payload(protocol: str, model: str) -> tuple[str, dict]:
    """最小对话请求（max_tokens=256）；anthropic 走 /messages（system 为顶层字段，此处不需要）。"""
    if protocol == "anthropic":
        return "/messages", {"model": model, "max_tokens": 256,
                             "messages": [{"role": "user", "content": _PING_PROMPT}]}
    return "/chat/completions", {"model": model, "max_tokens": 256,
                                 "messages": [{"role": "user", "content": _PING_PROMPT}]}


def _chat_content(body: str) -> str:
    """从 200 响应体提取文本：openai=choices[0].message.content；anthropic=text 块拼接。"""
    try:
        data = json.loads(body)
    except Exception:
        return ""
    if not isinstance(data, dict):
        return ""
    choices = data.get("choices")
    if isinstance(choices, list) and choices:
        content = ((choices[0] or {}).get("message") or {}).get("content")
        return content if isinstance(content, str) else ""
    blocks = data.get("content")
    if isinstance(blocks, list):
        return "".join(b.get("text", "") for b in blocks
                       if isinstance(b, dict) and b.get("type") == "text")
    return ""


def _embed_dim(body: str) -> int | None:
    try:
        emb = json.loads(body)["data"][0]["embedding"]
        return len(emb) if isinstance(emb, list) else None
    except Exception:
        return None


def _check_local() -> dict:
    t0 = time.monotonic()
    try:
        ok = LocalStudioClient().available()
    except Exception as e:
        ms = (time.monotonic() - t0) * 1000.0
        return _item("本地 LM Studio", False, "warn",
                     f"检查异常: {_scrub(e)[:120]} (没开不算错)", ms)
    ms = (time.monotonic() - t0) * 1000.0
    if ok:
        return _item("本地 LM Studio", True, "info",
                     f"可用 ({settings.LM_STUDIO_URL})", ms)
    return _item("本地 LM Studio", False, "warn", "未开启或不可达 (不算错, 自动走云端/离线)", ms)


def _check_chat_models() -> dict:
    client = CloudClient(kind="chat", protocol=settings.CLOUD_PROTOCOL)
    if not client.api_key:
        return _item("主聊天 /models", False, "warn", "未配置 CLOUD_API_KEY (跳过请求)", 0.0)
    status, ms, body = _raw(client, "/models", None)
    snippet = _scrub(body, client.api_key)[:120]
    if status == 200:
        return _item("主聊天 /models", True, "info",
                     f"{_host(client)} | 模型 {client.model} | key {_mask(client.api_key)}", ms)
    if status is None:
        return _item("主聊天 /models", False, "warn", f"网络失败无响应 ({_host(client)})", ms)
    return _item("主聊天 /models", False, "warn",
                 f"HTTP {status}: {snippet} (网关可能无 /models, 聊天或仍可用)", ms)


def _talk_check(name: str, client, degraded: bool) -> dict:
    """对话通道公共检查：degraded=True（维护通道）失败降为 warn，主聊天为 error。"""
    if not client.api_key:
        return _item(name, False, "warn" if degraded else "error",
                     "未配置 API Key (跳过请求)", 0.0)
    path, payload = _chat_path_payload(client.protocol, client.model)
    status, ms, body = _raw(client, path, payload)
    snippet = _scrub(body, client.api_key)[:200]
    if status == 200:
        content = _chat_content(body).strip()
        if content:
            return _item(name, True, "info",
                         f"{client.model} 回复样例: {_scrub(content, client.api_key)[:60]}", ms)
        lvl = "warn" if degraded else "error"
        return _item(name, False, lvl, "HTTP 200 但无文本内容 (content 为空/思考耗尽)", ms)
    if status is None:
        return _item(name, False, "warn" if degraded else "error",
                     f"网络失败无响应 ({_host(client)})" + (" (维护可降级)" if degraded else ""), ms)
    suffix = " (维护可降级)" if degraded else ""
    return _item(name, False, "warn" if degraded else "error",
                 f"HTTP {status}: {snippet}{suffix}", ms)


def _check_chat() -> dict:
    client = CloudClient(kind="chat", protocol=settings.CLOUD_PROTOCOL)
    return _talk_check("主聊天对话", client, degraded=False)


def _check_maint() -> dict:
    client = CloudClient(url=settings.CLOUD_MAINT_URL, api_key=settings.CLOUD_MAINT_API_KEY,
                         model=settings.CLOUD_MAINT_MODEL, kind="chat",
                         protocol=settings.CLOUD_MAINT_PROTOCOL)
    return _talk_check("维护通道对话", client, degraded=True)


def _check_embed() -> dict:
    client = CloudClient(url=settings.CLOUD_EMBED_URL, api_key=settings.CLOUD_EMBED_API_KEY,
                         model=settings.CLOUD_EMBED_MODEL, kind="embed",
                         timeout=settings.CLOUD_EMBED_TIMEOUT)
    if not client.api_key:
        return _item("云端嵌入", False, "warn", "未配置 CLOUD_EMBED_API_KEY (跳过请求, 走哈希兜底)", 0.0)
    status, ms, body = _raw(client, "/embeddings",
                            {"model": client.embed_model, "input": [_PING_PROMPT]})
    snippet = _scrub(body, client.api_key)[:200]
    if status == 200:
        dim = _embed_dim(body)
        if dim and dim != settings.EMBED_FALLBACK_DIM:
            return _item("云端嵌入", True, "info", f"{client.embed_model} 维度 {dim}", ms)
        if dim == settings.EMBED_FALLBACK_DIM:
            return _item("云端嵌入", False, "warn",
                         f"维度 {dim}：实际走了哈希兜底 (与 {settings.EMBED_FALLBACK_DIM} 维本地哈希不可区分)", ms)
        return _item("云端嵌入", False, "warn", "HTTP 200 但响应无嵌入向量", ms)
    if status is None:
        return _item("云端嵌入", False, "warn", f"网络失败无响应 ({_host(client)}, 自动哈希兜底)", ms)
    return _item("云端嵌入", False, "warn", f"HTTP {status}: {snippet} (自动哈希兜底)", ms)


# (检查名, 检查函数)；检查名同时用于单项异常时的结果项命名
_CHECKS = (("本地 LM Studio", _check_local),
           ("主聊天 /models", _check_chat_models),
           ("主聊天对话", _check_chat),
           ("维护通道对话", _check_maint),
           ("云端嵌入", _check_embed))


def run_diag() -> list[dict]:
    """五项连通性检查，按固定顺序返回结果项；单项崩溃转结果项，本函数永不抛。"""
    out: list[dict] = []
    for name, fn in _CHECKS:
        try:
            out.append(fn())
        except Exception as e:  # 兜底：诊断自身异常也必须是可打印的一行（密钥已清洗）
            out.append(_item(name, False, "error", f"检查异常: {_scrub(e)[:150]}", 0.0))
    return out
