"""LLM 适配层统一入口。

聊天/梳理/嵌入三通道分别独立选择，优先级各自为：本地 LM Studio -> 云端 -> 降级。
主通道（chat）服务实时管线：门控打分、编码抽取、冲突消解；
梳理通道（maintenance）服务离线巩固：聚类摘要、语义蒸馏，可指向低配模型省钱；
聊天无任何可用客户端时返回 None（调用方自行降级到规则）；
嵌入无可用客户端时返回本地哈希嵌入（维度固定，保证总能写入）。
聊天/梳理通道可选厂商协议（settings.CLOUD_PROTOCOL / CLOUD_MAINT_PROTOCOL：
"openai" | "anthropic"）；嵌入通道固定 OpenAI 兼容（Anthropic 无嵌入端点）。

模块级 chat/chat_stream/maintenance_chat/embed/llm_available 是全项目唯一的
LLM 访问入口，测试通过 mock 这几个函数即可获得确定性输出。
可用性探测结果缓存 _PROBE_TTL 秒。
"""
from __future__ import annotations

import time

from memagent import settings
from memagent.adapters.llm.base import LLMClient
from memagent.adapters.llm.cloud_client import CloudClient
from memagent.adapters.llm.local_client import LocalStudioClient
from memagent.core.vectors import hash_embed

_PROBE_TTL = 300.0  # 秒
CHAT_BREAKER_N = 2  # 云端 chat 连续失败 N 次熔断本槽（E8，对称 embed 熔断）
_cache: dict[str, dict] = {"chat": {"client": None, "probed": False, "at": 0.0},
                           "maint": {"client": None, "probed": False, "at": 0.0},
                           "embed": {"client": None, "probed": False, "at": 0.0},
                           "local": {"client": None, "probed": False, "at": 0.0}}
_chat_failures = {"chat": 0, "maint": 0}  # 云端 chat 连续失败计数（成功即清零）


def _chat_candidates() -> list[LLMClient]:
    clients: list[LLMClient] = [LocalStudioClient()]
    cloud = CloudClient(kind="chat", protocol=settings.CLOUD_PROTOCOL)
    if cloud.configured:
        clients.append(cloud)
    return clients


def _maint_candidates() -> list[LLMClient]:
    clients: list[LLMClient] = [LocalStudioClient()]
    cloud = CloudClient(url=settings.CLOUD_MAINT_URL, api_key=settings.CLOUD_MAINT_API_KEY,
                        model=settings.CLOUD_MAINT_MODEL, kind="chat",
                        protocol=settings.CLOUD_MAINT_PROTOCOL)
    if cloud.configured:
        clients.append(cloud)
    return clients


def _embed_candidates() -> list[LLMClient]:
    clients: list[LLMClient] = [LocalStudioClient()]
    cloud = CloudClient(url=settings.CLOUD_EMBED_URL, api_key=settings.CLOUD_EMBED_API_KEY,
                        model=settings.CLOUD_EMBED_MODEL, kind="embed",
                        timeout=settings.CLOUD_EMBED_TIMEOUT)
    if cloud.configured:
        clients.append(cloud)
    return clients


def _probe(kind: str) -> LLMClient | None:
    candidates = {"chat": _chat_candidates, "maint": _maint_candidates,
                  "embed": _embed_candidates}[kind]()
    for client in candidates:
        try:
            if client.available():
                return client
        except Exception:
            continue
    return None


def _pick(kind: str) -> LLMClient | None:
    slot = _cache[kind]
    now = time.monotonic()
    if not slot["probed"] or now - slot["at"] > _PROBE_TTL:
        slot["client"] = _probe(kind)
        slot["at"] = now
        slot["probed"] = True
    return slot["client"]


def llm_available() -> bool:
    return _pick("chat") is not None


def maintenance_available() -> bool:
    return _pick("maint") is not None


def active_provider() -> str:
    chat_name = _client_name(_pick("chat"))
    maint_name = _client_name(_pick("maint"))
    embed_name = _client_name(_pick("embed"))
    return f"chat={chat_name} maint={maint_name} embed={embed_name}"


def _client_name(client: LLMClient | None) -> str:
    if isinstance(client, CloudClient):
        host = client.url.split('//')[-1].split('/')[0]
        # anthropic 协议显式标注，避免 status 里误读为 OpenAI 兼容网关
        prefix = "cloud-anthropic" if client.protocol == "anthropic" else "cloud"
        return f"{prefix}({host}:{client.model})"
    if isinstance(client, LocalStudioClient):
        return "local(LM-Studio)"
    return "fallback(hash-embed)" if client is None else "unknown"


def _chat_via(kind: str, cloud_url: str, cloud_key: str, primary_model: str,
              prompt: str, system: str, temperature: float, validate=None,
              protocol: str = "openai") -> str | None:
    """调用聊天模型；可选 validate(结果)->bool 做格式校验。

    免费网关存在「模型级确定性输出损坏」（如 Qwen 免费节点会篡改标点与数字，
    且同请求稳定复现、加扰动也绕不开）。校验失败时依次尝试 CLOUD_FALLBACK_MODELS
    中的备用模型；全部失败则把首个结果交还调用方，由调用方按各自管线降级。
    protocol 须传给备用模型的 CloudClient：否则 anthropic 主模型校验失败后，
    备用模型会用 openai 协议去打同一家网关（通道协议分裂，必然 404/401）。

    E8 chat 熔断（对称 V1.5.2 的 embed 熔断）：云端网络失败（返回 None）连续
    CHAT_BREAKER_N 次即置空本槽直到探测 TTL 过期——网络黑洞期不再逐次撞网干等
    （在线实测：探测 60s + 请求 60s 串行超时 = CLI 挂死 120s 的根因）。本地
    LM Studio 失败不计数（探测缓存已兜底，本地超时也仅 30s）。
    """
    client = _pick(kind)
    if client is None:
        return None
    first = client.chat(prompt, system=system, temperature=temperature)
    if isinstance(client, CloudClient):
        if first is None:
            _chat_failures[kind] += 1
            if _chat_failures[kind] >= CHAT_BREAKER_N:
                _cache[kind].update(client=None, at=time.monotonic(), probed=True)
                _chat_failures[kind] = 0
        else:
            _chat_failures[kind] = 0
    if first is None or validate is None or validate(first):
        return first
    for model in settings.CLOUD_FALLBACK_MODELS.split(","):
        model = model.strip()
        if not model or model == primary_model:
            continue
        fallback = CloudClient(url=cloud_url, api_key=cloud_key, model=model, kind="chat",
                               protocol=protocol)
        if not fallback.configured:
            continue
        out = fallback.chat(prompt, system=system, temperature=temperature)
        if out is not None and validate(out):
            return out
    return first


def chat(prompt: str, system: str = "你是严谨的记忆整理助手。", temperature: float = 0.2,
         validate=None) -> str | None:
    """主通道：实时管线（门控/编码/冲突）使用。"""
    return _chat_via("chat", settings.CLOUD_URL, settings.CLOUD_API_KEY,
                     settings.CLOUD_MODEL, prompt, system, temperature, validate,
                     protocol=settings.CLOUD_PROTOCOL)


def chat_stream(prompt: str, system: str = "你是严谨的记忆整理助手。", temperature: float = 0.2,
                validate=None, on_delta=None) -> str | None:
    """主通道流式版本：增量经 on_delta(kind, text) 实时推送，返回拼接后的完整 answer。

    kind 三种："thinking"（模型自发产生的思考增量）、"answer"（正式回答增量）、
    "reset"（text 恒为 ""：validate 失败换备用模型重流前发出，UI 必须清空已渲染
    的部分输出再收新的）。熔断与 fallback 语义与 chat()（_chat_via）逐行同款：
    云端 None 计入 _chat_failures["chat"]（≥ CHAT_BREAKER_N 置空槽清零计数）、
    成功清零；validate 校验失败先发 reset 再逐个重流 CLOUD_FALLBACK_MODELS
    （备用 CloudClient 必须携带 protocol=settings.CLOUD_PROTOCOL，避免通道协议
    分裂），全败返回首个结果。on_delta 的异常不捕获（调用方 UI，崩了让它崩）。
    """
    client = _pick("chat")
    if client is None:
        return None
    first = client.chat_stream(prompt, system=system, temperature=temperature,
                               on_delta=on_delta)
    if isinstance(client, CloudClient):
        if first is None:
            _chat_failures["chat"] += 1
            if _chat_failures["chat"] >= CHAT_BREAKER_N:
                _cache["chat"].update(client=None, at=time.monotonic(), probed=True)
                _chat_failures["chat"] = 0
        else:
            _chat_failures["chat"] = 0
    if first is None or validate is None or validate(first):
        return first
    if on_delta is not None:
        on_delta("reset", "")
    for model in settings.CLOUD_FALLBACK_MODELS.split(","):
        model = model.strip()
        if not model or model == settings.CLOUD_MODEL:
            continue
        fallback = CloudClient(url=settings.CLOUD_URL, api_key=settings.CLOUD_API_KEY,
                               model=model, kind="chat", protocol=settings.CLOUD_PROTOCOL)
        if not fallback.configured:
            continue
        out = fallback.chat_stream(prompt, system=system, temperature=temperature,
                                   on_delta=on_delta)
        if out is not None and validate(out):
            return out
    return first


def maintenance_chat(prompt: str, system: str = "你是严谨的记忆整理助手。",
                     temperature: float = 0.2, validate=None) -> str | None:
    """梳理通道：巩固（聚类摘要/蒸馏）使用，可指向低配模型。"""
    return _chat_via("maint", settings.CLOUD_MAINT_URL, settings.CLOUD_MAINT_API_KEY,
                     settings.CLOUD_MAINT_MODEL, prompt, system, temperature, validate,
                     protocol=settings.CLOUD_MAINT_PROTOCOL)


def local_chat(prompt: str, system: str = "你是严谨的记忆整理助手。",
               temperature: float = 0.2, validate=None) -> str | None:
    """E8 本地快车道：只试本地 LM Studio，绝不碰云端（独立探测槽）。

    供经验绿色通道的技能抽取使用——经验写入是高频低延迟诉求，本地 8B 模型
    足够，云端（尤其经代理黑洞时）只会带来 60s 级挂死。不可用立即返回 None，
    由调用方走规则兜底：经验写入永不因网络挂死。校验失败不换模型重试——
    本地快车道要么可用要么降级，不做第二次网络等待。
    """
    slot = _cache["local"]
    now = time.monotonic()
    if not slot["probed"] or now - slot["at"] > _PROBE_TTL:
        client = LocalStudioClient()
        try:
            slot["client"] = client if client.available() else None
        except Exception:
            slot["client"] = None
        slot["at"], slot["probed"] = now, True
    client = slot["client"]
    if client is None:
        return None
    out = client.chat(prompt, system=system, temperature=temperature)
    if out is not None and validate is not None and not validate(out):
        return None
    return out


_embed_failures = {"count": 0}  # 云端嵌入连续失败计数（熔断器状态，成功即清零）


def embed(text: str) -> list[float]:
    """嵌入唯一入口。云端连续失败熔断：CloudClient.embed 网络失败时静默退哈希
    （维度 = EMBED_FALLBACK_DIM，与 bge-m3 的 1024 维可区分），连续 2 次失败即
    置空嵌入槽直到探测 TTL 过期——网络黑洞期不再逐次撞网干等（实测探测 60s +
    嵌入 60s 串行超时 = CLI 挂死 120s 的根因）。TTL 到期自动重探恢复。
    注意：若未来配置了恰为 64 维的云端嵌入模型，需改用显式失败信号。
    """
    client = _pick("embed")
    if client is None:
        return hash_embed(text, settings.EMBED_FALLBACK_DIM)
    vec = client.embed(text)
    if isinstance(client, CloudClient) and len(vec) == settings.EMBED_FALLBACK_DIM:
        _embed_failures["count"] += 1
        if _embed_failures["count"] >= 2:
            _cache["embed"].update(client=None, at=time.monotonic(), probed=True)
            _embed_failures["count"] = 0
    else:
        _embed_failures["count"] = 0
    return vec


def _backend_of(slot: dict, now: float | None = None) -> str:
    """由嵌入通道的缓存槽判定后端（纯函数，无任何 I/O，供测试直测）。

    槽未探测过 / 缓存已过期（TTL 300s）→ 视为无信息，保守返回 "hash"；
    已探测 → 按探测到的客户端类型如实回报 "cloud" / "local" / "hash"。
    """
    now = time.monotonic() if now is None else now
    client = slot["client"] if (slot.get("probed")
                                and now - slot.get("at", 0.0) <= _PROBE_TTL) else None
    if isinstance(client, CloudClient):
        return "cloud"
    if isinstance(client, LocalStudioClient):
        return "local"
    return "hash"


def embed_backend() -> str:
    """当前嵌入后端标识："cloud" / "local" / "hash"（零网络，检索路径安全调用）。

    只读 _pick 已建立的探测缓存，绝不在检索路径上发起新的可用性探测；
    未探测过或缓存已过期（TTL 300s）时返回保守值 "hash"——嵌入通道
    一旦真实 embed 过（缓存必已建立），此处读到的即是事实后端。
    E7 置信阈值分档（retriever.confident_bar）消费它：哈希兜底档 0.30
    按离线嵌入标定，真实稠密嵌入（bge-m3 等）跨主题余弦噪声底高
    （在线实测 0.56~0.64），须换 0.70 档。
    """
    return _backend_of(_cache["embed"])


def reset_probes() -> None:
    """清空全部探测缓存与熔断计数（供 TUI 改设置后强制重探）。

    四个槽（chat/maint/embed/local）统一回到未探测态（client=None, probed=False,
    at=0），下次访问按新设置重新探测；chat/maint 熔断计数与 embed 熔断计数一并
    清零——改设置是用户显式动作，旧失败记录不再代表新配置的行为。
    """
    for slot in _cache.values():
        slot.update(client=None, probed=False, at=0.0)
    _chat_failures.update(chat=0, maint=0)
    _embed_failures["count"] = 0


__all__ = ["LLMClient", "chat", "chat_stream", "maintenance_chat", "local_chat", "embed",
           "embed_backend", "llm_available", "maintenance_available", "active_provider",
           "reset_probes"]
