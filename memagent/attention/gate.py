"""注意力门控：类比前额叶，判断事件是否值得写入长期记忆。

规则打分（可解释、离线可用）+ 可选 LLM 复核（更准但需要模型）。
"""
from __future__ import annotations

from memagent import settings
from memagent.adapters import llm
from memagent.attention.scorers import score_by_rules
from memagent.core.domain import Event, EXPLICIT_TYPES, GREEN_TYPES

_LLM_SCORE_PROMPT = (
    "判断下面这条信息是否值得长期记住（用户身份背景（我是谁/职业/人生阶段）、用户偏好、"
    "重要事实、安全要求、经验教训值得记；闲聊、瞬时状态不值得记）。只输出一个 0~1 的数字。\n"
    "信息：{content}\n重要性："
)


def _parse_score(raw: str) -> float | None:
    try:
        return max(0.0, min(1.0, float(raw.strip())))
    except ValueError:
        return None


def score_by_llm(event: Event) -> float:
    """LLM 打分；模型不可用或输出不合法时返回 -1 表示降级到规则。"""
    raw = llm.chat(
        _LLM_SCORE_PROMPT.format(content=event.content),
        temperature=0.0,
        validate=lambda s: _parse_score(s) is not None,
    )
    if raw is None:
        return -1.0
    score = _parse_score(raw)
    return -1.0 if score is None else score


def attention_gate(event: Event, use_llm: bool = True) -> Event:
    """返回带 importance 的事件；importance 决定去向：
    >= WRITE_THRESHOLD  -> 长期记忆
    >= WORKING_THRESHOLD -> 仅工作记忆（本版由调用方丢弃或轻量处理）
    <  WORKING_THRESHOLD -> 丢弃

    用户显式声明类型（instruction / preference_statement / identity_statement）与
    E8 绿色通道类型（experience / env_statement，AI 点名沉淀的自身经验）不受低分
    拦截：门控过滤的是观察流，点名要记的不需要模型同意。

    E8 效率红线：绿色类型跳过 LLM 打分——保底已保证入库，云端打分（可能经
    代理黑洞 60s 挂死）对结果是零贡献的纯等待。经验写入必须秒级。
    """
    if event.type in GREEN_TYPES:
        event.importance = round(max(score_by_rules(event), settings.WRITE_THRESHOLD), 4)
        return event
    llm_score = score_by_llm(event) if use_llm else -1.0
    if llm_score >= 0:
        event.importance = round((llm_score * 0.7 + score_by_rules(event) * 0.3), 4)
    else:
        event.importance = round(score_by_rules(event), 4)
    if event.type in EXPLICIT_TYPES or event.type in GREEN_TYPES:
        event.importance = max(event.importance, settings.WRITE_THRESHOLD)
    return event
