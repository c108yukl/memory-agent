"""语义抽取：LLM 抽取三元组；无模型时退回关键词规则。

E8 绿色类型（experience / env_statement）不走 LLM——事实键由规则确定性构造：
取代的可靠性要求键稳定（LLM 自由输出的 entity 换个措辞，冲突检测就落空，
取代退化成平行堆积）。键设计：
  env_statement → (内容中的 ASCII 工具名, env_state)   环境状态"最新即正确"
  experience    → (任务域, lesson)                     任务域内经验迭代

P2-2 条件上下文（validity_context）：只有 LLM 主抽取通道在原文确有条件/限定
表述时产出（normalize_validity_context 白名单守门）；离线规则与显式声明恒空串——
用户原话没有的上下文不虚构（宁缺勿造）。注入过滤暂缓（既有决策）：本任务只做
存储与展示。
"""
from __future__ import annotations

import json
import re

from memagent import settings
from memagent.adapters import llm
from memagent.core.clock import now_iso
from memagent.core.domain import Event, SemanticFact, GREEN_TYPES
from memagent.encoding.prompts import FACT_EXTRACT_PROMPT

# 环境状态的实体 = 内容中的工具名（ffmpeg / openrouter / clash 这类 ASCII 词）；
# 中文语境里工具名几乎总是原样英文，这是零模型依赖下最稳的锚点
_TOOL_WORD = re.compile(r"[a-z][a-z0-9_-]{1,}", re.IGNORECASE)

# LLM 抽取的 relation 归一表（窄别名，只收语义明确等价的）：不归一的代价是版本链
# 断裂——「偏好」类事实被 LLM 写成自由 relation（实测：turn1 存 [user] prefers 分点列举，
# turn10 改偏好被存成 [user] 回答偏好 简洁为主），不同键互不冲突、不会取代，两条矛盾
# 偏好同时 active，检索两头都召回。离线路径本来就硬编码 prefers——此表把在线/离线
# 口径接平（C4 同一哲学：键要稳定，不让模型自由命名）。键空间里的其余自由 relation
# 不强行归一（宁窄勿宽，归错比漏归更伤）。
_RELATION_ALIASES = {
    "偏好": "prefers", "回答偏好": "prefers", "回答习惯": "prefers",
    "喜好": "prefers", "喜欢": "prefers",
    "不喜欢": "dislikes", "讨厌": "dislikes",
    "经验": "lesson", "教训": "lesson", "心得": "lesson",
    "环境状态": "env_state", "环境": "env_state",
}


def _normalize_relation(relation: str) -> str:
    return _RELATION_ALIASES.get(relation.strip(), relation.strip())


def _parse_facts(raw: str) -> list[dict]:
    try:
        return json.loads(raw[raw.find("["): raw.rfind("]") + 1])
    except (ValueError, IndexError):
        return []


# P2-2 条件上下文（认识论限定）白名单：LLM 自由输出的结构不可信——未知键丢弃、
# 类型不对丢该键、全空归空串，任何解析异常返回 ""（宁缺勿造：没有的条件绝不虚构）。
# LLM 产出的 validity_context 必须过本关才允许入库。
_VC_LIST_KEYS = ("preconditions", "environment")
_VC_STR_KEYS = ("expires_if",)


def normalize_validity_context(raw) -> str:
    """把抽取侧拿到的条件上下文规范化为紧凑 JSON 字符串（不可信输入的守门员）。

    - 输入：LLM JSON 里解析出的 validity_context 对象（dict；字符串则先尝试
      再解析一层，其余类型一律不要）；
    - 白名单键：preconditions / environment 各为 str 列表（非 str 元素剔除）、
      expires_if 为 str；未知键丢弃、类型不对丢该键、项全空丢该键；
    - 全空返回 ""（= 无条件限定，永久有效）；输出紧凑 JSON
      （ensure_ascii=False，排序键）；任何解析异常返回 ""。
    """
    try:
        if isinstance(raw, str):
            raw = json.loads(raw)
        if not isinstance(raw, dict):
            return ""
        out: dict = {}
        for key in _VC_LIST_KEYS:
            val = raw.get(key)
            if not isinstance(val, list):
                continue
            items = [v.strip() for v in val if isinstance(v, str) and v.strip()]
            if items:
                out[key] = items
        for key in _VC_STR_KEYS:
            val = raw.get(key)
            if isinstance(val, str) and val.strip():
                out[key] = val.strip()
        if not out:
            return ""
        return json.dumps(out, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    except Exception:  # 垃圾输入/解析异常一律按「无上下文」处理，绝不带病入库
        return ""


def _green_fact(event: Event) -> SemanticFact:
    value = event.content.strip()
    if event.type == "env_statement":
        m = _TOOL_WORD.search(event.content)
        entity = m.group(0).lower() if m else "env"
        relation = "env_state"
        # B5-a 值限长（写入侧唯一挂点）：只裁 env_statement 的 value——真实库曾
        # 出现一条巨型 env_state 塞全量机器信息（IP/面板/SSH 密钥），而 (工具名,
        # env_state) 的 value 是检索与注入的承载面（见 settings.ENV_STATE_MAX_CHARS）。
        # 情景记忆原文不裁（encode_episodic 另行保全），experience/显式声明也不裁：
        # 它们没有「最新即正确」的取代语义，截断等于永久丢信息。
        if len(value) > settings.ENV_STATE_MAX_CHARS:
            value = value[:settings.ENV_STATE_MAX_CHARS] + "…(已截断)"
    else:  # experience：任务域即实体（--context 传入），缺省归到通用经验域
        entity = event.task_context.strip() or "experience"
        relation = "lesson"
    # validity_context 恒空串（字段缺省）：E8 绿色类型不走 LLM，且环境状态/经验
    # 的语义是「最新即正确」，挂条件限定反而破坏取代链——用户原话没有的上下文不虚构
    return SemanticFact(
        entity=entity, relation=relation, value=value,
        confidence=0.8, valid_from=now_iso(),
        embedding=llm.embed(event.content))


def extract_semantic_facts(event: Event) -> list[SemanticFact]:
    facts: list[SemanticFact] = []
    if event.type in GREEN_TYPES:
        return [_green_fact(event)]
    if llm.llm_available():
        raw = llm.chat(FACT_EXTRACT_PROMPT.format(content=event.content),
                       validate=lambda s: len(_parse_facts(s)) > 0 or s.strip().endswith("[]"))
        if raw:
            for item in _parse_facts(raw):
                if isinstance(item, dict) and item.get("entity") and item.get("value"):
                    # P2-2：validity_context 只在 LLM 明确抽出时携带，且必须过
                    # normalize 白名单（未知键/错型/垃圾一律归空串）才允许入库
                    facts.append(SemanticFact(
                        entity=str(item["entity"]).strip(),
                        relation=_normalize_relation(str(item.get("relation", "fact"))),
                        value=str(item["value"]).strip(),
                        confidence=float(item.get("confidence", 0.8)),
                        valid_from=now_iso(),
                        embedding=llm.embed(f"{item['entity']} {item.get('relation', '')} {item['value']}"),
                        validity_context=normalize_validity_context(item.get("validity_context")),
                    ))
        return facts

    # 离线关键词兜底（无模型）：只有偏好关键词的确定性构键——用户原话里没有的
    # 条件上下文不虚构（宁缺勿造），validity_context 恒空串
    for keyword, rel, val in [
        ("喜欢", "prefers", "喜欢"),
        ("希望", "prefers", "希望"),
        ("不要", "dislikes", "不喜欢"),
        ("偏好", "prefers", "偏好"),
    ]:
        if keyword in event.content:
            facts.append(SemanticFact(
                entity="user", relation=rel, value=event.content.strip(),
                confidence=0.6, valid_from=now_iso(),
                embedding=llm.embed(event.content)))

    # 显式声明（"记住/以后"类词语，或显式事件类型）即使没有偏好关键词也是一条偏好事实
    # （EXPLICIT_TYPES 不生成 validity_context：用户显式点名的就是无条件原话，宁缺勿造）
    if not facts and (any(w in event.content for w in ("记住", "以后", "今后", "别再"))
                      or event.type in ("instruction", "preference_statement")):
        facts.append(SemanticFact(
            entity="user", relation="prefers", value=event.content.strip(),
            confidence=0.6, valid_from=now_iso(),
            embedding=llm.embed(event.content)))
    return facts
