"""实体解析（规则版）：实体/关系/值的轻量归一，为冲突检测与检索质量打基础。

安全边界：代词映射只做「整体相等」判断，绝不做子串替换
（"我朋友" 不能被当成 "我"）。语义级归一（formal==正式）推迟到有评测数据后。
"""
from __future__ import annotations

import unicodedata

# 仅精确匹配的代词映射（单用户助手场景）
PRONOUN_MAP = {
    "我": "user", "本人": "user", "我自己": "user", "用户": "user",
    "你": "assistant", "您": "assistant",
}

# 关系别名：把 LLM 抽取时自由发挥的中文关系收到少量标准关系上
RELATION_ALIASES = {
    "偏好": "prefers", "喜欢": "prefers", "更喜欢": "prefers", "倾向": "prefers",
    "不喜欢": "dislikes", "讨厌": "dislikes",
    "事实": "fact", "是": "is", "有": "has",
}


def _normalize_text(s: str) -> str:
    """空白折叠、全半角归一（NFKC）、英文小写、去首尾空白。"""
    if not s:
        return s
    s = unicodedata.normalize("NFKC", s)
    return " ".join(s.split()).strip().lower()


def normalize_entity(name: str, aliases: dict[str, str] | None = None) -> str:
    """实体归一：文本归一 -> 精确代词匹配 -> 用户别名表。"""
    n = _normalize_text(name)
    if n in PRONOUN_MAP:
        return PRONOUN_MAP[n]
    if aliases:
        normalized_aliases = {_normalize_text(k): v for k, v in aliases.items()}
        if n in normalized_aliases:
            return normalized_aliases[n]
    return n


def normalize_relation(relation: str) -> str:
    n = _normalize_text(relation)
    return RELATION_ALIASES.get(n, n)


def normalize_value(value: str) -> str:
    """值归一：仅文本层面（空白/全半角/大小写），不做语义映射。"""
    return _normalize_text(value)


def resolve_fact(fact, aliases: dict[str, str] | None = None):
    """原地归一一条 SemanticFact 的 entity/relation/value，raw_* 记录归一前原值。"""
    if not fact.raw_entity:
        fact.raw_entity = fact.entity
    if not fact.raw_value:
        fact.raw_value = fact.value
    fact.entity = normalize_entity(fact.entity, aliases)
    fact.relation = normalize_relation(fact.relation)
    fact.value = normalize_value(fact.value)
    return fact
