"""情感分析器：valence（效价，-1消极~+1积极）+ arousal（唤醒度，0平静~1激动）。

与门控同构的双模模式：规则打分（可解释、离线保底）+ 可选 LLM（更准但需模型）。
唤醒度进入强度模型的 W_EMOTION 分项与半衰期延长系数（E2 闪光灯记忆），
valence 目前只做符号化落库（Event.emotion），供后续阶段（E5 情感记忆优先巩固）消费。
"""
from __future__ import annotations

import json
import re

from memagent.adapters import llm

# 通用情感关键词（常用表述，非任何评测文本定制）
_POSITIVE_WORDS = ("开心", "高兴", "快乐", "兴奋", "激动", "满意", "喜欢", "幸福",
                   "感动", "自豪", "骄傲", "庆幸", "惊喜", "顺利", "成功", "如愿",
                   "梦寐以求", "圆梦")
_NEGATIVE_WORDS = ("难过", "失望", "崩溃", "愤怒", "生气", "焦虑", "担心", "心痛",
                   "伤心", "痛苦", "沮丧", "后悔", "害怕", "恐惧", "倒霉", "糟糕",
                   "委屈", "烦躁")

# 唤醒度启发式信号
_INTENSITY_MARKS = ("非常", "特别", "极其", "超级", "十分")  # 强度副词
_INTENSITY_PATTERN = re.compile(r"太[^，。！!？?]{1,6}了")    # 「太…了」句式
_AROUSAL_BODY_MARKS = ("没睡", "失眠", "睡不着", "彻夜", "整晚", "一晚",
                       "发抖", "流泪", "哭了", "心跳")       # 身体反应（情绪溢出到生理）

_LLM_EMOTION_PROMPT = (
    "分析下面这条信息的情感。只输出一个 JSON 对象，不要任何多余文字：\n"
    '{{"valence": <-1~1 的效价，-1 极度消极 / 0 中性 / 1 极度积极>, '
    '"arousal": <0~1 的唤醒度，0 平静 / 1 极度激动>}}\n'
    "信息：{content}\nJSON："
)


def _parse_emotion(raw: str) -> tuple[float, float] | None:
    """解析模型输出（写法仿 semantic_extractor._parse_facts：截取首尾大括号）。"""
    try:
        obj = json.loads(raw[raw.find("{"): raw.rfind("}") + 1])
        valence = max(-1.0, min(1.0, float(obj["valence"])))
        arousal = max(0.0, min(1.0, float(obj["arousal"])))
        return round(valence, 4), round(arousal, 4)
    except (ValueError, KeyError, TypeError, IndexError):
        return None


def analyze_by_llm(content: str) -> tuple[float, float] | None:
    """LLM 打分；模型不可用或输出不合法时返回 None 表示降级到规则。"""
    raw = llm.chat(
        _LLM_EMOTION_PROMPT.format(content=content),
        temperature=0.0,
        validate=lambda s: _parse_emotion(s) is not None,
    )
    if raw is None:
        return None
    return _parse_emotion(raw)


def analyze_by_rules(content: str) -> tuple[float, float]:
    """规则打分（离线确定性）：valence 正负关键词差，arousal 信号累加封顶。

    唤醒度启发式：出现情感词即有基础唤醒 0.4；强度副词/「太…了」句式、
    身体反应（失眠/发抖等）、感叹号逐项加成——生理溢出与强调语气是
    高唤醒最可靠的表层线索。
    """
    pos = sum(1 for w in _POSITIVE_WORDS if w in content)
    neg = sum(1 for w in _NEGATIVE_WORDS if w in content)
    valence = max(-1.0, min(1.0, 0.6 * (pos - neg)))

    arousal = 0.4 if (pos or neg) else 0.0
    marks = sum(1 for w in _INTENSITY_MARKS if w in content)
    if _INTENSITY_PATTERN.search(content):
        marks += 1
    arousal += 0.2 * min(2, marks)
    arousal += 0.3 * min(2, sum(1 for w in _AROUSAL_BODY_MARKS if w in content))
    arousal += 0.1 * min(2, content.count("!") + content.count("！"))
    return round(valence, 4), round(max(0.0, min(1.0, arousal)), 4)


def analyze_emotion(content: str, use_llm: bool = True) -> tuple[float, float]:
    """返回 (valence ∈ [-1,1], arousal ∈ [0,1])；LLM 不可用/解析失败降级规则。"""
    if use_llm:
        got = analyze_by_llm(content)
        if got is not None:
            return got
    return analyze_by_rules(content)


def emotion_label(valence: float) -> str:
    """valence 符号化：±0.05 内视为 neutral（规则打分步长 0.6，不会落在模糊带）。"""
    if valence > 0.05:
        return "positive"
    if valence < -0.05:
        return "negative"
    return "neutral"
