"""强度模型（正式化）：strength = w1·importance + w2·confidence + w3·freq(饱和) + w4·recency(半衰)
+ w5·emotion(唤醒)，冲突未决记忆乘惩罚系数。输出可解释分项，便于调参与审计。

设计要点：
- 频率项 log(1+access)/cap 封顶——防止「越召回越强、越强越召回」的自我强化；
- 时间项指数半衰——遗忘不再是独立的线性衰减，而是强度的内生属性；
- 高唤醒半衰期延长（E2 闪光灯记忆）：half_life × (1 + K·arousal)，情绪事件抗遗忘；
- 一切输入都是可观测量（重要度/置信度/访问次数/最近访问时间/唤醒度），强度随时可复现。
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from memagent import settings


@dataclass
class StrengthResult:
    score: float
    importance_part: float
    confidence_part: float
    frequency_part: float
    recency_part: float
    emotion_part: float               # E2：唤醒度分项（W_EMOTION·arousal）
    conflict_penalty: float  # 0.0 无惩罚；否则被扣除的比例


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def frequency_component(access_count: int,
                        cap: float = settings.FREQ_ACCESS_CAP) -> float:
    """访问频率分量：log 饱和，cap 次访问后不再增长。"""
    if access_count <= 0:
        return 0.0
    return min(math.log1p(access_count), cap) / cap


def recency_component(last_access_ts: float,
                      half_life_days: float = settings.STRENGTH_HALF_LIFE_DAYS,
                      now: float | None = None) -> float:
    """新近度分量：半衰期指数衰减；无时间信息记 0。"""
    if last_access_ts <= 0:
        return 0.0
    from memagent.core.clock import now_ts
    now = now if now is not None else now_ts()
    days = max(0.0, (now - last_access_ts) / 86400.0)
    return math.exp(-math.log(2) * days / half_life_days)


def half_life_of(category: str = "", arousal: float = 0.0) -> float:
    """E8 分层半衰期：经验层（category="experience"）用短半衰（LFU 快淘汰），
    普通层沿用 30 天；arousal 延长系数两层共用（E2 闪光灯记忆）。
    编码/重算/再巩固所有调用点统一走这里——杜绝「写入时高、重算后掉」的不一致。
    """
    base = (settings.EXPERIENCE_HALF_LIFE_DAYS if category == "experience"
            else settings.STRENGTH_HALF_LIFE_DAYS)
    return base * (1.0 + settings.EMOTION_HALF_LIFE_BOOST_K * _clamp01(arousal))


def compute_strength(importance: float, confidence: float, access_count: int,
                     last_access_ts: float, has_conflict: bool = False,
                     now: float | None = None, arousal: float = 0.0,
                     category: str = "") -> StrengthResult:
    """arousal 高唤醒双重作用：加 W_EMOTION 分项 + 延长新近度半衰期（闪光灯记忆）。

    E8 category 分层半衰期：experience 经验层短半衰（LFU），其余默认 30 天。
    注：加情感项后五权重和 1.10，满格组合超出 1 的部分由 _clamp01 截断。
    """
    imp = _clamp01(importance)
    conf = _clamp01(confidence)
    aro = _clamp01(arousal)
    freq = frequency_component(access_count)
    half_life = half_life_of(category, aro)
    rec = recency_component(last_access_ts, half_life_days=half_life, now=now)

    emotion_part = settings.W_EMOTION * aro
    score = (settings.W_IMPORTANCE * imp
             + settings.W_CONFIDENCE * conf
             + settings.W_FREQUENCY * freq
             + settings.W_RECENCY * rec
             + emotion_part)
    penalty = settings.CONFLICT_PENALTY_FACTOR if has_conflict else 0.0
    score *= (1.0 - penalty)

    return StrengthResult(
        score=round(_clamp01(score), 4),
        importance_part=round(settings.W_IMPORTANCE * imp, 4),
        confidence_part=round(settings.W_CONFIDENCE * conf, 4),
        frequency_part=round(settings.W_FREQUENCY * freq, 4),
        recency_part=round(settings.W_RECENCY * rec, 4),
        emotion_part=round(emotion_part, 4),
        conflict_penalty=penalty,
    )
