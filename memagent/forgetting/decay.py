"""记忆强度的时间演化：遗忘不再是独立的线性衰减，而是统一强度模型的重算。

一条记忆的强度由可观测量（重要度/访问次数/最近访问时间）随时重算得出，
长期未访问的记忆因新近度项衰减而自然走低。
"""
from __future__ import annotations

from memagent.core.clock import iso_to_ts
from memagent.learning.strength import compute_strength


def recompute_strength(mem) -> float:
    """按统一强度模型重算一条情景记忆的强度（事件无独立置信度，以重要度代理）。

    E2：arousal 随记忆持久化，重算时既加情感分项又延长新近度半衰期
    （写入时高、重算后也高，不会出现「写入时高、重算后掉」的不一致）。
    E8：经验层（category="experience"）重算走短半衰——分层策略在重算路径同样生效。
    """
    ref_ts = iso_to_ts(mem.last_access_at or mem.created_at)
    res = compute_strength(mem.importance, mem.importance, mem.access_count, ref_ts,
                           arousal=mem.arousal,
                           category=getattr(mem, "category", ""))
    return res.score
