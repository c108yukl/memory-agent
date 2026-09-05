"""工作记忆（E3）：纯内存、会话作用域的瞬时记忆 scratchpad。

刻意不入 SQLite——易失性即设计目标：进程退出（= 会话结束）自然蒸发，
未过门控的瞬时信息由此有了去处，也只在这一场会话里可查。容量 7±2
取上沿（Miller, 1956），满则淘汰 salience 最低者；salience 三要素
（情感唤醒 + 重要度 + 新近度）在读取时计算——不存衰减状态、不开后台
线程，「tick」就是每次读取时的现算。时间统一走 core.clock，评测的
时间旅行对它同样生效（超龄条目视为已蒸发，不干扰多天时间线）。
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

from memagent import settings
from memagent.core.clock import now_ts
from memagent.core.vectors import cosine, hash_embed


@dataclass
class WorkingEntry:
    """一条工作记忆：写入时刻 + 原文 + salience 的两个静态要素。

    新近度不是字段——它是 (now - created_ts) 的函数，读取时现算。
    id 为会话内创建序号（只增不复用），保证检索两轮 top1 的 id 稳定。
    """

    id: int
    text: str
    importance: float = 0.0
    arousal: float = 0.0
    created_ts: float = 0.0
    embedding: list[float] = field(default_factory=list)


class WorkingMemory:
    """会话级工作记忆：随 SqliteStore 实例生死（新场景新 store，天然隔离）。"""

    def __init__(self) -> None:
        self.entries: list[WorkingEntry] = []
        self._next_id = 1

    # ---------- 写 ----------
    def add(self, text: str, importance: float = 0.0, arousal: float = 0.0,
            embedding: list[float] | None = None) -> WorkingEntry:
        """写入一条；容量满时淘汰 salience 最低者（此刻现算，即「此刻」的淘汰）。

        embedding 由调用方（pipeline）经 llm.embed 传入——本层不 import
        adapters（支撑层不知道基础设施细节），仅兜底哈希嵌入。
        """
        entry = WorkingEntry(
            id=self._next_id, text=text,
            importance=max(0.0, min(1.0, importance)),
            arousal=max(0.0, min(1.0, arousal)),
            created_ts=now_ts(),
            embedding=list(embedding) if embedding is not None
            else hash_embed(text, settings.EMBED_FALLBACK_DIM),
        )
        self._next_id += 1
        self.entries.append(entry)
        while len(self.entries) > settings.WORKING_CAPACITY:
            victim = min(self.entries, key=lambda e: self.salience(e))
            self.entries.remove(victim)
        return entry

    # ---------- 读 ----------
    def salience(self, entry: WorkingEntry, now: float | None = None) -> float:
        """突显度（读取时计算的纯函数）：重要度 + 情感唤醒 + 指数新近衰减。"""
        now = now_ts() if now is None else now
        age_min = max(0.0, now - entry.created_ts) / 60.0
        recency = math.exp(-math.log(2) * age_min / settings.WORKING_HALF_LIFE_MIN)
        return (settings.WORKING_W_IMPORTANCE * entry.importance
                + settings.WORKING_W_AROUSAL * entry.arousal
                + settings.WORKING_W_RECENCY * recency)

    def _expired(self, entry: WorkingEntry, now: float) -> bool:
        return (now - entry.created_ts) > settings.WORKING_MAX_AGE_HOURS * 3600.0

    def search(self, query_vec: list[float],
               limit: int = settings.WORKING_RETRIEVE_LIMIT) -> list[tuple[float, WorkingEntry]]:
        """向量检索未过期条目，按 salience×相似度排序，返回 [(相似度, 条目)]。

        过期条目（age > WORKING_MAX_AGE_HOURS）视为已蒸发，不返回——
        排序键是 salience×相似度，但返回的相似度是裸余弦（调用方拿去做
        RetrievalHit.score），置顶与否不依赖高分。
        """
        now = now_ts()
        scored: list[tuple[float, float, WorkingEntry]] = []
        for e in self.entries:
            if self._expired(e, now):
                continue
            sim = cosine(query_vec, e.embedding) if e.embedding else 0.0
            scored.append((sim, self.salience(e, now) * sim, e))
        scored.sort(key=lambda t: t[1], reverse=True)  # 稳定排序：并列保持写入序
        return [(sim, e) for sim, _rank, e in scored[:limit]]

    def __len__(self) -> int:
        return len(self.entries)
