"""间隔重复（SM-2 简化版）：高价值记忆定期"提取练习"，越用越牢。

对应学习策略：每次主动回忆都是给记忆打补丁，效果远好于被动重读。
时间一律取自 core.clock（评测可整体平移，保证 due 队列确定性）。
"""
from __future__ import annotations

import datetime as dt
import sqlite3

from memagent import settings
from memagent.core.clock import now_dt, now_iso


class SpacedRepetition:
    """为语义记忆条目维护 SM-2 复习计划。"""

    TABLE = """
    CREATE TABLE IF NOT EXISTS repetition (
        memory_id INTEGER PRIMARY KEY,
        ease REAL DEFAULT 2.5,
        interval_days INTEGER DEFAULT 0,
        repetitions INTEGER DEFAULT 0,
        due_at TEXT DEFAULT '',
        last_review TEXT DEFAULT ''
    )"""

    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn
        conn.execute(self.TABLE)
        conn.commit()

    def review(self, memory_id: int, quality: int) -> dict:
        """quality: 0~5（5=完美回忆，0=完全忘记），越界值钳制到边界——
        99/-3 这类值会直接扭曲 ease/interval 曲线。返回更新后的计划。"""
        quality = max(0, min(5, int(quality)))
        row = self.conn.execute(
            "SELECT ease, interval_days, repetitions FROM repetition WHERE memory_id=?",
            (memory_id,)).fetchone()

        if row is None:
            ease, interval, reps = settings.SM2_EASE_INIT, 0, 0
        else:
            ease, interval, reps = row
            # 自愈：历史无上限 SM-2 可能已写入越界值（interval×ease 指数增长会
            # 溢出 datetime；ease 无上界加速爆炸）。读侧钳制，本次写回即固化。
            ease = max(settings.SM2_EASE_MIN, min(float(ease), settings.SM2_EASE_MAX))
            interval = max(0, min(int(interval), settings.SM2_MAX_INTERVAL))

        if quality >= 3:
            if reps == 0:
                interval = settings.SM2_MIN_INTERVAL
            elif reps == 1:
                interval = settings.SM2_MIN_INTERVAL * 6
            else:
                interval = round(interval * ease)
            reps += 1
            ease = max(settings.SM2_EASE_MIN,
                       min(settings.SM2_EASE_MAX,
                           ease + (0.1 - (5 - quality) * (0.08 + (5 - quality) * 0.02))))
        else:
            reps = 0
            interval = settings.SM2_MIN_INTERVAL
            ease = max(settings.SM2_EASE_MIN, ease - 0.2)

        # 写入侧钳制兜底：interval 有物理上限（now+timedelta 永不溢出），ease 封顶
        interval = max(settings.SM2_MIN_INTERVAL, min(int(interval), settings.SM2_MAX_INTERVAL))
        ease = max(settings.SM2_EASE_MIN, min(float(ease), settings.SM2_EASE_MAX))

        now = now_dt()
        due = now + dt.timedelta(days=interval)
        self.conn.execute(
            """INSERT INTO repetition (memory_id, ease, interval_days, repetitions, due_at, last_review)
               VALUES (?,?,?,?,?,?)
               ON CONFLICT(memory_id) DO UPDATE SET
                 ease=excluded.ease, interval_days=excluded.interval_days,
                 repetitions=excluded.repetitions, due_at=excluded.due_at,
                 last_review=excluded.last_review""",
            (memory_id, ease, interval, reps, due.isoformat(timespec="seconds"),
             now.isoformat(timespec="seconds")))
        self.conn.commit()
        return {"memory_id": memory_id, "ease": ease, "interval_days": interval,
                "repetitions": reps, "due_at": due.isoformat(timespec="seconds")}

    def due(self, limit: int = 20) -> list[int]:
        """返回到期的记忆 id 列表（今天的提取练习队列）。"""
        rows = self.conn.execute(
            "SELECT memory_id FROM repetition WHERE due_at <= ? ORDER BY due_at LIMIT ?",
            (now_iso(), limit)).fetchall()
        return [r[0] for r in rows]

    def status(self) -> list[dict]:
        rows = self.conn.execute("SELECT * FROM repetition ORDER BY due_at").fetchall()
        return [dict(r) for r in rows]

    def status_row(self, memory_id: int) -> dict | None:
        row = self.conn.execute(
            "SELECT * FROM repetition WHERE memory_id=?", (memory_id,)).fetchone()
        return dict(row) if row else None
