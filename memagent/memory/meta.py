"""元记忆（审计日志）：谁在何时对哪条记忆做了什么。"""
from __future__ import annotations

import sqlite3

from memagent.core.clock import now_iso


class AuditLog:
    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn

    def log(self, memory_type: str, memory_id: int, action: str, detail: str = "") -> None:
        self.conn.execute(
            "INSERT INTO meta (memory_type, memory_id, action, detail, created_at) VALUES (?,?,?,?,?)",
            (memory_type, memory_id, action, detail, now_iso()),
        )
        self.conn.commit()

    def count_trailing(self, action: str, detail_prefix: str, detail: str) -> int:
        """尾部连续计数（只读，P2-3 L3 双次失败翻转的连续性判定专用）。

        取 action 匹配的审计行按 id 倒序（最新在前）逐行检查：
        - detail 不以 detail_prefix 开头 -> 其他对象/其他任务域的行，**跳过不打断**
          （前缀把连续性隔离在对象内部：跨域穿插的工具结果互不干扰连续性）；
        - detail 恰为 detail_prefix + detail -> 计数 +1；
        - 其余（同前缀但不同值，如 success）-> 停止：成功天然清零连续失败计数。

        匹配用 Python startswith/全等而非 SQL LIKE：任务域含 %/_ 等 LIKE 通配符
        也不会误配。行数有限（同一 action 的审计流水），全量回扫代价可忽略。
        """
        rows = self.conn.execute(
            "SELECT detail FROM meta WHERE action=? ORDER BY id DESC", (action,)).fetchall()
        target = detail_prefix + detail
        n = 0
        for row in rows:
            d = row["detail"]
            if not d.startswith(detail_prefix):
                continue
            if d != target:
                break
            n += 1
        return n
