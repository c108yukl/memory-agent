"""冲突记录仓储：pending 冲突独立成表，可审查、可裁决，不散落在语义字段里。"""
from __future__ import annotations

from memagent.core.clock import now_iso
from memagent.storage.base import BaseRepo


class ConflictRepo(BaseRepo):
    def create(self, old_id: int, new_id: int, conflict_type: str = "value_conflict",
               reason: str = "") -> int:
        cur = self.conn.execute(
            "INSERT INTO memory_conflict (old_id, new_id, conflict_type, status, reason, created_at) "
            "VALUES (?,?,?,?,?,?)",
            (old_id, new_id, conflict_type, "pending", reason, now_iso()))
        self.conn.commit()
        return cur.lastrowid

    def get(self, conflict_id: int) -> dict | None:
        row = self.conn.execute(
            "SELECT * FROM memory_conflict WHERE conflict_id=?", (conflict_id,)).fetchone()
        return dict(row) if row else None

    def fetch_all(self, status: str | None = None) -> list[dict]:
        if status:
            rows = self.conn.execute(
                "SELECT * FROM memory_conflict WHERE status=? ORDER BY conflict_id DESC",
                (status,)).fetchall()
        else:
            rows = self.conn.execute(
                "SELECT * FROM memory_conflict ORDER BY conflict_id DESC").fetchall()
        return [dict(r) for r in rows]

    def pending_for_new(self, new_id: int) -> list[dict]:
        """该事实作为「新值」是否还有待裁冲突——V1.7 P1 D1 的 A/B 分类判据。

        pending 有两种语义，转正规则必须分开：
        - 本方法非空 = A 类·冲突待裁：新值与已有 active 事实互斥，转正即取代旧
          事实，只能由裁决（accept-new / keep-old / both）决定，**永不自动转正**；
        - 本方法为空 = B 类·低置信新事实：与谁都不冲突，只是证据不足——这正是
          「试用期」，可以靠使用频率转正。

        只认 status='pending'：已裁决的冲突行不阻塞转正（keep-old 会把新事实
        归档，那时它已不是 pending；both 裁决后两条共存是用户明确的意思表示）。
        """
        rows = self.conn.execute(
            "SELECT * FROM memory_conflict WHERE new_id=? AND status='pending' "
            "ORDER BY conflict_id", (new_id,)).fetchall()
        return [dict(r) for r in rows]

    def mark_resolved(self, conflict_id: int, resolution: str) -> None:
        self.conn.execute(
            "UPDATE memory_conflict SET status='resolved', resolved_at=?, resolution=? WHERE conflict_id=?",
            (now_iso(), resolution, conflict_id))
        self.conn.commit()
        self.audit("conflict", conflict_id, "resolve", resolution)
