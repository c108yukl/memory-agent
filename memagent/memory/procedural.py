"""程序记忆仓储：技能模板的沉淀、复用与统计。"""
from __future__ import annotations

from memagent.core.domain import ProceduralSkill
from memagent.storage.base import BaseRepo


class ProceduralSkillRepo(BaseRepo):
    def add(self, skill: ProceduralSkill) -> int:
        cur = self.conn.execute(
            "INSERT INTO procedural (name, trigger, policy, success_count, failure_count, usage_count, success_rate, status) "
            "VALUES (?,?,?,?,?,?,?,?)",
            (skill.name, skill.trigger, skill.policy, skill.success_count,
             skill.failure_count, skill.usage_count, skill.success_rate, skill.status))
        self.conn.commit()
        self.audit("procedural", cur.lastrowid, "create", skill.name)
        return cur.lastrowid

    def find(self, name: str) -> ProceduralSkill | None:
        row = self.conn.execute(
            "SELECT * FROM procedural WHERE name=? AND status='active'", (name,)).fetchone()
        return self._from_row(row) if row else None

    def get(self, skill_id: int) -> ProceduralSkill | None:
        row = self.conn.execute("SELECT * FROM procedural WHERE id=?", (skill_id,)).fetchone()
        return self._from_row(row) if row else None

    def set_status(self, skill_id: int, status: str) -> None:
        """V1.7.4：技能软状态（memory_archive 定向归档走这里）。技能检索只取
        active（fetch/find 均按 status 过滤），归档即从带出与复用里退场，行保留可回溯。"""
        self.conn.execute("UPDATE procedural SET status=? WHERE id=?", (status, skill_id))
        self.conn.commit()
        self.audit("procedural", skill_id, f"status->{status}")

    def update_stats(self, skill_id: int, success: bool) -> None:
        self.conn.execute(
            "UPDATE procedural SET usage_count=usage_count+1, "
            "success_count=success_count+?, failure_count=failure_count+?, "
            "success_rate=success_count*1.0/(usage_count+1) WHERE id=?",
            (1 if success else 0, 0 if success else 1, skill_id))
        self.conn.commit()
        self.audit("procedural", skill_id, "use", "success" if success else "failure")

    def update_policy(self, skill_id: int, policy: str) -> None:
        """E8 高度迭代：同名技能带新做法时更新 policy（经验进化，统计保留累计）。"""
        self.conn.execute("UPDATE procedural SET policy=? WHERE id=?", (policy, skill_id))
        self.conn.commit()
        self.audit("procedural", skill_id, "policy_update", policy[:60])

    def touch_usage(self, skill_id: int) -> None:
        """E8 程序记忆激活：检索带出记一次使用——只 +usage_count，不动成功率
       （被想起不等于被执行成功，成功率只由真实 outcome 观测累计）。"""
        self.conn.execute("UPDATE procedural SET usage_count=usage_count+1 WHERE id=?",
                          (skill_id,))
        self.conn.commit()
        self.audit("procedural", skill_id, "use", "retrieval")

    def fetch(self, status: str = "active") -> list[ProceduralSkill]:
        rows = self.conn.execute(
            "SELECT * FROM procedural WHERE status=? ORDER BY usage_count DESC", (status,)).fetchall()
        return [self._from_row(r) for r in rows]

    @staticmethod
    def _from_row(row) -> ProceduralSkill:
        return ProceduralSkill(
            id=row["id"], name=row["name"], trigger=row["trigger"], policy=row["policy"],
            success_count=row["success_count"], failure_count=row["failure_count"],
            usage_count=row["usage_count"], success_rate=row["success_rate"], status=row["status"])
