"""实体别名仓储：用户可维护的别名 -> 标准实体映射。"""
from __future__ import annotations

from memagent.core.clock import now_iso
from memagent.storage.base import BaseRepo


class EntityAliasRepo(BaseRepo):
    def add(self, alias: str, canonical: str, source: str = "manual") -> bool:
        """新增别名；别名已存在且指向不同实体时返回 False（不允许一别名多主）。"""
        row = self.conn.execute("SELECT canonical FROM entity_alias WHERE alias=?", (alias,)).fetchone()
        if row:
            return row["canonical"] == canonical
        self.conn.execute(
            "INSERT INTO entity_alias (alias, canonical, created_at, source) VALUES (?,?,?,?)",
            (alias, canonical, now_iso(), source))
        self.conn.commit()
        self.audit("alias", 0, "add", f"{alias}->{canonical}")
        return True

    def remove(self, alias: str) -> bool:
        cur = self.conn.execute("DELETE FROM entity_alias WHERE alias=?", (alias,))
        self.conn.commit()
        return cur.rowcount > 0

    def fetch_all(self) -> list[dict]:
        rows = self.conn.execute("SELECT * FROM entity_alias ORDER BY alias").fetchall()
        return [dict(r) for r in rows]

    def as_map(self) -> dict[str, str]:
        return {r["alias"]: r["canonical"] for r in self.fetch_all()}
