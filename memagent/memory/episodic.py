"""情景记忆仓储：具体事件的存取、检索与状态流转。"""
from __future__ import annotations

import json

from memagent import settings
from memagent.core.clock import now_iso
from memagent.core.domain import EpisodicMemory
from memagent.core.text import bigram
from memagent.core.vectors import cosine
from memagent.storage.base import BaseRepo


class EpisodicMemoryRepo(BaseRepo):
    def add(self, mem: EpisodicMemory) -> int:
        cur = self.conn.execute(
            """INSERT INTO episodic (summary, context, action, outcome, importance, created_at,
               access_count, last_access_at, strength, status, embedding,
               source_ids, summarized_by, is_summary, arousal, category, source)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (mem.summary, mem.context, mem.action, mem.outcome, mem.importance,
             mem.created_at or now_iso(), mem.access_count, mem.last_access_at,
             mem.strength, mem.status, json.dumps(mem.embedding),
             json.dumps(mem.source_ids), mem.summarized_by, int(mem.is_summary),
             mem.arousal, mem.category, mem.source),
        )
        mem.id = cur.lastrowid
        self._sync_fts(mem.id, mem.status, mem.summary)
        self.conn.commit()
        self.audit("episodic", mem.id, "create", mem.summary[:60])
        return mem.id

    def _fts_text(self, mem_id: int) -> str | None:
        row = self.conn.execute(
            "SELECT summary FROM episodic WHERE id=?", (mem_id,)).fetchone()
        return row["summary"] if row else None

    def _sync_fts(self, mem_id: int, status: str, summary: str | None = None) -> None:
        """让可检索状态的记忆在 FTS 索引里有条目（幂等，可重复调用）。

        维护规则是**实测**得出的，改动前请重跑 data/ 下的同类验证，别凭直觉动：
        本表是 external content 表（content='episodic'），行为有三条反直觉之处——
          1. INSERT 忽略传入值，回表取主表当前值建索引 → 传空串清不掉条目；
             好处是内容变更后重新 INSERT 即可刷新（update 走这条路）。
          2. 无法自省索引状态：裸 SELECT 从 content 表读，主表有这行就查得到。
          3. 单条 DELETE、以及 UPDATE 到空条目，都会损坏索引
             （DatabaseError: database disk image is malformed）。
        因此本方法**只做 INSERT，不做任何删除**。失效记忆（archived / summarized /
        deleted）的索引条目会残留并占据 search_fts 的 limit 名额，这是向 FTS5 约束
        妥协的已知取舍——代价由 settings.FTS_CANDIDATE_FACTOR 放大候选池抵消，
        正确性由 retriever 侧的 status 过滤保证。本方法不 commit。
        """
        text = summary if summary is not None else self._fts_text(mem_id)
        if text is None:
            return
        if status in settings.RETRIEVABLE_STATUSES:
            self.conn.execute(
                "INSERT INTO episodic_fts (rowid, summary) VALUES (?,?)",
                (mem_id, bigram(text)))

    def get(self, mem_id: int) -> EpisodicMemory | None:
        row = self.conn.execute("SELECT * FROM episodic WHERE id=?", (mem_id,)).fetchone()
        return self._from_row(row) if row else None

    def fetch(self, status: str = "active", limit: int = 1000) -> list[EpisodicMemory]:
        rows = self.conn.execute(
            "SELECT * FROM episodic WHERE status=? ORDER BY id DESC LIMIT ?", (status, limit)).fetchall()
        return [self._from_row(r) for r in rows]

    def update(self, mem: EpisodicMemory) -> None:
        self.conn.execute(
            """UPDATE episodic SET summary=?, context=?, action=?, outcome=?, importance=?,
               access_count=?, last_access_at=?, strength=?, status=?, embedding=?,
               source_ids=?, summarized_by=?, is_summary=?, arousal=?, category=?, source=?
               WHERE id=?""",
            (mem.summary, mem.context, mem.action, mem.outcome, mem.importance,
             mem.access_count, mem.last_access_at, mem.strength, mem.status,
             json.dumps(mem.embedding), json.dumps(mem.source_ids),
             mem.summarized_by, int(mem.is_summary), mem.arousal, mem.category,
             mem.source, mem.id),
        )
        self._sync_fts(mem.id, mem.status, mem.summary)
        self.conn.commit()

    def set_status(self, mem_id: int, status: str) -> None:
        self.conn.execute("UPDATE episodic SET status=? WHERE id=?", (status, mem_id))
        self._sync_fts(mem_id, status)  # 状态进出可检索集合时，索引行随之增删
        self.conn.commit()
        self.audit("episodic", mem_id, f"status->{status}")

    def search_fts(self, query: str, limit: int = 20) -> list[tuple[int, float]]:
        """FTS5 关键词检索（中文 2-gram，OR 连接——MATCH 多 token 默认 AND 会漏召回）。

        token 必须加引号：2-gram 会切出 "x-"/"-a" 这类含连字符片段，
        裸拼进 OR 表达式会被 FTS5 当作 NOT 运算符导致语法错误。
        """
        tokens = [t.replace('"', "") for t in bigram(query).split() if t.strip()]
        q = " OR ".join(f'"{t}"' for t in tokens if t)
        if not q:
            return []
        rows = self.conn.execute(
            "SELECT rowid, bm25(episodic_fts) AS score FROM episodic_fts "
            "WHERE episodic_fts MATCH ? ORDER BY score LIMIT ?", (q, limit)).fetchall()
        return [(r["rowid"], r["score"]) for r in rows]

    def cosine_search(self, vec: list[float], limit: int = 20) -> list[tuple[int, float]]:
        """暴力余弦检索：数据量小，够用且直观。维度不匹配（换过嵌入模型）的记录显式跳过。

        P0-1：候选集口径与 FTS 索引统一（settings.RETRIEVABLE_STATUSES）——两个检索
        通道必须看到同一份数据，否则 P1 让 pending 进 FTS 后会出现「关键词能召回、
        向量召回不到」的通道分裂。
        """
        pool: list = []
        for status in settings.RETRIEVABLE_STATUSES:
            pool.extend(self.fetch(status=status, limit=5000))
        results = []
        for mem in pool:
            if mem.embedding and len(mem.embedding) == len(vec):
                results.append((mem.id, cosine(vec, mem.embedding)))
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:limit]

    @staticmethod
    def _from_row(row) -> EpisodicMemory:
        keys = row.keys()
        return EpisodicMemory(
            id=row["id"], summary=row["summary"], context=row["context"],
            action=row["action"], outcome=row["outcome"], importance=row["importance"],
            created_at=row["created_at"], access_count=row["access_count"],
            last_access_at=row["last_access_at"], strength=row["strength"],
            status=row["status"], embedding=json.loads(row["embedding"] or "[]"),
            source_ids=json.loads(row["source_ids"]) if "source_ids" in keys and row["source_ids"] else [],
            summarized_by=row["summarized_by"] if "summarized_by" in keys else 0,
            is_summary=bool(row["is_summary"]) if "is_summary" in keys else False,
            arousal=row["arousal"] if "arousal" in keys else 0.0,
            category=row["category"] if "category" in keys and row["category"] else "",
            source=row["source"] if "source" in keys and row["source"] else "")
