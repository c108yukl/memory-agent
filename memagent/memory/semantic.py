"""语义记忆仓储：事实三元组的存取、去重、冲突检测与失效。"""
from __future__ import annotations

import json

from memagent import settings
from memagent.core.clock import now_iso
from memagent.core.domain import SemanticFact
from memagent.core.text import bigram
from memagent.core.vectors import cosine
from memagent.storage.base import BaseRepo


class SemanticMemoryRepo(BaseRepo):
    def upsert(self, fact: SemanticFact) -> tuple[int, bool]:
        """存在相同 (entity, relation, value) 且未失效的记录则累计证据，否则插入。
        返回 (id, 是否新建)。已存在分支不覆盖 embedding（回填走 update_embedding）。

        P0-1：FTS 索引行改由 _sync_fts 按 status 幂等维护——不再无条件插入
        （此前 pending/superseded 也进索引，白占 search_fts 的 limit 名额）。
        """
        row = self.conn.execute(
            "SELECT id, status, evidence_count, source_event_ids, validity_context FROM semantic "
            "WHERE entity=? AND relation=? AND value=? AND status IN ('active','pending')",
            (fact.entity, fact.relation, fact.value)).fetchone()
        if row:
            merged_ids = sorted(set(json.loads(row["source_event_ids"] or "[]"))
                                | set(fact.source_event_ids))
            # P2-2：条件上下文只在来值非空时才覆盖（防御式，同 P0-4 source 的写法）——
            # 离线/显式等不产 vc 的路径续证时不得抹掉既有条件限定；来值非空 = 同一
            # 事实最新一次观测携带的条件（「雨云不可靠，除非配了 S3 备份」的更新版）
            vc = fact.validity_context or row["validity_context"] or ""
            self.conn.execute(
                "UPDATE semantic SET confidence=?, valid_to='', evidence_count=?, "
                "source_event_ids=?, validity_context=? WHERE id=?",
                (max(fact.confidence, self._read_confidence(row["id"])),
                 row["evidence_count"] + 1, json.dumps(merged_ids), vc, row["id"]))
            self._sync_fts(row["id"], row["status"], fact.value)
            self.conn.commit()
            self.audit("semantic", row["id"], "renew",
                       f"{fact.entity} {fact.relation} {fact.value} evidence+1")
            return row["id"], False
        cur = self.conn.execute(
            "INSERT INTO semantic (entity, relation, value, confidence, valid_from, valid_to, "
            "conflict_note, embedding, status, raw_entity, raw_value, superseded_by, "
            "evidence_count, source_event_ids, hit_count, validity_context) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (fact.entity, fact.relation, fact.value, fact.confidence, fact.valid_from or now_iso(),
             fact.valid_to, fact.conflict_note, json.dumps(fact.embedding), fact.status,
             fact.raw_entity or fact.entity, fact.raw_value or fact.value, fact.superseded_by,
             fact.evidence_count, json.dumps(fact.source_event_ids), fact.hit_count,
             fact.validity_context))
        self._sync_fts(cur.lastrowid, fact.status, fact.value)
        self.conn.commit()
        self.audit("semantic", cur.lastrowid, "create", f"{fact.entity} {fact.relation} {fact.value}")
        return cur.lastrowid, True

    def _fts_text(self, fact_id: int) -> str | None:
        row = self.conn.execute(
            "SELECT value FROM semantic WHERE id=?", (fact_id,)).fetchone()
        return row["value"] if row else None

    def _sync_fts(self, fact_id: int, status: str, value: str | None = None) -> None:
        """让可检索状态的事实在 FTS 索引里有条目（幂等，可重复调用）。

        P0-1：此前写入路径无条件写索引（含 pending / superseded），而 rebuild_fts
        只重建 active，两侧口径不一致，导致「跑一次迁移，检索行为就变一次」。

        维护规则为**实测**结论，详见 episodic._sync_fts 的完整说明：external content
        表的 INSERT 会回表取当前值，单条 DELETE 与 UPDATE 到空条目都会损坏索引、且
        无法自省条目是否存在。故本方法**只做 INSERT 不做删除**；失效条目残留的代价
        由 settings.FTS_CANDIDATE_FACTOR 放大候选池抵消，正确性由 retriever 侧
        status 过滤保证。本方法不 commit，由调用方与业务更新同事务提交。
        """
        text = value if value is not None else self._fts_text(fact_id)
        if text is None:
            return
        if status in settings.RETRIEVABLE_STATUSES:
            self.conn.execute(
                "INSERT INTO semantic_fts (rowid, value) VALUES (?,?)",
                (fact_id, bigram(text)))

    def _read_confidence(self, fact_id: int) -> float:
        row = self.conn.execute("SELECT confidence FROM semantic WHERE id=?", (fact_id,)).fetchone()
        return row["confidence"] if row else 0.0

    def renew_variant(self, fact_id: int, fact: SemanticFact, similarity: float = 0.0) -> None:
        """近义措辞变体续证（V1.6.2 相似度预去重）：新观测与既有事实几乎同义但 value
        字符串不同——不建新行、不挂冲突，只累计证据。与 upsert 的 renew 分支互补
        （renew 按 (entity,relation,value) 精确匹配，这里按嵌入相似度匹配后由调用方触发）：
        evidence_count+1、confidence 取 max、source_event_ids 并集合并（保留溯源）；
        不覆盖 value/embedding/FTS——旧表达继续代表该事实。审计 action=renew_variant，
        detail 含触发相似度，可溯源。
        """
        row = self.conn.execute(
            "SELECT evidence_count, source_event_ids FROM semantic WHERE id=?",
            (fact_id,)).fetchone()
        if not row:
            return
        merged_ids = sorted(set(json.loads(row["source_event_ids"] or "[]"))
                            | set(fact.source_event_ids))
        self.conn.execute(
            "UPDATE semantic SET confidence=?, evidence_count=?, source_event_ids=? WHERE id=?",
            (max(fact.confidence, self._read_confidence(fact_id)),
             row["evidence_count"] + 1, json.dumps(merged_ids), fact_id))
        self.conn.commit()
        self.audit("semantic", fact_id, "renew_variant",
                   f"{fact.entity} {fact.relation} {fact.value} sim={similarity:.3f} evidence+1")

    def find_same_key(self, entity: str, relation: str, value: str) -> SemanticFact | None:
        """同键（(entity, relation, value) 三元组全同）且未失效（active/pending）的既有行。

        B1 跨键吸收的前置探测：命中说明本次写入将走 upsert 的 renew 续证分支，
        不得被跨键吸收劫持（同键路径语义原样，见 conflict_resolver._store_fact）。
        查询口径与 upsert 的已存在分支完全一致。
        """
        row = self.conn.execute(
            "SELECT * FROM semantic WHERE entity=? AND relation=? AND value=? "
            "AND status IN ('active','pending') LIMIT 1",
            (entity, relation, value)).fetchone()
        return self._from_row(row) if row else None

    def absorb_into(self, keeper_id: int, incoming_confidence: float, detail: str = "") -> None:
        """跨键近重复合并（B1 写入侧吸收）：新观测与既有 active 事实几乎同义但落在
        不同的 (entity, relation) 键下——不建新行，证据并入 keeper。与 renew_variant
        （同键近义措辞续证）互补：那按同键相似度匹配，这里跨键、由调用方触发。
        evidence_count+1（一次真实观测）、confidence 取 max；value/embedding/FTS 全部
        不动——keeper 的既有表达继续代表该事实，value 未变则 FTS 索引无需同步。
        审计 action=dedupe_absorb，detail 摘要来值的 entity/relation/value，可溯源
        「这条证据从哪个键并进来」。
        """
        row = self.conn.execute(
            "SELECT evidence_count, confidence FROM semantic WHERE id=?", (keeper_id,)).fetchone()
        if not row:
            return
        self.conn.execute(
            "UPDATE semantic SET confidence=?, evidence_count=? WHERE id=?",
            (max(incoming_confidence, row["confidence"]), row["evidence_count"] + 1, keeper_id))
        self.conn.commit()
        self.audit("semantic", keeper_id, "dedupe_absorb", detail)

    def bump_evidence(self, fact_id: int, delta: int) -> None:
        """证据并入（B1 存量治理）：把被吸收记录的 evidence_count 整体加到 keeper。
        每份记录都是一次独立真实观测，求和合理；evidence 进联想激活 base 有 log 封顶
        （FREQ_ACCESS_CAP），合并不会让激活失控。只动 evidence_count——confidence
        与 status 由 keeper 选择规则与 expire/set_status 各自负责，互不代偿。
        """
        self.conn.execute(
            "UPDATE semantic SET evidence_count=evidence_count+? WHERE id=?", (delta, fact_id))
        self.conn.commit()

    def touch_confidence(self, fact_id: int, confidence: float) -> None:
        """只更新置信度（E6 检索再巩固专用）：不动 evidence_count/embedding/FTS/审计。

        检索不是新证据——evidence_count 只由真实写入观测累积。此前 _reinforce 走
        upsert 续证路径，每次检索除 confidence+0.02 外还 evidence+1，而 evidence 又是
        联想激活 base 的输入，构成「检索放大证据、证据抬高激活」的自增强回路。
        对既有调用方零影响：upsert 语义不变，本方法是纯新增通道。
        """
        self.conn.execute("UPDATE semantic SET confidence=? WHERE id=?", (confidence, fact_id))
        self.conn.commit()

    def bump_hit_count(self, fact_id: int) -> int:
        """试用期命中计数 +1，返回累计值（V1.7 P1 转正通道的度量，D3 取绝对次数）。

        与 evidence_count 严格分开、互不代偿：evidence 是「真实观测到几次」
        （写入侧 renew 累积，进联想激活 base），hit_count 是「被想起几次」
        （检索侧累积，只喂转正判定）。混用会重蹈 E6 的自增强回路——检索放大
        证据、证据抬高激活。故本方法只动 hit_count：不改 confidence（由
        _reinforce 的 touch_confidence 负责）、不动 FTS、不写审计（审计由调用
        方按语义记 probation_hit / promote，便于分口径统计转正漏斗）。
        """
        self.conn.execute("UPDATE semantic SET hit_count=hit_count+1 WHERE id=?", (fact_id,))
        self.conn.commit()
        row = self.conn.execute("SELECT hit_count FROM semantic WHERE id=?", (fact_id,)).fetchone()
        return row["hit_count"] if row else 0

    def demote_confidence(self, fact_id: int, delta: float,
                          floor: float | None = None) -> float | None:
        """置信度按增量下调（带下限保护）：E6 干扰降权（RIF / 线索过载）的共用写通道。

        只动 confidence——不改 status、不删行、不动版本链（valid_from/valid_to）；
        已在下限处不动作（返回 None），抑制可被写入观测（renew 续证）反转。
        """
        if floor is None:
            floor = settings.RIF_CONF_FLOOR
        current = self._read_confidence(fact_id)
        target = max(floor, round(current + delta, 4))
        if target >= current:  # 已在下限：不再动作，保持幂等可审计
            return None
        self.conn.execute("UPDATE semantic SET confidence=? WHERE id=?", (target, fact_id))
        self.conn.commit()
        return target

    def active_facts_of(self, entity: str) -> list[SemanticFact]:
        """某实体（精确匹配，别名归一由调用方负责）的全部 active 事实。"""
        rows = self.conn.execute(
            "SELECT * FROM semantic WHERE entity=? AND status='active' ORDER BY valid_from, id",
            (entity,)).fetchall()
        return [self._from_row(r) for r in rows]

    def apply_cue_overload(self, entity: str) -> list[int]:
        """线索过载降权（E6 写入侧预防）：该实体 active 事实数超过 CUE_OVERLOAD_N 时，
        最旧（valid_from 最早）的 active 事实降 CUE_OVERLOAD_PENALTY，每超一条降一条。

        同一实体挂靠事实过多 = 检索线索区分度下降（cue overload，经典干扰论遗忘），
        先降权最旧者而非删除——confidence 跌到遗忘阈值以下自然走归档管线。
        审计记 cue_overload，可溯源。返回被降权的事实 id 列表（未过载返回空）。
        """
        facts = self.active_facts_of(entity)
        excess = len(facts) - settings.CUE_OVERLOAD_N
        demoted: list[int] = []
        for fact in facts[:max(0, excess)]:
            target = self.demote_confidence(fact.id, -settings.CUE_OVERLOAD_PENALTY)
            if target is not None:
                demoted.append(fact.id)
                self.audit("semantic", fact.id, "cue_overload",
                           f"{entity} active={len(facts)}>{settings.CUE_OVERLOAD_N} "
                           f"{fact.confidence:.2f}->{target:.2f}")
        return demoted


    def find_conflicts(self, entity: str, relation: str, value: str) -> list[SemanticFact]:
        """同一实体同一关系下，value 不同的活跃事实（即潜在冲突）。"""
        rows = self.conn.execute(
            "SELECT * FROM semantic WHERE entity=? AND relation=? AND value<>? AND status='active'",
            (entity, relation, value)).fetchall()
        return [self._from_row(r) for r in rows]

    def expire(self, fact_id: int, note: str = "", superseded_by: int = 0,
               status: str = "archived") -> None:
        """事实失效：记录失效时间与原因；被取代时 status='superseded' 并指向新版本。"""
        self.conn.execute(
            "UPDATE semantic SET valid_to=?, conflict_note=?, status=?, superseded_by=? WHERE id=?",
            (now_iso(), note, status, superseded_by, fact_id))
        self._sync_fts(fact_id, status)  # 必须在 commit 之前，否则索引行残留
        self.conn.commit()
        self.audit("semantic", fact_id, status, note)

    def set_status(self, fact_id: int, status: str) -> None:
        self.conn.execute("UPDATE semantic SET status=? WHERE id=?", (status, fact_id))
        self._sync_fts(fact_id, status)  # 状态进出可检索集合时，索引行随之增删
        self.conn.commit()
        self.audit("semantic", fact_id, f"status->{status}")

    def fetch_history(self, entity: str, relation: str | None = None) -> list[SemanticFact]:
        """某实体（可选关系）的全部版本，按生效时间排序（含 superseded/archived/pending）。"""
        if relation:
            rows = self.conn.execute(
                "SELECT * FROM semantic WHERE entity=? AND relation=? ORDER BY valid_from, id",
                (entity, relation)).fetchall()
        else:
            rows = self.conn.execute(
                "SELECT * FROM semantic WHERE entity=? ORDER BY relation, valid_from, id",
                (entity,)).fetchall()
        return [self._from_row(r) for r in rows]

    def get(self, fact_id: int) -> SemanticFact | None:
        row = self.conn.execute("SELECT * FROM semantic WHERE id=?", (fact_id,)).fetchone()
        return self._from_row(row) if row else None

    def fetch(self, status: str = "active", limit: int = 2000) -> list[SemanticFact]:
        rows = self.conn.execute(
            "SELECT * FROM semantic WHERE status=? ORDER BY id DESC LIMIT ?", (status, limit)).fetchall()
        return [self._from_row(r) for r in rows]

    def update_embedding(self, fact_id: int, embedding: list[float]) -> None:
        """仅更新嵌入（回填/换嵌入模型用）；upsert 的已存在分支不覆盖 embedding。"""
        self.conn.execute("UPDATE semantic SET embedding=? WHERE id=?",
                          (json.dumps(embedding), fact_id))
        self.conn.commit()

    def update_entity(self, fact_id: int, entity: str, relation: str, raw_entity: str) -> None:
        """实体归一迁移专用：只改实体标识，不动 value/valid 期（FTS 索引 value 无需更新）。"""
        self.conn.execute(
            "UPDATE semantic SET entity=?, relation=?, raw_entity=? WHERE id=?",
            (entity, relation, raw_entity, fact_id))
        self.conn.commit()
        self.audit("semantic", fact_id, "entity-resolved", f"->{entity}")

    def search_fts(self, query: str, limit: int = 20) -> list[tuple[int, float]]:
        tokens = [t.replace('"', "") for t in bigram(query).split() if t.strip()]
        q = " OR ".join(f'"{t}"' for t in tokens if t)
        if not q:
            return []
        rows = self.conn.execute(
            "SELECT rowid, bm25(semantic_fts) AS score FROM semantic_fts "
            "WHERE semantic_fts MATCH ? ORDER BY score LIMIT ?", (q, limit)).fetchall()
        return [(r["rowid"], r["score"]) for r in rows]

    def cosine_search(self, vec: list[float], limit: int = 20) -> list[tuple[int, float]]:
        # P0-1：候选集口径与 FTS 索引统一（见 episodic.cosine_search 的同名说明）
        pool: list = []
        for status in settings.RETRIEVABLE_STATUSES:
            pool.extend(self.fetch(status=status, limit=5000))
        results = []
        for fact in pool:
            if fact.embedding and len(fact.embedding) == len(vec):
                results.append((fact.id, cosine(vec, fact.embedding)))
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:limit]

    @staticmethod
    def _from_row(row) -> SemanticFact:
        keys = row.keys()
        return SemanticFact(
            id=row["id"], entity=row["entity"], relation=row["relation"], value=row["value"],
            confidence=row["confidence"], valid_from=row["valid_from"], valid_to=row["valid_to"],
            conflict_note=row["conflict_note"], embedding=json.loads(row["embedding"] or "[]"),
            status=row["status"],
            raw_entity=row["raw_entity"] if "raw_entity" in keys else "",
            raw_value=row["raw_value"] if "raw_value" in keys else "",
            superseded_by=row["superseded_by"] if "superseded_by" in keys else 0,
            evidence_count=row["evidence_count"] if "evidence_count" in keys else 1,
            source_event_ids=json.loads(row["source_event_ids"]) if "source_event_ids" in keys and row["source_event_ids"] else [],
            hit_count=row["hit_count"] if "hit_count" in keys else 0,
            # P2-2 防御式读取（同 P0-4）：列不存在（旧结构行）按空串 = 无条件限定
            validity_context=row["validity_context"] if "validity_context" in keys else "")
