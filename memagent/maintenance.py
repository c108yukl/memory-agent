"""维护任务：嵌入回填（B5）与实体归一迁移（B2）。

约定：默认 dry-run 只出报告；apply 前自动备份并写 migration_log。
"""
from __future__ import annotations

import json

from memagent import settings
from memagent.adapters import llm
from memagent.consolidation.cluster import _DSU
from memagent.core.text import bigram
from memagent.core.vectors import cosine
from memagent.encoding.entity_resolver import normalize_entity, normalize_relation
from memagent.storage import SqliteStore
from memagent.storage.sqlite_repo import FTS_TABLE_DDL


def _dim_stats(vectors: list[list[float]]) -> dict[int, int]:
    stats: dict[int, int] = {}
    for v in vectors:
        stats[len(v)] = stats.get(len(v), 0) + 1
    return stats


def reembed(store: SqliteStore, dry_run: bool = True, batch_size: int = 50) -> dict:
    """用当前嵌入服务重算全部 active 记忆的向量，统一维度。

    episodic 以 summary 为嵌入文本（与 FTS 检索面一致），
    semantic 以 "entity relation value" 为嵌入文本（与抽取时一致）。
    """
    episodes = store.episodic.fetch(status="active", limit=10 ** 9)
    facts = store.semantic.fetch(status="active", limit=10 ** 9)

    report = {
        "dry_run": dry_run,
        "episodic_total": len(episodes),
        "semantic_total": len(facts),
        "episodic_dims": _dim_stats([m.embedding for m in episodes]),
        "semantic_dims": _dim_stats([f.embedding for f in facts]),
        "embedded": 0, "failed": [],
    }
    if dry_run:
        return report

    backup_path = store.backup("reembed")
    done = 0
    for batch_start in range(0, len(episodes), batch_size):
        for mem in episodes[batch_start:batch_start + batch_size]:
            try:
                mem.embedding = llm.embed(mem.summary)
                store.episodic.update(mem)
                done += 1
            except Exception as e:
                report["failed"].append({"kind": "episodic", "id": mem.id, "error": str(e)})
    for batch_start in range(0, len(facts), batch_size):
        for fact in facts[batch_start:batch_start + batch_size]:
            try:
                store.semantic.update_embedding(
                    fact.id, llm.embed(f"{fact.entity} {fact.relation} {fact.value}"))
                done += 1
            except Exception as e:
                report["failed"].append({"kind": "semantic", "id": fact.id, "error": str(e)})

    report["embedded"] = done
    # 回填后校验：维度一致、无空嵌入
    eps2 = store.episodic.fetch(status="active", limit=10 ** 9)
    fac2 = store.semantic.fetch(status="active", limit=10 ** 9)
    report["episodic_dims_after"] = _dim_stats([m.embedding for m in eps2])
    report["semantic_dims_after"] = _dim_stats([f.embedding for f in fac2])
    report["dim_consistent"] = (
        len(report["episodic_dims_after"]) <= 1 and len(report["semantic_dims_after"]) <= 1
        and 0 not in report["episodic_dims_after"] and 0 not in report["semantic_dims_after"])

    dim = next(iter(report["episodic_dims_after"]), 0)
    store.log("system", 0, "reembed", f"model=cloud embed_dim={dim} embedded={done}")
    store.log_migration("reembed", "ok" if report["dim_consistent"] else "warn",
                        affected_rows=done, backup_path=backup_path,
                        detail=json.dumps(report["failed"][:20], ensure_ascii=False))
    report["backup_path"] = backup_path
    return report


def rebuild_fts(store: SqliteStore, dry_run: bool = True) -> dict:
    """存量 FTS 索引重建为中文 2-gram 分词（FTS5 默认整句成 token，中文检索失效）。

    P0-1：重建口径统一走 settings.RETRIEVABLE_STATUSES——此前硬编码 "active"，而写入
    路径无条件索引（含 archived / superseded / deleted），两侧分歧导致「跑一次迁移，
    检索行为变一次」。现在写入路径（仓储 _sync_fts）与本函数共用同一状态集合，重建幂等。
    """
    episodes: list = []
    facts: list = []
    for status in settings.RETRIEVABLE_STATUSES:
        episodes.extend(store.episodic.fetch(status=status, limit=10 ** 9))
        facts.extend(store.semantic.fetch(status=status, limit=10 ** 9))
    report = {"dry_run": dry_run, "episodic": len(episodes), "semantic": len(facts),
              "statuses": list(settings.RETRIEVABLE_STATUSES)}
    if dry_run:
        return report

    backup_path = store.backup("rebuild-fts")
    # 重建走 DROP + CREATE 而非 DELETE 全表：后者在主表存在无索引条目的行时会损坏
    # 索引（external content 表实测行为，详见 storage.sqlite_repo.FTS_TABLE_DDL 注释）。
    for table in ("episodic_fts", "semantic_fts"):
        store.conn.execute(f"DROP TABLE IF EXISTS {table}")
        store.conn.execute(FTS_TABLE_DDL[table])
    for m in episodes:
        store.conn.execute("INSERT INTO episodic_fts (rowid, summary) VALUES (?,?)",
                           (m.id, bigram(m.summary)))
    for f in facts:
        store.conn.execute("INSERT INTO semantic_fts (rowid, value) VALUES (?,?)",
                           (f.id, bigram(f.value)))
    store.conn.commit()
    store.log_migration("rebuild-fts", "ok",
                        affected_rows=len(episodes) + len(facts), backup_path=backup_path)
    report["backup_path"] = backup_path
    return report


def resolve_entities(store: SqliteStore, dry_run: bool = True) -> dict:
    """把库中未归一的实体/关系按当前规则与别名表归一（幂等，可重复执行）。

    raw_* 保留原值；合并后若出现完全同 (entity,relation,value) 的重复 active，
    保留小 id、多余者归档（同 upsert 去重语义）。同 (entity,relation) 不同 value
    的潜在冲突只统计不处理（B1 的冲突消解负责）。
    """
    aliases = store.aliases.as_map()
    facts = store.semantic.fetch(status="active", limit=10 ** 9)

    changes: list[dict] = []
    for fact in facts:
        new_entity = normalize_entity(fact.entity, aliases)
        new_relation = normalize_relation(fact.relation)
        if new_entity != fact.entity or new_relation != fact.relation:
            changes.append({"id": fact.id, "entity": fact.entity, "relation": fact.relation,
                            "new_entity": new_entity, "new_relation": new_relation})

    report = {"dry_run": dry_run, "aliases": len(aliases), "changes": changes,
              "affected": len(changes)}
    if dry_run:
        return report

    backup_path = store.backup("resolve-entities")
    for ch in changes:
        fact = store.semantic.get(ch["id"])
        if fact is None:
            continue
        store.semantic.update_entity(
            ch["id"], ch["new_entity"], ch["new_relation"],
            raw_entity=fact.raw_entity or fact.entity)

    # 归一可能产生的精确重复：保留小 id
    seen: dict[tuple, int] = {}
    merged = 0
    for fact in store.semantic.fetch(status="active", limit=10 ** 9):
        key = (fact.entity, fact.relation, fact.value)
        if key in seen:
            store.semantic.expire(fact.id, note=f"merged->{seen[key]}")
            merged += 1
        else:
            seen[key] = fact.id
    report["merged_duplicates"] = merged

    store.log_migration("resolve-entities", "ok", affected_rows=len(changes) + merged,
                        backup_path=backup_path, detail=f"merged={merged}")
    report["backup_path"] = backup_path
    return report


# P0-3 存量垃圾清理允许的动作键：plan 里出现之外的键一律报错（防手滑写错清单）。
_CLEANUP_ACTION_KEYS = ("archive_episodic", "fix_episodic_category",
                        "archive_semantic", "archive_procedural")


def _normalize_cleanup_plan(plan: dict) -> dict:
    """校验并归一清理清单：未知动作键、id 类型错误都清晰报错（宁炸不猜）。

    归一规则：
    - 三个归档键的 id 必须是 int（bool 也是 int 的子类，显式拒绝）；
    - fix_episodic_category 的键经 JSON 往来必是字符串（"25"），须能转 int；
      目标 category 必须是 str（''=普通层是合法目标）；
    - 列表内重复 id 去重保序（同一行归档两次只会审计一次，报告才诚实）。
    """
    if not isinstance(plan, dict):
        raise TypeError(f"plan 须为 dict 清单，收到 {type(plan).__name__}")
    unknown = sorted(set(plan) - set(_CLEANUP_ACTION_KEYS))
    if unknown:
        raise ValueError(f"plan 含未知动作键 {unknown}，允许的键: {list(_CLEANUP_ACTION_KEYS)}")

    normalized: dict = {}
    for key in ("archive_episodic", "archive_semantic", "archive_procedural"):
        ids = plan.get(key) or []
        if not isinstance(ids, (list, tuple)):
            raise TypeError(f"plan[{key!r}] 须为 id 列表，收到 {type(ids).__name__}: {ids!r}")
        clean: list[int] = []
        for i in ids:
            if isinstance(i, bool) or not isinstance(i, int):
                raise TypeError(f"plan[{key!r}] 含非整数 id: {i!r}")
            if i not in clean:
                clean.append(i)
        normalized[key] = clean

    fix = plan.get("fix_episodic_category") or {}
    if not isinstance(fix, dict):
        raise TypeError(f"plan['fix_episodic_category'] 须为 {{id: 目标category}} dict，"
                        f"收到 {type(fix).__name__}: {fix!r}")
    clean_fix: dict[int, str] = {}
    for k, v in fix.items():
        try:
            mem_id = int(k)
        except (TypeError, ValueError):
            raise ValueError(f"plan['fix_episodic_category'] 键须为可转整数的 id，收到 {k!r}") from None
        if not isinstance(v, str):
            raise TypeError(f"plan['fix_episodic_category'][{k!r}] 的目标 category 须为字符串，"
                            f"收到 {v!r}")
        clean_fix[mem_id] = v
    normalized["fix_episodic_category"] = clean_fix
    return normalized


def cleanup_garbage(store: SqliteStore, plan: dict, apply: bool = False) -> dict:
    """P0-3 存量垃圾清理：按显式清单归档误入库的垃圾行、修正情景 category。

    plan 是显式清单 dict（四个动作键全部可选；未知键 / id 类型错误直接报错）：
      archive_episodic:       [id...]              情景记忆 status -> archived
      fix_episodic_category:  {"id": 目标category}  情景 category 修正（JSON 键为字符串）
      archive_semantic:       [id...]              语义事实 status -> archived
      archive_procedural:     [id...]              技能 status -> archived

    dry-run（默认）只逐条核对目标行现状（id / 当前 status·category / 目标值），
    不写任何东西；行不存在或已处于目标状态 -> skipped 带原因，不算错误。
    apply=True 先 store.backup() 热备份一次（整个命令一次，只在确有变更时备，
    幂等重跑全 skipped 不产生无谓备份文件），再逐条执行；每条变更写审计
    （action=cleanup_garbage，detail 含 旧值->新值 + 计划来源 P0-3）。
    幂等：同一清单重复 apply，第二次全 skipped 零报错。

    FTS 不手工动：归档一律走各仓储既有 set_status（episodic/semantic 的 _sync_fts
    对不可检索状态本就不插索引、external content 表也禁止删索引——归档行的残留
    索引条目由检索侧 status 过滤兜底，V1.7 P0-1 三铁律）；category 修正走
    get -> 改 category -> update（update 内部按行当前 status 幂等同步 FTS）。
    """
    plan = _normalize_cleanup_plan(plan)
    changes: list[dict] = []
    skipped: list[dict] = []

    # 阶段一（只读）：逐条核对现状，产出将执行的变更与 skipped 明细
    for mem_id in plan["archive_episodic"]:
        mem = store.episodic.get(mem_id)
        if mem is None:
            skipped.append({"kind": "episodic", "id": mem_id, "reason": "行不存在"})
        elif mem.status == "archived":
            skipped.append({"kind": "episodic", "id": mem_id, "reason": "已处于目标状态 archived"})
        else:
            changes.append({"kind": "episodic", "id": mem_id, "action": "archive",
                            "current_status": mem.status, "current_category": mem.category,
                            "target": "archived"})
    for mem_id, target_cat in plan["fix_episodic_category"].items():
        mem = store.episodic.get(mem_id)
        if mem is None:
            skipped.append({"kind": "episodic", "id": mem_id, "reason": "行不存在"})
        elif mem.category == target_cat:
            skipped.append({"kind": "episodic", "id": mem_id,
                            "reason": f"category 已为 {target_cat!r}，无需修正"})
        else:
            changes.append({"kind": "episodic", "id": mem_id, "action": "fix_category",
                            "current_status": mem.status, "current_category": mem.category,
                            "target": target_cat})
    for fact_id in plan["archive_semantic"]:
        fact = store.semantic.get(fact_id)
        if fact is None:
            skipped.append({"kind": "semantic", "id": fact_id, "reason": "行不存在"})
        elif fact.status == "archived":
            skipped.append({"kind": "semantic", "id": fact_id, "reason": "已处于目标状态 archived"})
        else:
            changes.append({"kind": "semantic", "id": fact_id, "action": "archive",
                            "current_status": fact.status, "target": "archived"})
    for skill_id in plan["archive_procedural"]:
        skill = store.procedural.get(skill_id)
        if skill is None:
            skipped.append({"kind": "procedural", "id": skill_id, "reason": "行不存在"})
        elif skill.status == "archived":
            skipped.append({"kind": "procedural", "id": skill_id, "reason": "已处于目标状态 archived"})
        else:
            changes.append({"kind": "procedural", "id": skill_id, "action": "archive",
                            "current_status": skill.status, "target": "archived"})

    report = {
        "dry_run": not apply,
        "changes": changes,
        "skipped": skipped,
        "archived": sum(1 for c in changes if c["action"] == "archive"),
        "fix_category": sum(1 for c in changes if c["action"] == "fix_category"),
        "skipped_count": len(skipped),
    }
    if not apply or not changes:
        return report

    # 阶段二：热备份一次（整个命令一次），再逐条执行。备份在阶段一之后、
    # 任何写入之前，捕获的是变更前的完整状态。
    backup_path = store.backup("cleanup-garbage")
    for ch in changes:
        if ch["action"] == "archive":
            repo = {"episodic": store.episodic, "semantic": store.semantic,
                    "procedural": store.procedural}[ch["kind"]]
            repo.set_status(ch["id"], "archived")
            store.log(ch["kind"], ch["id"], "cleanup_garbage",
                      f"status: {ch['current_status']}->archived（计划来源 P0-3）")
        else:  # fix_category：get 后改 category 再 update（update 内部幂等同步 FTS）
            mem = store.episodic.get(ch["id"])
            if mem is None:  # 理论不可达（changes 来自同调用内的现状核对），防御兜底
                continue
            mem.category = ch["target"]
            store.episodic.update(mem)
            store.log("episodic", ch["id"], "cleanup_garbage",
                      f"category: {ch['current_category']!r}->{ch['target']!r}（计划来源 P0-3）")
    report["backup_path"] = backup_path
    return report


def dedupe_facts(store: SqliteStore, dry_run: bool = True) -> dict:
    """存量语义事实跨键近重复合并（B1）：value 嵌入两两余弦 ≥ DEDUP_ABSORB_SIM 的
    active 事实按并查集归簇（复用 consolidation/cluster._DSU 的连通分量语义——
    传递相似归一簇，与巩固聚类同一把尺子），每组保留一份、其余并入。

    keeper 选择（evidence_count 降序 → confidence 降序 → id 升序）：证据最多者优先，
    同票取置信高者，再同票取最老（id 最小）——最老最稳者留。被并者走 expire 置
    superseded_by=keeper.id + status='superseded'（与冲突消解的取代路径同一出口，
    _sync_fts 自动同步索引，不手动碰 FTS）；keeper.evidence_count 整体并入被吸收者
    各自的计数（每份记录都是一次独立真实观测，求和合理，evidence 进联想 base 有
    log 封顶）。pending 不参与（试用期事实不是平行堆积，由转正/遗忘通道管）。
    审计 dedupe_merge 带组摘要，dry-run 只出报告不改库。
    """
    facts = store.semantic.fetch(status="active", limit=10 ** 9)
    n = len(facts)
    dsu = _DSU(n)
    for i in range(n):
        for j in range(i + 1, n):
            a, b = facts[i], facts[j]
            if a.embedding and b.embedding and \
                    cosine(a.embedding, b.embedding) >= settings.DEDUP_ABSORB_SIM:
                dsu.union(i, j)

    components: dict[int, list] = {}
    for i, f in enumerate(facts):
        components.setdefault(dsu.find(i), []).append(f)

    groups: list[dict] = []
    for members in components.values():
        if len(members) < 2:
            continue
        members.sort(key=lambda f: (-f.evidence_count, -f.confidence, f.id))
        keeper, absorbed = members[0], members[1:]
        absorbed.sort(key=lambda f: f.id)  # 报告按 id 升序定序（证据求和与顺序无关）
        groups.append({
            "keeper": {"id": keeper.id, "entity": keeper.entity, "relation": keeper.relation,
                       "value": keeper.value, "evidence_count": keeper.evidence_count},
            "absorbed": [{"id": a.id, "entity": a.entity, "relation": a.relation,
                          "value": a.value, "evidence_count": a.evidence_count}
                         for a in absorbed],
        })
    affected = sum(len(g["absorbed"]) for g in groups)
    report = {"dry_run": dry_run, "total_active": n, "groups": groups, "affected": affected}
    if dry_run:
        return report

    backup_path = store.backup("dedupe-facts")
    for g in groups:
        keeper_id, merged_evidence = g["keeper"]["id"], 0
        for a in g["absorbed"]:
            store.semantic.expire(a["id"], note=f"deduped->{keeper_id}",
                                  superseded_by=keeper_id, status="superseded")
            merged_evidence += a["evidence_count"]
        store.semantic.bump_evidence(keeper_id, merged_evidence)
        store.log("semantic", keeper_id, "dedupe_merge",
                  f"组 keeper#{keeper_id} 并入 "
                  f"{[a['id'] for a in g['absorbed']]}，evidence+{merged_evidence}")
    store.log_migration("dedupe-facts", "ok", affected_rows=affected,
                        backup_path=backup_path,
                        detail=f"groups={len(groups)} keepers={[g['keeper']['id'] for g in groups]}")
    report["backup_path"] = backup_path
    return report
