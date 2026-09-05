"""冲突消解：写入时即判定，同实体同关系不同取值按来源分级处理。

策略（错不如旧——自动覆盖只发生在高可信来源）：
- explicit（用户显式指令/偏好声明）          -> 新版生效，旧版 superseded（保留可查）
- model 且 confidence >= 阈值(默认 0.8)      -> 同上
- model 低置信 / consolidation（巩固期蒸馏） -> 新版置 pending 不参与检索，记 memory_conflict 待裁决
- 无冲突或同值重复                           -> 正常写入/累计证据

返回 outcome dict 供 CLI 与测试断言。
"""
from __future__ import annotations

from memagent import settings
from memagent.core.domain import SemanticFact
from memagent.core.vectors import cosine
from memagent.storage import SqliteStore


def dedupe_same_key(facts: list[SemanticFact]) -> tuple[list[SemanticFact], list[SemanticFact]]:
    """同一事件/同一簇抽出的多条同 (entity, relation) 事实只保留置信度最高的一条。

    防止一次输入内部的自我取代链（A 取代旧版后，同事件的 B 又立刻取代 A）。
    被跳过的互补信息仍保留在情景记忆原文中。
    """
    best: dict[tuple[str, str], SemanticFact] = {}
    skipped: list[SemanticFact] = []
    for f in facts:
        key = (f.entity, f.relation)
        if key not in best:
            best[key] = f
        elif f.confidence > best[key].confidence:
            skipped.append(best[key])
            best[key] = f
        else:
            skipped.append(f)
    return list(best.values()), skipped


def store_fact_with_conflict_check(store: SqliteStore, fact: SemanticFact,
                                   source: str = "model") -> dict:
    """写入一条语义事实并处理冲突。source: explicit | model | consolidation。

    E6 线索过载（写入侧预防）：落库后检查该实体 active 事实数，超过
    CUE_OVERLOAD_N 时最旧者降权——同一实体挂靠事实过多意味着检索线索
    区分度下降，先降权最旧者（每超一条降一条），不删除、不动版本链。
    评测场景每实体事实数远低于阈值，此机制靠单元测试背书。
    """
    result = _store_fact(store, fact, source)
    store.semantic.apply_cue_overload(fact.entity)
    return result


def _find_absorb_keeper(store: SqliteStore, fact: SemanticFact) -> SemanticFact | None:
    """B1 跨键查重：取与来值嵌入余弦最高且 ≥ DEDUP_ABSORB_SIM 的 active 既有事实。

    只吸收进 active——pending 是证据未足的试用期事实（或冲突待裁的 A 类），不配当
    keeper，也不该被写入侧悄悄壮大。cosine_search 候选池按 RETRIEVABLE_STATUSES
    取数（含 pending），故命中后回查 status 复核。结果按相似度降序，跌破阈值即可
    停止；嵌入为空/维度不齐时 cosine 返回 0.0，天然落在阈值之下（保守：漏并 >
    误并，错合并会破坏冲突消解语义）。
    """
    if not fact.embedding:
        return None
    for cand_id, sim in store.semantic.cosine_search(fact.embedding, limit=5):
        if sim < settings.DEDUP_ABSORB_SIM:
            break
        cand = store.semantic.get(cand_id)
        if cand is not None and cand.status == "active":
            return cand
    return None


def _store_fact(store: SqliteStore, fact: SemanticFact, source: str = "model") -> dict:
    conflicts = store.semantic.find_conflicts(fact.entity, fact.relation, fact.value)

    if not conflicts:
        # B1 跨键近重复合并（写入侧防线）：仅当本次写入确定要"新建行"时才跨键查重——
        # 同键已有同值 active/pending 行（将走 upsert 续证）时完全不动；同键不同值的
        # 冲突分支在上面已处理。命中吸收则不建新行，证据并入 keeper；未命中走原
        # create 路径，行为零变化。
        # 来值必须以 active 身份入库才参与吸收：pending 写入（B 类试用期注入 /
        # 低置信蒸馏）的证据不足是**裁决过的状态**，吸收会把它偷换成正职证据抬高
        # keeper，且试用期事实必须保留独立行攒 hit_count（转正通道挂在自己身上）。
        if (fact.status == "active"
                and store.semantic.find_same_key(fact.entity, fact.relation, fact.value) is None):
            keeper = _find_absorb_keeper(store, fact)
            if keeper is not None:
                store.semantic.absorb_into(
                    keeper.id, fact.confidence,
                    detail=f"{fact.entity} {fact.relation} {fact.value} "
                           f"<- keeper#{keeper.id} {keeper.entity}/{keeper.relation}")
                return {"fact_id": keeper.id, "action": "dedupe_absorbed",
                        "superseded": [], "conflict_ids": []}
        fact_id, created = store.semantic.upsert(fact)
        return {"fact_id": fact_id, "action": "created" if created else "renewed",
                "superseded": [], "conflict_ids": []}

    can_supersede = (
        source == "explicit"
        or (source == "model" and fact.confidence >= settings.CONFLICT_AUTO_SUPERSEDE_CONFIDENCE)
    )

    if can_supersede:
        fact_id, created = store.semantic.upsert(fact)
        if not created:
            # 同值行已存在（active/pending）-> 这次写入只是续证，不是"新版"。
            # 不 expire 冲突方：双方关系已由既有状态定义——同值行 active 说明它是
            # both 裁决后的共存方（一次重复观测无权撤销用户裁决），同值行 pending
            # 则本就有待裁冲突在队列里（轮不到这次写入抢跑裁决）。
            return {"fact_id": fact_id, "action": "renewed",
                    "superseded": [], "conflict_ids": []}
        for old in conflicts:
            store.semantic.expire(old.id, note=f"superseded->{fact.value}",
                                  superseded_by=fact_id, status="superseded")
        return {"fact_id": fact_id, "action": "superseded",
                "superseded": [c.id for c in conflicts], "conflict_ids": []}

    # 相似度预去重（V1.6.2 根因 B）：近义不同措辞的蒸馏产物不再按字符串不等挂冲突。
    # 位置在 can_supersede 判定之后：explicit/高置信来源已在上面分支处理完毕——
    # 用户显式声明的原话必须按取代语义处理，不能被相似度吞掉。
    # 嵌入任一为空或维度不匹配时 cosine 返回 0.0，天然跳过该条（保守：漏并 > 误并）。
    best_old, best_sim = None, 0.0
    for old in conflicts:
        sim = cosine(fact.embedding, old.embedding)
        if sim >= settings.FACT_DEDUP_SIMILARITY and sim > best_sim:
            best_old, best_sim = old, sim
    if best_old is not None:
        store.semantic.renew_variant(best_old.id, fact, similarity=best_sim)
        return {"fact_id": best_old.id, "action": "renewed_variant",
                "superseded": [], "conflict_ids": []}

    # 低可信来源：不覆盖旧版，新版挂起待裁决
    fact.status = "pending"
    fact_id, created = store.semantic.upsert(fact)
    if not created:
        # 根因 A（V1.6.2）：upsert 命中已有同值行走了 renew 分支——续证不是新事实，
        # 绝不对冲突方再建 memory_conflict 行（否则用户 both 裁决后每轮巩固都会
        # 把同一对事实重新挂上冲突，队列反复增长）。真实库证据：冲突 #6 在 #5
        # 裁决 both 后 5 分钟由一次巩固的 renew 重复产生。
        return {"fact_id": fact_id, "action": "renewed",
                "superseded": [], "conflict_ids": []}
    conflict_ids = [
        store.conflicts.create(old.id, fact_id, "value_conflict",
                               f"{old.value!r} vs {fact.value!r}")
        for old in conflicts
    ]
    return {"fact_id": fact_id, "action": "pending",
            "superseded": [], "conflict_ids": conflict_ids}


def resolve_conflict(store: SqliteStore, conflict_id: int, resolution: str) -> dict | None:
    """裁决一条 pending 冲突。resolution: accept-new | keep-old | both。

    both = 判定为误报（互补偏好被粗粒度键误配）：双方共存——新事实转 active
    但不取代旧版（无 supersede 链接，版本链不动）。适用于"收藏什么"与"怎么
    存放"这类不同 aspect 的事实：两条同时 active、同时可检索才是正确状态；
    后续同键再写入时，两条 active 都会作为冲突方正常参与消解。
    """
    row = store.conflicts.get(conflict_id)
    if not row or row["status"] != "pending":
        return None

    if resolution == "accept-new":
        new_fact = store.semantic.get(row["new_id"])
        if new_fact:
            store.semantic.set_status(row["new_id"], "active")
        old = store.semantic.get(row["old_id"])
        if old:
            store.semantic.expire(old.id, note=f"superseded->{old.value}",
                                  superseded_by=row["new_id"], status="superseded")
    elif resolution == "keep-old":
        store.semantic.set_status(row["new_id"], "archived")
    elif resolution == "both":
        new_fact = store.semantic.get(row["new_id"])
        if new_fact:
            store.semantic.set_status(row["new_id"], "active")
            store.log("semantic", row["new_id"], "coexist",
                      f"conflict #{conflict_id} 判定误报，与 #{row['old_id']} 共存")
    else:
        raise ValueError(f"未知裁决: {resolution}")

    store.conflicts.mark_resolved(conflict_id, resolution)
    return row
