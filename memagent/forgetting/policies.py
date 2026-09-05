"""主动遗忘策略：强度重算（时间内生衰减）+ 归档 + 摘要降级/硬删。

硬删候选的处理顺序（保守）：
1. LLM 不可用（离线模式）-> 维持旧行为直接硬删；
2. 在线：可聚成簇（≥2 条相似、跨度合理、摘要成功）-> 降级为摘要记忆，原始记忆标 summarized 保留；
3. 落单的候选 -> 硬删；
4. 摘要失败的簇 -> 保持 archived 原状，绝不写半成品。

V1.7 P1：新增试用期（pending 语义事实）出口——长期无人问津者归档。此前转正通道
不通，pending 事实既不参与检索也不会被清理，只进不出；接通后必须配套出口，否则
试用期的垃圾会永久占位。语义层无时间衰减（见 ARCHITECTURE §1.1），故这里独立
计时，不复用强度模型。
"""
from __future__ import annotations

from memagent import settings
from memagent.adapters import llm
from memagent.core.clock import iso_to_ts, now_ts
from memagent.forgetting.decay import recompute_strength
from memagent.forgetting.degrade import cluster_candidates, summarize_cluster
from memagent.storage import SqliteStore


def _forget_threshold(mem) -> float:
    """E8 分层归档阈值：经验层用更严的尺子（LFU 低频优先淘汰）。

    绿色类型走门控保底 importance=0.50，强度地板 ≈0.30，普通 0.15 阈值永远
    够不着——经验层必须按自己的半衰与阈值淘汰（0 次访问约 14 天沉底，
    访问续命靠频率项与 last_access 新近度）。
    """
    if getattr(mem, "category", "") == "experience":
        return settings.EXPERIENCE_FORGET_THRESHOLD
    return settings.FORGET_THRESHOLD


def archive_stale_probation(store: SqliteStore) -> int:
    """试用期事实的遗忘侧出口：入库超过 PROBATION_MAX_DAYS 且**一次都没被命中**
    的 pending 事实归档（复用既有 set_status + 审计，不新增通道）。

    为什么只清 hit_count == 0 的：已被想起过的事实说明有人需要它，只是还差次数
    （慢热不等于没用），清掉等于惩罚「被用过但用得少」；而一次都没被想起的才是
    「长期无人问津」。命中不重置计时——重置会让时间上限形同虚设，_pending 事实
    只要偶尔蹭到一次检索就能无限续期。

    为什么不在强度模型里做：语义层无时间衰减（ARCHITECTURE §1.1），confidence
    只由证据/干扰演化，把「年龄」塞进强度模型会污染可解释性。

    冲突待裁型（A 类）同样适用：裁决不该被遗忘抢跑——但归档不等于删除，行仍在
    库里、版本链与冲突行都完整，裁决仍可把它捞回来。
    """
    if settings.PROBATION_MAX_DAYS <= 0:
        return 0
    cutoff = now_ts() - settings.PROBATION_MAX_DAYS * 86400.0
    archived = 0
    for fact in store.semantic.fetch(status="pending", limit=10000):
        if fact.hit_count > 0:
            continue
        if iso_to_ts(fact.valid_from) > cutoff:
            continue
        store.semantic.set_status(fact.id, "archived")
        store.log("semantic", fact.id, "forgot->archive",
                  f"试用期超 {settings.PROBATION_MAX_DAYS} 天无人问津")
        archived += 1
    return archived


def run_forgetting(store: SqliteStore) -> dict:
    report = {"archived": 0, "deleted": 0, "episodic_active": 0, "summarized": 0}
    # 试用期出口独立调用、不并入 report：report 是 test_regression 锁定的迁移快照
    # （结构变更即破黄金值）。归档结果走审计（forgot->archive），与 episodic 遗忘
    # 同口径可溯源；健康报告侧汇总留给 V1.7 Phase 3。
    archive_stale_probation(store)

    for mem in store.episodic.fetch(status="active", limit=10000):
        mem.strength = recompute_strength(mem)
        store.episodic.update(mem)
        if mem.strength < _forget_threshold(mem):
            store.episodic.set_status(mem.id, "archived")
            store.log("episodic", mem.id, "forgot->archive",
                      f"strength={mem.strength} category={mem.category or 'normal'}")
            report["archived"] += 1
        else:
            report["episodic_active"] += 1

    archived = store.episodic.fetch(status="archived", limit=10000)
    candidates = []
    for mem in archived:
        mem.strength = recompute_strength(mem)
        store.episodic.update(mem)
        if mem.strength < settings.ARCHIVE_THRESHOLD:
            candidates.append(mem)

    if not candidates:
        return report

    if not llm.llm_available():
        for mem in candidates:
            store.episodic.set_status(mem.id, "deleted")
            store.log("episodic", mem.id, "forgot->deleted", f"strength={mem.strength}")
            report["deleted"] += 1
        return report

    for cluster in cluster_candidates(candidates):
        summary = summarize_cluster(store, cluster)
        if summary is not None:
            report["summarized"] += len(cluster)
        elif len(cluster) == 1:
            mem = cluster[0]
            store.episodic.set_status(mem.id, "deleted")
            store.log("episodic", mem.id, "forgot->deleted", f"strength={mem.strength}")
            report["deleted"] += 1
        # 摘要失败的簇：保持 archived，等待下次
    return report
