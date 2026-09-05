"""摘要降级：硬删前把可聚类的低价值历史记忆压缩成带溯源的摘要记忆。

保守原则：
- 只有簇内 ≥2 条、时间跨度合理、LLM 摘要成功才降级；
- 原始记忆标 summarized 保留（不硬删），summarized_by 双向可溯；
- 任何一步失败（无 LLM / 摘要为空 / 跨度过大）保持原状，绝不写半成品。
"""
from __future__ import annotations

from memagent import settings
from memagent.adapters import llm
from memagent.core.clock import now_iso, now_ts
from memagent.core.domain import EpisodicMemory
from memagent.core.vectors import cosine
from memagent.learning.strength import compute_strength
from memagent.storage import SqliteStore

DEGRADE_PROMPT = (
    "把下面 {n} 条同主题的旧记忆压缩成一句事实性概括（≤60字，说清主题和规律）。"
    "直接输出，不要解释。\n{items}"
)


def cluster_candidates(mems: list[EpisodicMemory]) -> list[list[EpisodicMemory]]:
    """对硬删候选做贪心余弦聚类（含落单的簇）。"""
    clusters: list[list[EpisodicMemory]] = []
    used: set[int] = set()
    for m in mems:
        if m.id in used:
            continue
        cluster = [m]
        used.add(m.id)
        for other in mems:
            if other.id in used or not m.embedding or not other.embedding:
                continue
            if len(m.embedding) == len(other.embedding) and \
                    cosine(m.embedding, other.embedding) >= settings.CONSOLIDATE_MIN_SIMILARITY:
                cluster.append(other)
                used.add(other.id)
        clusters.append(cluster)
    return clusters


def _span_days(cluster: list[EpisodicMemory]) -> float:
    from memagent.core.clock import iso_to_ts
    times = [iso_to_ts(m.created_at) for m in cluster if m.created_at]
    if not times:
        return 0.0
    return (max(times) - min(times)) / 86400.0


def summarize_cluster(store: SqliteStore, cluster: list[EpisodicMemory]) -> EpisodicMemory | None:
    """把一簇待删记忆降级为摘要记忆；不满足条件或失败返回 None（调用方保持原状）。"""
    if len(cluster) < settings.SUMMARY_DEGRADE_MIN_CLUSTER:
        return None
    if _span_days(cluster) > settings.MAX_SUMMARY_SPAN_DAYS:
        return None
    if not llm.llm_available():
        return None

    items = "\n".join(f"- {m.summary}" for m in cluster)
    try:
        text = llm.chat(DEGRADE_PROMPT.format(n=len(cluster), items=items))
    except Exception:
        return None  # 任何异常都保持原状，绝不写半成品
    if not text:
        return None

    max_imp = max(m.importance for m in cluster)  # 概括记忆承担代表簇的责任，取 max 不取均值
    start = min(m.created_at for m in cluster)[:10]
    end = max(m.created_at for m in cluster)[:10]
    summary = EpisodicMemory(
        summary=f"[摘要] {start}~{end}，{len(cluster)} 条相关记忆：{text}",
        importance=max_imp,
        created_at=now_iso(),
        strength=compute_strength(max_imp, max_imp, 0, now_ts()).score,
        status="active",
        is_summary=True,
        source_ids=sorted(m.id for m in cluster),
        embedding=llm.embed(text),
    )
    summary_id = store.episodic.add(summary)

    for m in cluster:
        m.summarized_by = summary_id
        m.status = "summarized"
        store.episodic.update(m)
        store.log("episodic", m.id, "degraded->summary", f"#{summary_id}")
    store.log("episodic", summary_id, "summary-created",
              f"sources={summary.source_ids}")
    return summary
