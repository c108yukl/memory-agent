"""记忆健康报告：状态分布、强度分布、版本链概况、待裁决冲突、试用期转正（P1/P3）、
REM 联想（昨夜之梦）、检索空缺（E7「我知道我不知道」）、最近审计动作。

应用层模块：CLI/TUI 在巩固完成后调用 write_health_report 生成到 data/reports/
（机制层不依赖本模块，依赖规则见 ARCHITECTURE.md）。
独立模块（不进 eval 包）以避免巩固管线的循环导入。
"""
from __future__ import annotations

import ast
import os
from collections import Counter
from datetime import datetime, timezone

from memagent import settings
from memagent.learning.spaced_repetition import SpacedRepetition
from memagent.storage import SqliteStore


def _chain_length(store: SqliteStore, fact_id: int, supersede_map: dict[int, int]) -> int:
    length, cur, seen = 0, fact_id, set()
    while cur in supersede_map and cur not in seen:
        seen.add(cur)
        cur = supersede_map[cur]
        length += 1
    return length


def _last_rem_associations(store: SqliteStore) -> list[dict]:
    """读最近一次巩固审计（consolidation_done）里的 rem_associations（E5）。

    巩固统计以 str(report) 落审计 detail，用 literal_eval 安全还原（无网络、
    无执行）；从未巩固过或旧版本报告无该键时返回空列表。"""
    row = store.conn.execute(
        "SELECT detail FROM meta WHERE action='consolidation_done' "
        "ORDER BY id DESC LIMIT 1").fetchone()
    if not row or not row["detail"]:
        return []
    try:
        report = ast.literal_eval(row["detail"])
    except (ValueError, SyntaxError):
        return []
    if not isinstance(report, dict):
        return []
    rem = report.get("rem_associations", [])
    return [r for r in rem if isinstance(r, dict)
            and {"entities", "facts", "strength"} <= r.keys()]


def build_health_report(store: SqliteStore) -> str:
    lines: list[str] = []
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    lines.append(f"# 记忆健康报告 {ts}")

    # ---- 状态分布 ----
    ep = dict(store.conn.execute(
        "SELECT status, COUNT(*) FROM episodic GROUP BY status").fetchall())
    se = dict(store.conn.execute(
        "SELECT status, COUNT(*) FROM semantic GROUP BY status").fetchall())
    skills_n = store.conn.execute("SELECT COUNT(*) FROM procedural").fetchone()[0]
    summaries_n = store.conn.execute(
        "SELECT COUNT(*) FROM episodic WHERE is_summary=1").fetchone()[0]
    lines.append("")
    lines.append("## 总量")
    lines.append(f"- 情景记忆: {ep}（其中摘要记忆 {summaries_n} 条）")
    lines.append(f"- 语义记忆: {se}")
    lines.append(f"- 程序记忆: {skills_n} 条")

    # ---- 强度分布（活跃情景）----
    actives = store.episodic.fetch(status="active", limit=10 ** 9)
    buckets = Counter(min(int(m.strength * 5), 4) for m in actives)
    lines.append("")
    lines.append("## 强度分布（活跃情景记忆）")
    for i in range(5):
        label = f"{i * 0.2:.1f}-{(i + 1) * 0.2:.1f}"
        bar = "#" * buckets.get(i, 0)
        lines.append(f"- {label}: {buckets.get(i, 0)} {bar}")
    if actives:
        avg_access = sum(m.access_count for m in actives) / len(actives)
        top = sorted(actives, key=lambda m: -m.access_count)[:3]
        lines.append(f"- 平均访问次数: {avg_access:.2f}")
        lines.append("- 最高访问: " + ", ".join(f"#{m.id}({m.access_count}次)" for m in top))

    # ---- 版本链 ----
    rows = store.conn.execute(
        "SELECT id, entity, relation, superseded_by FROM semantic").fetchall()
    supersede_map = {r["id"]: r["superseded_by"] for r in rows if r["superseded_by"]}
    if rows:
        longest = max(_chain_length(store, r["id"], supersede_map) for r in rows)
        lines.append("")
        lines.append("## 版本链")
        lines.append(f"- 语义事实总数: {len(rows)}，被取代: {len(supersede_map)}，最长链: {longest}")
        keys = {(r["entity"], r["relation"]) for r in rows}
        lines.append(f"- 活跃关系键: {len(keys)} 个")

    # ---- 待裁决冲突 ----
    pending = store.conflicts.fetch_all(status="pending")
    lines.append("")
    lines.append(f"## 待裁决冲突: {len(pending)} 条")
    for r in pending[:5]:
        old, new = store.semantic.get(r["old_id"]), store.semantic.get(r["new_id"])
        lines.append(f"- #{r['conflict_id']} [{r['created_at']}] "
                     f"{old.value if old else '?'} vs {new.value if new else '?'}")
    if len(pending) > 5:
        lines.append(f"- ...另有 {len(pending) - 5} 条")

    # ---- 试用期转正（V1.7 P1/P3）----
    # 三个数全部从既有痕迹读出，零新表、零新审计动作：
    #   转正数   = meta 审计 action='promote'（_try_promote 是唯一写入点，天然去重）
    #   试用归档 = meta 审计 action='forgot->archive' 且 detail 以「试用期」开头
    #             （archive_stale_probation 的专用前缀，与情景侧遗忘区分）
    #   待观察   = 当前 status='pending' 的语义事实，按 D1 分两类——A 类（冲突待裁）
    #             不参与自动转正只能裁决，B 类才是「试用期」本体；判据与检索侧
    #             同源（conflicts.fetch_all 的 new_id 集合），两处口径不会分叉。
    promoted = store.conn.execute(
        "SELECT COUNT(*) FROM meta WHERE action='promote' AND memory_type='semantic'"
    ).fetchone()[0]
    probation_archived = store.conn.execute(
        "SELECT COUNT(*) FROM meta WHERE action='forgot->archive' "
        "AND memory_type='semantic' AND detail LIKE '试用期%'").fetchone()[0]
    pending_ids = {r["id"] for r in store.conn.execute(
        "SELECT id FROM semantic WHERE status='pending'").fetchall()}
    conflict_a = len(pending_ids & {
        c["new_id"] for c in store.conflicts.fetch_all(status="pending")})
    lines.append("")
    lines.append("## 试用期转正")
    lines.append(f"- 累计转正: {promoted} 条（B 类低置信事实命中达标自动转正，审计 promote）")
    lines.append(f"- 试用期归档: {probation_archived} 条"
                 f"（超 PROBATION_MAX_DAYS={settings.PROBATION_MAX_DAYS} 天无人问津）")
    lines.append(f"- 待观察: {len(pending_ids)} 条 pending"
                 f"（其中冲突待裁 A 类 {conflict_a} 条，不参与自动转正）")

    # ---- 今日回忆清单（SM-2 到期队列，E1）----
    due_ids = SpacedRepetition(store.conn).due()
    lines.append("")
    lines.append(f"## 今日回忆清单: {len(due_ids)} 条到期")
    for mem_id in due_ids[:10]:
        fact = store.semantic.get(mem_id)
        desc = f"#{fact.entity} {fact.relation} {fact.value}" if fact else "#?(已删除)"
        lines.append(f"- {desc}")
    if len(due_ids) > 10:
        lines.append(f"- ...另有 {len(due_ids) - 10} 条")

    # ---- REM 联想（昨夜之梦，E5）----
    rem = _last_rem_associations(store)
    lines.append("")
    lines.append(f"## REM 联想（昨夜之梦）: {len(rem)} 条")
    for r in rem[:5]:
        e_a, e_b = r["entities"][0], r["entities"][1]
        lines.append(f"- [{e_a}] × [{e_b}]"
                     f"（事实 #{r['facts'][0]} ↔ #{r['facts'][1]}，激活 {r['strength']}）")
    if len(rem) > 5:
        lines.append(f"- ...另有 {len(rem) - 5} 条")

    # ---- 检索空缺（E7「我知道我不知道」）----
    # 读 meta 表 action='retrieval_gap' 的审计（retrieve 完全落空时写入），
    # 去重 query 按时间倒序取前 10——这些是检索答不上来的问题，即高价值待巩固
    # 方向（补写入观测或巩固蒸馏）。v1 取舍：全量记录不做高重要度启发式过滤
    # （重要度在检索空时无从判断，误过滤比多展示更糟），靠去重 + 条数上限控噪。
    gap_rows = store.conn.execute(
        "SELECT detail FROM meta WHERE action='retrieval_gap' ORDER BY id DESC"
    ).fetchall()
    gaps: list[str] = []
    for r in gap_rows:
        q = (r["detail"] or "").strip()
        if q and q not in gaps:
            gaps.append(q)
        if len(gaps) >= 10:
            break
    lines.append("")
    lines.append(f"## 检索空缺（我知道我不知道）: {len(gaps)} 条")
    for q in gaps:
        lines.append(f"- {q}")
    if len(gap_rows) > len(gaps):
        lines.append(f"- （另有重复落空 {len(gap_rows) - len(gaps)} 次，已去重）")
    if gaps:
        lines.append("- 以上为检索完全落空的问题，属高价值待巩固方向（补写入或巩固蒸馏）")

    # ---- 最近审计动作 ----
    actions = [r["action"].split("->")[0].split("-")[0]
               for r in store.conn.execute(
                   "SELECT action FROM meta ORDER BY id DESC LIMIT 200").fetchall()]
    lines.append("")
    lines.append("## 最近审计动作分布（最多 200 条）")
    for action, n in Counter(actions).most_common(8):
        lines.append(f"- {action}: {n}")

    return "\n".join(lines) + "\n"


def write_health_report(store: SqliteStore, out_dir: str | None = None) -> str:
    out_dir = out_dir or os.path.join(settings.DATA_DIR, "reports")
    os.makedirs(out_dir, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = os.path.join(out_dir, f"health-{ts}.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write(build_health_report(store))
    store.log("system", 0, "health-report", path)
    return path
