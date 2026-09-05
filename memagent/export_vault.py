"""导出 Obsidian vault 快照：实体/技能/情景记忆 → markdown 笔记 + [[wiki 链接]]。

只读导出（快照语义，单向不回流）：Obsidian 打开导出目录即得免费的可视化——
实体=笔记、三元组=链接、REM 联想=边，图谱视图（graph view）零 UI 代码可用。

笔记布局（标题全局唯一，wiki 链接按标题解析）：
  INDEX.md            总览：计数 + 入口链接（user 实体置顶）
  entities/<实体>.md  每个规范实体一页：active 事实 / 待裁决 / 版本历史 / 邻居
  skills/技能-<名>.md 每个技能一页（标题加「技能-」前缀防与实体重名）
  memories/记忆-<id>.md 每条 active 情景一页，链接到其产出事实的实体

值内连边复用 E4 规则：值中精确出现的已知实体名转 [[链接]]（宁可少连不可误伤）。
文件名清洗 Windows 非法字符，冲突标题追加 -2 去重。
"""
from __future__ import annotations

import os
import re

from memagent.retrieval.activation import build_entity_graph
from memagent.storage import SqliteStore

_ILLEGAL = re.compile(r'[\\/:*?"<>|#^\[\]]')


def _safe_title(name: str, used: dict[str, str]) -> str:
    """实体名 -> 安全且唯一的笔记标题（同名实体清洗后冲突时追加序号）。"""
    title = _ILLEGAL.sub("_", name).strip() or "未命名"
    base, n = title, 2
    while title in used and used[title] != name:
        title = f"{base}-{n}"
        n += 1
    used[title] = name
    return title


def _link_value(value: str, own_entity: str, titles: dict[str, str],
                known: dict[str, str]) -> str:
    """把值中精确出现的已知实体名替换为 [[链接]]（长名优先；已链接区间保护）。"""
    names = sorted((n for n in known
                    if known[n] != own_entity and n in value),
                   key=len, reverse=True)
    protected: list[tuple[int, int]] = []

    def _inside(i: int, j: int) -> bool:
        return any(not (j <= s or i >= e) for s, e in protected)

    for name in names:
        start = 0
        while True:
            i = value.find(name, start)
            if i < 0:
                break
            j = i + len(name)
            if not _inside(i, j):
                linked = f"[[{titles[known[name]]}]]"
                value = value[:i] + linked + value[j:]
                shift = len(linked) - len(name)
                protected = [(s + shift if s >= j else s, e + shift if e >= j else e)
                             for s, e in protected]
                protected.append((i, i + len(linked)))
                start = i + len(linked)
            else:
                start = i + len(name)
    return value


def _frontmatter(pairs: list[tuple[str, object]]) -> str:
    lines = ["---"]
    for k, v in pairs:
        lines.append(f"{k}: {v}")
    lines.append("---")
    return "\n".join(lines)


def export_vault(store: SqliteStore, out_dir: str) -> dict:
    """把当前记忆库导出为 Obsidian vault 快照，返回统计。确定性排序，可重复执行。"""
    graph = build_entity_graph(store)
    used: dict[str, str] = {}  # 标题 -> 实体名（全局唯一，防 entities/skills 撞名）
    titles = {e: _safe_title(e, used) for e in graph.facts_by_entity}

    stats = {"entities": 0, "facts": 0, "pending": 0, "memories": 0,
             "skills": 0, "files": 0, "dir": os.path.abspath(out_dir)}
    entities_dir = os.path.join(out_dir, "entities")
    skills_dir = os.path.join(out_dir, "skills")
    memories_dir = os.path.join(out_dir, "memories")
    for d in (out_dir, entities_dir, skills_dir, memories_dir):
        os.makedirs(d, exist_ok=True)

    pending_by_entity: dict[str, list] = {}
    for fact in store.semantic.fetch(status="pending"):
        e = graph.aliases.get(fact.entity, fact.entity)
        pending_by_entity.setdefault(e, []).append(fact)
    history_by_entity: dict[str, list] = {}
    for entity in graph.facts_by_entity:
        history_by_entity[entity] = [
            f for f in store.semantic.fetch_history(entity) if f.status == "superseded"]

    # ---------- 技能标题预分配（实体笔记反链需要，先于实体写入保证标题唯一性）----------
    skills = store.procedural.fetch(status="active")
    skill_titles = [(s, _safe_title(f"技能-{s.name}", used)) for s in skills]
    skill_link_by_entity: dict[str, list[str]] = {}
    for s, t in skill_titles:
        if s.name in titles:  # 绿色通道技能名 = 任务域实体，互链
            skill_link_by_entity.setdefault(titles[s.name], []).append(t)

    # ---------- 实体笔记 ----------
    for entity in sorted(graph.facts_by_entity):
        facts = sorted(graph.facts_of(entity), key=lambda f: (-f.confidence, f.id))
        lines = [_frontmatter([("type", "entity"), ("facts", len(facts)),
                               ("base_strength", round(graph.base.get(entity, 0.0), 3))]), ""]
        lines.append(f"# {entity}")
        lines.append("")
        lines.append("## 事实")
        for f in facts:
            linked = _link_value(f.value, entity, titles, _known_names(graph))
            lines.append(f"- **{f.relation}** :: {linked} "
                         f"`(conf {f.confidence:.2f} / 证据×{f.evidence_count})`")
            stats["facts"] += 1
        if entity in pending_by_entity:
            lines.append("")
            lines.append("## 待裁决")
            for f in sorted(pending_by_entity[entity], key=lambda x: -x.confidence):
                lines.append(f"- **{f.relation}** :: {f.value} `(pending conf {f.confidence:.2f})`")
                stats["pending"] += 1
        if history_by_entity.get(entity):
            lines.append("")
            lines.append("## 版本历史")
            for f in history_by_entity[entity]:
                period = f"{f.valid_from[:10]} ~ {f.valid_to[:10] if f.valid_to else '今'}"
                lines.append(f"- **{f.relation}** :: {f.value} `({period}，被 #{f.superseded_by} 取代)`")
        neighbors = sorted(graph.edges.get(entity, ()))
        if neighbors:
            lines.append("")
            lines.append("## 邻居")
            lines.append(" ".join(f"[[{titles[n]}]]" for n in neighbors))
        if titles[entity] in skill_link_by_entity:
            lines.append("")
            lines.append("## 技能")
            lines.append(" ".join(f"[[{t}]]" for t in skill_link_by_entity[titles[entity]]))
        _write(os.path.join(entities_dir, f"{titles[entity]}.md"), "\n".join(lines))
        stats["entities"] += 1

    # ---------- 技能笔记 ----------
    for skill, title in skill_titles:
        lines = [_frontmatter([("type", "skill"), ("usage", skill.usage_count),
                               ("success_rate", f"{skill.success_rate:.0%}")]), ""]
        lines.append(f"# 技能：{skill.name}")
        lines.append("")
        if skill.trigger:
            lines.append(f"**触发**：{skill.trigger}")
        lines.append(f"**做法**：{skill.policy}")
        lines.append(f"`(用过 {skill.usage_count} 次，成功 {skill.success_count} / "
                     f"失败 {skill.failure_count})`")
        if skill.name in titles:
            lines.append("")
            lines.append(f"相关实体：[[{titles[skill.name]}]]")
        _write(os.path.join(skills_dir, f"{title}.md"), "\n".join(lines))
        stats["skills"] += 1

    # ---------- 情景记忆笔记 ----------
    facts_of_event: dict[int, list] = {}
    for entity in graph.facts_by_entity:
        for f in graph.facts_of(entity):
            for sid in f.source_event_ids:
                facts_of_event.setdefault(sid, []).append(entity)
    for mem in store.episodic.fetch(status="active", limit=10 ** 9):
        title = f"记忆-{mem.id}"
        used[title] = title
        lines = [_frontmatter([("type", "episodic"), ("id", mem.id),
                               ("created", mem.created_at[:10]),
                               ("importance", round(mem.importance, 3)),
                               ("strength", round(mem.strength, 3)),
                               ("category", mem.category or "normal")]), ""]
        lines.append(f"# {mem.summary}")
        lines.append("")
        meta = [f"发生于 {mem.created_at[:10]}"]
        if mem.context:
            meta.append(f"上下文 {mem.context}")
        if mem.outcome:
            meta.append(f"结果 {mem.outcome}")
        lines.append("`" + "，".join(meta) + "`")
        linked_entities = sorted({e for e in facts_of_event.get(mem.id, ())
                                  if e in titles})
        if linked_entities:
            lines.append("")
            lines.append("相关实体：" + " ".join(f"[[{titles[e]}]]" for e in linked_entities))
        _write(os.path.join(memories_dir, f"{title}.md"), "\n".join(lines))
        stats["memories"] += 1

    # ---------- INDEX ----------
    stats["files"] = (stats["entities"] + stats["skills"]
                      + stats["memories"] + 1)
    lines = [_frontmatter([("type", "index"), *[(k, v) for k, v in stats.items()
                                                if k != "dir"]]), ""]
    lines.append("# 记忆库快照")
    lines.append("")
    lines.append(f"- 实体 {stats['entities']} 个 / 语义事实 {stats['facts']} 条"
                 f"（另有待裁决 {stats['pending']} 条）")
    lines.append(f"- 情景记忆 {stats['memories']} 条（active）")
    lines.append(f"- 技能 {stats['skills']} 个")
    lines.append(f"- 导出于 {store.db_path}（快照单向，Obsidian 内修改不会回流）")
    lines.append("")
    lines.append("## 实体")
    ordered = sorted(titles, key=lambda t: (t != "user", t))
    lines.append(" ".join(f"[[{t}]]" for t in ordered) or "（无）")
    lines.append("")
    lines.append("## 技能")
    lines.append(" ".join(f"[[{t}]]" for t in
                          (t for _, t in sorted(skill_titles,
                                                key=lambda st: -st[0].usage_count))) or "（无）")
    _write(os.path.join(out_dir, "INDEX.md"), "\n".join(lines))

    return stats


def _known_names(graph) -> dict[str, str]:
    """E4 连边同款已知名表：实体名与已挂靠的别名 -> 规范实体。"""
    known = {e: e for e in graph.facts_by_entity}
    for alias, canonical in graph.aliases.items():
        if canonical in graph.facts_by_entity:
            known.setdefault(alias, canonical)
    return known


def _write(path: str, text: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(text + "\n")
