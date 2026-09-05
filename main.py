"""CLI 入口：记忆系统的日常操作。

用法：
  python main.py init                初始化数据库
  python main.py add <内容>          写入一条事件（自动门控+编码）
  python main.py retrieve <问题>     检索相关记忆（--deep 用 LLM 展开查询多路召回）
  python main.py inspect             查看全部记忆
  python main.py consolidate         睡眠巩固（离线整理）
  python main.py forget              主动遗忘（衰减清理）
  python main.py repetition          查看间隔重复队列 / 复习
  python main.py review <id> <0-5>   对某条语义记忆做一次复习打分
  python main.py status              LLM 可用性 + 记忆统计
  python main.py test                LLM 通道连通性诊断（五项检查，主聊天对话失败退出码 1）
  python main.py agent               记忆原生对话循环（注入→生成→录入 + 记忆工具，默认流式，--no-stream 整段）
"""
from __future__ import annotations

import argparse
import json
import os
import sys

from memagent import settings
from memagent.adapters import llm
from memagent.consolidation import consolidate
from memagent.consolidation.conflict_resolver import resolve_conflict
from memagent.forgetting import run_forgetting
from memagent.learning import SpacedRepetition
from memagent.maintenance import reembed, resolve_entities
from memagent.pipeline import ingest_event
from memagent.retrieval import build_context, format_validity_context, retrieve
from memagent.retrieval.deep import deep_retrieve
from memagent.storage import SqliteStore


def _storage() -> SqliteStore:
    os.makedirs(settings.DATA_DIR, exist_ok=True)
    return SqliteStore()


def cmd_init(store: SqliteStore, args) -> int:
    store.log("system", 0, "init")
    print(f"数据库已初始化: {settings.DB_PATH}")
    print(f"LLM 可用: {llm.llm_available()} (本地 LM Studio)")
    return 0


def cmd_add(store: SqliteStore, args) -> int:
    r = ingest_event(store, args.content, source=args.source, type=args.type,
                     task_context=args.context, outcome=args.outcome, use_llm=not args.no_llm)
    print(f"门控打分: importance={r['importance']:.3f}")

    if r["gated"]:
        # E3 双写：两档事件在门控前均已进会话工作记忆（本会话内可检索，进程
        # 退出即蒸发），差别只在长期记忆是否入库——文案如实反映去向。
        if r["reason"] == "dropped":
            print("低价值，不入长期记忆（同在本会话工作记忆中，会话结束蒸发）")
        else:
            print("中等价值，已暂存会话工作记忆（本会话内可检索，会话结束蒸发）")
        return 0

    mem = store.episodic.get(r["episodic_id"])
    print(f"情景记忆已写入: #{r['episodic_id']} [{mem.summary}]")

    for s in r["skipped_facts"]:
        print(f"同键低置信已跳过: {s['value']} (conf={s['confidence']})")
    for f in r["facts"]:
        label = f"#{f['fact_id']} [{f['entity']}] {f['relation']} = {f['value']}"
        if f["action"] == "created":
            print(f"语义记忆新建: {label} (conf={f['confidence']})")
        elif f["action"] == "renewed":
            print(f"语义记忆复证: {label} (证据+1)")
        elif f["action"] == "superseded":
            print(f"语义记忆变更: {label}，取代旧版 {f['superseded']}")
        else:
            print(f"语义记忆待裁: {label}，冲突 {f['conflict_ids']}（旧版保留，可用 conflicts resolve 裁决）")

    for sk in r["skills"]:
        if sk["reused"]:
            print(f"技能已复用: {sk['name']} (使用次数+1)")
        else:
            print(f"技能已沉淀: {sk['name']} <- {sk['policy']}")
    return 0


def cmd_retrieve(store: SqliteStore, args) -> int:
    """检索记忆。P1-4：--deep 走深搜（LLM 展开查询多路召回，慢且消耗一次对话
    调用；只读探索——deep_retrieve 内部恒 boost_access=False，深搜是「查」不是
    「复习」）；默认快搜（行为原样）。深搜离线/失败自动静默降级快搜。"""
    if getattr(args, "deep", False):
        hits = deep_retrieve(store, args.query, top_k=args.top_k)
    else:
        hits = retrieve(store, args.query, top_k=args.top_k)
    if not hits:
        print("未检索到相关记忆")
        return 0
    if any(h.meta.get("uncertain") for h in hits if h.kind != "working"):
        # E7 检索置信度表面化：top 直接命中最高分低于置信线时显式示警
        print("⚠ 检索置信度低，以下结果可能不相关（feeling-of-knowing: 不确定）")
    print(build_context(hits, max_chars=2000, store=store))
    return 0


def cmd_inspect(store: SqliteStore, args) -> int:
    episodes = store.episodic.fetch(status="active", limit=200)
    facts = store.semantic.fetch(status="active", limit=200)
    pending = store.semantic.fetch(status="pending", limit=10 ** 9)
    superseded = store.semantic.fetch(status="superseded", limit=10 ** 9)
    skills = store.procedural.fetch()

    print(f"== 情景记忆 ({len(episodes)}) ==")
    for m in episodes:
        tag = f" [摘要<-{m.source_ids}]" if m.is_summary else ""
        print(f"  #{m.id} [{m.strength:.2f}] {m.summary} (次数:{m.access_count}){tag}")
    print(f"== 语义记忆 (活跃 {len(facts)} / 待裁 {len(pending)} / 已取代 {len(superseded)}) ==")
    for f in facts:
        # P2-2：带条件上下文的事实追加条件串（全量显示；无该字段输出逐字符不变）
        vc = format_validity_context(f.validity_context)
        cond = f" 条件:{vc}" if vc else ""
        print(f"  #{f.id} [{f.entity}] {f.relation} = {f.value} (conf={f.confidence}){cond}")
    print(f"== 程序记忆 ({len(skills)}) ==")
    for s in skills:
        print(f"  {s.id} {s.name} 触发:{s.trigger} 策略:{s.policy} 使用:{s.usage_count} 成功:{s.success_rate:.0%}")
    return 0


def cmd_consolidate(store: SqliteStore, args) -> int:
    from memagent.reports import write_health_report
    report = consolidate(store)
    rem = report.get("rem_associations", [])
    print(f"睡眠巩固完成（NREM×{report.get('nrem_rounds', 0)} + REM）: "
          f"聚类={report['clusters']}, 蒸馏事实={report['distilled_facts']}, "
          f"摘要替代={report['summarized']}, REM联想={len(rem)}, REM写入={report.get('rem_facts', 0)}")
    for r in rem[:5]:
        print(f"  💭 [{r['entities'][0]}] × [{r['entities'][1]}]"
              f"（事实 #{r['facts'][0]} ↔ #{r['facts'][1]}，激活 {r['strength']}）")
    if len(rem) > 5:
        print(f"  ...另有 {len(rem) - 5} 条 REM 联想")
    due = report.get("due_reviews", [])
    if due:
        print(f"今日回忆清单（SM-2 重演）: {len(due)} 条")
        for d in due[:10]:
            mark = "✓唤回" if d["recalled"] else "✗遗忘"
            print(f"  [{mark}] #{d['fact_id']} [{d['entity']}] {d['relation']} = {d['value']}"
                  f" (下次间隔 {d['interval_days']} 天)")
    try:
        print(f"健康报告: {write_health_report(store)}")
    except Exception as e:  # 报告失败不阻断巩固结果
        print(f"健康报告生成失败: {e!r}")
    return 0


def cmd_forget(store: SqliteStore, args) -> int:
    report = run_forgetting(store)
    print(f"遗忘完成: 归档={report['archived']}, 硬删={report['deleted']}, "
          f"活跃={report['episodic_active']}")
    return 0


def cmd_repetition(store: SqliteStore, args) -> int:
    sr = SpacedRepetition(store.conn)
    if args.review is not None:  # 显式判断：--review 0 也要走存在性校验，不能当 falsy 跳过
        if not store.semantic.get(args.review):
            print(f"记忆 #{args.review} 不存在，已拒绝记录复习（防幽灵排期）")
            return 1
        info = sr.review(args.review, args.quality)
        print(f"复习已记录: #{info['memory_id']} 下次 {info['due_at']} (间隔 {info['interval_days']} 天, 难度 {info['ease']:.2f})")
        return 0
    due = sr.due()
    print(f"今日到期复习: {len(due)} 条")
    for mem_id in due:
        fact = store.semantic.get(mem_id)
        if fact:
            print(f"  #{mem_id} [{fact.entity}] {fact.relation} = {fact.value}")
    for row in sr.status():
        print(f"  计划: #{row['memory_id']} 间隔={row['interval_days']}天 下次={row['due_at']}")
    return 0


def cmd_reembed(store: SqliteStore, args) -> int:
    report = reembed(store, dry_run=not args.apply, batch_size=args.batch_size)
    if report["dry_run"]:
        print(f"[dry-run] 情景记忆 {report['episodic_total']} 条，维度分布 {report['episodic_dims']}")
        print(f"[dry-run] 语义记忆 {report['semantic_total']} 条，维度分布 {report['semantic_dims']}")
        print("加 --apply 执行回填（自动备份到 data/backups/）")
        return 0
    print(f"已重嵌入 {report['embedded']} 条；维度一致: {report['dim_consistent']}")
    print(f"维度分布(情景/语义): {report['episodic_dims_after']} / {report['semantic_dims_after']}")
    if report["failed"]:
        print(f"失败 {len(report['failed'])} 条: {report['failed'][:5]}")
    print(f"备份: {report['backup_path']}")
    return 0


def cmd_alias(store: SqliteStore, args) -> int:
    if args.alias_cmd == "add":
        existing = store.aliases.as_map().get(args.alias)
        if existing == args.canonical:
            print(f"别名已存在且指向一致，无需添加: {args.alias} -> {args.canonical}")
            return 0
        if not store.aliases.add(args.alias, args.canonical):
            print(f"别名已存在且指向不同实体，未修改: {args.alias}（现指向 {existing}）")
            return 1
        print(f"别名已添加: {args.alias} -> {args.canonical}")
    elif args.alias_cmd == "remove":
        print(f"{'已删除' if store.aliases.remove(args.alias) else '别名不存在'}: {args.alias}")
    else:  # list
        rows = store.aliases.fetch_all()
        print(f"别名表 ({len(rows)} 条):")
        for r in rows:
            print(f"  {r['alias']} -> {r['canonical']} ({r['source']})")
    return 0


def cmd_resolve_entities(store: SqliteStore, args) -> int:
    report = resolve_entities(store, dry_run=not args.apply)
    print(f"别名表: {report['aliases']} 条")
    if report["dry_run"]:
        print(f"[dry-run] 将归一 {len(report['changes'])} 条:")
        for ch in report["changes"][:20]:
            print(f"  #{ch['id']} [{ch['entity']}]{ch['relation']} -> [{ch['new_entity']}]{ch['new_relation']}")
        print("加 --apply 执行归一（自动备份）")
        return 0
    print(f"已归一 {report['affected']} 条，合并精确重复 {report.get('merged_duplicates', 0)} 条")
    print(f"备份: {report['backup_path']}")
    return 0


def cmd_history(store: SqliteStore, args) -> int:
    facts = store.semantic.fetch_history(args.entity, args.relation)
    if not facts:
        print(f"没有 [{args.entity}] 的记忆")
        return 0
    print(f"== [{args.entity}] 版本链 ==")
    for f in facts:
        period = f"{f.valid_from} ~ {f.valid_to or '今'}"
        extra = f" <-被#{f.superseded_by}取代" if f.superseded_by else ""
        evidence = f" 证据x{f.evidence_count}" if f.evidence_count > 1 else ""
        # P2-2：带条件上下文的事实追加条件串（跟随行内后缀风格；无该字段输出不变）
        vc = format_validity_context(f.validity_context)
        cond = f" 条件:{vc}" if vc else ""
        print(f"  #{f.id} [{f.status}] {f.relation} = {f.value} ({period}){evidence}{extra}{cond}")
    return 0


def cmd_conflicts(store: SqliteStore, args) -> int:
    if args.conflicts_cmd == "resolve":
        row = resolve_conflict(store, args.conflict_id, args.resolution)
        if row is None:
            print(f"冲突 #{args.conflict_id} 不存在或已裁决")
            return 1
        print(f"已裁决 #{args.conflict_id}: {args.resolution} (旧#{row['old_id']} / 新#{row['new_id']})")
        return 0
    rows = store.conflicts.fetch_all(status="pending")
    print(f"待裁决冲突 ({len(rows)} 条):")
    for r in rows:
        old, new = store.semantic.get(r["old_id"]), store.semantic.get(r["new_id"])
        print(f"  #{r['conflict_id']} [{r['created_at']}] "
              f"旧#{r['old_id']}:{old.value if old else '?'} vs 新#{r['new_id']}:{new.value if new else '?'}")
        print(f"      resolve --accept-new 新版生效 | resolve --keep-old 保留旧版 | resolve --both 误报共存")
    return 0


def cmd_eval_mini(store: SqliteStore, args) -> int:
    from memagent.eval import run_mini
    report = run_mini()
    print(f"mini golden: {report['cases']} 个场景")
    print(f"偏好命中率:       {report['preference_hit_rate']:.2%} (基线 ≥85%)")
    print(f"冲突消解准确率:   {report['conflict_resolution_accuracy']:.2%} (基线 ≥80%)")
    print(f"版本链完整率:     {report['history_integrity']:.2%} (基线 =100%)")
    if report["failures"]:
        print(f"失败场景: {list(report['failures'])}")
    print("结论: " + ("达标 ✅" if report["passed"] else "未达标 ❌"))
    return 0 if report["passed"] else 1


def cmd_eval(store: SqliteStore, args) -> int:
    from memagent.eval.harness import run_harness
    import os
    golden = os.path.join(settings.BASE_DIR, "evals", "golden", "*.jsonl")
    scenarios = os.path.join(settings.BASE_DIR, "evals", "scenarios", "*.jsonl")
    report = run_harness([golden, scenarios])
    m = report["metrics"]
    print(f"评测文件: {report['files']} 个")
    print(f"偏好命中率:     {m['preference']['rate']:.2%} ({m['preference']['pass']}/{m['preference']['total']})")
    print(f"冲突消解正确率: {m['conflict']['rate']:.2%} ({m['conflict']['pass']}/{m['conflict']['total']})")
    print(f"版本链完整率:   {m['integrity']['rate']:.2%} ({m['integrity']['pass']}/{m['integrity']['total']})")
    print(f"重复提问稳定率: {m['repeat']['rate']:.2%} ({m['repeat']['pass']}/{m['repeat']['total']})")
    print("扩展指标（V1.5 拟人度 + V1.6 经验通道）:")
    labels = {"retention": "保留曲线(SM-2)", "emotion": "情感区分度",
              "association": "联想召回率", "working": "会话命中率",
              "metacognition": "元认知校准", "experience": "经验通道命中率",
              "promotion": "试用期转正", "consistency": "跨轮记忆一致性",
              "growth": "记忆增长率", "dedup": "近重复合并(B1)",
              "inject_gate": "注入门槛(B3)", "assistant_noise": "助手噪声入库率"}
    for key, label in labels.items():
        v = m[key]
        print(f"  {label}: {v['rate']:.2%} ({v['pass']}/{v['total']})")
    if report["failures"]:
        print(f"失败断言 ({len(report['failures'])}):")
        for f_ in report["failures"][:10]:
            print(f"  - {f_}")
    passed = (m["preference"]["rate"] >= 0.85 and m["conflict"]["rate"] >= 0.80
              and m["integrity"]["rate"] >= 1.0 and m["repeat"]["rate"] >= 0.90)
    print("结论: " + ("达标 ✅" if passed else "未达标 ❌"))
    return 0 if passed else 1


def cmd_export_vault(store: SqliteStore, args) -> int:
    from memagent.export_vault import export_vault
    r = export_vault(store, args.dir)
    print(f"已导出 Obsidian vault 快照 -> {r['dir']}")
    print(f"实体 {r['entities']} / 语义事实 {r['facts']}（待裁决 {r['pending']}）"
          f" / 情景记忆 {r['memories']} / 技能 {r['skills']}，共 {r['files']} 个文件")
    print(f"用 Obsidian 打开该目录（作为 vault），图谱视图即可浏览记忆与联想")
    return 0


def cmd_report(store: SqliteStore, args) -> int:
    from memagent.reports import build_health_report
    print(build_health_report(store))
    return 0


def cmd_rebuild_fts(store: SqliteStore, args) -> int:
    from memagent.maintenance import rebuild_fts
    report = rebuild_fts(store, dry_run=not args.apply)
    if report["dry_run"]:
        print(f"[dry-run] 将重建 FTS 索引：情景 {report['episodic']} 条 / 语义 {report['semantic']} 条")
        print("加 --apply 执行（自动备份）")
        return 0
    print(f"FTS 已重建（情景 {report['episodic']} / 语义 {report['semantic']}），备份: {report['backup_path']}")
    return 0


def cmd_dedupe_facts(store: SqliteStore, args) -> int:
    """B1 存量治理：跨键近重复的 active 语义事实按并查集归簇合并，每组留一份。"""
    from memagent.maintenance import dedupe_facts
    report = dedupe_facts(store, dry_run=not args.apply)
    if report["dry_run"]:
        print(f"[dry-run] active 语义事实 {report['total_active']} 条，"
              f"近重复组 {len(report['groups'])} 个（{report['affected']} 条将被并入 keeper）：")
        for g in report["groups"]:
            k = g["keeper"]
            print(f"  keeper #{k['id']} [{k['entity']}/{k['relation']}] {k['value']} "
                  f"(evidence={k['evidence_count']})")
            for a in g["absorbed"]:
                print(f"    <- #{a['id']} [{a['entity']}/{a['relation']}] {a['value']} "
                      f"(evidence={a['evidence_count']})")
        print("加 --apply 执行（自动备份）")
        return 0
    print(f"近重复合并完成：{report['affected']} 条并入 {len(report['groups'])} 个 keeper，"
          f"备份: {report['backup_path']}")
    return 0


def _load_cleanup_plan(raw: str) -> dict:
    """--plan 的两种形态：JSON 字符串或 JSON 文件路径（命中现有路径按文件读）。"""
    if os.path.isfile(raw):
        with open(raw, encoding="utf-8") as f:
            text = f.read()
    else:
        text = raw
    try:
        plan = json.loads(text)
    except json.JSONDecodeError as e:
        raise SystemExit(f"--plan 既不是存在的文件，也无法按 JSON 解析: {e}") from None
    if not isinstance(plan, dict):
        raise SystemExit("--plan 须为 JSON 对象（清理清单 dict），收到 "
                         f"{type(plan).__name__}")
    return plan


def cmd_cleanup_garbage(store: SqliteStore, args) -> int:
    """P0-3 存量垃圾清理：按显式清单归档垃圾行/修正 category（默认 dry-run）。"""
    from memagent.maintenance import cleanup_garbage
    report = cleanup_garbage(store, _load_cleanup_plan(args.plan), apply=args.apply)
    kind_label = {"episodic": "情景", "semantic": "语义", "procedural": "技能"}
    if report["dry_run"]:
        print(f"[dry-run] 将归档 {report['archived']} 条，修正 category {report['fix_category']} 条，"
              f"skipped {report['skipped_count']} 条")
        for ch in report["changes"]:
            if ch["action"] == "archive":
                # 情景行带 category 一起示出（本次垃圾多为误标 experience 层，便于人工复核）
                cat = (f", category={ch['current_category']!r}"
                       if "current_category" in ch else "")
                print(f"  将归档 {kind_label[ch['kind']]} #{ch['id']} "
                      f"[status={ch['current_status']}{cat}] -> archived")
            else:
                print(f"  将修正 {kind_label[ch['kind']]} #{ch['id']} category "
                      f"{ch['current_category']!r} -> {ch['target']!r} (status={ch['current_status']})")
        for s in report["skipped"]:
            print(f"  跳过 {kind_label[s['kind']]} #{s['id']}: {s['reason']}")
        print("dry-run 未写入；确认后加 --apply 执行（执行前自动热备份）")
        return 0
    if report.get("backup_path"):
        print(f"备份: {report['backup_path']}")
    else:
        # 幂等复跑：目标行均已处于目标状态，cleanup_garbage 早退不产生 backup_path——
        # 此分支不是异常，是幂等语义的正常输出（V1.5.2 教训：CLI 入口最容易漏测）
        print("无待执行变更（均已处于目标状态），未备份未写入")
    print(f"已归档 {report['archived']} 条，修正 category {report['fix_category']} 条，"
          f"skipped {report['skipped_count']} 条")
    for s in report["skipped"]:
        print(f"  跳过 {kind_label[s['kind']]} #{s['id']}: {s['reason']}")
    return 0


def cmd_status(store: SqliteStore, args) -> int:
    episodes = store.episodic.fetch(status="active", limit=10 ** 9)
    facts = store.semantic.fetch(status="active", limit=10 ** 9)
    print(f"LLM: {'可用' if llm.llm_available() else '不可用(本地哈希嵌入模式)'}")
    print(f"服务商: {llm.active_provider()}")
    print(f"情景记忆(活跃): {len(episodes)}  语义记忆(活跃): {len(facts)}")
    print(f"数据文件: {settings.DB_PATH}")
    return 0


def cmd_test(store: SqliteStore, args) -> int:
    """通道连通性诊断：五项检查逐行输出，主聊天对话通过即退出码 0（脚本可判定）。"""
    from memagent.adapters.llm.diag import run_diag
    mark = {"info": "OK ", "warn": "WARN", "error": "ERR "}
    results = run_diag()
    for r in results:
        print(f"[{mark[r['level']]}] {r['name']} ({r['latency_ms'] / 1000:.1f}s): {r['detail']}")
    n = {"info": 0, "warn": 0, "error": 0}
    for r in results:
        n[r["level"]] += 1
    chat_ok = next(r["ok"] for r in results if r["name"] == "主聊天对话")
    print(f"汇总: {n['info']} 正常 / {n['warn']} 告警 / {n['error']} 失败"
          + ("" if chat_ok else "（主聊天对话未通过，退出码 1）"))
    return 0 if chat_ok else 1


def cmd_tui(store: SqliteStore, args) -> int:
    from memagent.tui import run
    return run(store)


class StreamPrinter:
    """cmd_agent 的流式打印器（纯类，模块级便于单测）。

    对接 agent/loop 的 on_delta 合同：kind ∈ thinking / answer / reset。
    换段规则：
      本轮首个增量   -> 先打 "\\n助手: " 段标签（与 --no-stream 整段模式的「助手: 」对齐）；
      thinking 首次出现 -> 先打 "[思考] " 段头（ANSI dim 弱化色，Windows 终端
                           开启 VT 后支持，同 tui.Style.dim）；
      answer   首次出现 -> 先解除 dim 并打 "\\n" 脱离思考段；
      tool     首次出现 -> 打 dim 的 "[工具] 名称+首行结果" 行（模型发起了工具调用）；
      tool_result -> 在工具行后补 "（结果已回填）"，下一轮生成从新行开始；
      reset    -> 打 "\\n[重新生成]\\n" 并复位分段状态（此后增量按首次出现处理，
                  但段标签只在本轮首个增量前打一次）。
    增量原样经 emit 逐片写出（默认 print(end="", flush=True) 实时上屏；单测传
    list.append 收集断言）。printed 标记本轮是否已输出过内容，供 CLI 决定收尾时
    补换行还是整段打印（离线兜底轮没有任何增量，回退整段输出）。
    """

    _DIM, _RESET = "\x1b[2m", "\x1b[0m"

    def __init__(self, emit=None):
        self._emit = emit if emit is not None else (
            lambda t: print(t, end="", flush=True))
        self.seg = None          # 当前段：None / "thinking" / "answer"
        self.printed = False     # 本轮是否已输出过任何内容

    def feed(self, kind: str, text: str) -> None:
        if not self.printed:
            self._emit("\n助手: ")   # 本轮首个增量前打段标签，只打一次
        self.printed = True
        if kind == "reset":
            if self.seg == "thinking":
                self._emit(self._RESET)      # 先退出 dim 再打复位提示
            self._emit("\n[重新生成]\n")
            self.seg = None
            return
        if kind == "tool":
            if self.seg == "thinking":
                self._emit(self._RESET)
            self._emit(f"\n{self._DIM}[工具] {text}\x1b[0m")
            self.seg = "tool"
            return
        if kind == "tool_result":
            self._emit(f"{self._DIM}（结果已回填）\x1b[0m")
            self.seg = "tool_result"    # 下一轮 thinking/answer 会先换行
            return
        if kind == "thinking":
            if self.seg != "thinking":
                self._emit(f"\n{self._DIM}[思考] ")
                self.seg = "thinking"
        else:  # answer
            if self.seg == "thinking":
                self._emit(self._RESET)
            if self.seg != "answer":
                self._emit("\n")
                self.seg = "answer"
        self._emit(text)


def _warmup_probes() -> None:
    """后台预热 LLM 三通道探测（照 TUI start_probe 模式，V1.7.4）。

    首轮对话前有 4~5 个串行探测请求（本地 LM Studio 尝试 × 槽 + 云端 /models），
    全部推迟到首次 LLM 访问时才发——用户第一句话要白等这笔钱（实测 ~10s，其中
    真正干活的不到 2s）。放在 input() 阻塞等用户打字期间跑，预热大概率在首句
    之前完成；探测本身有超时与异常兜底，失败按离线降级，不影响使用。
    """
    try:
        llm.active_provider()   # 依次探 chat / maint / embed 三槽
        llm.embed("预热")        # embed 槽真实请求走一遍（后端选择落缓存）
        llm.local_chat("预热")   # local 快车道槽（经验技能抽取用）
    except Exception:
        pass  # 预热失败不算事：各槽 TTL 过期后自然重探，调用方自有降级


def _print_record_sides(turn) -> None:
    """--debug 的录入去向显示（用户/助手两侧）。

    命令轮（/deep）不注入不录入，Turn.record 为空 dict——显示层必须用 .get
    防御而非假设每轮都有双侧录入（实测翻车：/deep 开启轮 KeyError 'user'；
    V1.5.2 教训重演：bug 集中在 CLI 入口对非典型轮次的假设）。"""
    if not turn.record:
        print(f"  [命令] {turn.user_text}（会话命令，不注入不录入）")
        return
    for side, label in (("user", "用户"), ("assistant", "助手")):
        rec = turn.record.get(side)
        if rec is None:
            continue
        extra = f" signals={rec['signals']}" if rec.get("signals") else ""
        print(f"  [{label}] {rec['action']}{extra}")


def cmd_agent(store: SqliteStore, args) -> int:
    """记忆原生对话循环：注入 → 生成（可调记忆工具）→ 录入。

    每轮自动注入是默认行为，不是选项——模型不需要「想起来要查记忆」；
    工具是注入漏了时的补丁（主动检索/点名写入/睡眠整理等，V1.7.3），
    --no-tools 关闭工具只保留注入与静默录入。
    默认流式输出（thinking 段 dim 弱化、answer 增量实时上屏、工具调用打
    [工具] 行，见 StreamPrinter），--no-stream 回到生成完毕后整段打印。
    --debug 打印框架注入了什么、录入了什么、执行了哪些工具（写入去向：
    ingested / working_only / restatement_skipped / gated），用于观察
    静默优先与复述识别是否按预期工作。
    """
    from memagent.agent import AgentLoop
    printer = None if args.no_stream else StreamPrinter()
    loop = AgentLoop(store, task_context=args.context, use_llm=not args.no_llm,
                     enable_tools=not args.no_tools,
                     tool_max_rounds=args.tool_rounds,
                     on_delta=printer.feed if printer else None)
    print("记忆原生循环（输入空行或 Ctrl-C 退出；--debug 看注入/录入/工具；"
          "--no-tools 关工具；/deep 切换深搜记忆检索）")
    print("（LLM 通道探测正在后台预热，可直接开始输入）")
    import threading
    threading.Thread(target=_warmup_probes, daemon=True).start()
    try:
        while True:
            try:
                user_text = input("\n你: ").strip()
            except EOFError:
                break
            if not user_text:
                break
            turn = loop.turn(user_text)
            if args.debug:
                ctx = turn.injection.context or "（无相关记忆）"
                print(f"[注入 {len(turn.injection.injected_texts)} 条]\n{ctx}")
            if printer is not None and printer.printed:
                print()   # 流式轮收尾：最后一段增量行未以换行结束，补打一个
            else:
                print(f"助手: {turn.assistant_text}")
            if args.debug:
                for tc in turn.tool_calls:
                    print(f"  [工具] {tc['name']} ok={tc['ok']} args={tc['args']}"
                          f" -> {tc['result'][:100]}")
                _print_record_sides(turn)
    except KeyboardInterrupt:
        print()
    s = loop.stats
    print(f"\n本轮会话: {s['turns']} 轮，长期写入 {s['ingested']} 次，"
          f"仅工作记忆 {s['working_only']} 次，复述拦截 {s['restatement_skipped']} 次，"
          f"工具调用 {s['tools']} 次")
    return 0


def _positive_int(value: str) -> int:
    n = int(value)
    if n <= 0:
        raise argparse.ArgumentTypeError(f"须为正整数，收到 {value!r}")
    return n


def _build_parser() -> argparse.ArgumentParser:
    """CLI 参数解析（P1-4 从 main() 抽出为模块级函数：单测可直接 parse_args
    验证旗标，不必跑起整个入口；行为与原内联构造逐行一致）。"""
    parser = argparse.ArgumentParser(description="memory-agent: 类脑持续学习记忆系统")
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("init", help="初始化数据库")
    sub.add_parser("tui", help="交互式终端界面（全屏 TUI）")
    p_agent = sub.add_parser("agent", help="记忆原生对话循环（自动注入记忆 + 静默优先录入 + 记忆工具）")
    p_agent.add_argument("--context", default="", help="任务域（经验/技能的事实键前缀）")
    p_agent.add_argument("--debug", action="store_true", help="打印注入的片段、本轮录入去向与工具调用")
    p_agent.add_argument("--no-stream", action="store_true",
                         help="禁用流式输出（等生成完毕后整段打印）")
    p_agent.add_argument("--no-llm", action="store_true", help="禁用 LLM 生成（只看记忆注入）")
    p_agent.add_argument("--no-tools", action="store_true",
                         help="禁用模型工具调用（只保留自动注入与静默录入）")
    p_agent.add_argument("--tool-rounds", type=int, default=None,
                         help="工具调用轮数上限（默认读 MEMAGENT_TOOL_MAX_ROUNDS，缺省 3；0=禁用工具轮）")
    p_add = sub.add_parser("add", help="写入一条事件")
    p_add.add_argument("content")
    p_add.add_argument("--source", default="user")
    p_add.add_argument("--type", default="observation")
    p_add.add_argument("--context", default="")
    p_add.add_argument("--outcome", default="", choices=["", "success", "failure"])
    p_add.add_argument("--no-llm", action="store_true", help="禁用 LLM 打分")

    p_ret = sub.add_parser("retrieve", help="检索记忆")
    p_ret.add_argument("query")
    p_ret.add_argument("--top-k", type=_positive_int, default=settings.RETRIEVE_TOP_K,
                       help="返回条数（正整数；负数会触发列表反向切片，已禁止）")
    p_ret.add_argument("--deep", action="store_true",
                       help="深搜：用 LLM 展开查询多路召回（慢且消耗一次对话调用；"
                            "只读不写回，LLM 不可用时自动降级普通检索）")

    sub.add_parser("inspect", help="查看全部记忆")
    sub.add_parser("consolidate", help="睡眠巩固")
    sub.add_parser("forget", help="主动遗忘")

    p_rep = sub.add_parser("repetition", help="间隔重复队列")
    p_rep.add_argument("--review", type=int, help="复习某条语义记忆的 id")
    p_rep.add_argument("--quality", type=int, default=4, choices=range(6),
                       help="回忆质量 0~5（越界会扭曲 SM-2 曲线，已禁止）")

    sub.add_parser("status", help="系统状态")
    sub.add_parser("test", help="LLM 通道连通性诊断（五项检查，主聊天对话失败退出码 1）")

    p_ree = sub.add_parser("reembed", help="嵌入回填（统一向量维度）")
    p_ree.add_argument("--apply", action="store_true", help="执行回填（默认 dry-run）")
    p_ree.add_argument("--batch-size", type=_positive_int, default=50)

    p_alias = sub.add_parser("alias", help="实体别名表管理")
    p_alias_sub = p_alias.add_subparsers(dest="alias_cmd", required=True)
    p_add_alias = p_alias_sub.add_parser("add")
    p_add_alias.add_argument("alias")
    p_add_alias.add_argument("canonical")
    p_alias_sub.add_parser("remove").add_argument("alias")
    p_alias_sub.add_parser("list")

    p_res = sub.add_parser("resolve-entities", help="实体归一迁移")
    p_res.add_argument("--apply", action="store_true", help="执行归一（默认 dry-run）")

    p_fts = sub.add_parser("rebuild-fts", help="FTS 索引重建为中文 2-gram 分词")
    p_fts.add_argument("--apply", action="store_true", help="执行重建（默认 dry-run）")

    p_dedup = sub.add_parser("dedupe-facts", help="语义事实跨键近重复合并（B1 存量治理）")
    p_dedup.add_argument("--apply", action="store_true", help="执行合并（默认 dry-run）")

    p_clean = sub.add_parser("cleanup-garbage",
                             help="存量垃圾清理（P0-3：按显式清单归档/修正，默认 dry-run）")
    p_clean.add_argument("--plan", required=True,
                         help="清理清单：JSON 字符串或 JSON 文件路径（archive_episodic / "
                              "fix_episodic_category / archive_semantic / archive_procedural "
                              "四个动作键可选）")
    p_clean.add_argument("--apply", action="store_true",
                         help="执行清理（默认 dry-run；执行前自动热备份到 data/backups/）")

    p_his = sub.add_parser("history", help="查看事实版本链")
    p_his.add_argument("entity")
    p_his.add_argument("relation", nargs="?", default=None)

    p_conf = sub.add_parser("conflicts", help="冲突裁决")
    p_conf_sub = p_conf.add_subparsers(dest="conflicts_cmd")
    p_res2 = p_conf_sub.add_parser("resolve")
    p_res2.add_argument("conflict_id", type=int)
    p_res2.add_argument("--accept-new", dest="resolution", action="store_const",
                        const="accept-new", default=None)
    p_res2.add_argument("--keep-old", dest="resolution", action="store_const",
                        const="keep-old")
    p_res2.add_argument("--both", dest="resolution", action="store_const",
                        const="both", help="判定误报：双方共存（互补事实同时生效，不取代）")
    p_conf_sub.add_parser("list")

    sub.add_parser("eval-mini", help="运行 mini golden 评测（离线确定性）")
    sub.add_parser("eval", help="运行完整评测（golden + scenarios，离线确定性）")
    sub.add_parser("report", help="输出记忆健康报告")
    p_vault = sub.add_parser("export-vault", help="导出 Obsidian vault 快照（记忆图谱可视化）")
    p_vault.add_argument("dir", help="输出目录（Obsidian vault 根）")
    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()
    store = _storage()
    try:
        handler = {
            "init": cmd_init, "tui": cmd_tui, "agent": cmd_agent,
            "add": cmd_add, "retrieve": cmd_retrieve,
            "inspect": cmd_inspect, "consolidate": cmd_consolidate,
            "forget": cmd_forget, "repetition": cmd_repetition, "status": cmd_status,
            "test": cmd_test,
            "reembed": cmd_reembed, "alias": cmd_alias, "resolve-entities": cmd_resolve_entities,
            "history": cmd_history, "conflicts": cmd_conflicts, "eval-mini": cmd_eval_mini,
            "eval": cmd_eval, "report": cmd_report, "rebuild-fts": cmd_rebuild_fts,
            "dedupe-facts": cmd_dedupe_facts,
            "cleanup-garbage": cmd_cleanup_garbage,
            "export-vault": cmd_export_vault,
        }[args.cmd]
        return handler(store, args)
    finally:
        store.close()


if __name__ == "__main__":
    sys.exit(main())
