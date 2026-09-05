"""通用评测 harness：用场景 DSL 回放完整记忆管线并打分。

DSL 操作（每行一个步骤）：
  add            {content, type?, outcome?}          走完整写入管线（离线规则）
  add_fact       {entity, relation, value, confidence, source, status?}  直接注入事实（测分级策略）
  alias          {alias, canonical}
  retrieve       {query, top_k?}                     记录最近一次检索结果
  resolve        {resolution, conflict_id?}          裁决 pending 冲突（缺省第一条）
  add_archived   {summary, importance?, created_at?} 种入陈旧归档记忆（测摘要降级）
  forget                                             离线遗忘（LLM 不可用 -> 硬删路径）
  forget_online  {summary_text}                      在线遗忘（摘要文本固定，确定性）
  consolidate                                        睡眠巩固
  time_travel    {days}                              平移系统时钟 N 天（确定性测保留曲线/SM-2）
断言（计入四项指标）：
  expect_retrieved     {value}            -> 偏好命中率
  expect_active        {value, entity?}   -> 冲突消解正确率
  expect_pending       {value}            -> 冲突消解正确率
  expect_no_conflict   {}                 -> 冲突消解正确率
  expect_history       {count, entity?}   -> 版本链完整率
  expect_evidence      {value, count}     -> 版本链完整率
  expect_repeat_stable {query}            -> 重复提问稳定率（在线指标的离线代理）
拟人度扩展指标（V1.5 E0，红灯基线期不计入总达标）：
  expect_sm2_enrolled         {value}                -> 保留曲线（检索命中应 enroll 进 SM-2）
  expect_sm2_interval_growing {value}                -> 保留曲线（间隔随复习单调增长）
  expect_retention_advantage  {recalled, control}    -> 保留曲线（被检索记忆强度 > 同龄对照）
  expect_emotion_advantage    {emotional, neutral}   -> 情感区分度（E2 前红灯）
  expect_associated           {value}                -> 联想召回率（E4 前红灯）
  expect_working_hit          {value}                -> 会话命中率（E3 前红灯）
  expect_uncertain            {query}                -> 元认知校准（E7：检索不可信时系统自知）
  expect_confident            {query}                -> 元认知校准（E7：有把握时无不确定标记）
经验通道扩展指标（V1.6 E8，不计入总达标）：
  expect_experience_hit  {value}          -> 经验通道（E8 前红灯：绿色类型过不了门控）
  expect_env_state       {entity, active_contains, rows}
                                            -> 经验通道（环境状态快取代：同键新版接管旧版）
  expect_experience_archived {count}       -> 经验通道（LFU：经验层低频先沉底）
  expect_skill_hit       {value}           -> 经验通道（程序记忆激活：检索带出技能）
试用期转正扩展指标（V1.7 P1，红灯基线期计入总达标待定）：
  expect_promoted            {value}       -> 转正精确率（B 类低置信事实够用即转正）
  expect_still_pending       {value}       -> 转正精确率（未达阈值不得转正）
  expect_no_promote_on_conflict {value}    -> 转正精确率（A 类冲突待裁型豁免自动转正）
  expect_probation           {value}       -> 转正精确率（试用期事实打折进场且带 meta.probation）
Agent 循环扩展指标（V1.7 P2，离线可测——助手文本由场景给出，不依赖真实模型）：
  agent_turn    {user, assistant?, context?, outcome?}  走一轮循环（注入 → 生成/回放 → 录入）
  expect_consistent          {value}       -> 跨轮记忆一致性（第 N 轮再问仍能注入该事实）
  expect_no_contradiction    {value, alt}  -> 跨轮记忆一致性（被纠正的旧值不再以语义事实注入）
  expect_memory_growth       {episodic_min?, episodic_max?, semantic_min?, semantic_max?, procedural_max?}
                                           -> 记忆增长率（长期记忆随信号增长，不随轮次爆炸）
  expect_restatement_skipped {count}       -> 记忆增长率（复述注入内容不沉淀的累计次数）
跨键近重复合并扩展指标（B1，不计入总达标）：
  expect_semantic_active_count {max?, exact?}
                                           -> 近重复合并（同一知识点的跨键变体收敛后
                                              的库内 active 语义事实总数）
注入门槛扩展指标（B3，不计入总达标）：
  expect_injection_absent_longterm {}      -> 注入门槛（寒暄/零信息轮不得把长期记忆
                                              （语义+情景）注入上下文；working 不限）
助手噪声入库扩展指标（P0-5，防复发尺子——既有指标全在测「该记的记没记」，
这一项量「不该记的进没进」）：
  say            {source?, text}           助手轮次：走真实 record_turn 录入（复述拦截、
                                            信号检测、working 双写原样生效，不绕机制）
  expect_no_assistant_ingest {}            -> 助手噪声入库率（助手未点名内容零长期入库：
                                            experience 情景层与技能表零新增，0 条 = 满分）
"""
from __future__ import annotations

import glob
import json
import os
import tempfile
from unittest import mock

from memagent import settings
from memagent.adapters import llm
from memagent.consolidation.conflict_resolver import resolve_conflict, store_fact_with_conflict_check
from memagent.core import EpisodicMemory, SemanticFact
from memagent.core import clock
from memagent.core.vectors import hash_embed
from memagent.encoding.entity_resolver import normalize_value, resolve_fact
from memagent.eval.mini import offline_mode
from memagent.forgetting import run_forgetting
from memagent.consolidation import consolidate as run_consolidate
from memagent.pipeline import ingest_event
from memagent.retrieval import retrieve
from memagent.storage import SqliteStore

# 前四项为 V0.4 核心指标（计入达标）；后五项为 V1.5 拟人度扩展；experience 为 V1.6
# 经验通道扩展；promotion 为 V1.7 P1 试用期转正扩展；consistency / growth 为 V1.7
# P2 Agent 循环扩展（离线可测：助手文本由场景脚本给出，不依赖真实模型）；
# dedup 为 B1 跨键近重复合并扩展、inject_gate 为 B3 注入门槛扩展（不入达标门槛，
# 与 promotion 等扩展指标惯例一致）；assistant_noise 为 P0-5 防复发尺子——十五项
# 指标全在测「该记的记没记」，这一项补上「不该记的进没进」（助手未点名内容零入库）
METRIC_KEYS = ("preference", "conflict", "integrity", "repeat",
               "retention", "emotion", "association", "working", "metacognition",
               "experience", "promotion", "consistency", "growth", "dedup",
               "inject_gate", "assistant_noise")

# 断言 -> 指标归属：场景中途异常时，剩余断言按此表计入失败（红灯必须被完整计量）
_METRIC_OF = {
    "expect_retrieved": "preference",
    "expect_active": "conflict", "expect_pending": "conflict", "expect_no_conflict": "conflict",
    "expect_history": "integrity", "expect_evidence": "integrity",
    "expect_summary": "integrity", "expect_episodic_status": "integrity",
    "expect_repeat_stable": "repeat",
    "expect_sm2_enrolled": "retention", "expect_sm2_interval_growing": "retention",
    "expect_retention_advantage": "retention",
    "expect_emotion_advantage": "emotion",
    "expect_associated": "association",
    "expect_working_hit": "working",
    "expect_uncertain": "metacognition", "expect_confident": "metacognition",
    "expect_experience_hit": "experience", "expect_env_state": "experience",
    "expect_experience_archived": "experience", "expect_skill_hit": "experience",
    "expect_promoted": "promotion", "expect_still_pending": "promotion",
    "expect_no_promote_on_conflict": "promotion", "expect_probation": "promotion",
    "expect_consistent": "consistency", "expect_no_contradiction": "consistency",
    "expect_memory_growth": "growth", "expect_restatement_skipped": "growth",
    "expect_semantic_active_count": "dedup",
    "expect_injection_absent_longterm": "inject_gate",
    "expect_no_assistant_ingest": "assistant_noise",
}


class Harness:
    def __init__(self):
        self.store: SqliteStore | None = None
        self.last_hits = []
        self.last_context = ""
        self._loop = None
        self.failures: list[str] = []
        # P0-5 助手噪声入库：_assistant_noise 逐条留痕 say 轮的沉淀违规；
        # _noise_base 是「经验层情景 + 技能表」的基线快照（最近一次非 say 步骤后重摄）
        self._assistant_noise: list[str] = []
        self._noise_base: tuple[int, int] = (0, 0)
        self.metrics = {k: [0, 0] for k in METRIC_KEYS}  # [pass, total]

    # ---------- 断言辅助 ----------
    def _check(self, metric: str, ok: bool, label: str) -> None:
        self.metrics[metric][1] += 1
        if ok:
            self.metrics[metric][0] += 1
        else:
            self.failures.append(label)

    # ---------- 操作 ----------
    def op_add(self, step):
        ingest_event(self.store, step["content"], type=step.get("type", "observation"),
                     task_context=step.get("context", ""),
                     outcome=step.get("outcome", ""), use_llm=False)

    def op_add_fact(self, step):
        # status 可选：缺省 active；显式 "pending" 用于种入 B 类「低置信新事实」
        # （与谁都不冲突、只是证据不足）——P1 转正通道的作用对象。A 类冲突待裁型
        # 由 _store_fact 在有冲突时自动置 pending，无需场景指定。
        fact = SemanticFact(entity=step["entity"], relation=step.get("relation", "prefers"),
                            value=step["value"], confidence=step.get("confidence", 0.9),
                            status=step.get("status", "active"),
                            embedding=hash_embed(step["value"], settings.EMBED_FALLBACK_DIM))
        resolve_fact(fact, self.store.aliases.as_map())
        store_fact_with_conflict_check(self.store, fact, source=step.get("source", "model"))

    def op_alias(self, step):
        self.store.aliases.add(step["alias"], step["canonical"], source="scenario")

    def op_retrieve(self, step):
        self.last_hits = retrieve(self.store, step["query"], top_k=step.get("top_k", 5))

    def op_resolve(self, step):
        cid = step.get("conflict_id")
        if cid is None:
            pending = self.store.conflicts.fetch_all(status="pending")
            if not pending:
                raise AssertionError("没有待裁决冲突")
            cid = pending[0]["conflict_id"]
        resolve_conflict(self.store, cid, step["resolution"])

    def op_add_archived(self, step):
        self.store.episodic.add(EpisodicMemory(
            summary=step["summary"], importance=step.get("importance", 0.05),
            created_at=step.get("created_at", "2020-01-01T00:00:00+00:00"),
            strength=0.01, status="archived",
            embedding=hash_embed(step["summary"], settings.EMBED_FALLBACK_DIM)))

    def op_forget(self, step):
        with mock.patch.object(llm, "llm_available", return_value=False):
            run_forgetting(self.store)

    def op_forget_online(self, step):
        text = step.get("summary_text", "用户多次表达同类偏好")
        with mock.patch.object(llm, "llm_available", return_value=True), \
             mock.patch.object(llm, "chat", return_value=text), \
             mock.patch.object(llm, "embed",
                               side_effect=lambda t: hash_embed(t, settings.EMBED_FALLBACK_DIM)):
            run_forgetting(self.store)

    def op_consolidate(self, step):
        run_consolidate(self.store)

    def op_time_travel(self, step):
        clock.set_offset(days=step.get("days", 0), hours=step.get("hours", 0))

    # ---------- 断言 ----------
    def _active_values(self):
        return [f.value for f in self.store.semantic.fetch(status="active")]

    def op_expect_retrieved(self, step):
        value = normalize_value(step["value"])
        # 对称归一比较：事实值入库时 NFKC 归一，情景 summary 保留原文
        ok = any(value in normalize_value(h.text) for h in self.last_hits)
        self._check("preference", ok, f"expect_retrieved 未命中: {step['value']}")

    def op_expect_active(self, step):
        value = normalize_value(step["value"])
        self._check("conflict", value in self._active_values(),
                    f"expect_active 失败: {step['value']} / actives={self._active_values()}")

    def op_expect_pending(self, step):
        value = normalize_value(step["value"])
        pending = [f.value for f in self.store.semantic.fetch(status="pending")]
        self._check("conflict", value in pending,
                    f"expect_pending 失败: {step['value']} / pending={pending}")

    def op_expect_no_conflict(self, step):
        n = len(self.store.conflicts.fetch_all(status="pending"))
        self._check("conflict", n == 0, f"expect_no_conflict 失败: 存在 {n} 条待裁决冲突")

    def op_expect_summary(self, step):
        summaries = [m for m in self.store.episodic.fetch(status="active") if m.is_summary]
        ok = len(summaries) == 1 and len(summaries[0].source_ids) == step["source_count"]
        self._check("integrity", ok,
                    f"expect_summary 失败: {[m.source_ids for m in summaries]}")

    def op_expect_episodic_status(self, step):
        n = len(self.store.episodic.fetch(status=step["status"], limit=10 ** 9))
        self._check("integrity", n == step["count"],
                    f"expect_episodic_status[{step['status']}]: {n} != {step['count']}")

    def op_expect_history(self, step):
        entity = step.get("entity", "user")
        total = len(self.store.conn.execute(
            "SELECT id FROM semantic WHERE entity=?", (entity,)).fetchall())
        self._check("integrity", total == step["count"],
                    f"expect_history 失败: {entity} 有 {total} 行，期望 {step['count']}")

    def op_expect_evidence(self, step):
        value = normalize_value(step["value"])
        facts = [f for f in self.store.semantic.fetch() if f.value == value]
        ok = facts and facts[0].evidence_count == step["count"]
        self._check("integrity", ok,
                    f"expect_evidence 失败: {step['value']} 证据 {facts[0].evidence_count if facts else 0}，期望 {step['count']}")

    def op_expect_repeat_stable(self, step):
        first = retrieve(self.store, step["query"], top_k=5)
        second = retrieve(self.store, step["query"], top_k=5, boost_access=False)
        top1_same = bool(first and second) and first[0].id == second[0].id and first[0].kind == second[0].kind
        self._check("repeat", top1_same,
                    f"expect_repeat_stable 失败: {step['query']} 两轮 top1 不一致")

    # ---------- 拟人度扩展断言（V1.5 E0）----------
    def _semantic_by_value(self, value: str):
        value = normalize_value(value)
        for f in self.store.semantic.fetch(status="active"):
            if value in normalize_value(f.value):
                return f
        return None

    def _episodic_strength(self, value: str) -> float | None:
        value = normalize_value(value)
        for m in self.store.episodic.fetch(status="active", limit=10 ** 9):
            if value in normalize_value(m.summary):
                return m.strength
        return None

    def _repetition_row(self, memory_id: int) -> dict | None:
        row = self.store.conn.execute(
            "SELECT * FROM repetition WHERE memory_id=?", (memory_id,)).fetchone()
        return dict(row) if row else None

    def op_expect_sm2_enrolled(self, step):
        fact = self._semantic_by_value(step["value"])
        row = self._repetition_row(fact.id) if fact else None
        self._check("retention", bool(row) and row["repetitions"] >= 1,
                    f"expect_sm2_enrolled 失败: {step['value']} 未进入 SM-2 复习计划")

    def op_expect_sm2_interval_growing(self, step):
        fact = self._semantic_by_value(step["value"])
        row = self._repetition_row(fact.id) if fact else None
        ok = bool(row) and row["repetitions"] >= 2 and row["interval_days"] > 1
        self._check("retention", ok,
                    f"expect_sm2_interval_growing 失败: {step['value']} 计划={dict(row) if row else None}（间隔未随复习增长）")

    def op_expect_retention_advantage(self, step):
        s_recalled = self._episodic_strength(step["recalled"])
        s_control = self._episodic_strength(step["control"])
        ok = s_recalled is not None and s_control is not None and s_recalled > s_control
        self._check("retention", ok,
                    f"expect_retention_advantage 失败: 被检索 {s_recalled} vs 对照 {s_control}")

    def op_expect_emotion_advantage(self, step):
        s_emo = self._episodic_strength(step["emotional"])
        s_neu = self._episodic_strength(step["neutral"])
        ok = s_emo is not None and s_neu is not None and s_emo > s_neu
        self._check("emotion", ok,
                    f"expect_emotion_advantage 失败: 情感事件 {s_emo} vs 中性事件 {s_neu}（强度模型无情感分项）")

    def op_expect_associated(self, step):
        # 断言"联想通道"而非文本碰巧出现：联想命中必须带 associated 标记
        # （E4 activation 落地前不存在该通道，确定性红灯）
        value = normalize_value(step["value"])
        ok = any(value in normalize_value(h.text) and h.meta.get("associated")
                 for h in self.last_hits)
        self._check("association", ok,
                    f"expect_associated 失败: {step['value']} 未经联想通道带出（直接检索不达且无激活扩散）")

    def op_expect_working_hit(self, step):
        value = normalize_value(step["value"])
        ok = any(value in normalize_value(h.text) for h in self.last_hits)
        self._check("working", ok,
                    f"expect_working_hit 失败: {step['value']} 未命中（会话级工作记忆缺失，未过门控即蒸发）")

    # ---------- 元认知校准断言（V1.5 E7）----------
    def op_expect_uncertain(self, step):
        # 反面：检索不到/检索不可信时系统必须自知——命中为空，或所有直接命中都带
        # uncertain 标记（top 直接命中最高分 < 置信线，系统承认这些结果可能是噪声）
        self.last_hits = retrieve(self.store, step["query"], top_k=step.get("top_k", 5))
        direct = [h for h in self.last_hits if h.kind != "working"]
        ok = not direct or all(h.meta.get("uncertain") for h in direct)
        self._check("metacognition", ok,
                    f"expect_uncertain 失败: {step['query']} 检索不可信却无自知（"
                    f"直接命中={[f'{h.kind}:{h.score}' for h in direct]}）")

    def op_expect_confident(self, step):
        # 正面：有把握时别犹疑——命中非空且无任何 uncertain 标记
        self.last_hits = retrieve(self.store, step["query"], top_k=step.get("top_k", 5))
        ok = bool(self.last_hits) and not any(h.meta.get("uncertain") for h in self.last_hits)
        self._check("metacognition", ok,
                    f"expect_confident 失败: {step['query']} 应有把握命中却标了不确定（"
                    f"hits={[f'{h.kind}:{h.score}' for h in self.last_hits]}）")

    # ---------- 经验通道断言（V1.6 E8）----------
    def op_expect_experience_hit(self, step):
        # 断言"经验绿色通道"而非文本碰巧出现：长期命中必须带 meta.green 通道标记
        # （E8 落地前绿色类型过不了门控，只有 working 会话级命中，确定性红灯）
        value = normalize_value(step["value"])
        ok = any(value in normalize_value(h.text) and h.meta.get("green")
                 for h in self.last_hits)
        self._check("experience", ok,
                    f"expect_experience_hit 失败: {step['value']} 未从长期经验层命中（"
                    f"绿色类型被门控拦在长期记忆之外）")

    def op_expect_env_state(self, step):
        # 环境状态快取代：同 (工具名, env_state) 键下，新版 active、旧版 superseded、
        # 总行数 = rows（版本链保留可回溯，不是平行堆积）
        entity = step["entity"]
        rows = self.store.conn.execute(
            "SELECT value, status FROM semantic WHERE entity=? AND relation='env_state'",
            (entity,)).fetchall()
        active = [r["value"] for r in rows if r["status"] == "active"]
        ok = (len(rows) == step["rows"] and len(active) == 1
              and normalize_value(step["active_contains"]) in normalize_value(active[0]))
        self._check("experience", ok,
                    f"expect_env_state 失败: {entity} rows={[dict(r) for r in rows]}，"
                    f"期望 {step['rows']} 行且 active 含「{step['active_contains']}」")

    def op_expect_experience_archived(self, step):
        # LFU：经验层低频先沉底——category='experience' 且已归档的条数
        rows = self.store.conn.execute(
            "SELECT id, summary FROM episodic WHERE category='experience' "
            "AND status='archived'").fetchall()
        self._check("experience", len(rows) == step["count"],
                    f"expect_experience_archived 失败: 经验层归档 {len(rows)} 条 "
                    f"({[r['summary'][:20] for r in rows]})，期望 {step['count']}")

    def op_expect_skill_hit(self, step):
        # 程序记忆激活：检索结果里带出 kind='procedural' 的技能命中
        # （E8 前检索只回 working/episodic/semantic，procedural 存而不用，确定性红灯）
        value = normalize_value(step["value"])
        ok = any(h.kind == "procedural" and value in normalize_value(h.text)
                 for h in self.last_hits)
        self._check("experience", ok,
                    f"expect_skill_hit 失败: {step['value']} 未被程序记忆激活带出"
                    f"（技能库存而不取，经验消费闭环断裂）")

    # ---------- 试用期转正断言（V1.7 P1）----------
    def _values_of(self, status: str) -> list[str]:
        return [f.value for f in self.store.semantic.fetch(status=status)]

    def _fact_by_value(self, value: str):
        value = normalize_value(value)
        for status in ("pending", "active", "archived", "superseded"):
            for f in self.store.semantic.fetch(status=status):
                if value in normalize_value(f.value):
                    return f
        return None

    def op_expect_promoted(self, step):
        # B 类低置信新事实：被反复检索（hit_count 达标）应自动转正为 active
        self._check("promotion", normalize_value(step["value"]) in self._values_of("active"),
                    f"expect_promoted 失败: {step['value']} 仍在试用期（"
                    f"active={self._values_of('active')} / pending={self._values_of('pending')}）")

    def op_expect_still_pending(self, step):
        # 反面：命中次数不足（或根本没被想起）的事实必须留在试用期——
        # 转正是「够用」的证明，不是时间到了自动发放
        self._check("promotion", normalize_value(step["value"]) in self._values_of("pending"),
                    f"expect_still_pending 失败: {step['value']} 不该转正（"
                    f"active={self._values_of('active')} / pending={self._values_of('pending')}）")

    def op_expect_no_promote_on_conflict(self, step):
        # A 类冲突待裁型豁免：memory_conflict 表存在以该事实为 new_id 的 pending 行
        # ——转正意味着取代与之互斥的 active 旧事实，只能由裁决（accept-new/keep-old/
        # both）决定，命中次数再多也不得自动转正。
        fact = self._fact_by_value(step["value"])
        if fact is None:
            self._check("promotion", False,
                        f"expect_no_promote_on_conflict 失败: 找不到事实 {step['value']}")
            return
        conflicts = [c for c in self.store.conflicts.fetch_all(status="pending")
                     if c["new_id"] == fact.id]
        ok = fact.status == "pending" and bool(conflicts)
        self._check("promotion", ok,
                    f"expect_no_promote_on_conflict 失败: #{fact.id} {step['value']} "
                    f"status={fact.status} 待裁冲突={len(conflicts)}（冲突型被自动转正=取代旧事实）")

    def op_expect_probation(self, step):
        # 试用期事实「打折进场」：可检索（否则永远转不了正），但同分事实中必须
        # 排在 active 之后，且带 meta.probation 供上层识别「这条还在试用期」。
        value = normalize_value(step["value"])
        matched = [h for h in self.last_hits
                   if h.kind == "semantic" and value in normalize_value(h.text)]
        probation = [h for h in matched if h.meta.get("probation")]
        full = [h for h in matched if not h.meta.get("probation")]
        ok = (bool(probation) and bool(full)
              and max(h.score for h in full) > max(h.score for h in probation))
        self._check("promotion", ok,
                    f"expect_probation 失败: {step['value']} 试用期事实未打折进场（"
                    f"正式={[h.score for h in full]} 试用={[h.score for h in probation]}）")

    # ---------- Agent 循环（V1.7 P2）----------
    def op_agent_turn(self, step):
        """走一轮记忆原生循环：注入 → 生成（场景脚本给定则回放）→ 录入。

        助手文本由场景给出是**确定性的来源**：离线无真实模型，若让循环自己生成，
        测的就不是记忆机制而是随机性。生成路径本身由单测覆盖。
        """
        from memagent.agent import AgentLoop  # 入口层：按需导入，避免与被测模块循环
        if self._loop is None:
            self._loop = AgentLoop(self.store, use_llm=False)
        turn = self._loop.turn(step["user"], assistant_text=step.get("assistant"),
                               task_context=step.get("context", ""),
                               outcome=step.get("outcome", ""))
        self.last_hits = turn.injection.hits
        self.last_context = turn.injection.context

    def op_expect_consistent(self, step):
        # 跨轮一致性：第 N 轮再问同一件事，框架自动注入的上下文里仍须有该事实。
        # 退化（答不出）与矛盾（答出旧值）是本指标要挡的两种失败形态。
        value = normalize_value(step["value"])
        ok = value in normalize_value(self.last_context)
        self._check("consistency", ok,
                    f"expect_consistent 失败: 注入上下文中没有「{step['value']}」"
                    f"（跨轮记忆退化）context={self.last_context[:200]!r}")

    def op_expect_no_contradiction(self, step):
        # 反面：被纠正的旧值不得再以**语义事实**形态注入（冲突消解取代链生效）。
        # 只约束语义层——情景层保留原始事件是分层设计的固有属性（ARCH §1.1
        # 「情景记忆：冲突策略不适用（保留原始事件）」），不由本断言覆盖。
        value, alt = normalize_value(step["value"]), normalize_value(step["alt"])
        sem = [normalize_value(h.text) for h in self.last_hits if h.kind == "semantic"]
        ok = any(value in t for t in sem) and not any(alt in t for t in sem)
        self._check("consistency", ok,
                    f"expect_no_contradiction 失败: 语义命中={sem}，"
                    f"期望含「{step['value']}」且不含「{step['alt']}」")

    def op_expect_memory_growth(self, step):
        # 记忆增长率：20 轮对话后长期记忆不爆炸。双侧断言——上界防「每轮都写」的
        # 灌水（信号检测太松会让绿色通道变成漏斗，保底入库不经门控过滤），
        # 下界防「什么都不敢写」的假绿（静默优先过头 = 记忆系统空转）。
        epi = len(self.store.episodic.fetch(status="active", limit=10 ** 9))
        sem = len(self.store.semantic.fetch(status="active", limit=10 ** 9)) + \
            len(self.store.semantic.fetch(status="pending", limit=10 ** 9))
        proc = len(self.store.procedural.fetch())
        bad = []
        for key, actual in (("episodic", epi), ("semantic", sem), ("procedural", proc)):
            lo, hi = step.get(f"{key}_min"), step.get(f"{key}_max")
            if lo is not None and actual < lo:
                bad.append(f"{key}={actual} < {lo}")
            if hi is not None and actual > hi:
                bad.append(f"{key}={actual} > {hi}")
        self._check("growth", not bad,
                    f"expect_memory_growth 失败: {', '.join(bad)}"
                    f"（情景={epi} 语义={sem} 技能={proc}）")

    def op_expect_restatement_skipped(self, step):
        # 自增强防护：模型复述刚注入的内容不得沉淀。累计次数由循环统计（审计表里
        # 另有 restatement_skipped 逐条留痕，此处比对数而非查表，避免耦合存储细节）。
        actual = self._loop.stats["restatement_skipped"] if self._loop else 0
        self._check("growth", actual == step["count"],
                    f"expect_restatement_skipped 失败: 拦截 {actual} 次，期望 {step['count']}")

    # ---------- 跨键近重复合并断言（B1）----------
    def op_expect_semantic_active_count(self, step):
        # 库内 active 语义事实总数：同一知识点的多个措辞变体经写入侧吸收 / 存量
        # 治理后应收敛到一份（max=上限防平行堆积）；exact 用于反面对照——
        # 相似度不足的不同知识不得被误并（宁漏勿错杀）。
        n = len(self.store.semantic.fetch(status="active", limit=10 ** 9))
        ok = True
        if "exact" in step:
            ok = ok and n == step["exact"]
        if "max" in step:
            ok = ok and n <= step["max"]
        want = step["exact"] if "exact" in step else f"≤{step['max']}"
        self._check("dedup", ok,
                    f"expect_semantic_active_count 失败: active={n}，期望 {want}")

    # ---------- 注入门槛断言（B3）----------
    def op_expect_injection_absent_longterm(self, step):
        # 注入门槛·反面（B3）：寒暄/零信息轮次不得把长期记忆（语义+情景）拼进
        # 上下文——「自动注入前的筛选才是关键：只注入高相关/高唤醒/未过时的事实，
        # 否则挤占上下文窗口」。判定对象是本轮注入的命中（agent_turn 的
        # turn.injection.hits，与 expect_consistent 消费的 last_context 同源）：
        # working 是当下上下文（永远相关）不受此约束；procedural 技能带出跟随
        # 触发词而非相关度，也不在语义/情景之列。正向防误杀断言复用 P2 的
        # expect_consistent（有相关查询时该事实仍须被注入）。
        leaked = [f"{h.kind}:{h.text[:24]}" for h in self.last_hits
                  if h.kind in ("semantic", "episodic")]
        self._check("inject_gate", not leaked,
                    f"expect_injection_absent_longterm 失败: 零信息轮注入了长期记忆 "
                    f"{leaked}（注入相关性门槛失效）")

    # ---------- 助手噪声入库（P0-5：量「不该记的进没进」）----------
    def _noise_counts(self) -> tuple[int, int]:
        """长期库「经验层情景 + 技能表」行数快照——助手未点名内容仅有的两个落点。

        experience 是 E8 绿色类型（宽进严出）：信号/词表一旦命中即无任何闸门能拦，
        P0 三层污染事故灌进的就是这两张表（情景 category='experience' + 技能名
        截断行），故噪声口径只对这两处计数。
        """
        exp = self.store.conn.execute(
            "SELECT COUNT(*) FROM episodic WHERE category='experience'").fetchone()[0]
        return exp, len(self.store.procedural.fetch())

    def op_say(self, step):
        """助手轮次（P0-5）：走真实 record_turn 录入助手发言——口径神圣，测的就是
        真实机制：复述拦截、信号检测、working 双写全部原样生效，不经 op_add 直写、
        不造场景替身。say 前后对噪声两张表做快照，助手文本一旦沉淀进长期记忆即
        记为一次违规（含助手侧唯一自动信号 env_state——它不在三条点名通路之列，
        按口径同样算未点名入库；环境快照应由用户侧 add --type env_statement 进入）。
        """
        if step.get("source", "assistant") != "assistant":
            raise ValueError(
                f"say 只支持助手侧发言（source={step.get('source')!r}）；"
                f"用户侧输入请用 agent_turn（带完整注入与双侧录入）")
        from memagent.agent.recorder import record_turn  # 入口层按需导入（同 op_agent_turn）
        before = self._noise_counts()
        record_turn(self.store, assistant_text=step["text"], use_llm=False)
        after = self._noise_counts()
        exp_d, proc_d = after[0] - before[0], after[1] - before[1]
        if exp_d or proc_d:
            self._assistant_noise.append(
                f"say「{step['text'][:24]}」沉淀 experience+{exp_d}/procedural+{proc_d}")

    def op_expect_no_assistant_ingest(self, step):
        """P0-5 尺子：助手未点名内容零长期入库，0 条 = 满分。两道检查：
        (1) say 轮累计违规（哪句话灌进来的，逐条留痕）；
        (2) 断言时点实际行数 vs 基线快照（最近一次非 say 步骤后重摄）——抓 say
        之外绕过点名塞进来的行（旧行为代理：直写 experience 的旁路一律现形）。
        显式点名写入（add --type experience 等合法通路）由 run_scenario 的基线
        重摄吸收，不会误伤——正例对照防「修过头」把正路也堵死。
        """
        exp, proc = self._noise_counts()
        leaks = list(self._assistant_noise)
        exp_d, proc_d = exp - self._noise_base[0], proc - self._noise_base[1]
        if exp_d > 0 or proc_d > 0:
            leaks.append(f"断言时点较基线新增 experience+{exp_d}/procedural+{proc_d}")
        self._check("assistant_noise", not leaks,
                    f"expect_no_assistant_ingest 失败: 助手未点名内容入库 {len(leaks)} 处"
                    f"（{'; '.join(leaks[:3])}）——experience/技能只认 memory_remember / "
                    f"add --type experience / remember() 三条点名通路")

    # ---------- 执行 ----------
    def run_scenario(self, scenario: dict) -> None:
        self.store = SqliteStore(os.path.join(tempfile.mkdtemp(), "scenario.db"))
        self.last_hits = []
        self.last_context = ""
        self._loop = None
        self._assistant_noise = []   # P0-5：say 轮违规留痕按场景重置
        clock.clear_offset()
        try:
            self._noise_base = self._noise_counts()  # P0-5：噪声基线 = 场景起点两张表行数
            for i, step in enumerate(scenario["steps"]):
                op = step["op"]
                handler = getattr(self, f"op_{op}", None)
                if handler is None:
                    raise ValueError(f"未知操作: {op}")
                handler(step)
                if op != "say":
                    # 非 say 步骤都是场景作者的显式意图——add --type experience 等
                    # 合法点名通路可能正当增多两张表行数，重摄基线吸收之（正例对照
                    # 不误伤）；say 不重摄：其沉淀由 op_say 留痕、由断言终审。
                    self._noise_base = self._noise_counts()
        except Exception as e:  # 场景内异常：剩余断言按所属指标计为失败
            self.failures.append(f"scenario[{scenario['scenario']}] 步骤异常: {e!r}")
            for step in scenario["steps"][i + 1:]:
                metric = _METRIC_OF.get(step["op"])
                if metric:
                    self._check(metric, False,
                                f"scenario[{scenario['scenario']}] {step['op']} 未执行（前序步骤异常）")
        finally:
            clock.clear_offset()
            self.store.close()


def run_harness(paths: list[str]) -> dict:
    """离线确定性执行一批场景文件，聚合核心四指标 + 拟人度扩展五指标。"""
    h = Harness()
    files = []
    for p in paths:
        files.extend(sorted(glob.glob(p)) if any(c in p for c in "*?") else [p])
    with offline_mode():
        for path in files:
            with open(path, encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        h.run_scenario(json.loads(line))
    return {
        "files": len(files),
        "metrics": {k: {"pass": v[0], "total": v[1],
                        "rate": round(v[0] / v[1], 4) if v[1] else 1.0}
                    for k, v in h.metrics.items()},
        "failures": h.failures,
    }
