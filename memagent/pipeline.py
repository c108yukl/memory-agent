"""记忆写入主管线：门控 -> 情感打分 -> 情景编码 -> 语义抽取（去重/归一/冲突消解）-> 技能沉淀
-> 信念分级自动修正（P2-3：L2 用户显式反驳降权 / L3 工具结果步进与双次失败翻转）。

CLI 与评测 harness 共用同一入口，保证被评测的就是被使用的。

P2-3（bounded belief revision，D4-a）：修正发生在语义事实落库**之后**——先让 L1 的
同键取代自然发生，修正只作用于「取代之后仍然成立」的信念。触发面各自收窄：
L2 只认显式参数 corrects=True（纠正信号由调用方下发，pipeline 是 L4、recorder 是 L5，
**不许 pipeline import recorder**——分层正确姿势）；L3 只认 source=tool + outcome 的
沙箱特有组合。其余路径三个 belief_* 结果键恒为空列表，行为逐位不变。
**纯模型推演永不改写 active 事实（L4 现状即断路器）**：model 来源/纯推演/低置信冲突
照旧走 pending，修正只由「用户显式反驳」与「真实工具结果」两类已证实信号驱动。
"""
from __future__ import annotations

from memagent import settings
from memagent.adapters import llm
from memagent.attention import attention_gate
from memagent.attention.emotion import analyze_emotion, emotion_label
from memagent.attention.scorers import IDENTITY_PATTERN
from memagent.consolidation.conflict_resolver import (
    dedupe_same_key,
    store_fact_with_conflict_check,
)
from memagent.core import Event
from memagent.core.domain import EXPLICIT_TYPES, GREEN_TYPES
from memagent.encoding import encode_episodic, extract_semantic_facts, extract_skills
from memagent.encoding.entity_resolver import normalize_entity, resolve_fact
from memagent.storage import SqliteStore


def _revise_user_belief(store: SqliteStore, correct_vec: list[float],
                        excluded: set[int]) -> list[int]:
    """L2 用户显式反驳降权（P2-3，D4-a）：纠正信号在场时，从 active 语义事实里找与
    纠正原文余弦 ≥ DEDUP_ABSORB_SIM 的 top-1，置信度降 BELIEF_REVISE_STEP。

    断路器·信念（安全阀，缺一不可）：
    - 触发只认 ingest_event 的显式参数 corrects=True（纠正信号在场，非文本再猜）；
    - 目标必须是 **active** 事实（cosine_search 候选池含 pending，命中后回查 status
      复核，同 conflict_resolver._find_absorb_keeper 的保守口径）；
    - 本次写入刚 supersede/新建/续证的事实一律排除（excluded）——L1 的同键取代已经
      表达了「以新说法为准」，对取代链上的行再降权是重复惩罚；刚写入的新说法更不能
      因「与纠正原文最像」而被误伤；
    - **最多降 1 条（top-1）**；只降置信不改状态不删行，下限 RIF_CONF_FLOOR 不许
      降穿（demote_confidence 已在下限时返回 None -> 不动作不审计，幂等，同 RIF 惯例）；
    - 无候选 -> 空列表，什么都不做（纠正照旧走 L1 显式声明通路）。
    审计 belief_revised(user) 可溯源。返回被降权的事实 id 列表（0 或 1 个）。
    """
    for cand_id, sim in store.semantic.cosine_search(correct_vec, limit=20):
        if sim < settings.DEDUP_ABSORB_SIM:
            break  # 按相似度降序，跌破阈值即可停止（同 B1 keeper 查法）
        if cand_id in excluded:
            continue
        cand = store.semantic.get(cand_id)
        if cand is None or cand.status != "active":
            continue
        target = store.semantic.demote_confidence(cand_id, -settings.BELIEF_REVISE_STEP)
        if target is None:  # 已在下限：不动作不审计，不再找次优（top-1 就是目标）
            return []
        store.log("semantic", cand_id, "belief_revised",
                  f"user 纠正 -{settings.BELIEF_REVISE_STEP:g} -> {target:g}")
        return [cand_id]
    return []


def _step_tool_beliefs(store: SqliteStore, task_context: str,
                       outcome: str) -> tuple[list[int], list[int]]:
    """L3 工具结果步进（P2-3，D4-a）：沙箱工具结果（source=tool + outcome）对同任务域
    的 active 经验事实做有界置信度步进；连续两次失败把整域信念收回待重证。

    - 触发面：source=="tool" 且 outcome ∈ {success, failure} 是 run_command 沙箱经
      remember() 进来的特有组合；user/assistant 等其他来源带 outcome（如显式
      add --outcome）一概不触发，现状逐位不变；
    - 步进对象 = 同任务域（entity=task_context，口径与写入侧 _green_fact 完全一致：
      strip 或缺省通用经验域后做实体归一，保证「写在哪、步进就在哪」不漂移）且
      relation=="lesson" 的全部 **active** 事实（含本次刚写入/续证的）：
      success -> touch_confidence（+0.2 封顶 1.0）；failure -> demote_confidence
      （-0.2，下限 RIF_CONF_FLOOR；已在下限不动作不审计，幂等同 RIF 惯例）；
    - **双次失败翻转**（信念收回待重证）：连续性用自己的审计行判定——每次动作前先写
      tool_outcome 审计，再数同任务域尾部连续 failure 行数（success 天然清零）；
      ≥2 -> 该域全部 active lesson 事实 status 置 pending（不是删除、不是 superseded：
      信念收回但保留行，经检索直接命中 3 次可走 P1 转正通道挣回来）+ 审计
      belief_flipped(tool)。set_status 自带的 status->pending 审计照常，双痕可溯。
    返回 (被步进的事实 id, 被翻转的事实 id)。
    """
    # 连续性凭证先行落库（先记后数：本次 outcome 必须计入尾部连续序列）
    store.log("tool", 0, "tool_outcome", f"{task_context} {outcome}")
    failures = store.audit_log.count_trailing(
        "tool_outcome", f"{task_context} ", "failure")
    domain = normalize_entity(task_context.strip() or "experience",
                              store.aliases.as_map())
    lessons = [f for f in store.semantic.active_facts_of(domain)
               if f.relation == "lesson"]
    stepped: list[int] = []
    for fact in lessons:
        if outcome == "success":
            target = min(1.0, fact.confidence + settings.BELIEF_REVISE_STEP)
            store.semantic.touch_confidence(fact.id, target)
            stepped.append(fact.id)
            store.log("semantic", fact.id, "belief_revised",
                      f"tool {outcome} {fact.confidence:.2f}->{target:.2f}")
        else:
            target = store.semantic.demote_confidence(fact.id, -settings.BELIEF_REVISE_STEP)
            if target is not None:
                stepped.append(fact.id)
                store.log("semantic", fact.id, "belief_revised",
                          f"tool {outcome} {fact.confidence:.2f}->{target:.2f}")
    flipped: list[int] = []
    if failures >= 2:
        # 步进只动 confidence 不动 status：lessons 仍是本次的全量 active 集合，直接翻转
        for fact in lessons:
            store.semantic.set_status(fact.id, "pending")
            flipped.append(fact.id)
            store.log("semantic", fact.id, "belief_flipped",
                      f"tool 连续 {failures} 次失败，任务域「{task_context}」"
                      f"信念收回待重证（检索命中 {settings.PROMOTE_MIN_HITS} 次可转正）")
    return stepped, flipped


def ingest_event(store: SqliteStore, content: str, source: str = "user",
                 type: str = "observation", task_context: str = "", outcome: str = "",
                 use_llm: bool = True, corrects: bool = False) -> dict:
    """走完整写入管线，返回结构化结果（供 CLI 展示与评测断言）。

    corrects（P2-3 L2，默认 False=现状逐位不变）：本轮是用户显式反驳（纠正信号命中，
    由调用方——recorder 的 correction 信号——显式传参）。True 时在语义事实落库后，
    对与纠正原文最相似的 top-1 active 事实降权一步（见 _revise_user_belief）。
    结果键 belief_revised / belief_stepped / belief_flipped 恒在（空列表=无动作）。
    """
    # 用户直陈自我描述（"我是…"）按身份显式声明处理：add 的内容是用户原话而非观察流。
    # 仅在 use_llm=True（真实 CLI 路径）启用；评测 harness 一律 use_llm=False，行为不变。
    if use_llm and type == "observation" and IDENTITY_PATTERN.search(content):
        type = "identity_statement"
    event = Event(content=content, source=source, type=type,
                  task_context=task_context, outcome=outcome)
    event = attention_gate(event, use_llm=use_llm)
    # 情感打分（E2）：门控之后、分流之前——被拦截的事件同样带情感字段返回，
    # 后续阶段（E3 工作记忆 salience / E5 情感记忆优先巩固）要消费它。
    # E8：绿色类型走纯规则情感（经验条目近乎中性，不值得等一次云端打分）
    event.valence, event.arousal = analyze_emotion(
        event.content, use_llm=(use_llm and event.type not in GREEN_TYPES))
    event.emotion = emotion_label(event.valence)
    result = {"importance": event.importance, "gated": False, "reason": "",
              "episodic_id": 0, "facts": [], "skipped_facts": [], "skills": [],
              "belief_revised": [], "belief_stepped": [], "belief_flipped": []}

    # E3 双写：门控之前先进工作记忆——会话内说过的都在（含被下方门控拦截的
    # 瞬时信息：working 里有、长期没有）。纯内存易失，进程退出即蒸发；
    # 长期仓储仍走既有门控，result 结构不变。
    # 嵌入向量只算一次（P2-3：correct_vec 复用同一结果，L2 不再重复过嵌入通道）。
    correct_vec = llm.embed(event.content)
    store.working.add(event.content, importance=event.importance,
                      arousal=event.arousal, embedding=correct_vec)

    if event.importance < settings.WORKING_THRESHOLD:
        result.update(gated=True, reason="dropped")
        return result
    if event.importance < settings.WRITE_THRESHOLD:
        result.update(gated=True, reason="working_only")
        return result

    mem = encode_episodic(event)
    result["episodic_id"] = store.episodic.add(mem)

    facts, skipped = dedupe_same_key(extract_semantic_facts(event))
    # E8 绿色类型与显式声明同级：explicit 源 = 同键新值自动取代旧版（高度迭代——
    # 环境状态最新即正确，新版经验直接接管旧版；版本链保留可回溯）
    fact_source = ("explicit" if event.type in EXPLICIT_TYPES or event.type in GREEN_TYPES
                   else "model")
    for fact in facts:
        if not fact.embedding:
            fact.embedding = llm.embed(fact.value)
        resolve_fact(fact, store.aliases.as_map())
        fact.source_event_ids = [result["episodic_id"]]
        outcome_f = store_fact_with_conflict_check(store, fact, source=fact_source)
        result["facts"].append({
            "entity": fact.entity, "relation": fact.relation, "value": fact.value,
            "confidence": fact.confidence, **outcome_f,
        })
    for s in skipped:
        result["skipped_facts"].append({"value": s.value, "confidence": s.confidence})

    # ---- P2-3 信念分级自动修正（bounded belief revision，D4-a）----
    # 时机在语义事实落库之后：先让 L1 的同键取代自然发生，修正只作用于「取代之后
    # 仍然成立」的信念。门控拦截的事件（上面两处提前 return）什么都不修正。
    if corrects:
        # L2 排除集 = 本次写入涉及的全部事实 id（新建/续证/吸收 keeper + 被取代的旧版）
        excluded = set()
        for f in result["facts"]:
            excluded.add(f["fact_id"])
            excluded.update(f.get("superseded", []))
        result["belief_revised"] = _revise_user_belief(store, correct_vec, excluded)
    if source == "tool" and outcome in ("success", "failure"):
        # L3 步进与双次失败翻转（见 _step_tool_beliefs）
        result["belief_stepped"], result["belief_flipped"] = _step_tool_beliefs(
            store, task_context, outcome)

    for skill in extract_skills(event):
        existed = store.procedural.find(skill.name)
        if existed:
            store.procedural.update_stats(existed.id, success=(event.outcome == "success"))
            # E8 高度迭代：同名技能带新做法时更新 policy（经验进化而非只累计统计）
            if skill.policy and skill.policy != existed.policy:
                store.procedural.update_policy(existed.id, skill.policy)
            skill.id = existed.id
        else:
            skill.id = store.procedural.add(skill)
        result["skills"].append({"id": skill.id, "name": skill.name,
                                 "policy": skill.policy, "reused": bool(existed)})
    return result
