"""睡眠巩固（编排，E5 多轮化 + P2-1 图式归纳）：预处理（高唤醒优先）
→ NREM×3（情景→语义转移，每轮以上轮蒸馏产出为参照，同键事实不重复蒸馏）
→ REM 阶段（用 E4 激活图做跨簇联想重组——「梦」的工程化；LLM 不可用时纯统计不写回）
→ 图式归纳（P2-1：把本夜聚出的跨时间相似情景提炼为 relation="rule" 的高阶规律，
带数量/跨度/LLM 三重断路器，一律 pending 保守落库）→ 既有技能沉淀
→ 消费 SM-2 到期队列（今日回忆清单：睡眠期重演，提取练习重排复习计划）。"""
from __future__ import annotations

from itertools import combinations

from memagent import settings
from memagent.adapters import llm
from memagent.consolidation.cluster import cluster_episodes
from memagent.consolidation.conflict_resolver import (
    dedupe_same_key,
    store_fact_with_conflict_check,
)
from memagent.consolidation.summarizer import (
    DISTILL_PROMPT,
    RULE_INDUCE_PROMPT,
    _parse_rule,
    parse_fact_json,
    summarize_cluster,
)
from memagent.core.clock import iso_to_ts, now_iso
from memagent.core.domain import SemanticFact
from memagent.core.vectors import cosine
from memagent.encoding.entity_resolver import resolve_fact
from memagent.learning.spaced_repetition import SpacedRepetition
from memagent.retrieval import retrieve
from memagent.retrieval.activation import build_entity_graph
from memagent.storage import SqliteStore

REM_ASSOC_PROMPT = (
    "睡眠联想（REM）阶段：下面两条记忆来自不同主题簇，但在实体激活图上被同时激活。\n"
    "判断它们之间是否存在值得长期记住的关联（共同主题 / 因果 / 相似经验）。"
    "有则输出 JSON 数组 [{{\"value\": \"一句关联描述（≤40字）\", \"confidence\": 0~1}}]"
    "（至多一条），没有则输出 []。只输出 JSON。\n"
    "记忆1: {fact_a}\n记忆2: {fact_b}"
)


def consolidate(store: SqliteStore, limit: int = settings.CONSOLIDATE_DAILY_LIMIT) -> dict:
    """执行一轮多轮巩固（NREM×3 + REM + 图式归纳），返回统计报告。

    旧统计键（clusters/distilled_facts/skills/expired_conflicts/summarized/due_reviews）
    语义不变，只加键：nrem_rounds（实际执行的 NREM 轮数）、rem_associations（REM 联想对）、
    rem_facts（REM 经 LLM 裁决写回的联想事实数）、rules_induced（P2-1：本夜写入的
    rule 数）、rule_candidates（P2-1：通过数量/跨度断路器的合格组数，含被 LLM
    拒绝/解析失败的）、skipped_vague（B5-b：低信息量蒸馏产物丢弃计数）。"""
    report = {"clusters": 0, "distilled_facts": 0, "skills": 0, "expired_conflicts": 0,
              "summarized": 0, "nrem_rounds": 0, "rem_associations": [], "rem_facts": 0,
              "rules_induced": 0, "rule_candidates": 0, "skipped_vague": 0}

    distilled_keys: set[tuple[str, str]] = set()   # 本夜已蒸馏 (entity, relation) 键，跨轮去重
    distilled_entities: set[str] = set()           # REM 种子原料：蒸馏触碰过的实体
    night_clusters: list[list] = []                # 本夜全部轮次的簇（REM 跨簇判定原料）

    for _round in range(settings.NREM_ROUNDS):
        episodes = store.episodic.fetch(status="active", limit=limit)
        if not episodes:
            break
        # 预处理：高唤醒优先（消费 E2 产出——情感记忆先入簇、并作为簇代表保留原文；
        # 元数据相同按 id 降序稳定，保证确定性）
        episodes.sort(key=lambda e: (-e.arousal, -e.id))
        clusters = cluster_episodes(episodes, store)
        report["nrem_rounds"] += 1
        report["clusters"] += len(clusters)
        night_clusters.extend(clusters)
        _nrem_round(store, clusters, distilled_keys, distilled_entities, report)

    _rem_associations(store, night_clusters, distilled_entities, report)
    _induce_rules(store, night_clusters, report)
    _promote_skills(store, report)
    report["due_reviews"] = _replay_due_facts(store)
    store.log("episodic", 0, "consolidation_done", str(report))
    # 健康报告由调用方（CLI/TUI 应用层）在巩固后生成——机制层不反向依赖应用层
    return report


def _nrem_round(store: SqliteStore, clusters: list[list], distilled_keys: set,
                distilled_entities: set, report: dict) -> None:
    """单轮 NREM：逐簇蒸馏（以上轮产出为参照——同键事实不重复蒸馏）+ 摘要替代。"""
    for cluster in clusters:
        items_text = "\n".join(f"- {e.summary}" for e in cluster)

        # 1) 蒸馏为语义事实（LLM；失败则跳过）；同键多值只保留置信度最高，防自我取代；
        #    跨轮参照：上一轮已蒸馏出的同键事实不再重复蒸馏（一夜不嚼同一口）
        if llm.maintenance_available():
            raw = llm.maintenance_chat(DISTILL_PROMPT.format(items=items_text),
                                       validate=lambda s: len(parse_fact_json(s)) > 0 or s.strip().endswith("[]"))
            if raw:
                candidates = [
                    SemanticFact(**f, valid_from=now_iso(),
                                 embedding=llm.embed(f"{f['entity']} {f['relation']} {f['value']}"))
                    for f in parse_fact_json(raw)
                ]
                kept, _skipped = dedupe_same_key(candidates)
                for fact in kept:
                    # B5-b 低信息量守门（在 dedupe 同键去重这条既有校验链之后追加）：
                    # value 去空白后有效长度不足的蒸馏产物直接丢弃——实测真实库出现
                    # 「[系统] 优化 性能与稳定性」（value 仅 6 字）这类空话事实，被
                    # 检索续抬到 conf=1.0 干扰排序（见 settings.DISTILL_MIN_VALUE_CHARS）。
                    # 守门放在 distilled_keys 登记之前：被丢的键不占「本夜已蒸馏」名额，
                    # 下一轮若蒸馏出同键的像样表述仍可入库。只作用于巩固蒸馏通道，
                    # 主抽取（用户输入）与显式声明是用户原话，不做此裁剪。
                    if len("".join(fact.value.split())) < settings.DISTILL_MIN_VALUE_CHARS:
                        report["skipped_vague"] += 1
                        continue
                    resolve_fact(fact, store.aliases.as_map())
                    if (fact.entity, fact.relation) in distilled_keys:
                        continue
                    distilled_keys.add((fact.entity, fact.relation))
                    distilled_entities.add(fact.entity)
                    store_fact_with_conflict_check(store, fact, source="consolidation")
                    report["distilled_facts"] += 1
        # 2) 相似度足够高的事件，用簇摘要替代（保留代表事件——高唤醒者优先当代表）
        if len(cluster) >= settings.CONSOLIDATE_MIN_CLUSTER:
            summary = summarize_cluster(cluster)
            if summary:
                keeper = cluster[0]
                keeper.summary = f"[综合] {summary}"
                store.episodic.update(keeper)
                for extra in cluster[1:]:
                    store.episodic.set_status(extra.id, "archived")
                    store.log("episodic", extra.id, "consolidated->archive", summary)
                report["summarized"] += len(cluster) - 1


def _rem_associations(store: SqliteStore, night_clusters: list[list],
                      distilled_entities: set, report: dict) -> None:
    """REM 阶段（「梦」的工程化）：实体激活图（E4）上找跨簇联想对。

    种子 = 本夜蒸馏触碰的实体 + 与本夜处理过的情景记忆同源（source_event_ids）
    的语义事实实体；两事实「图上近邻」（共享激活实体或相邻实体）而「嵌入上远邻」
    （低于聚类阈值、分属不同簇）即为跨簇联想对。纯统计、只读——写回仅当
    maintenance LLM 可用且经其裁决认可，绝不无证据写语义事实。"""
    graph = build_entity_graph(store)
    processed_ids = {ep.id for cl in night_clusters for ep in cl}
    seeds = {graph.aliases.get(e, e) for e in distilled_entities}
    for fact in store.semantic.fetch(status="active"):
        if processed_ids.intersection(fact.source_event_ids):
            seeds.add(graph.aliases.get(fact.entity, fact.entity))
    report["rem_associations"] = []
    report["rem_facts"] = 0
    if not seeds:
        return
    act = graph.activate({e: 1.0 for e in seeds})

    # 事实的簇归属（经 source_event_ids 链到本夜情景簇；无链路 = 无簇）
    cluster_ids_of: dict[int, set[int]] = {}
    for idx, cl in enumerate(night_clusters):
        for ep in cl:
            cluster_ids_of.setdefault(ep.id, set()).add(idx)

    def _clusters_of(fact) -> set[int]:
        got: set[int] = set()
        for sid in fact.source_event_ids:
            got |= cluster_ids_of.get(sid, set())
        return got

    best: dict[frozenset[int], dict] = {}

    def _consider(f1, f2, strength: float) -> None:
        if f1.id == f2.id:
            return
        if _clusters_of(f1) & _clusters_of(f2):
            return  # 同一夜的同一簇：已一起处理过，不是跨簇联想
        if f1.embedding and f2.embedding and \
                cosine(f1.embedding, f2.embedding) >= settings.CONSOLIDATE_MIN_SIMILARITY:
            return  # 嵌入近邻：本就可能同簇，不是「图近嵌入远」的梦
        key = frozenset((f1.id, f2.id))
        if key not in best or strength > best[key]["strength"]:
            best[key] = {"entities": [f1.entity, f2.entity],
                         "facts": sorted((f1.id, f2.id)),
                         "strength": round(strength, 4)}

    for entity, a in sorted(act.items(), key=lambda kv: (-kv[1], kv[0])):
        if a < settings.ACTIVATION_MIN:
            continue
        facts = graph.facts_of(entity)
        # 1) 共享同一激活实体的事实对（图上一跳邻居）
        for f1, f2 in combinations(facts, 2):
            _consider(f1, f2, a)
        # 2) 激活实体与其图邻居两侧的事实对（每条边按字典序只处理一次）
        for nb in sorted(graph.edges.get(entity, ())):
            if nb <= entity:
                continue
            link = min(a, act.get(nb, 0.0))
            if link < settings.ACTIVATION_MIN:
                continue
            for f1 in facts:
                for f2 in graph.facts_of(nb):
                    _consider(f1, f2, link)

    pairs = sorted(best.values(), key=lambda p: (-p["strength"], p["facts"]))[:settings.REM_MAX_ASSOC]
    report["rem_associations"] = pairs
    if pairs and llm.maintenance_available():
        report["rem_facts"] = _rem_writeback(store, pairs)


def _rem_writeback(store: SqliteStore, pairs: list[dict]) -> int:
    """REM 保守写回：联想对交给 maintenance LLM 裁决，认可才生成低置信
    associated_with 事实（source=consolidation，走既有冲突检查——默认 pending，
    绝不自动取代）。LLM 输出格式损坏 / 不认可 / 抛异常都只统计不写。"""
    written = 0
    written_keys: set[tuple[str, str, str]] = set()  # 本夜已写的联想事实键（幂等）
    for pair in pairs:
        f1 = store.semantic.get(pair["facts"][0])
        f2 = store.semantic.get(pair["facts"][1])
        if not f1 or not f2:
            continue
        try:
            raw = llm.maintenance_chat(
                REM_ASSOC_PROMPT.format(
                    fact_a=f"[{f1.entity}] {f1.relation} = {f1.value}",
                    fact_b=f"[{f2.entity}] {f2.relation} = {f2.value}"),
                validate=lambda s: len(parse_fact_json(s)) <= 1)
        except Exception:
            continue  # 梦写不进去就算了：绝不因整理动作破坏既有记忆
        items = parse_fact_json(raw) if raw else []
        for item in items[:1]:
            value = str(item.get("value", "")).strip()
            if not value:
                continue
            confidence = max(0.0, min(settings.REM_ASSOC_MAX_CONFIDENCE,
                                      float(item.get("confidence", 0.5))))
            key = (pair["entities"][0], "associated_with", value)
            if key in written_keys:
                continue
            written_keys.add(key)
            fact = SemanticFact(entity=pair["entities"][0], relation="associated_with",
                                value=value, confidence=confidence, valid_from=now_iso(),
                                embedding=llm.embed(f"{pair['entities'][0]} associated_with {value}"))
            resolve_fact(fact, store.aliases.as_map())
            store_fact_with_conflict_check(store, fact, source="consolidation")
            written += 1
    return written


def _induce_rules(store: SqliteStore, night_clusters: list[list], report: dict) -> None:
    """P2-1 图式归纳（睡眠第四阶段）：把本夜 NREM 聚出的「同类经历」跨时间归纳为
    relation="rule" 的高阶规律事实——扁平三元组只记「发生了什么」，rule 记
    「从多次经历中学到什么」（如多次备份失败 + 客服响应慢 -> 「选 VPS 厂商先验证
    备份机制与客服响应」）。

    候选分组对原方案的一处适配（报告已确认）：原方案写「按（实体，outcome）分组」，
    但情景记忆没有实体字段（P1-3 的实体锚是检索时现算的），且 P0-1 后情景 outcome
    多为空（成败信号已收敛到显式点名通路）——实际用**本夜 NREM 聚类本身作分组**：
    相似度 ≥ CONSOLIDATE_MIN_SIMILARITY 的连通分量就是「同类经历」，语义与原意
    （跨事件、同主题）一致且零额外计算。

    断路器（错不如旧在归纳侧的表达：各条同时满足才产 rule）：
    (a) 组内情景数 ≥ RULE_MIN_EPISODES——单次事件不成规律；
    (b) 组内最早与最晚情景的 created_at 跨度 ≥ RULE_MIN_SPAN_DAYS——同期巧合
        （同一天的多条相似记录）不算跨事件归纳（时间统一走 core.clock，评测可偏移）；
    (c) maintenance LLM 可用且输出合法——离线整段跳过（零写回、零候选统计），
        失败/格式坏/entity=user/空 value 只统计候选不写；
    (d) rule 的 source_event_ids 机械地等于参与归纳的情景 id 全集——证据链不许
        凭空，也绝不遗漏；
    (e) B5-c 防抖·一：组内含二手概括行（is_summary=True 的降级摘要行，或 summary
        被 "[综合] " 改写的 E5 keeper 行）整组跳过——摘要已经是上一轮巩固/降级的
        二手概括，再拿它归纳就是「规律的规律」（真实库垃圾规律的已知来源）；
    (f) B5-c 防抖·二：单夜写入 rule 数达 RULE_MAX_PER_NIGHT 后，剩余合格组只统计
        不写（候选计数照常、不再请求 LLM）——防连续多簇连环产规律刷库。

    保守落库：一律 status="pending"（试用期，靠 P1 检索转正通道攒使用证据）、
    confidence ≤ RULE_MAX_CONFIDENCE、走 store_fact_with_conflict_check(
    source="consolidation")——同键旧 rule 冲突时挂 memory_conflict 待裁，绝不自动
    取代；原始情景零改动（rule 与 episode 并存，归纳永不降权情景）。跨轮去重：
    NREM×3 在摘要替代失败时会反复取出同一批情景，同一情景组一夜只归纳一次。
    """
    if not llm.maintenance_available():
        return  # 断路器 c 前置：离线整段跳过，与 NREM 蒸馏/REM 写回同一取向
    considered: set[frozenset[int]] = set()          # 已看过的情景组（跨 NREM 轮去重）
    written_keys: set[tuple[str, str, str]] = set()  # 本夜已写 rule 键（幂等，仿 REM 写回）
    for cluster in night_clusters:
        ids = frozenset(ep.id for ep in cluster)
        if len(cluster) < settings.RULE_MIN_EPISODES or ids in considered:
            continue
        considered.add(ids)
        # 断路器 e（B5-c 防抖·一）：组内混入二手概括行则整组跳过。识别面有两个：
        # is_summary=True（B4 摘要降级行）与 "[综合] " 前缀（E5 keeper 行——它
        # is_summary 恒为 0，巩固只改写 summary 不改标记，实测只能靠前缀认出）。
        # 两者都是上一轮巩固的产物，拿它们归纳出的是「规律的规律」——错归纳
        # 不如不归纳，宁可等这些主题再有原始情景进簇。
        if any(ep.is_summary or (ep.summary or "").startswith("[综合] ")
               for ep in cluster):
            continue
        # 断路器 b：时间跨度。created_at 残缺的组无法证明「跨时间」，保守跳过
        created = [iso_to_ts(ep.created_at) for ep in cluster]
        if not all(created):
            continue
        if (max(created) - min(created)) / 86400.0 < settings.RULE_MIN_SPAN_DAYS:
            continue
        report["rule_candidates"] += 1  # 合格组：即使 LLM 随后拒绝也计数（报告口径）
        # 断路器 f（B5-c 防抖·二）：单夜 rule 写入上限。达到后剩余合格组只统计
        # 不写、也不再请求 LLM（省一次调用）——report 键照常推进，下一夜继续攒
        if report["rules_induced"] >= settings.RULE_MAX_PER_NIGHT:
            continue
        items_text = "\n".join(f"- {e.summary}" for e in cluster)
        try:
            raw = llm.maintenance_chat(RULE_INDUCE_PROMPT.format(items=items_text))
        except Exception:
            continue  # 梳理通道抖动：只统计不写（与 REM 写回同一取向）
        parsed = _parse_rule(raw)
        if parsed is None:
            continue  # 断路器 c：格式坏/键约束违反 -> 零写
        key = (parsed["entity"], "rule", parsed["value"])
        if key in written_keys:
            continue
        written_keys.add(key)
        fact = SemanticFact(
            entity=parsed["entity"], relation="rule", value=parsed["value"],
            confidence=parsed["confidence"], status="pending", valid_from=now_iso(),
            embedding=llm.embed(f"{parsed['entity']} rule {parsed['value']}"),
            source_event_ids=sorted(ep.id for ep in cluster))  # 断路器 d：证据链=全组
        resolve_fact(fact, store.aliases.as_map())
        store_fact_with_conflict_check(store, fact, source="consolidation")
        report["rules_induced"] += 1


def _replay_due_facts(store: SqliteStore) -> list[dict]:
    """今日回忆清单（E1）：消费 SM-2 到期队列，对每条到期事实做自提示提取练习——
    以事实自身为线索检索，命中（score≥阈值）即隐式复习并重排计划（睡眠期重演）；
    连自提示都唤不回的记为 forgotten，重置间隔等待下一轮。"""
    sr = SpacedRepetition(store.conn)
    results = []
    for fact_id in sr.due(limit=settings.SM2_RECALL_LIMIT):
        fact = store.semantic.get(fact_id)
        if not fact or fact.status != "active":
            continue
        cue = fact.value  # 以事实内容自身为线索重演（睡眠期 replay）
        hits = retrieve(store, cue)
        recalled = next((h for h in hits
                         if h.kind == "semantic" and h.id == fact_id
                         and h.score >= settings.SM2_IMPLICIT_THRESHOLD), None)
        if recalled is None:
            sr.review(fact_id, quality=1)  # 唤不回：低质量复习，间隔重置
        plan = sr.status_row(fact_id)
        results.append({"fact_id": fact_id, "entity": fact.entity,
                        "relation": fact.relation, "value": fact.value,
                        "recalled": bool(recalled),
                        "interval_days": plan["interval_days"] if plan else 0})
    return results


def _promote_skills(store: SqliteStore, report: dict) -> None:
    """把高成功率的既有技能置为启用（v1 简化：仅汇总统计）。"""
    for skill in store.procedural.fetch():
        report["skills"] += skill.usage_count
