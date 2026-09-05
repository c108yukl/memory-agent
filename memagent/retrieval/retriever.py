"""检索器：混合检索（关键词 + 向量）→ 归一化排序 → 强度提升（再巩固）→ SM-2 隐式复习。

E3：工作记忆命中置顶；其再巩固「穿透」到文本相同的长期情景记忆——
working 条目替代了被去重的直接命中，提取练习的效果不应因替代而丢失。

E4：联想激活通道——真直接命中（关键词命中，或向量相似度超过噪声线）所挂
实体作种子，实体图上两跳扩散；被激活实体下未直接命中的事实作「联想命中」
追加在直接命中之后（独立返回，不占 top_k 名额、不做任何写回——联想是只读通道）。

E6：干扰论遗忘。① semantic 再巩固不再走 upsert 续证（那会 evidence_count+1，
而 evidence 又是联想激活 base 的输入——自增强回路），只 touch confidence；
② 提取诱发遗忘（RIF）：回忆 A 时，与 A 同实体但本次未被够到（不在直接命中
也不在联想带出中）的竞争者置信度小幅下调——回忆会抑制同类竞争者。

E7：元认知 v1（读侧增强，不动写路径）。① 检索置信度表面化——top 直接命中
最高分 < 置信阈值时，直接命中与联想命中标 meta.uncertain
（feeling-of-knowing：与其静默返回噪声级结果，不如显式返回「不确定」）；
阈值按嵌入后端分档（confident_bar：hash→0.30 / local·cloud→0.70）；
② 检索完全落空记 retrieval_gap 审计——「我知道我不知道」，健康报告的
检索空缺区块消费它作为高价值待巩固方向。

E8：程序记忆激活——技能（trigger/policy 文本）与查询 bigram 匹配者追加在
联想命中之后（不占 top_k 名额、不进 RIF、豁免 uncertain——技能带出跟随
触发词而非相关度）；带出即 touch_usage（被想起记一次使用，不动成功率）。
经验层情景命中带 meta.green 通道标记（分层可见性）。

V1.7 P1：试用期转正通道——pending 语义事实进入检索（此前被 status 过滤挡在
门外，永远拿不到「被频繁使用」的证据，转正通道只进不出），分数打折进场
（meta.probation）；直接命中累计 hit_count，达标且非冲突型则转 active。
A 类冲突待裁型（memory_conflict 有以它为 new_id 的 pending 行）永不自动转正
——转正即取代互斥的 active 旧事实，只能由冲突消解决定。

B3：读侧 meta 补充「通道证据」——semantic/episodic 直接命中带 meta.vec（向量
相似度）与 meta.fts（是否 FTS 关键词命中），联想命中不带（它们只有激活值、
没有查询相关度）。这是注入门槛（agent.injector）的判定依据，也是「评的与用
的同口径」的一部分：只加 meta，排序/分数/阈值零变化——CLI retrieve 与评测
的直接检索路径行为完全不变。

P1-1：检索融合去 confidence 偏置，两段式——
① 分池保底名额（结构修复）：episodic 池与 semantic 池各自内部排序后，top-k
组装保证 episodic 至少 EPISODIC_MIN_SLOTS 个非 working 名额（池里有候选且
预算允许时），剩余名额按全局分数补齐；池内相对排序、联想/技能追加通道、
working 置顶都不变，名额只决定成员资格、最终仍按分数降序。背景：semantic
融合分的 confidence 项让高置信噪声（巩固蒸馏产物 conf 0.8~1.0）白拿
0.32~0.4 分、融合分 0.67~0.74 永久压过 vec 0.61 的相关情景记忆（融合仅
~0.52），top-5 截断后真记忆不可见。名额不 binding（episodic 天然进前列 /
episodic 池为空）时与旧算法逐位一致。
② 权重微调（评测护航后执行）：semantic 融合改三项式
VECTOR_WEIGHT×vec + SEMANTIC_CONF_WEIGHT×confidence
+ SEMANTIC_EVIDENCE_WEIGHT×evidence_component（evidence 项 log 饱和）——
confidence 降权、真实证据计数补位；十六项指标任一下降即整体回退（D6）。

P1-3：联想种子扩展——`if seeds:` 门槛不再只认 semantic 直接命中（此前零
semantic 直接命中联想完全熄火，真实事故：用户说「我打算换个云厂商」，雨云
相关记忆唤不醒）。三路种子：① semantic 真直接命中（原 E4）；② 别名直启
（查询含别名 → 规范实体作种子，ACTIVATION_SEED_DEFAULT=0.5，别名是查询词
与实体名的字面重叠）；③ episodic 实体锚（episodic 真命中的 summary 含已知
实体名 → 该实体作种子；检索时现算，零 schema 变更对存量立即生效）。联想
命中的零写回承诺原样（不进 top、不 _reinforce、不 RIF、不 touch_usage）。

P1-5：episodic 命中的 meta 透传写入来源 source（P0-4 落地的列）——注入侧
据此拦截 assistant 来源的经验层自动注入（防复发纵深，见 agent.injector）。

P1-4：结构重构（行为零变化）——把原 retrieve 的第 0 步（working 检索）、
第 1/2 步（双路候选收集）与第 3~8 步（联想/组装/写回/元认知/技能）抽成
公共函数：collect_candidates（单条线索的双路候选收集）、_episodic_hit /
_semantic_hit（融合公式与 meta，快搜与深搜同一套）、finalize_retrieval
（后处理）。快搜就是 cues=[query] 的特例；深搜（retrieval/deep.py）用同一
套口径做多线索召回。快搜行为逐位不变由 tests/unit/test_deep.py 的金样
（重构前采集的固定记忆集输出）逐位锁定，既有全量测试（基线 528 项）为
硬门槛。
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

from memagent import settings
from memagent.adapters import llm
from memagent.core.clock import iso_to_ts, now_iso
from memagent.core.domain import RetrievalHit, SemanticFact
from memagent.core.text import bigram
from memagent.learning.spaced_repetition import SpacedRepetition
from memagent.learning.strength import compute_strength
from memagent.retrieval.activation import build_entity_graph
from memagent.retrieval.ranker import recency_boost
from memagent.storage import SqliteStore


def _semantic_evidence_component(evidence_count: int) -> float:
    """P1-1 第二段：semantic 融合的证据项 = min(1, log1p(证据数)/log1p(5))。

    只认真实观测累积的 evidence_count（E6 契约：检索不续证）。log 饱和与
    strength 频率项（FREQ_ACCESS_CAP）、联想激活 base 的证据加成同一取舍：
    5 次真实观测即满格 1.0，之后不再加价——防「证据多→分高→更易被检索」
    的自增强回路。单测 tests/unit/test_p1_fusion.py 锁定纯数学行为。
    """
    return min(1.0, math.log1p(max(0, evidence_count)) / math.log1p(5))


def confident_bar(backend: str | None = None) -> float:
    """E7 置信阈值按嵌入后端分档（纯读侧，不动排序）：

    - hash（离线哈希兜底）：RETRIEVAL_CONFIDENT_BAR=0.30，按哈希嵌入噪声底标定；
    - local / cloud（真实稠密嵌入，如 bge-m3）：RETRIEVAL_CONFIDENT_BAR_DENSE=0.70，
      在线实测跨主题余弦高达 0.56~0.64，沿用 0.30 会让 uncertain 永不触发。
    backend 缺省时实时查询 llm.embed_backend()（零网络，只读探测缓存）——
    retrieve 内先 embed 过，缓存必已反映本次查询实际使用的后端。
    """
    if backend is None:
        backend = llm.embed_backend()
    if backend in ("local", "cloud"):
        return settings.RETRIEVAL_CONFIDENT_BAR_DENSE
    return settings.RETRIEVAL_CONFIDENT_BAR


def _match_skills(store: SqliteStore, query: str, boost_access: bool) -> list[RetrievalHit]:
    """E8 程序记忆激活：技能文本（name/trigger/policy）与查询的 bigram 匹配。

    匹配度 = 查询 bigram 被技能文本覆盖的比例（查询视角：这条技能解释了
    提问的多大一部分），score = 0.6·覆盖 + 0.4·成功率（用过且靠谱的技能优先）。
    零覆盖不出——带出必须有事前触发相关性，不做模糊联想。boost_access 时
    touch_usage（被想起记一次使用，成功率不动——想起≠执行成功）。
    空 trigger 技能不参与匹配（P0-2）：无事前触发相关性，带出即违反本函数契约。
    """
    q_grams = set(bigram(query).split())
    if not q_grams:
        return []
    scored = []
    for skill in store.procedural.fetch(status="active"):
        if not skill.trigger.strip():
            continue  # P0-2：空 trigger = 无事前触发语义，带出即违反本函数的匹配契约
        s_grams = set(bigram(f"{skill.name} {skill.trigger} {skill.policy}").split())
        if not s_grams:
            continue
        coverage = len(q_grams & s_grams) / len(q_grams)
        if coverage <= 0:
            continue
        score = round(0.6 * coverage + 0.4 * skill.success_rate, 4)
        scored.append((score, skill))
    scored.sort(key=lambda t: (-t[0], t[1].id))
    hits = []
    for score, skill in scored[:settings.EXPERIENCE_SKILL_TOP_N]:
        if boost_access:
            store.procedural.touch_usage(skill.id)
        hits.append(RetrievalHit(
            kind="procedural", id=skill.id,
            text=f"{skill.name}｜{skill.policy}",
            score=score, strength=skill.success_rate,
            meta={"skill": True, "success_rate": skill.success_rate,
                  "usage": skill.usage_count, "trigger": skill.trigger}))
    return hits


# ---------- P1-4 公共检索原语（快搜与深搜共用的唯一口径） ----------

def _working_hits(store: SqliteStore, qvec: list[float]) -> list[RetrievalHit]:
    """E3 工作记忆命中（原 retrieve 第 0 步原样抽取）：未过期条目按
    salience×相似度取前几条，全部置顶。过期条目（>24h）视为已蒸发——
    time_travel 多天场景里它自然消失。深搜同样只以**原查询**跑一次
    working 检索（线索不进 working 检索——working 是当下上下文，只认用户原话）。
    """
    return [
        RetrievalHit(kind="working", id=entry.id, text=entry.text,
                     score=round(sim, 4),
                     meta={"working": True, "importance": entry.importance,
                           "arousal": entry.arousal})
        for sim, entry in store.working.search(qvec, limit=settings.WORKING_RETRIEVE_LIMIT)
    ]


@dataclass
class CueCandidates:
    """collect_candidates 的产物：单条线索的双路候选（库内行 + 召回证据）。

    只带证据不带融合分——融合在 _episodic_hit / _semantic_hit（快搜与深搜
    共用同一套公式），深搜需先合并各线索证据（vec 取 max、fts 取并）再构建
    命中。行是库内真行（红线：进结果的条目必须有库内 id，禁止凭线索造条目）：
    episodic 项 = (EpisodicMemory, 向量分, 是否FTS命中)；semantic 项同理。
    """

    episodic: list = field(default_factory=list)
    semantic: list = field(default_factory=list)


def collect_candidates(store: SqliteStore, cue: str, qvec: list[float] | None = None,
                       working_texts: frozenset | set = frozenset()) -> CueCandidates:
    """P1-4：单条线索的双路候选收集——快搜与深搜共用的唯一召回口径。

    自 retrieve 第 1/2 步原样抽取：episodic「FTS5 关键词 + 向量余弦」去重
    合并，semantic 同款双路（A 类冲突待裁过滤、working 同文去重、候选池放大
    系数全部原样）。qvec 缺省时现场 embed(cue)（主查询调用方复用已算好的
    向量，避免同一文本重复过嵌入通道）。候选保持原遍历序（FTS 命中在前、
    向量候选补后，(kind, id) 去重）。
    """
    if qvec is None:
        qvec = llm.embed(cue)
    pool = settings.FTS_CANDIDATE_FACTOR
    seen: set[tuple[str, int]] = set()

    # 1) 情景记忆候选：FTS5 关键词 + 向量余弦，去重合并
    # P0-1：候选池放大——失效记忆的索引条目删不掉，会占名额，用冗余换召回完整度
    fts_ids = [rowid for rowid, _ in store.episodic.search_fts(cue, limit=20 * pool)]
    vec_ids = [rowid for rowid, _ in store.episodic.cosine_search(qvec, limit=20)]
    fts_id_set = set(fts_ids)  # B3：通道证据标记用（只做集合判定，不影响候选与排序）
    # 每条的向量分从「本线索的 top-50 余弦」一次取齐（原实现逐候选重跑
    # cosine_search 的 next() 查找；cosine_search 每行恰出现一次，dict 取值
    # 与 next() 逐位等价，只是把 O(n²) 摊平成 O(n)）
    epi_vec_scores = dict(store.episodic.cosine_search(qvec, 50))
    episodic: list = []
    for mem_id in fts_ids + vec_ids:
        mem = store.episodic.get(mem_id)
        if not mem or mem.status not in settings.RETRIEVABLE_STATUSES or ("episodic", mem.id) in seen:
            continue
        if mem.summary in working_texts:  # 同一内容既在长期又在 working：保留置顶条目
            continue
        seen.add(("episodic", mem.id))
        episodic.append((mem, epi_vec_scores.get(mem.id, 0.0), mem_id in fts_id_set))

    # 2) 语义记忆候选：同样双路合并（记录每条的通道证据，供联想通道判定归属）
    s_fts = [rowid for rowid, _ in store.semantic.search_fts(cue, limit=10 * pool)]
    s_fts_set = set(s_fts)
    vec_scores = dict(store.semantic.cosine_search(qvec, limit=50))
    # P1：A 类·冲突待裁型 pending 不进检索结果——它的新值与已有 active 事实互斥，
    # 与旧值同台只会误导消费方（助手会同时看到两个互相矛盾的答案）；且它按 D1
    # 永不自动转正，可检索对它没有任何通路价值（唯一出口是裁决队列）。
    # 只有 B 类（与谁都不冲突、只是证据不足）需要「被想起」来攒转正证据。
    # 注意：这是**语义**过滤，不是状态过滤——RETRIEVABLE_STATUSES 仍是索引与
    # 过滤的唯一口径（D2 的单点不被破坏）。
    conflict_pending_ids = {c["new_id"] for c in store.conflicts.fetch_all(status="pending")}
    semantic: list = []
    for fact_id in s_fts + list(vec_scores)[:10]:
        fact = store.semantic.get(fact_id)
        if not fact or fact.status not in settings.RETRIEVABLE_STATUSES or ("semantic", fact.id) in seen:
            continue
        if fact.status == "pending" and fact.id in conflict_pending_ids:
            continue
        if f"[{fact.entity}] {fact.relation} {fact.value}" in working_texts:
            continue  # 与 working 条目逐字相同的直接命中不重复出现
        seen.add(("semantic", fact.id))
        semantic.append((fact, vec_scores.get(fact.id, 0.0), fact_id in s_fts_set))
    return CueCandidates(episodic=episodic, semantic=semantic)


def _episodic_hit(mem, vec_score: float, fts_hit: bool) -> RetrievalHit:
    """episodic 候选 → 命中：P1-1 融合公式 + B3/P1-5 meta（快搜与深搜同一套）。"""
    recency = recency_boost(iso_to_ts(mem.created_at))
    fused = (settings.VECTOR_WEIGHT * vec_score
             + 0.15 * settings.FTS_WEIGHT * min(1.0, mem.importance)
             + 0.2 * mem.strength
             + 0.1 * recency)
    # B3：通道证据进 meta（vec=向量相似度、fts=关键词命中）——注入门槛的判定
    # 依据，纯读侧标记，不参与任何分数计算。
    # P1-5：meta.source = 写入来源（P0-4 落地的列，读侧透传）——注入侧据此
    # 拦截「assistant 来源的经验层」自动注入（防复发纵深，见 agent.injector），
    # 与 vec/fts 同性质：只加标记，排序/分数/阈值零变化
    return RetrievalHit(kind="episodic", id=mem.id, text=mem.summary,
                        score=round(fused, 4), strength=mem.strength,
                        meta={"importance": mem.importance, "created_at": mem.created_at,
                              "vec": vec_score,
                              "source": mem.source,
                              **({"fts": True} if fts_hit else {}),
                              **({"green": True} if getattr(mem, "category", "") == "experience"
                                 else {})})


def _semantic_hit(fact: SemanticFact, vec_score: float, fts_hit: bool) -> RetrievalHit:
    """semantic 候选 → 命中：P1-1 三项式融合 + 试用期打折 + B3 meta（共用）。"""
    # P1-1 第二段（权重微调）：confidence 是「写入时对抽取的自信」，不是
    # 「与查询的相关度」——旧口径 0.4×confidence 白送高置信噪声 0.32~0.4 分
    # （巩固蒸馏产物 conf 普遍 0.8~1.0，跨主题噪声融合 0.67~0.74，实测压过
    # vec 0.61 的真记忆），降权到 0.25 后相关度由向量项主导；被砍掉的份额由
    # 真实证据计数补位（log 饱和，见 _semantic_evidence_component）。权重
    # 取值见 settings.SEMANTIC_CONF_WEIGHT 注释；回退条款 D6（任一评测指标
    # 下降即整体回退本段）。
    fused = (settings.VECTOR_WEIGHT * vec_score
             + settings.SEMANTIC_CONF_WEIGHT * fact.confidence
             + settings.SEMANTIC_EVIDENCE_WEIGHT
             * _semantic_evidence_component(fact.evidence_count))
    # P1：试用期（pending）事实「打折进场」——必须可检索（否则永远拿不到被
    # 频繁使用的证据，试用期只进不出），但不与正式事实争位。折扣在融合分之
    # 后、排序之前施加，故试用与正式的相对次序确定（同分时正式在前）。
    probation = fact.status == "pending"
    if probation:
        fused *= settings.PENDING_SCORE_PENALTY
    # B3：同 episodic——通道证据进 meta，供注入门槛判定（纯读侧标记）
    return RetrievalHit(kind="semantic", id=fact.id,
                        text=f"[{fact.entity}] {fact.relation} {fact.value}",
                        score=round(fused, 4), strength=fact.confidence,
                        meta={"entity": fact.entity,
                              "vec": vec_score,
                              **({"fts": True} if fts_hit else {}),
                              **({"probation": True,
                                  "hit_count": fact.hit_count} if probation else {})})


def retrieve(store: SqliteStore, query: str, top_k: int = settings.RETRIEVE_TOP_K,
             boost_access: bool = True) -> list[RetrievalHit]:
    """检索流程：候选收集 → 分数融合排序 → 命中后强度提升（突触可塑性）。

    E3：会话内工作记忆命中置顶（当下上下文压过历史记忆，不依赖高分），
    与直接命中文本完全相同者去重——同一信息只出现一次，且以 working 面目出现。
    E4：直接命中截断完成后追加联想命中（激活扩散带出、未直接命中的事实）。
    P1-4：候选收集与后处理已抽成公共函数（collect_candidates /
    finalize_retrieval）——本函数就是 cues=[query] 的特例，行为逐位不变
    （tests/unit/test_deep.py 金样锁定）；深搜见 retrieval/deep.py。
    """
    qvec = llm.embed(query)
    working_hits = _working_hits(store, qvec)
    working_texts = {h.text for h in working_hits}
    cands = collect_candidates(store, query, qvec=qvec, working_texts=working_texts)
    hits = [_episodic_hit(mem, vec, fts) for mem, vec, fts in cands.episodic]
    hits += [_semantic_hit(fact, vec, fts) for fact, vec, fts in cands.semantic]
    return finalize_retrieval(store, query, working_hits, hits,
                              top_k=top_k, boost_access=boost_access)


def finalize_retrieval(store: SqliteStore, query: str, working_hits: list[RetrievalHit],
                       hits: list[RetrievalHit], top_k: int = settings.RETRIEVE_TOP_K,
                       boost_access: bool = True,
                       audit_gap: bool = True) -> list[RetrievalHit]:
    """检索后处理（原 retrieve 第 3~8 步原样抽取，快搜与深搜共用）：

    联想激活（种子来自传入直接命中的真直接命中）→ 分池保底组装 → 再巩固
    （boost_access 时）→ E7 元认知 → retrieval_gap 审计（audit_gap 时）→
    技能激活。入参 hits = episodic + semantic 直接命中候选（episodic 在前，
    同分稳定排序的次序来源）；working_hits 由调用方以查询向量跑
    _working_hits 得到（深搜只以原查询跑，线索不进 working 检索）。
    audit_gap=False 供深搜用：空检索审计是一次 meta 写入，与深搜「只读探索、
    库字节不变」的红线相抵（快搜照常记录「我知道我不知道」）。
    """
    working_texts = {h.text for h in working_hits}

    # 3) E4 联想激活：真直接命中的实体作种子，实体图上两跳扩散。
    #    真直接命中 = 命中 FTS 关键词，或向量相似度超过噪声线
    #    （小库时 cosine_search 按排名返回，纯碰撞噪声 ~0.1 的候选也会混进
    #    直接命中列表——它们不是被查询够到的，是被联想带出的，改走联想通道）。
    #
    #    P1-3 种子扩展（三路种子，全部离线确定性），流程重构为「先建图 → 收集
    #    种子 → if seeds: 扩散」。为什么先建图：别名种子需要别名表（graph.aliases
    #    就是它的快照）、情景锚需要图中已知实体名集合——都得有图在场才能收集；
    #    而建图只读零副作用（activation 模块注释明说每次检索现建、数据量小不做
    #    缓存），提前建图无额外语义、无额外成本量级。
    #
    #    P1-4：真直接命中判定改读每条命中自带的通道证据 meta（fts / vec）——
    #    与旧的「id ∈ s_fts_set 或 vec_scores ≥ 噪声线」逐位等价（meta 由
    #    collect_candidates 的同源证据构建）；深搜传入的命中其 meta 是多线索
    #    合并后的最强证据，种子语义随之自然成立（种子来自合并后的真直接命中）。
    genuine_ids = {h.id for h in hits if h.kind == "semantic"
                   and (h.meta.get("fts")
                        or h.meta.get("vec", 0.0) >= settings.ACTIVATION_VEC_DIRECT_BAR)}
    graph = build_entity_graph(store)
    seeds: dict[str, float] = {}
    alias_seeded: set[str] = set()       # 经别名种子激活的实体（归一后）：联想命中打 meta.alias_seed
    episodic_anchored: set[str] = set()  # 经情景锚激活的实体（归一后）：联想命中打 meta.episodic_anchor

    def _canon(entity: str) -> str:
        return graph.aliases.get(entity, entity)

    def _seed(entity: str, value: float) -> str:
        # 种子实体名入图前先经别名归一（既有行为：activate 内同款归一）；同实体
        # 多路种子取 max 不叠加（seeds 字典的既有惯例）。这里把归一提到收集侧做，
        # 使三路种子的 max 合并在归一后的规范实体上真正成立——否则「原始实体名」
        # 与「别名规范名」两个键会在 activate 的归一里互相顶掉（后者覆盖前者），
        # max 惯例名存实亡。
        canonical = _canon(entity)
        seeds[canonical] = max(seeds.get(canonical, 0.0), value)
        return canonical

    # 路一（原 E4）：semantic 真直接命中所挂实体，激活值 = 命中分
    for h in hits:
        if h.kind == "semantic" and h.id in genuine_ids and h.meta.get("entity"):
            _seed(h.meta["entity"], min(1.0, h.score))

    # 路二（P1-3 别名直启）：查询文本包含某别名 → 其规范实体作种子。真实事故：
    # 用户说「我打算换个云厂商」，库里雨云相关记忆唤不醒——查询措辞与实体名/
    # 事实值零词汇重叠时 semantic 双路全灭，联想随之熄火（种子只来自 semantic
    # 直接命中，`if seeds:` 门槛不过）。别名表是用户亲手维护的「查询词 ↔ 实体名」
    # 词典（CLI alias add），查询里出现别名本身就是字面词汇证据；激活值取
    # settings.ACTIVATION_SEED_DEFAULT（低于真直接命中、高于扩散衰减）。关键
    # 语义：别名种子计入 `if seeds:` 门槛——只有别名命中、零 semantic 直接命中
    # 时联想也要点火（这正是「换云厂商」场景的解法）。比对口径：alias 是短词，
    # 子串包含判定；ASCII 大小写归一后比对，中文原样（str.lower 对中文是恒等）。
    # 深搜注：只扫原查询（用户自己的话才是字面词汇证据；LLM 展开的线索不充当
    # 别名触发词），query 实参恒为用户原查询。
    q_lower = query.lower()
    for alias, target in graph.aliases.items():
        if alias and alias.lower() in q_lower:
            alias_seeded.add(_seed(target, settings.ACTIVATION_SEED_DEFAULT))

    # 路三（P1-3 episodic 实体锚）：episodic「真命中」（meta.fts 或 meta.vec ≥
    # ACTIVATION_VEC_DIRECT_BAR，与 semantic 真命中同一把噪声线）的 summary 文本
    # 含已知实体名（图节点 + 别名键）→ 该实体作种子，激活值 = min(1.0, 命中分)
    # （与 semantic 种子同惯例）。检索时现算而非编码时抽实体存新列——对原方案的
    # 显式偏离，理由：① 零 schema 变更、零回填问题，对存量记忆立即生效（编码时
    # 抽实体只惠及新记忆，真实库存量 active 情景全是历史数据）；② 子串匹配与
    # build_entity_graph 连边用同款口径（known 实体名 in 文本），不引入新的抽取
    # 不确定性；③ 成本可忽略（episodic 命中数 ≤ top_k 量级 × 实体名有限）。
    # known 的构建与连边处同款：实体名优先于别名键（连边用 setdefault，实体名先
    # 占位）；别名键仅收规范实体确在图中的（连边同款过滤——锚到空实体无意义）。
    known: dict[str, str] = {a: _canon(a) for a in graph.aliases
                             if _canon(a) in graph.facts_by_entity}
    known.update({e: e for e in graph.facts_by_entity})
    for h in hits:
        if h.kind != "episodic":
            continue
        if not (h.meta.get("fts") or h.meta.get("vec", 0.0) >= settings.ACTIVATION_VEC_DIRECT_BAR):
            continue  # 噪声级情景候选（无关键词证据且低于噪声线）不构成种子
        for name, canonical in known.items():
            if name and name in h.text:
                episodic_anchored.add(_seed(canonical, min(1.0, h.score)))

    associated: list[RetrievalHit] = []
    if seeds:  # 三路种子全空才不联想——无中生有的噪声不如没有（别名/情景锚也算「有」）
        act = graph.activate(seeds)
        entity_of = lambda h: graph.aliases.get(h.meta.get("entity", ""), h.meta.get("entity", ""))

        # 3a) 认领噪声级候选：向量通道噪声混入的语义事实，其实体被激活时
        #     从直接命中中移出（不占直接命中名额与 top_k 截断），改由联想通道外送
        claimed: dict[int, tuple[float, SemanticFact]] = {}
        direct: list[RetrievalHit] = []
        for h in hits:
            a = act.get(entity_of(h), 0.0) if h.kind == "semantic" else 0.0
            if h.kind == "semantic" and h.id not in genuine_ids and a >= settings.ACTIVATION_MIN:
                fact = store.semantic.get(h.id)
                if fact:
                    claimed[fact.id] = (a, fact)
            else:
                direct.append(h)
        hits = direct

        # 3b) 直接命中排序 boost：所挂实体被激活者小幅加成（对同查询确定性一致，
        #     直接命中之间的相对顺序稳定；working 置顶不受影响）
        for h in hits:
            if h.kind == "semantic":
                a = act.get(entity_of(h), 0.0)
                if a > 0:
                    h.score = round(h.score + settings.ACTIVATION_BOOST * min(1.0, a), 4)

        # 3c) 联想命中候选：激活实体下未直接命中的 active 事实（含被认领者），
        #     按激活值竞争取前 ACTIVATION_TOP_N
        candidates: list[tuple[float, SemanticFact]] = list(claimed.values())
        for entity, a in act.items():
            if a < settings.ACTIVATION_MIN:
                continue
            for fact in graph.facts_of(entity):
                if fact.id in genuine_ids or fact.id in claimed:
                    continue
                if f"[{fact.entity}] {fact.relation} {fact.value}" in working_texts:
                    continue  # working 已置顶同文本：联想通道不重复外送
                candidates.append((a, fact))
        candidates.sort(key=lambda t: (-t[0], -t[1].confidence, t[1].id))
        for a, fact in candidates[:settings.ACTIVATION_TOP_N]:
            # P1-3 种子来源标记（审计/调试可见 + 注入侧连坐豁免的判定依据）：
            # alias_seed = 该实体的激活有别名种子参与——「查询词与实体名的字面
            # 重叠」，词汇证据等价 FTS 命中（连坐豁免只认这一路）；episodic_anchor =
            # 情景锚参与（纯审计，不参与豁免——情景文本是另一条记忆的措辞，不构成
            # 对本查询的词汇证据）。
            meta = {"associated": True, "entity": fact.entity}
            if _canon(fact.entity) in alias_seeded:
                meta["alias_seed"] = True
            if _canon(fact.entity) in episodic_anchored:
                meta["episodic_anchor"] = True
            associated.append(RetrievalHit(
                kind="semantic", id=fact.id,
                text=f"[{fact.entity}] {fact.relation} {fact.value}",
                score=round(min(1.0, a), 4), strength=fact.confidence,
                meta=meta))

    # 4) 排序 + 截断 + 置顶组装：working 命中在最前，直接命中补齐剩余名额。
    #    负数 top_k 钳为 0——裸切片会把 -3 变成「去掉末尾 3 条」，语义错乱（CLI 层已拒绝，此处兜底 API 调用方）
    #
    #    P1-1 分池保底名额：非 working 预算中保证 episodic 池至少 EPISODIC_MIN_SLOTS
    #    个名额（池里有候选且预算允许时）。为什么需要：semantic 融合分含 confidence
    #    项（见第 2 步），巩固蒸馏产物的 confidence 普遍 0.8~1.0，跨主题噪声语义事实
    #    仅置信项就白拿 0.32~0.4 分、融合分 0.67~0.74，永久压过相关情景记忆——episodic
    #    融合是 0.6×vec + 0.15×FTS×importance + 0.2×strength + 0.1×recency，实测相关
    #    记忆 vec 0.61 融合仅 ~0.52，top-5 截断后真记忆不可见（真实事故：查询「雨云
    #    服务器备份怎么样」，vec 0.612 的真条目进不了 top-5，vec 0.47/conf 1.0 的噪声
    #    事实得 0.68+）。分池保底让两个记忆系统在结果面上都保底有声量，而不必先动
    #    融合公式（权重微调是 P1-1 第二段，须评测护航后另行落地）。
    #    取舍：名额只决定**成员资格**、不改顺序语义——入选集最终仍按分数降序排列，
    #    同分按既有 sort 稳定性惯例保序（episodic 恒先于 semantic 追加进 hits，稳定
    #    排序下同分时 episodic 在前，与旧算法逐位一致）。
    #    幂等性：episodic 天然占据前列（保底名额不 binding）或 episodic 池为空时，
    #    入选集与旧算法「全局排序 + 截断」的成员与次序逐位一致——保底只在保底真的
    #    缺位（episodic 被 semantic 挤出预算线）时才改变结果（tests/unit/test_p1_fusion.py 锁定）。
    hits.sort(key=lambda h: h.score, reverse=True)
    budget = max(top_k, 0) - len(working_hits)  # 非 working 名额（钳负：working 可溢出截断）
    episodic_pool = [h for h in hits if h.kind == "episodic"]
    other_pool = [h for h in hits if h.kind != "episodic"]  # 现阶段只有 semantic
    episodic_slots = min(settings.EPISODIC_MIN_SLOTS, len(episodic_pool), max(budget, 0))
    # 补位池：保底名额之外的剩余候选。两个子列各自是全局序的子序列（相对次序不变），
    # 稳定排序后同分相对次序仍与全局序一致（episodic 在 hits 中恒先于 semantic）。
    fill = episodic_pool[episodic_slots:] + other_pool
    fill.sort(key=lambda h: -h.score)
    chosen = episodic_pool[:episodic_slots] + fill[:max(budget - episodic_slots, 0)]
    chosen.sort(key=lambda h: -h.score)  # 名额只定成员资格：最终仍按分数降序（见上取舍）
    top = (working_hits + chosen)[:max(top_k, 0)]

    # 5) 再巩固：成功检索提升强度（类比记忆被提取后重新巩固）。
    #    联想命中不参与——v1 联想通道只读，写回留给后续证据。
    #    E6 RIF：直接命中完成再巩固后，抑制同实体未被够到的竞争者（干扰论遗忘）。
    #    深搜恒传 boost_access=False：深搜是「查」不是「复习」，只读探索零写回。
    if boost_access:
        for h in top:
            _reinforce(store, h)
        _suppress_competitors(store, top, associated)

    # 6) E7 元认知 v1：检索置信度表面化。top 直接命中最高分低于阈值 = 没有一条
    #    是被这个查询真正够到的（小库噪声候选照常返回，但系统自知不可信）——
    #    直接命中与联想命中标 meta.uncertain（联想跟随直接命中的判定）。
    #    working 命中除外：当下上下文是「看见的」不是「想起的」，无需自信心标注。
    #    只标 meta，不改分数、不改排序、不触发任何写回行为变化。
    #    阈值按嵌入后端分档（confident_bar）：哈希嵌入离线标定 0.30，真实稠密
    #    嵌入（bge-m3 等）跨主题噪声底高取 0.70——同一机制，两把尺子。
    bar = confident_bar()
    if not any(h.kind != "working" and h.score >= bar for h in top):
        for h in top + associated:
            if h.kind != "working":
                h.meta["uncertain"] = True

    # 7) E7「我知道我不知道」：检索完全落空记入审计（既有 meta 表，不加新表）。
    #    健康报告「检索空缺」区块读它提示高价值待巩固方向。v1 全量记录不做
    #    重要度启发式过滤（避免误报），取舍见 ROADMAP E7。
    #    audit_gap=False（深搜）时跳过：审计行是一次 DB 写入，会破坏深搜
    #    「前后库字节不变」断路器——空手而归的深搜不留任何痕迹。
    if audit_gap and not top and not associated:
        store.log("system", 0, "retrieval_gap", query)

    # 8) E8 程序记忆激活：技能匹配追加在最尾——不占 top_k、不影响 uncertain
    #    判定（第 6 步已过）、不进 RIF（只看 semantic）。经验消费闭环：写进去
    #    的技能要能被想起，检索带出即 touch_usage 续命（LFU 的访问侧）。
    #    匹配文本恒为原查询（技能触发词认用户原话；深搜的 LLM 线索不充当触发词）。
    skill_hits = _match_skills(store, query, boost_access=boost_access)

    return top + associated + skill_hits


def _reinforce(store: SqliteStore, h: RetrievalHit) -> None:
    """检索后写回（再巩固）：episodic 按统一强度模型重算（频率项 log 饱和，防自我强化）；
    semantic 置信度小幅上调。

    E1 隐式复习：semantic 命中分 ≥ SM2_IMPLICIT_THRESHOLD 视为一次成功回忆，
    quality=score×5 映射喂给 SM-2（人脑的提取练习是内隐的，不依赖手动打卡）。
    只对 semantic 生效——repetition 表以语义事实为主键，episodic 的可塑性
    由强度模型的频率项承担，两类 id 混用会碰撞。

    E6 去自增强：semantic 写回走 touch_confidence 只调置信度，不再 upsert 续证
    （续证会 evidence_count+1，而 evidence 是联想激活 base 的输入——检索放大
    证据、证据抬高激活的自增强回路）。evidence 回归「只由真实观测累积」；
    被写入观测续证时按既有 renew 逻辑恢复/抬升 confidence，抑制可被新证据反转。
    """
    if h.kind == "episodic":
        mem = store.episodic.get(h.id)
        if mem:
            mem.access_count += 1
            mem.last_access_at = now_iso()
            res = compute_strength(mem.importance, mem.importance,
                                   mem.access_count, iso_to_ts(mem.last_access_at),
                                   arousal=mem.arousal,
                                   category=getattr(mem, "category", ""))
            mem.strength = res.score
            store.episodic.update(mem)
            store.log("episodic", mem.id, "retrieve")
    elif h.kind == "semantic":
        fact = store.semantic.get(h.id)
        if fact:
            fact.confidence = min(1.0, fact.confidence + 0.02)
            store.semantic.touch_confidence(fact.id, fact.confidence)
            store.log("semantic", fact.id, "retrieve")
            if fact.status == "pending":
                # P1 转正通道：试用期事实被想起一次记一次，够数且非冲突型则转正。
                # 只在直接命中路径计数（联想命中不进 top、不调 _reinforce，E4/C3
                # 零写回承诺天然成立）；状态判定取本次命中前的 status。
                _count_probation_hit(store, fact)
            if h.score >= settings.SM2_IMPLICIT_THRESHOLD:
                quality = max(0, min(5, round(h.score * 5)))
                SpacedRepetition(store.conn).review(h.id, quality)
                store.log("semantic", fact.id, "implicit_review", f"q={quality} score={h.score}")
    elif h.kind == "working":
        # E3 再巩固穿透：working 命中代表「同文本直接命中被去重」——把提取练习
        # 写回文本相同的活跃情景记忆。只存在于 working 的瞬时内容（被门控拦截）
        # 无长期痕迹可写，自然无操作；SM-2 不穿透（repetition 表以语义为主键）。
        for mem in store.episodic.fetch(status="active", limit=10 ** 9):
            if mem.summary != h.text:
                continue
            mem.access_count += 1
            mem.last_access_at = now_iso()
            res = compute_strength(mem.importance, mem.importance,
                                   mem.access_count, iso_to_ts(mem.last_access_at),
                                   arousal=mem.arousal,
                                   category=getattr(mem, "category", ""))
            mem.strength = res.score
            store.episodic.update(mem)
            store.log("episodic", mem.id, "retrieve", "via_working")


def _count_probation_hit(store: SqliteStore, fact: SemanticFact) -> None:
    """试用期事实被检索命中一次：计数 +1，达标则尝试转正。

    计数走 semantic.bump_hit_count（只动 hit_count，不碰 evidence_count）——
    E6 契约：检索是「被想起」不是「被观测」，两者混用会让检索放大证据、证据
    抬高联想激活 base，形成自增强回路。转正门槛取绝对次数（D3），自增强风险
    由该契约兜底而非靠命中率上限。
    """
    hits = store.semantic.bump_hit_count(fact.id)
    store.log("semantic", fact.id, "probation_hit",
              f"hits={hits}/{settings.PROMOTE_MIN_HITS}")
    if hits >= settings.PROMOTE_MIN_HITS:
        _try_promote(store, fact)


def _try_promote(store: SqliteStore, fact: SemanticFact) -> bool:
    """转正判定：达标 + 非冲突型 + 转正前复查无冲突，三条同时满足才转 active。

    D1 分类（pending 的两种语义必须分开处理）：
    - A 类·冲突待裁：memory_conflict 表有以本事实为 new_id 的 pending 行，新值
      与已有 active 事实互斥 → **绝不自动转正**（转正即取代旧事实，只能由冲突
      消解决定，见 conflict_resolver.resolve_conflict）；
    - B 类·低置信新事实：与谁都不冲突，只是证据不足 → 可以靠使用频率转正。

    转正前复查用 conflict_resolver 同一套原语（semantic.find_conflicts 是
    _store_fact 的冲突探测入口，conflicts.create 是它挂待裁的同一张表）——
    本事实入库时无冲突，但此后可能已有 active 事实占了同一个 (entity, relation)
    键（写入侧只拿 active 事实比对，pending 不入比对集，这个空档查不到）。
    复查发现冲突则挂待裁、维持 pending，把决定权交回既有裁决路径。
    不复用 store_fact_with_conflict_check 本身：那会走 upsert，检索路径踩进
    写入路径的 renew 分支正是 E6 明令禁止的自增强入口。
    """
    if store.conflicts.pending_for_new(fact.id):
        store.log("semantic", fact.id, "promote_blocked",
                  "A 类冲突待裁：转正即取代旧事实，须由裁决决定")
        return False

    conflicts = store.semantic.find_conflicts(fact.entity, fact.relation, fact.value)
    if conflicts:
        for old in conflicts:
            store.conflicts.create(old.id, fact.id, "value_conflict",
                                   f"promote-check: {old.value!r} vs {fact.value!r}")
        store.log("semantic", fact.id, "promote_blocked",
                  f"转正复查发现 {len(conflicts)} 条 active 冲突，挂待裁维持 pending")
        return False

    store.semantic.set_status(fact.id, "active")
    store.log("semantic", fact.id, "promote",
              f"{fact.entity} {fact.relation} {fact.value} 试用期转正（命中达标）")
    return True


def _suppress_competitors(store: SqliteStore, top: list[RetrievalHit],
                          associated: list[RetrievalHit]) -> None:
    """E6 提取诱发遗忘（RIF）：回忆过的实体，其本次未被够到的同类竞争者小幅降权。

    语义（认知心理学的 retrieval-induced forgetting）：成功提取 A 会抑制与 A
    共享检索线索（实体）的竞争项——「想起了 A，就想不起同类的 B」。

    作用面（宁窄勿宽，对照干净是底线）：
    - 只作用于检索写回路径（boost_access），不碰写入管线；working 命中无实体
      语义，不构成 RIF 种子；只处理 status=active 事实，pending/superseded 不波及；
    - 竞争者 = 同实体（经别名归一）、本次既不在直接命中 top、也不在联想带出
      associated 中的 active 事实——被联想带出者保持 E4 零写回承诺，不抑制；
    - 不同实体的事实绝不波及；同实体只有一条 active 事实时无竞争，不生效；
    - 只降 confidence（-RIF_PENALTY，下限 RIF_CONF_FLOOR 不许降穿）：不改
      status、不删行、不动版本链。抑制可被新证据反转——事实再次被写入观测
      （renew 续证）时按既有 upsert 逻辑恢复/抬升 confidence。
    """
    alias_map = store.aliases.as_map()

    def _entity_of(meta_entity: str) -> str:
        return alias_map.get(meta_entity, meta_entity)

    # 被够到的事实（直接命中或联想带出）与其挂靠实体——它们是被回忆/被联想的对象，不抑制
    recalled_fact_ids: set[int] = set()
    recalled_entities: set[str] = set()
    for h in top + associated:
        if h.kind == "semantic":
            recalled_fact_ids.add(h.id)
            if h.meta.get("entity"):
                recalled_entities.add(_entity_of(h.meta["entity"]))
    if not recalled_entities:
        return

    active_facts = store.semantic.fetch(status="active")
    for entity in recalled_entities:
        # 同实体（含别名归一到同一规范实体）的 active 事实群；单条无竞争
        group = [f for f in active_facts if _entity_of(f.entity) == entity]
        if len(group) <= 1:
            continue
        for fact in group:
            if fact.id in recalled_fact_ids:
                continue
            target = store.semantic.demote_confidence(fact.id, -settings.RIF_PENALTY)
            if target is not None:
                store.log("semantic", fact.id, "rif_suppress",
                          f"competitor of [{entity}] {fact.confidence:.2f}->{target:.2f}")
