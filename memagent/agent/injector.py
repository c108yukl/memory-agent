"""前额叶注入（V1.7 P2 读侧）：每轮 turn 开始时由**框架**自动取记忆并拼进提示词。

为什么是框架侧而不是模型侧：让模型「想起来要查记忆」必然不可靠——它不知道库里
有什么、什么时候值得查、查回来该留多少。注入时机、召回条数、字符预算、工作记忆
名额，全部在这里一次性定好，模型只管消费。

复用而非另起炉灶：注入的内容就是一次标准 `retrieve` 的结果（同排序、同分数、
同 E7 元认知标记、同 P1 试用期打折），经 `build_context` 渲染。换一套口径等于
「评的与用的不是同一个东西」，这是本项目一贯的红线。

框架侧只做四处裁剪：
  1. 工作记忆置顶名额 AGENT_WORKING_INJECT_LIMIT（B3 后与检索侧
     WORKING_RETRIEVE_LIMIT 对齐为兜底——长期记忆的名额挤压已被门槛 #3 取代）；
  2. 字符预算 AGENT_INJECT_MAX_CHARS（build_context 的既有裁剪点）；
  3. 注入相关性门槛（B3）：直接命中须 FTS 关键词命中或向量相似度达后端档位
     （INJECT_MIN_VEC_HASH / INJECT_MIN_VEC_DENSE 双档）才注入；零条直接命中
     过门槛时联想命中一并连坐——「自动注入前的筛选才是关键：只注入高相关/
     高唤醒/未过时的事实，否则挤占上下文窗口」（用户架构方向原话）。门槛只
     落在注入、不进检索：CLI retrieve 与评测直接检索路径行为完全不变；
  4. 来源过滤（P1-5）：assistant 来源的经验层情景命中不自动注入（防复发
     纵深——助手回答即使未来意外入库，也不自动回到模型眼前形成自引用；
     可被 retrieve / 工具显式查到）。

注入的内容一律打 `meta.injected` 标记：recorder 靠它（更准确说是靠本模块返回的
injected_texts）识别「模型在复述刚注入的内容」，从而拒绝沉淀——自增强回路的
第一道闸。被字符预算裁掉的命中不算「注入过」（它没进模型），标记会撤掉。
"""
from __future__ import annotations

from dataclasses import dataclass, field

from memagent import settings
from memagent.adapters import llm
from memagent.core.domain import RetrievalHit
from memagent.retrieval import build_context, retrieve
from memagent.storage import SqliteStore


@dataclass
class Injection:
    """一轮注入的产物：query 是注入依据，context 是拼进提示词的文本。

    injected_texts 是**真正出现在 context 里**的命中原文（被字符预算裁掉的不算），
    recorder 用它做复述识别——没进模型的文本不可能被复述。
    """

    query: str = ""
    hits: list[RetrievalHit] = field(default_factory=list)
    context: str = ""
    injected_texts: list[str] = field(default_factory=list)

    @property
    def empty(self) -> bool:
        return not self.injected_texts


def inject(store: SqliteStore, query: str, top_k: int | None = None,
           max_chars: int | None = None, retriever_fn=retrieve) -> Injection:
    """取记忆 → 来源过滤 → 相关性门槛 → 框架侧裁剪 → 渲染成可注入文本。

    P1-5：检索结果先过 _drop_assistant_echo（assistant 来源的经验层不自动注入，
    且先于门槛——被拦内容不得充当联想锚点），再过 _relevance_gate（不够相关的
    直接命中与无锚点的联想命中不进上下文），最后做工作记忆名额与字符预算裁剪。
    boost_access 保持默认 True：注入即「被想起」，P1 转正通道与 E3 再巩固都靠
    这次写回计数（只读预览请显式调 retrieve(..., boost_access=False)）。

    P1-4：retriever_fn 可注入检索器（默认 retrieve = 现状，逐位兼容）——
    AgentLoop 深搜开启时传 retrieval.deep.deep_retrieve（LLM 展开查询多路召回）。
    深搜是只读探索（deep_retrieve 内部恒 boost_access=False），其结果的
    meta.vec / meta.fts 是多线索合并后的最强证据，注入门槛照常适用；本函数
    其余逻辑零改动。
    """
    top_k = settings.AGENT_INJECT_TOP_K if top_k is None else top_k
    max_chars = settings.AGENT_INJECT_MAX_CHARS if max_chars is None else max_chars

    hits = retriever_fn(store, query, top_k=top_k)
    kept = _trim_working(_relevance_gate(_drop_assistant_echo(hits)))
    for h in kept:
        h.meta["injected"] = True

    context = build_context(kept, max_chars=max_chars, store=store)

    def _injected_form(h):
        # 「真出现在 context 里」的比对文本。rule 命中是特例渲染（P2-1：[规律]
        # 前缀 + 规律原文，与 hit.text 的三元组形态不同）——必须用 rule_value
        # 比对，否则 rule 永远被判「被预算裁掉」：injected 标记被撤、复述拦截
        # 的参照系缺规律原文，模型复述规律就绕过了自增强防护（P2-1 交付时
        # 自查发现，验收轮当场修）。
        return h.meta.get("rule_value") or h.text

    injected = [t for h in kept if (t := _injected_form(h)) and t in context]
    for h in kept:
        if _injected_form(h) not in context:
            h.meta.pop("injected", None)  # 被预算裁掉：标记撤回，不算注入过
    return Injection(query=query, hits=kept, context=context, injected_texts=injected)


def _inject_vec_floor() -> float:
    """B3 注入向量门槛按嵌入后端分档（仿 retriever.confident_bar 的双档模式）：

    - hash（离线哈希兜底）：INJECT_MIN_VEC_HASH=0.30，与 RETRIEVAL_CONFIDENT_BAR
      同源标定——哈希 2-gram 同题 ~0.35+、噪声底 0.2~0.3；
    - local / cloud（真实稠密嵌入，如 bge-m3）：INJECT_MIN_VEC_DENSE=0.65，
      跨主题噪声底实测 0.56~0.64，门槛压在噪声底之上。
    实时查询 llm.embed_backend()（零网络，只读探测缓存）——inject 内先 embed 过，
    缓存必已反映本次查询实际使用的后端。
    """
    backend = llm.embed_backend()
    if backend in ("local", "cloud"):
        return settings.INJECT_MIN_VEC_DENSE
    return settings.INJECT_MIN_VEC_HASH


def _drop_assistant_echo(hits: list[RetrievalHit]) -> list[RetrievalHit]:
    """P1-5 来源过滤（防复发纵深）：assistant 来源的经验层情景命中不自动注入。

    为什么拦：P0 五层防线保证了「助手未点名内容零长期入库」（assistant 侧成败
    词表已移除，整段汇报只进工作记忆）——但这是写入侧纪律，注入侧不能再假设它
    永远成立。若未来任何翻车形态让助手回答意外沉淀为 experience 情景（绿色通道
    宽进、无门闸可拦），自动注入会让它回到模型眼前形成自引用：助手引用自己上次
    的发言当「用户记忆」，越引越像事实。命中带 meta.source（P0-4 落地的列）+
    meta.green（category=experience，E8 分层标记），两个条件同时成立才拦：
    - 只拦 experience：普通情景（用户叙述的事件）即使 source 误标也不拦——
      宁可窄不可宽，本过滤只针对「助手自引用」这一种翻车形态；
    - 只拦自动注入：内容照常在检索结果里（retrieve / 工具显式查询可达），
      「不自动塞给模型」不是「看不见」。

    放行口径（防修过头堵正路）：
    - memory_remember 工具写入的 experience 走 source="tool"（agent.tools 注释
      明说），不受影响——那是模型点名写入的合法通路；
    - 存量行 source='unknown'（P0-4 之前的库）不受影响；
    - user / system / feedback 来源不受影响。

    挂点在 _relevance_gate 之前：被拦内容不得充当联想锚点（否则拦了本体、
    放了影子——它照样通过 anchored 拉起一批联想命中进上下文）。
    """
    return [h for h in hits
            if not (h.kind == "episodic"
                    and h.meta.get("source") == "assistant"
                    and h.meta.get("green"))]


def _relevance_gate(hits: list[RetrievalHit]) -> list[RetrievalHit]:
    """B3 注入相关性门槛：不够相关的直接命中不进上下文，寒暄轮自动只剩 working。

    动机（用户架构方向原话）：「自动注入前的筛选才是关键：只注入高相关/高唤醒/
    未过时的事实，否则挤占上下文窗口」。实测翻车形态：一句「你好！」被注入 8 条
    长期记忆，大半不相关——根因有二：①融合分 0.6×向量+0.4×置信度里置信度是
    常量底盘，高置信事实无论相关与否都能进 top_k；②联想通道的分数是激活值
    （hub 实体趋近 1.0）不是相关度，hub 成员两跳扩散后全员搭车。

    检索返回的噪声候选照常存在（retriever 口径零变化，E7 用 uncertain 表达「自知
    不可信」），但把它们拼进提示词是循环场景特有的代价——注入是唯一每轮自动发生
    的读路径，筛选只能落在这里。保留条件：

    - working 绕过：当下上下文永远是相关的；
    - procedural 绕过：技能带出跟随触发词而非相关度（bigram 覆盖 >0 才出，本身
      就是显式词汇证据，同 FTS 的豁免理由；E8 亦据此豁免其 uncertain）；
    - episodic/semantic 直接命中：meta.fts（FTS 关键词命中=显式词汇证据，措辞
      不同但同题的损失由关键词通道兜底）**或** meta.vec ≥ 当前后端档位值
      （_inject_vec_floor）；
    - 联想命中（meta.associated）：有直接命中过门槛才保留（E4 语义不变），零锚点
      则全部不注入——没有相关锚点就没有联想的意义。寒暄降级由此自动成立：零条
      直接命中过门槛 → 联想连坐丢弃，注入只剩 working，无需专门的寒暄检测器。
      P1-5 连坐豁免：带 meta.alias_seed 的联想命中单独放行——别名是「查询词与
      实体名的字面重叠」（P1-3 检索侧只给这路实体打标），证据强度等价 FTS 命中
      （显式词汇证据，同上 FTS 的豁免理由），不需要再借直接命中作锚。其余联想
      命中（含 meta.episodic_anchor——情景文本是另一条记忆的措辞，不构成对本
      查询的词汇证据）连坐语义原样。

    保持入参顺序不变（两遍扫描：先裁决锚点，再按原序筛选）。
    """
    floor = _inject_vec_floor()
    anchored = any(
        h.kind in ("episodic", "semantic") and not h.meta.get("associated")
        and (h.meta.get("fts") or h.meta.get("vec", 0.0) >= floor)
        for h in hits)
    kept: list[RetrievalHit] = []
    for h in hits:
        if h.meta.get("associated"):
            # P1-5 连坐豁免：alias_seed 命中自带词汇证据，不看 anchored（有锚点时
            # 它们本就保留，两种条件取并集语义一致）
            if anchored or h.meta.get("alias_seed"):
                kept.append(h)
        elif (h.kind in ("working", "procedural")
              or h.meta.get("fts") or h.meta.get("vec", 0.0) >= floor):
            kept.append(h)
        # 其余（无 FTS、向量不达标的直接命中）：不够相关，不注入
    return kept


def _trim_working(hits: list[RetrievalHit]) -> list[RetrievalHit]:
    """压掉超名额的工作记忆命中（其余通道原样保留，顺序不变）。"""
    kept: list[RetrievalHit] = []
    working = 0
    for h in hits:
        if h.kind == "working":
            working += 1
            if working > settings.AGENT_WORKING_INJECT_LIMIT:
                continue
        kept.append(h)
    return kept
