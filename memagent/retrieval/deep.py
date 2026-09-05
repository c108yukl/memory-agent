"""P1-4 深搜记忆检索（Deep Recall）：用户显式点名的 LLM 加深召回。

定位与红线（项目哲学：自动注入是主读路径，深搜绝不自动发生）：
- 深搜是**用户显式点名**的重检索：CLI ``retrieve --deep`` / 会话 ``/deep`` /
  memory_search 工具的 ``deep`` 参数，三处入口默认全关；
- **断路器一·不入库**：展开的线索只用于召回，绝不调 ingest、不写 meta 审计、
  不产生任何 INSERT/UPDATE（tests/unit/test_deep.py 以全表 count 前后一致锁定）；
- **断路器二·条目必有库内 id**：进结果/注入的条目全部由 collect_candidates 从
  库里取出的真行构建（RetrievalHit 结构天然保证），线索文本永远不出现在结果里，
  只以 meta.deep.via 记录「这条记忆是被哪条线索够到的」；
- **断路器三·只读探索**：深搜路径全程 boost_access=False——不 _reinforce、不
  RIF、不 touch_usage、不 SM-2、不写 retrieval_gap 审计（finalize_retrieval 的
  audit_gap=False）。深搜是「查」，不是「复习」；
- **静默降级**：LLM 不可用 / 失败 / 超时 / 输出不合法 → 线索退化为 [query]
  原样，直接走 retrieve() 同一条代码路径（等价快搜，逐位一致由构造保证），
  绝不报错中断对话。

机制（补快搜的最后一缺）：快搜召回失败的真实形态是「查询措辞与记忆文本零
词汇重叠且向量相似度不足」（真实事故：查询「我打算换个云厂商」唤不醒雨云
记忆；P1-1 缓解排序、P1-3 缓解联想，查询本身的召回缺口由本模块补）。深搜用
一次 llm.chat 把查询展开成 2~3 条假设性线索（改写 + 关联概念/上位词 + 相邻
场景），每条线索独立跑「FTS + 向量」双路候选收集（collect_candidates 与快搜
同一口径），按 (kind, id) 合并取最强证据，再按快搜同款融合公式与分池组装重排
（finalize_retrieval 共用）——联想/技能带出照常生效（种子来自合并后的真直接
命中；别名种子与技能触发词只认用户原查询，LLM 线索不充当词汇证据）。

超时：不另设。展开的耗时上限 = llm.chat 现有超时（本地 LM_STUDIO_TIMEOUT=30s
/ 云端 CLOUD_TIMEOUT=60s，连续失败熔断照常生效），适配层零改动（任务红线）。
"""
from __future__ import annotations

import json

from memagent import settings
from memagent.adapters import llm
from memagent.core.domain import RetrievalHit
from memagent.retrieval.retriever import (
    CueCandidates,
    _episodic_hit,
    _semantic_hit,
    _working_hits,
    collect_candidates,
    finalize_retrieval,
    retrieve,
)
from memagent.storage import SqliteStore

# ---------- 深搜展开提示词（常量放本模块——它是深搜机制的一部分，非全局配置） ----------
DEEP_EXPAND_SYSTEM = "你是记忆检索助手，输出必须是合法 JSON。"
DEEP_EXPAND_PROMPT = (
    "为下面的用户查询生成 2~3 条用于记忆库检索的短线索，帮助唤回措辞不同但"
    "主题相关的记忆：\n"
    "1. 换一种措辞复述查询原意；\n"
    "2. 补充关联概念或上位词（例：「换云厂商」→「服务器迁移 数据备份方案」）；\n"
    "3. 给出一个相邻场景或更具体的说法。\n"
    "每条不超过 20 个字。只输出 JSON 字符串数组（形如 [\"线索一\",\"线索二\"]），"
    "不要输出解释、编号或代码块标记。\n\n用户查询：{query}"
)

_VIA_MAX_CHARS = 40   # via 标记的截断长度（线索本体已限 40 字，对原查询再兜底一层）
_CUE_MAX_CHARS = 40   # 单条线索的长度上限（超出即丢弃——宁缺毋滥）


def deep_retrieve(store: SqliteStore, query: str, top_k: int | None = None,
                  expand_fn=None) -> list[RetrievalHit]:
    """深搜：LLM 展开查询为多条线索 → 多路召回合并 → 快搜同款后处理。

    - top_k：与 retrieve 同义（None 取 RETRIEVE_TOP_K）；
    - expand_fn：自定义线索展开器（测试注入 mock / 未来自定义展开器），
      缺省用内置 _expand（llm.chat 一次调用）。
    全程 boost_access=False（断路器三：只读探索）；线索绝不入库（断路器一）；
    返回条目全部来自库内真行（断路器二）。LLM 任何失败 → 等价快搜（静默降级）。
    """
    top_k = settings.RETRIEVE_TOP_K if top_k is None else top_k
    expansions = _expand(query, expand_fn)
    if not expansions:
        # 降级：线索退化为 [query] = 等价快搜。直接走 retrieve() 同一条代码
        # 路径（逐位一致由构造保证），仅把 boost_access 钉死为 False（红线 3）。
        return retrieve(store, query, top_k=top_k, boost_access=False)

    cues = [query] + expansions
    # working 置顶照旧：只以原查询跑一次 working 检索（线索不进 working 检索——
    # working 是「当下上下文」，只认用户原话，与快搜口径一致）。
    qvec = llm.embed(query)
    working_hits = _working_hits(store, qvec)
    working_texts = {h.text for h in working_hits}

    # 多路召回：每条线索独立跑与快搜同一套的双路候选收集。cue0 即原查询，
    # 复用已算好的 qvec（同一文本不重复过嵌入通道）。
    per_cue: list[tuple[str, CueCandidates]] = []
    for i, cue in enumerate(cues):
        cvec = qvec if i == 0 else llm.embed(cue)
        per_cue.append((cue, collect_candidates(store, cue, qvec=cvec,
                                                working_texts=working_texts)))

    merged_epi, merged_sem = _merge_cues(per_cue)
    hits: list[RetrievalHit] = []
    for mem, vec, fts, via in merged_epi:
        h = _episodic_hit(mem, vec, fts)
        h.meta["deep"] = {"via": via}   # 命中线索留痕（纯 meta 标记，不参与分数）
        hits.append(h)
    for fact, vec, fts, via in merged_sem:
        h = _semantic_hit(fact, vec, fts)
        h.meta["deep"] = {"via": via}
        hits.append(h)
    # 融合/组装/联想/技能后处理与快搜完全同一套（finalize_retrieval）；
    # boost_access=False + audit_gap=False：深搜全程只读，库前后字节不变。
    return finalize_retrieval(store, query, working_hits, hits,
                              top_k=top_k, boost_access=False, audit_gap=False)


def _merge_cues(per_cue: list[tuple[str, CueCandidates]]
                ) -> tuple[list, list]:
    """按 (kind, id) 合并各线索的候选（P1-4 第 3 步）。

    同一候选取各线索中的**最强证据**：vec 取 max、fts 任一线索命中即为真；
    meta.deep.via 记**证据最强的线索**文本（vec 相同时偏好带 FTS 证据者，再按
    线索先后取先者）——原查询排最前，它本身就以最强证据够到的候选 via=原查询
    （「不多标」）；小库里原查询对记忆只有 vec=0 的噪声级「够到」（cosine_search
    按排名返回所致）不算数，via 让位给真正提供证据的展开线索。
    融合分不在本函数计算：合并完成后由 _episodic_hit/_semantic_hit 按快搜同款
    公式用最强证据重建（其余分项 importance/strength/confidence/证据数都读自
    同一库内行，与线索无关）。返回 (合并后 episodic, 合并后 semantic)，元素为
    (行, vec, fts, via)，保持首见顺序（同分稳定排序的次序来源，确定性保证）。
    """
    merged: list[tuple[str, object, float, bool, str]] = []
    index: dict[tuple[str, int], int] = {}
    for cue, cands in per_cue:
        for kind, items in (("episodic", cands.episodic), ("semantic", cands.semantic)):
            for row, vec, fts in items:
                key = (kind, row.id)
                pos = index.get(key)
                if pos is None:
                    index[key] = len(merged)
                    merged.append((kind, row, vec, fts, _truncate_cue(cue)))
                    continue
                _, row0, vec0, fts0, via0 = merged[pos]
                # 同库同 id 的行内容相同（收集全程只读），保留首见行对象即可。
                # via 归属：本线索证据严格更强（vec 更高，或 vec 持平而本线索有
                # FTS 证据）才接管——原查询的噪声级够到让位给真证据线索
                better = vec > vec0 or (vec == vec0 and fts and not fts0)
                merged[pos] = (kind, row0, max(vec0, vec), fts0 or fts,
                               _truncate_cue(cue) if better else via0)
    epi = [(row, vec, fts, via) for kind, row, vec, fts, via in merged if kind == "episodic"]
    sem = [(row, vec, fts, via) for kind, row, vec, fts, via in merged if kind == "semantic"]
    return epi, sem


def _truncate_cue(cue: str) -> str:
    """via 记录用的线索文本截断（原查询可能远超线索长度上限）。"""
    return cue if len(cue) <= _VIA_MAX_CHARS else cue[:_VIA_MAX_CHARS]


# ---------- 线索展开（LLM 通道 + 严格校验 + 静默降级） ----------

def _expand(query: str, expand_fn=None) -> list[str]:
    """线索展开：LLM 把查询改写为 2~3 条检索线索；**任何失败都退化为 []**
    （调用方据此走等价快搜），绝不抛错——降级是深搜的默认姿态。

    expand_fn 的输出同样过 _validate_cues：断路器对任意线索来源一视同仁。
    """
    fn = _expand_via_llm if expand_fn is None else expand_fn
    try:
        raw = fn(query)
    except Exception:
        return []  # LLM 失败/超时/展开器异常：静默降级，绝不中断对话
    return _validate_cues(raw, query)


def _expand_via_llm(query: str) -> list[str]:
    """内置展开器：llm.chat 一次调用，要求输出 JSON 字符串数组。

    超时即 chat 通道现有超时（本地 30s / 云端 60s，见模块 docstring——不另设，
    适配层零改动）。离线时 chat 返回 None → _parse_cues 返回 [] → 降级快搜。
    查询截断 200 字（与 toolkit.memory_search 同款防灌水口径）。
    """
    out = llm.chat(DEEP_EXPAND_PROMPT.format(query=query[:200]),
                   system=DEEP_EXPAND_SYSTEM)
    return _parse_cues(out)


def _parse_cues(text) -> list:
    """从模型输出抠出 JSON 数组：整段解析失败时取首 ``[`` 到末 ``]`` 的子串
    重试（模型常在数组前后加说明文字；与 toolkit._extract_json_obj 同一容错
    立场，但只认数组——对象/标量一律视为无效，validate 严格）。"""
    if not isinstance(text, str) or not text.strip():
        return []
    try:
        data = json.loads(text)
    except ValueError:
        start, end = text.find("["), text.rfind("]")
        if start < 0 or end <= start:
            return []
        try:
            data = json.loads(text[start:end + 1])
        except ValueError:
            return []
    return data if isinstance(data, list) else []


def _validate_cues(raw, query: str) -> list[str]:
    """线索严格校验：JSON list[str]、去首尾空白、非空、每条 ≤40 字、与原查询
    去重（原样线索是无效展开，只会白付一次 embed）、截断到
    settings.DEEP_EXPANSIONS 条。不合格条目直接丢弃，全部不合格返回 []
    ——等价快搜。"""
    if not isinstance(raw, list):
        return []
    cues: list[str] = []
    seen = {query}
    for item in raw:
        if not isinstance(item, str):
            continue
        cue = item.strip()
        if not cue or len(cue) > _CUE_MAX_CHARS or cue in seen:
            continue
        seen.add(cue)
        cues.append(cue)
        if len(cues) >= settings.DEEP_EXPANSIONS:
            break
    return cues
