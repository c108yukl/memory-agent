"""排序与上下文构建：分数融合、新近度加成、检索结果注入。

E7 provenance 溯源：build_context 传入 store 时为每条命中读侧计算证据次数 /
距上次验证天数 / 过时标注并注入 meta（不写库）——「我为什么相信这条记忆」
随结果一起暴露给上层；uncertain 命中（低置信表面化）渲染为 [低置信] 前缀。
P2-1 图式归纳：relation="rule" 的命中由 inject_provenance 注入只读 rule meta，
build_context 渲染为 [规律] 前缀 + (规律·基于N起经历) 证据后缀（检索零改动）。
P2-2 条件上下文：semantic 事实带 validity_context（认识论限定，如「未配异地
备份时雨云不可靠，配了 S3 备份则可用」）时注入只读 meta.vc，build_context 在
既有 provenance 后缀后追加 ｜{条件串}（钳长防吃预算）；无该字段的事实渲染
逐字符不变。注入过滤（按条件裁剪命中）暂缓——既有决策，本任务只做存储与展示。
"""
from __future__ import annotations

import json
import math

from memagent import settings
from memagent.core.clock import iso_to_ts, now_ts
from memagent.core.domain import RetrievalHit

# build_context 行尾条件串的钳长（P2-2）：条件限定是背景信息不是主内容，
# 超长截断加省略号——防止长 vc 吃掉注入预算挤掉真正相关的记忆
_VC_MAX_CHARS = 60


def format_validity_context(vc: str) -> str:
    """把 validity_context JSON 渲染成人读的条件短串（读侧展示共用：meta.vc、CLI）。

    解析失败 / 非法结构 / 空串一律返回 ""（按无条件限定处理，渲染侧零输出）；
    键顺序固定 生效条件 → 环境 → 失效，段间用「；」。CLI 的 history/inspect
    展示全量串，build_context 侧另有钳长（_VC_MAX_CHARS）。
    """
    if not vc:
        return ""
    try:
        obj = json.loads(vc)
    except (ValueError, TypeError):
        return ""
    if not isinstance(obj, dict):
        return ""
    parts = []
    pre = obj.get("preconditions")
    if isinstance(pre, list) and pre:
        parts.append("生效条件: " + "、".join(str(x) for x in pre))
    env = obj.get("environment")
    if isinstance(env, list) and env:
        parts.append("环境: " + "、".join(str(x) for x in env))
    exp = obj.get("expires_if")
    if isinstance(exp, str) and exp.strip():
        parts.append("失效: " + exp.strip())
    return "；".join(parts)


def recency_boost(created_ts: float, half_life_days: float = 30.0) -> float:
    if not created_ts:
        return 0.0
    days = (now_ts() - created_ts) / 86400.0
    return math.exp(-math.log(2) * max(0.0, days) / half_life_days)


def inject_provenance(h: RetrievalHit, store=None) -> None:
    """E7 读侧 provenance：把「这条命中凭什么可信」注入 meta（纯读，不写库）。

    - semantic：meta.evidence = evidence_count（真实观测续证次数，检索不累积）、
      meta.days_since_valid = (now - valid_from).days，超 PROVENANCE_STALE_DAYS
      标 meta.stale（过时事实显式标注，上层可决定降权或反问）；
      P2-2：事实带 validity_context 时注入 meta.vc（format_validity_context
      拼好的条件短串，解析失败按空处理）——纯 meta 零行为变化，同 B3/rule 惯例；
    - episodic：证据次数语义不适用，days_since_valid 按创建时间起算，同样有 stale；
    - working：meta.ephemeral = True（会话级易失内容，无溯源可言）。
    时间一律走 core.clock（评测可时间旅行）；天数负值（时钟倒流）按 0 处理。
    store 为 None 时只标 working 的 ephemeral，长期命中保持原样（向后兼容）。
    """
    if h.kind == "working":
        h.meta["ephemeral"] = True
        return
    if store is None:
        return
    if h.kind == "semantic":
        fact = store.semantic.get(h.id)
        if not fact:
            return
        h.meta["evidence"] = fact.evidence_count
        # P2-1：rule 命中的读侧标记（纯 meta，排序/分数/阈值零变化，同 B3 惯例）——
        # build_context 据此改用 [规律] 形态渲染（值本体 + 「基于N起经历」证据后缀，
        # N=参与归纳的情景数，规律的证据链是 source_event_ids 而非 evidence_count）。
        # TUI 的检索页有自己的排版、不消费这些键（渲染不动，遗留）。
        if fact.relation == "rule":
            h.meta["rule"] = True
            h.meta["rule_value"] = fact.value
            h.meta["rule_episodes"] = len(fact.source_event_ids)
        # P2-2 条件上下文：只读 meta（纯展示，排序/阈值零变化）。rule 也可带条件
        #（规律同样只在特定前提下成立），渲染侧对 rule/普通事实一视同仁。
        vc = format_validity_context(fact.validity_context)
        if vc:
            h.meta["vc"] = vc
        anchor = fact.valid_from
    elif h.kind == "episodic":
        mem = store.episodic.get(h.id)
        if not mem:
            return
        anchor = mem.created_at
    else:
        return
    days = max(0.0, (now_ts() - iso_to_ts(anchor)) / 86400.0)
    h.meta["days_since_valid"] = int(days)
    if days > settings.PROVENANCE_STALE_DAYS:
        h.meta["stale"] = True


def provenance_suffix(h: RetrievalHit) -> str:
    """命中行的溯源后缀（meta 里没有 provenance 键时返回空，保持旧行为）。

    CLI 的 build_context 与 TUI 的检索页共用本函数——展示格式由各自排版，
    数据来源（inject_provenance 注入的 meta）必须同源。
    E8：procedural 技能命中的溯源是使用统计（成功率/用过几次），由 retriever
    的 _match_skills 直接注入 meta，此处只负责渲染。
    """
    if h.meta.get("ephemeral"):
        return "(会话级)"
    if h.meta.get("skill"):
        return (f"(技能·成功率{h.meta.get('success_rate', 0.0):.0%}"
                f"·用过{h.meta.get('usage', 0)}次)")
    if "days_since_valid" not in h.meta:
        return ""
    stale = "[久未验证]" if h.meta.get("stale") else ""
    if "evidence" in h.meta:
        return (f"(证据×{h.meta['evidence']} / "
                f"验证于{h.meta['days_since_valid']}天前){stale}")
    return f"(发生于{h.meta['days_since_valid']}天前){stale}"


_provenance_suffix = provenance_suffix  # 旧私有名兼容别名


def build_context(hits: list[RetrievalHit], max_chars: int = 800, store=None) -> str:
    """把检索结果压缩成可注入上下文的片段。

    E7：传 store 时每条命中注入 provenance 并以行尾后缀带出（证据×N /
    验证于X天前[久未验证]）；uncertain 命中加 [低置信] 前缀——低置信与
    过时是两个正交信号：前者是「这次检索可能不相关」，后者是「这条
    记忆本身可能已过期」。首行结构 `- (kind:score) text` 保持稳定。
    P2-1：relation="rule" 的 semantic 命中不用常规三元组渲染（[entity] rule
    value），改用 `[规律]` 前缀 + 规律原文，行尾证据后缀换成
    `(规律·基于N起经历)`（N=len(source_event_ids)）——规律的证据是参与归纳的
    情景全集，与普通事实的 evidence_count 是两回事。判定依据是 inject_provenance
    注入的只读 meta（rule / rule_value / rule_episodes），store 为 None（无溯源）
    或 TUI 自有排版时保持原样。
    P2-2：semantic 命中带 meta.vc（条件上下文短串）时在既有后缀后追加
    ` ｜{vc}`（钳长 _VC_MAX_CHARS，截断加省略号）——无 vc 的行逐字符不变；
    [规律] 行同样追加（规律也可带成立条件）。
    """
    parts = []
    total = 0
    for h in hits:
        inject_provenance(h, store)
        flag = "[低置信] " if h.meta.get("uncertain") else ""
        body, suffix = h.text, provenance_suffix(h)
        if h.kind == "semantic" and h.meta.get("rule"):
            flag += "[规律] "
            body = h.meta.get("rule_value") or h.text
            suffix = f"(规律·基于{h.meta.get('rule_episodes', 0)}起经历)"
        line = f"- {flag}({h.kind}:{h.score:.2f}) {body}"
        if suffix:
            line += f" {suffix}"
        # P2-2 条件上下文：追加在 provenance 后缀之后（meta 无 vc 键 = 无条件
        # 限定 = 既有输出逐字符不变，钳长防长条件吃掉注入预算）
        vc = h.meta.get("vc") or ""
        if vc:
            if len(vc) > _VC_MAX_CHARS:
                vc = vc[:_VC_MAX_CHARS - 1] + "…"
            line += f" ｜{vc}"
        if total + len(line) > max_chars:
            break
        parts.append(line)
        total += len(line)
    return "\n".join(parts)
