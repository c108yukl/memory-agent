"""检索层：混合检索（FTS5 关键词 + 向量余弦）→ 融合排序 → 命中后再巩固。

retriever：双路召回 + 分数融合 + 强度写回（突触可塑性）+ 联想命中追加（E4）+
E7 置信阈值按嵌入后端分档（confident_bar：hash 0.30 / 真实嵌入 0.70）；
ranker：新近度提升与上下文构建（build_context 供上层拼 prompt）；
activation：实体图构建与 ACT-R 式两跳扩散激活（E4；E5 巩固 REM 阶段复用）。
依赖 memory 仓储与 adapters.llm（嵌入），被 pipeline 之外的读路径共用（CLI/TUI/eval）。
"""
from memagent.retrieval.activation import EntityGraph, build_entity_graph, spread
from memagent.retrieval.ranker import (
    build_context,
    format_validity_context,
    inject_provenance,
    provenance_suffix,
    recency_boost,
)
from memagent.retrieval.retriever import confident_bar, retrieve

__all__ = ["retrieve", "build_context", "recency_boost", "confident_bar",
           "inject_provenance", "provenance_suffix", "format_validity_context",
           "EntityGraph", "build_entity_graph", "spread"]
