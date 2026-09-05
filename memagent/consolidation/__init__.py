"""巩固层（睡眠）：离线整理记忆。

cluster：相似聚类；summarizer：簇摘要 + 语义蒸馏（LLM 可选）；
conflict_resolver：写入时的冲突分级消解（显式取代 / conf≥0.8 取代 / 低置信挂起）；
job：一轮巩固的编排。注意：巩固后的健康报告由应用层（CLI/TUI）生成，
本层不反向依赖 reports（依赖规则见 ARCHITECTURE.md）。
"""
from memagent.consolidation.job import consolidate

__all__ = ["consolidate"]
