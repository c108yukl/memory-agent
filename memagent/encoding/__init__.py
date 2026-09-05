"""编码层（海马体）：事件 → 三类记忆的编码。

episodic_encoder / semantic_extractor / procedural_extractor 分别对应
情景摘要、语义三元组（含显式声明保底抽取）、程序技能沉淀；
entity_resolver 做实体归一（NFKC + 精确代词映射），prompts 存 LLM 提示词。
依赖 adapters.llm（可选增强）与 core，被 pipeline / consolidation 调用。
"""
from memagent.encoding.episodic_encoder import encode_episodic
from memagent.encoding.procedural_extractor import extract_skills
from memagent.encoding.semantic_extractor import extract_semantic_facts

__all__ = ["encode_episodic", "extract_semantic_facts", "extract_skills"]
