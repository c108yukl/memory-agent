"""注意力层（前额叶门控）：决定一条输入值不值得进长期记忆。

scorers：规则打分（离线保底，显式声明类型有分数下限）；
gate：规则 + LLM 混合打分，低于 WORKING_THRESHOLD 丢弃、低于
WRITE_THRESHOLD 只进工作记忆。被 pipeline（写入管线）调用。
"""
from memagent.attention.gate import attention_gate, score_by_llm
from memagent.attention.scorers import score_by_rules

__all__ = ["attention_gate", "score_by_llm", "score_by_rules"]
