"""遗忘层：强度重算 → 归档 → 摘要降级（原始记忆标 summarized 可溯源）→ 硬删。

decay：强度重算；degrade：摘要降级替代硬删（LLM 失败绝不写半成品）；
policies：遗忘策略编排。依赖 learning 强度模型与 memory 仓储，被 CLI/TUI/eval 调用。
"""
from memagent.forgetting.policies import run_forgetting

__all__ = ["run_forgetting"]
