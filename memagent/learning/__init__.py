"""学习层（突触可塑性）：记忆强度的可解释数学模型。

strength：strength = w1·importance + w2·confidence + w3·log(1+freq)
+ w4·recency(半衰) − 冲突折扣，分项可审计；编码/再巩固/遗忘三处统一调用。
spaced_repetition：SM-2 间隔重复调度。纯计算 + SQL，不依赖机制层。
"""
from memagent.learning.spaced_repetition import SpacedRepetition

__all__ = ["SpacedRepetition"]
