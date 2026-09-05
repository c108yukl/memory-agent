"""规则打分器：可解释、离线可用的重要性打分。"""
from __future__ import annotations

import re

from memagent import settings
from memagent.core.domain import Event

EXPLICIT_WORDS = ["记住", "请记住", "以后", "今后", "一直", "永远", "总是", "别再", "记得"]
PREFERENCE_WORDS = ["喜欢", "希望", "偏好", "更倾向", "讨厌", "不喜欢", "不要", "希望别", "比较喜欢",
                    "还是保持", "改回", "换成", "更偏好"]
RISK_WORDS = ["密码", "密钥", "token", "apikey", "api key", "隐私", "机密", "安全", "合规", "勿外传", "加密"]
SUCCESS_WORDS = ["成功", "搞定", "完成", "通过", "解决了", "太好了"]
FAILURE_WORDS = ["失败", "出错", "报错", "崩溃", "不行", "又失败了", "问题"]
# 自我描述（身份/阶段/属性）：用户是谁是记忆系统最核心的长期信息。
# "我是"后允许跨逗号长跨度匹配（"我是，即你认为的唯一用户，…，是一个准大一新生"）
IDENTITY_PATTERN = re.compile(r"我(是|叫|今年)[^。]{1,40}")


def score_by_rules(event: Event) -> float:
    """规则打分：0~1，多个维度取最大命中分（加和截断）。"""
    text = event.content.lower()
    score = 0.0

    explicit_hits = sum(1 for w in EXPLICIT_WORDS if w in text)
    if explicit_hits:
        score += settings.W_EXPLICIT * min(2, explicit_hits)
    if any(w in text for w in PREFERENCE_WORDS):
        score += settings.W_PREFERENCE
    if any(w in text for w in RISK_WORDS):
        score += settings.W_RISK
    if IDENTITY_PATTERN.search(text):
        score += settings.W_EXPLICIT
    if event.outcome == "success":
        score += settings.W_FEEDBACK
    if event.outcome == "failure":
        score += settings.W_FEEDBACK

    return min(1.0, score)
