"""编码层提示词集中管理。"""
from __future__ import annotations

SUMMARY_PROMPT = (
    "把下面的事件压缩成一句中文摘要（≤40字，保留关键信息：谁/什么事/结果）。"
    "直接输出摘要，不要解释。\n事件：{content}"
)

FACT_EXTRACT_PROMPT = (
    "从下面的事件中抽取用户偏好、事实或经验教训，输出 JSON 数组，"
    "每项格式：{{\"entity\": 主语, \"relation\": 关系, \"value\": 内容, \"confidence\": 0~1}}。"
    "每项可带可选字段 \"validity_context\"（事实成立的条件限定）："
    "{{\"preconditions\": [生效前提...], \"environment\": [环境限定...], "
    "\"expires_if\": \"失效条件\"}}——只在原文确有条件/限定表述时输出，"
    "抽不出就不要输出该键。没有可抽信息就输出 []。只输出 JSON。\n事件：{content}"
)

SKILL_EXTRACT_PROMPT = (
    "如果下面事件包含明确的做事方法/步骤/规则，输出一个技能对象 JSON："
    "{{\"name\": 技能名, \"trigger\": 触发条件, \"policy\": 操作策略}}；否则输出 null。只输出 JSON。\n事件：{content}"
)
