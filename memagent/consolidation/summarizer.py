"""簇摘要与事实蒸馏：LLM 归纳 + JSON 解析。

CLUSTER_PROMPT（簇摘要，面向情景压缩）/ DISTILL_PROMPT（扁平三元组蒸馏）/
RULE_INDUCE_PROMPT（P2-1 图式归纳：跨事件情景 -> 一条 relation="rule" 高阶规律，
_parse_rule 负责合法性校验——断路器的格式闸）三种提示词的单一来源。"""
from __future__ import annotations

import json

from memagent import settings
from memagent.adapters import llm
from memagent.core.domain import EpisodicMemory
from memagent.encoding.prompts import SUMMARY_PROMPT  # noqa: F401  (保持提示词单一来源)

CLUSTER_PROMPT = (
    "下面是若干条相似经历，请把它们归纳成一句通用事实或规律（≤50字），"
    "直接输出，不要解释。\n"
    "{items}"
)

DISTILL_PROMPT = (
    "从下面这些相似经历中抽取 1~3 条可长期记住的用户偏好/规律，输出 JSON 数组，"
    "每项：{{\"entity\": 主语, \"relation\": 关系, \"value\": 内容, \"confidence\": 0~1}}。"
    "键约束：只有用户本人的偏好/习惯才允许 entity=user 且 relation=prefers；"
    "任务经验、技术结论、工具用法请用恰当的实体名（工具名/任务名）和关系词"
    "（如 lesson/practice），不要挂到 user/prefers 上。只输出 JSON。\n{items}"
)

# P2-1 图式归纳提示词：输入是本夜 NREM 聚出的「同类经历」组（跨时间的相似情景），
# 要求产出一条跨事件的一般性规律（relation 固定 "rule"，不由模型决定）。与
# CLUSTER_PROMPT 的簇摘要（面向情景压缩）、DISTILL_PROMPT 的扁平三元组（面向单簇
# 蒸馏）并列为本模块第三种 LLM 归纳形态——本提示词是 rule 的单一来源（单一来源
# 惯例同上）。置信度上限 0.7 与 RULE_MAX_CONFIDENCE 一致：归纳是猜测不是观测。
RULE_INDUCE_PROMPT = (
    "下面是发生在不同时间的若干条相似经历（同一主题的多次事件）。请跨事件归纳出"
    "一条可复用的一般性规律——不是任何单条经历的复述，而是它们共同说明的道理。"
    "输出一个 JSON 对象：{{\"entity\": 任务域实体名（工具名/任务名/领域名，"
    "禁止用 user）, \"value\": 一句跨事件规律（≤50字）, \"confidence\": 0~0.7}}。"
    "只输出 JSON。\n{items}"
)


def summarize_cluster(cluster: list[EpisodicMemory]) -> str:
    if not llm.maintenance_available():
        return cluster[0].summary
    items = "\n".join(f"- {e.summary}" for e in cluster)
    return llm.maintenance_chat(CLUSTER_PROMPT.format(items=items)) or ""


def parse_fact_json(raw: str) -> list[dict]:
    try:
        start, end = raw.find("["), raw.rfind("]")
        return json.loads(raw[start:end + 1])
    except (ValueError, IndexError):
        return []


def _parse_rule(raw) -> dict | None:
    """解析并校验图式归纳（rule）的 LLM 输出（P2-1 断路器 c 的格式闸）。

    合法输出是单个 JSON 对象 {"entity", "value", "confidence"}；relation 固定
    "rule"，不由模型决定（模型只给实体、规律与自信度）。校验三关：
    - JSON 合法：裸对象，或容忍包在数组/赘语里的单个对象（取第一个 { 到最后一个
      } 之间解析）；解析失败 / 不是对象 -> None；
    - entity 非空且不为 "user"（大小写不敏感）：V1.6.2 的蒸馏键约束同样适用于
      rule——用户偏好不挂规律，规律的实体是任务域（工具名/任务名）；
    - value 非空；confidence 缺省 0.5、非数值视为格式损坏，合法值统一钳到
      [0, RULE_MAX_CONFIDENCE]（断路器 d：归纳的自信不许超过观测的自信）。
    任一不满足返回 None，调用方只计 rule_candidates 不写库（错归纳不如不归纳）。
    """
    if not raw or not isinstance(raw, str):
        return None
    try:
        start, end = raw.find("{"), raw.rfind("}")
        obj = json.loads(raw[start:end + 1]) if start >= 0 < end else None
    except (ValueError, IndexError):
        obj = None
    if not isinstance(obj, dict):
        return None
    entity = str(obj.get("entity", "")).strip()
    value = str(obj.get("value", "")).strip()
    if not entity or entity.lower() == "user" or not value:
        return None
    try:
        confidence = float(obj.get("confidence", 0.5))
    except (TypeError, ValueError):
        return None
    confidence = max(0.0, min(settings.RULE_MAX_CONFIDENCE, confidence))
    return {"entity": entity, "value": value, "confidence": confidence}
