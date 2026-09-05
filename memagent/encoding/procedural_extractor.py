"""程序抽取：LLM 识别技能模板；无模型时略过（v1 不强行规则）。

E8 绿色通道两处变化：
- 绿色类型（experience/env_statement）走 llm.local_chat 本地快车道——云端
  （尤其经代理黑洞时）一次 60s 超时就够毁掉整个写入体验，技能抽取不值得等；
- 绿色类型且带 outcome 时加规则兜底：本地 LLM 也不可用时，用 task_context 当
  trigger、内容本身当 policy 构造技能——确定性键（name=任务域）保证同域经验
  复用同一技能条目，下次带新 policy 自然走 update_policy 迭代。
  任务域必带（P0-2）：空任务域不产技能。
"""
from __future__ import annotations

import json

from memagent.adapters import llm
from memagent.core.domain import Event, ProceduralSkill, GREEN_TYPES
from memagent.encoding.prompts import SKILL_EXTRACT_PROMPT


def _parse_skill(raw: str) -> dict | None:
    try:
        obj = json.loads(raw[raw.find("{"): raw.rfind("}") + 1])
        return obj if isinstance(obj, dict) else None
    except (ValueError, IndexError):
        return None


def _rule_skill(event: Event) -> ProceduralSkill | None:
    """离线规则兜底：仅绿色类型且带 outcome（成功/失败是技能沉淀的语义前提）。

    任务域必带（P0-2）：空任务域时宁可不产技能。旧实现拿内容前 12 字截断当
    name、整段回答当 policy、trigger 留空——name 是半句话不可读；trigger 为空
    没有触发语义（无「事前触发相关性」，本就不该存在）；policy 是整段回答、
    bigram 覆盖面大反而更易被任意查询带出（实测污染：库中 7 个此类垃圾技能，
    其一被带出 22 次形成 touch_usage 自增强）。显式任务域同时是经验确定性键
    （任务域, lesson）的来源：没有它就没有「同域经验迭代同一技能」的语义。
    """
    if event.type not in GREEN_TYPES or not event.outcome:
        return None
    if not event.task_context.strip():  # P0-2：空任务域拒绝，不产垃圾技能
        return None
    name = event.task_context.strip()
    return ProceduralSkill(
        name=name,
        trigger=event.task_context.strip(),
        policy=event.content.strip(),
        success_count=1 if event.outcome == "success" else 0,
        usage_count=1,
        success_rate=1.0 if event.outcome == "success" else 0.0,
    )


def extract_skills(event: Event) -> list[ProceduralSkill]:
    chat_fn = llm.local_chat if event.type in GREEN_TYPES else llm.chat
    raw = chat_fn(SKILL_EXTRACT_PROMPT.format(content=event.content),
                  validate=lambda s: _parse_skill(s) is not None or s.strip() == "null")
    if not raw:
        skill = _rule_skill(event)
        return [skill] if skill else []
    obj = _parse_skill(raw)
    if not obj or not obj.get("name"):
        return []
    return [ProceduralSkill(
        name=str(obj["name"]).strip(),
        trigger=str(obj.get("trigger", "")).strip(),
        policy=str(obj.get("policy", "")).strip(),
        success_count=1 if event.outcome == "success" else 0,
        usage_count=1,
        success_rate=1.0 if event.outcome == "success" else 0.0,
    )]
