"""模型主动写入入口（V1.7 P2）：remember()。

「模型主动点名」是五类写入信号里唯一由模型发起的一路。设计与 PLAN-V1.7 一致：

- **只接受 GREEN_TYPES**（experience / env_statement）。其他 type 降级为普通写入
  走门控——不给保底、不进绿色通道，能不能入库由门控说了算。
- **滥用不是安全问题**（模型是自己人），不新增任何防护机制：兜底在遗忘侧，
  E8 的「宽进严出」（经验层半衰 7 天 + 归档阈值 0.35）保证从没被命中的经验
  两周沉底。
- **复述拦截**：模型若把刚注入的内容原样 remember 一遍，直接拒绝并记审计。
  这是「注入 → 复述 → 新事件 → 与原文相似 → 计数都涨」这条回路的最后一道闸
  （前两道是静默优先与 recorder 的复述识别）。

source 用 "tool"：这次写入的发起方是模型调用的工具通道，不是用户、也不是
观察流——审计里要能区分。
"""
from __future__ import annotations

from memagent.agent.recorder import is_restatement
from memagent.core.domain import GREEN_TYPES
from memagent.pipeline import ingest_event
from memagent.storage import SqliteStore


def remember(store: SqliteStore, content: str, type: str = "experience",
             context: str = "", outcome: str = "", injected_texts=(),
             use_llm: bool = False) -> dict:
    """模型点名写入，返回 ingest_event 的结果（外加 degraded / rejected 标记）。

    返回结构在 pipeline 结果之上多两个键：
      - degraded=True：type 不在白名单，已降级为普通写入走门控；
      - rejected=True：内容是刚注入记忆的复述，已拒绝写入。
    """
    if is_restatement(content, injected_texts):
        store.log("agent", 0, "remember_rejected", f"复述注入内容: {content[:60]}")
        return {"rejected": True, "degraded": False, "gated": True,
                "reason": "restatement", "importance": 0.0, "episodic_id": 0,
                "facts": [], "skipped_facts": [], "skills": []}

    degraded = type not in GREEN_TYPES
    if degraded:
        store.log("agent", 0, "remember_degraded",
                  f"type={type!r} 不在绿色通道白名单，降级走门控")
    result = ingest_event(store, content, source="tool", type=type,
                          task_context=context, outcome=outcome, use_llm=use_llm)
    result["degraded"] = degraded
    result["rejected"] = False
    return result
