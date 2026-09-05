"""情景编码：事件 -> 情景记忆（摘要 + 上下文 + 结果）。"""
from __future__ import annotations

from memagent.adapters import llm
from memagent.core.clock import now_iso, now_ts
from memagent.core.domain import EpisodicMemory, Event, GREEN_TYPES
from memagent.encoding.prompts import SUMMARY_PROMPT
from memagent.learning.strength import compute_strength


def encode_episodic(event: Event) -> EpisodicMemory:
    summary = event.content
    # E8 效率红线：绿色类型跳过 LLM 摘要——经验条目本就是凝练的（AI 自己写的
    # 教训），不值得为 40 字摘要等一次可能 60s 的云端调用；原文即最佳摘要
    if llm.llm_available() and event.type not in GREEN_TYPES:
        s = llm.chat(SUMMARY_PROMPT.format(content=event.content))
        if s:
            summary = s
    # 初始强度：重要度为主 + 新近度满格（无访问记录）；事件无独立置信度，以重要度代理
    # E2：唤醒度进强度分项（写入时即生效，情感事件先天更强）；arousal 持久化供重算半衰期延长
    # E8：绿色类型落经验层标记——LFU 短半衰与分层归档阈值按 category 生效
    category = "experience" if event.type in GREEN_TYPES else ""
    strength = compute_strength(event.importance, event.importance, 0, now_ts(),
                                arousal=event.arousal, category=category)
    return EpisodicMemory(
        summary=summary,
        context=event.task_context,
        action="",
        outcome=event.outcome,
        importance=event.importance,
        created_at=now_iso(),
        strength=strength.score,
        status="active",
        embedding=llm.embed(event.content),
        arousal=event.arousal,
        category=category,
        source=event.source,   # P0-4：来源透传（谁说的），落库防复发；检索/注入暂不消费（P1-5 启用）
    )
