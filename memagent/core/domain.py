"""数据类定义：事件、情景记忆、语义记忆、程序记忆、技能。"""
from __future__ import annotations

from dataclasses import dataclass, field

# 用户显式声明类型：门控保底 WRITE_THRESHOLD（门控过滤观察流，不过滤用户点名的），
# 语义事实按 explicit 源落库（同键新值自动取代旧版）。
EXPLICIT_TYPES = ("instruction", "preference_statement", "identity_statement")

# E8 工作经验绿色通道类型：AI/Agent 点名沉淀的自身经验，与显式声明同级信任——
# 门控保底 + explicit 级取代（高度迭代：新版经验直接接管旧版）。
#   experience    AI 工作经验/技巧（事实键 = 任务域 lesson，来自 task_context）
#   env_statement 环境状态声明（事实键 = 工具名 env_state，"最新即正确"）
GREEN_TYPES = ("experience", "env_statement")


@dataclass
class Event:
    """感知层标准化事件。

    E2 情感加权：emotion 为 valence 的符号化（positive/negative/neutral），
    valence/arousal 由 attention.emotion 双模打分填充（pipeline 门控后统一调用）。
    """
    content: str
    source: str = "user"                 # user | tool | system | feedback
    type: str = "observation"            # preference_statement | instruction | experience | ...
    task_context: str = ""
    entities: list[str] = field(default_factory=list)
    emotion: str = "neutral"             # positive | negative | neutral
    outcome: str = ""                    # success | failure | ""
    importance: float = 0.0              # 门控打分结果
    valence: float = 0.0                 # 情感效价 -1~1（E2）
    arousal: float = 0.0                 # 情感唤醒度 0~1（E2，进强度模型）


@dataclass
class EpisodicMemory:
    """情景记忆：在什么时间、什么场景、发生了什么、结果如何。

    is_summary 标记由摘要降级生成的概括记忆，source_ids 溯源到被压缩的原始记忆，
    summarized_by 是原始记忆反向指向摘要的链接。
    """
    id: int = 0
    summary: str = ""
    context: str = ""
    action: str = ""
    outcome: str = ""
    importance: float = 0.0
    created_at: str = ""                 # ISO 时间
    access_count: int = 0
    last_access_at: str = ""
    strength: float = 0.0
    status: str = "active"               # active | archived | summarized | deleted
    embedding: list[float] = field(default_factory=list)
    source_ids: list[int] = field(default_factory=list)
    summarized_by: int = 0
    is_summary: bool = False
    arousal: float = 0.0                 # 情感唤醒度 0~1（E2：重算半衰期延长需要持久化）
    category: str = ""                   # E8 记忆分层："" 普通 | "experience" 经验层
                                         # （LFU 短半衰 + 分层归档阈值消费它）
    source: str = ""                     # 来源（P0-4）：user|assistant|system|tool|feedback，
                                         # 编码时从 Event.source 透传；历史行为 'unknown'。
                                         # 注入侧已消费（P1-5）：assistant 来源的
                                         # 经验层情景不自动注入，防助手自引用


@dataclass
class SemanticFact:
    """语义记忆：抽象事实 (entity, relation, value)。

    raw_* 保留归一化前的原始表达；superseded_by 指向取代本条的新版本；
    evidence_count 是同值重复观测次数；source_event_ids 溯源到触发事件；
    hit_count 是试用期（status=pending）被检索命中的累计次数（P1 转正通道的度量，
    与 evidence_count 严格分开——检索是「被想起」不是「被观测」，见 E6 契约）。
    """
    id: int = 0
    entity: str = ""
    relation: str = ""
    value: str = ""
    confidence: float = 0.0
    valid_from: str = ""
    valid_to: str = ""
    conflict_note: str = ""
    embedding: list[float] = field(default_factory=list)
    status: str = "active"               # active | pending | superseded | archived
    raw_entity: str = ""
    raw_value: str = ""
    superseded_by: int = 0
    evidence_count: int = 1
    source_event_ids: list[int] = field(default_factory=list)
    hit_count: int = 0
    validity_context: str = ""           # 条件上下文（P2-2）：JSON 字符串，规范键
                                         # {"preconditions": [str...], "environment": [str...],
                                         # "expires_if": str}；空 = 无条件限定，永久有效。
                                         # 只由 LLM 主抽取通道产出（过 normalize_validity_context
                                         # 白名单才允许入库），显式声明/离线规则/蒸馏一律空——
                                         # 用户原话没有的上下文不虚构（宁缺勿造）。
                                         # 注入过滤暂缓（既有决策）：本任务只做存储与展示。


@dataclass
class ProceduralSkill:
    """程序记忆：如何做某事。"""
    id: int = 0
    name: str = ""
    trigger: str = ""
    policy: str = ""
    success_count: int = 0
    failure_count: int = 0
    usage_count: int = 0
    success_rate: float = 1.0
    status: str = "active"


@dataclass
class RetrievalHit:
    """一次检索命中。"""
    kind: str = "episodic"               # episodic | semantic | procedural | working（E3 会话级置顶，易失）
    id: int = 0
    text: str = ""
    score: float = 0.0
    strength: float = 0.0
    meta: dict = field(default_factory=dict)
