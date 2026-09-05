"""轮次 → 事件映射（V1.7 P2 写侧）：一轮对话里什么值得进记忆、进哪一层。

## 写入策略：静默优先

对话是低信息密度的——「嗯」「好的」「继续」占了真实对话的大半。默认只进工作记忆
（L0，纯内存、会话结束即蒸发）；只有检测到**信号**才走写入管线进长期记忆。

信号四类（与 PLAN-V1.7 §Phase 2 一致，全部规则判定、离线可测）：

| 信号 | 侧 | 落入类型 | 通道 |
|---|---|---|---|
| 用户纠正 | 用户 | instruction | 显式声明保底（EXPLICIT_TYPES）+ L2 信念降权（corrects=True 显式传参，P2-3） |
| 明确偏好 | 用户 | preference_statement | 显式声明保底 |
| 身份陈述 | 用户 | identity_statement | 显式声明保底 |
| 显式指令 | 用户 | instruction | 显式声明保底 |
| 环境状态 | 助手 | env_statement | E8 绿色通道（最新即正确） |
| 模型点名 | 助手 | 调用方给定 | remember()，只收 GREEN_TYPES |

映射成**已存在的类型**而不是新造类型，是为了让既有机制（门控保底、冲突消解、
E8 绿色通道、LFU 归档）原样接管：写入的唯一入口仍是 `pipeline.ingest_event`，
本模块不做任何编码，也不绕过门控——它只决定「这一轮要不要去敲那扇门」。

「结果成败」词表已移除（P0-1，2026-09）：词表命中≠点名——助手回答里**引用**
用户的投诉词「备份失败」或自述「搞定了」，都会把整段汇报当成 AI 自己的工作经验
入库（实测三层污染事故：情景/语义/技能全中，技能名甚至是回答前 12 个字符的
截断）。experience 是 E8 绿色类型（宽进严出），词表一旦命中就没有任何闸门能
拦它，所以这个口子必须在信号层关死。experience 只认三条点名通路：
`memory_remember` 工具 / `add --type experience` / `remember()` 显式传 outcome；
成败的语义判定将来由结构化工具结果（exit_code）承担，不由散文正则承担。

## 自增强防护：复述不沉淀

注入 → 模型复述 → 复述成为新事件 → 与原文相似 → 下次两者都命中 → 计数都涨。
这条回路的第一道闸在 `is_restatement`：助手文本若只是在复述刚注入的内容，
**任何地方都不写**（连工作记忆也不进——那条记忆本来就在库里，再存一份纯属
自我复制）。模型真想留痕，走 `remember()` 明说，而不是靠复述刷存在感。
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from memagent import settings
from memagent.adapters import llm
from memagent.attention import attention_gate
from memagent.attention.emotion import analyze_emotion
from memagent.attention.scorers import IDENTITY_PATTERN
from memagent.core import Event
from memagent.pipeline import ingest_event
from memagent.storage import SqliteStore

# ---------- 信号词表（规则判定，离线确定；与 attention.scorers 同源词汇，
# 但语义不同：那边是「值多少分」，这边是「要不要去敲长期记忆的门」）----------
_QUESTION_RE = re.compile(r"[?？]|什么|怎么|为什么|如何|哪[个里些种]|吗|呢")
_CORRECTION_RE = re.compile(
    r"不对|错了|说错了|写错了|记错了|搞错了|弄错了|不是这样|我说的是|应该是|其实是|纠正一下|纠正|别再")
_INSTRUCTION_RE = re.compile(
    r"记住|请记住|请记|帮我记|记到记忆|记进记忆|记入记忆|以后|今后|一直|永远|总是|记得|别再")
# 「把这条经验记到记忆里」是指令但旧词表不覆盖（无「记住」字样，V1.7.3 多轮实测：
# 显式写入请求只靠模型调 memory_remember 兜住，recorder 漏检）。补「记到/进/入记忆」
# 与「帮我记/请记」等指向 Agent 的窄形态；「记一下」「记下来」主语不明（自述动作
# 「我先记下来」会误触发），不加。
# 回忆请求不是指令：「你记得什么」「你还记得 X 吗」是在对系统说话——索取或核实
# 已存的记忆，不是在写新记忆。不豁免的话，「记得」撞上指令词表，问句本体会被
# 存成偏好/指令（实测翻车两次：先是一句「你还记得我偏好什么格式吗」顶掉真偏好，
# 后是「你记得什么」因不以吗/？结尾漏豁免又入库）。凡第二人称 + 记得/知道一律
# 视为回忆请求；祈使句（「记得明天提醒我」）无第二人称前缀，不受影响（宁窄勿宽
# 的反面在这里不适用：这类句式没有值得记的写入语义，全豁免是安全的）。
_RECALL_REQUEST_RE = re.compile(r"(?:你|您)(?:还)?(?:记得|知道)")
# 社交表达不是偏好：「我喜欢你」是对助手的**关系表达**，不是可执行的用户偏好——
# 入库会成为永远不会被"执行"的伪偏好，还会在寒暄轮被召回、甚至在 B3 门槛下当
# 联想锚点撑开噪声注入（V1.7.2 实测链路）。只抑制宾语为第二人称**且到句尾**的
# 形态：「我不喜欢你打断我」宾语后有实义内容，仍是偏好/反馈，不抑制（宁窄勿宽）。
_SOCIAL_PREFERENCE_RE = re.compile(
    r"我(?:其实|可能|偶尔|私下)?(?:喜欢|讨厌|爱)(?:上)?(?:你|您|你们|妳)(?:们)?\s*[!！。．~～]?\s*$")
# 让 Agent 查记忆库的请求不是偏好声明：「帮我查一下记忆库里关于回答格式的偏好」
# 是在对记忆系统说话（索取已存内容），不是在声明「我偏好 X」——句中「偏好」撞上
# 词表、又常不带疑问标记（无吗/？），会把查询请求原句存成偏好（V1.7.3 工具上线
# 实测：真机 smoke 一轮即翻车，且工具场景下这类请求只会更多）。只豁免「查询动词
# + 记忆」的紧邻形态，宁窄勿宽：「我偏好先查一下资料再回答」宾语不是记忆库，
# 不受豁免。
_MEMORY_QUERY_RE = re.compile(
    r"(?:查一下|查查|搜一下|搜搜|检索|搜[索找])[^\n。！？]{0,12}记忆|记忆[^\n。！？]{0,6}(?:查一下|查查|搜一下|检索)")
_PREFERENCE_RE = re.compile(
    r"我(?:其实|可能|偶尔|私下)?(?:喜欢|希望|偏好|更倾向|更喜欢|讨厌|不喜欢|不想要|想要|习惯|一般用|通常用)"
    r"|(?:改成|换成|改回|还是|保持|一律|以后都|都要用)|偏好")
# 环境状态：ASCII 工具名（中文语境里工具名几乎总是原样英文）+ 状态谓语。
# 与 encoding/semantic_extractor._TOOL_WORD 同口径——那个是取值侧，这个是判定侧
# 工具名至少 3 字符（{2,}）：2 字母的 "vs" 会在「A vs B」句式里被当工具名，
# 12 字符窗口内撞上「版本」等谓语就误发环境状态信号（V1.7.3 多轮实测翻车：
# 助手的冲突清单回答「旧表述 vs 新表述」整段被存成 [llm] env_state 事实）。
# 漏掉的极短工具名（go/R）本就该走观察流，宁窄勿宽。
_ENV_STATE_RE = re.compile(
    r"[A-Za-z][A-Za-z0-9_.\-]{2,}[^\n。！？]{0,12}?"
    r"(?:版本|已安装|不可用|已升级|已废弃|已切换|安装在|路径是|已更新)")


@dataclass(frozen=True)
class Signal:
    """一个写入信号：kind 用于审计与调试，type/outcome 是落进 Event 的字段。"""

    kind: str
    type: str
    outcome: str = ""


def detect_signals(text: str, source: str = "user") -> list[Signal]:
    """按优先级返回本轮命中的信号（第一个即采纳的那个），无信号返回空列表。

    疑问句抑制偏好/身份判定：「我偏好什么回答格式？」是在**问**，不是在声明偏好——
    把它当偏好写进去会得到一条 [user] prefers「我偏好什么回答格式？」的垃圾事实。
    指令类信号一般不受抑制（「请记住：以后别问这个？」依然是要求），但**回忆请求**
    除外：「你还记得 X 吗」是索取已存记忆不是写入（_RECALL_REQUEST_RE，人工验收
    实测：不豁免会问句取代真偏好）。**社交表达**同样豁免偏好：「我喜欢你！」是
    对助手的关系表达（_SOCIAL_PREFERENCE_RE），入库即伪偏好。**查库请求**亦然：
    「帮我查一下记忆里的偏好」是对记忆系统说话（_MEMORY_QUERY_RE，V1.7.3 工具
    场景实测翻车），句中的「偏好」只是查询对象，不是声明。
    """
    if not text or not text.strip():
        return []
    if source == "assistant":
        # 助手侧只认环境状态（P0-1 后成败词表已移除）：助手回答里的「失败/成功」
        # 字样是引用或汇报，不是点名——experience 只走三条点名通路（见文件头）。
        signals = []
        if _ENV_STATE_RE.search(text):
            signals.append(Signal("env_state", "env_statement"))
        return signals

    signals: list[Signal] = []
    if IDENTITY_PATTERN.search(text) and not _QUESTION_RE.search(text):
        signals.append(Signal("identity", "identity_statement"))
    if _INSTRUCTION_RE.search(text) and not _RECALL_REQUEST_RE.search(text):
        signals.append(Signal("instruction", "instruction"))
    if _CORRECTION_RE.search(text):
        # 纠正 = 对未来行为的指令（否则何必纠正），走显式声明保底，确保一定进长期
        signals.append(Signal("correction", "instruction"))
    if (_PREFERENCE_RE.search(text) and not _QUESTION_RE.search(text)
            and not _SOCIAL_PREFERENCE_RE.search(text)
            and not _MEMORY_QUERY_RE.search(text)):
        signals.append(Signal("preference", "preference_statement"))
    return signals


# ---------- 复述识别 ----------
_NON_TEXT_RE = re.compile(r"[^0-9A-Za-z\u4e00-\u9fff]")


def _normalize(text: str) -> str:
    """归一化：只留中日韩/字母数字字符并小写——标点与空格不承载「是否复述」的信息。"""
    return _NON_TEXT_RE.sub("", text or "").lower()


def _grams(text: str, n: int = 3) -> set[str]:
    """字符 n-gram 集合（n=3：逐字复述与换措辞重述在 3-gram 上才拉得开，见 settings）。"""
    t = _normalize(text)
    if not t:
        return set()
    if len(t) < n:
        return {t}
    return {t[i:i + n] for i in range(len(t) - n + 1)}


def restatement_score(text: str, injected_texts) -> float:
    """text 被注入文本覆盖的比例（字符 3-gram 容器率，分母是 text 自己）。

    取容器率而非 Jaccard：长答案里引用一句注入内容不应被判成复述（容器率的
    分母随答案变长而变大，天然抗「引用」）；只有**整段答案都由注入内容拼成**
    时才会接近 1。
    """
    grams = _grams(text)
    if not grams:
        return 0.0
    union: set[str] = set()
    for t in injected_texts or ():
        union |= _grams(t)
    if not union:
        return 0.0
    return len(grams & union) / len(grams)


def is_restatement(text: str, injected_texts) -> bool:
    """是否只是在复述刚注入的内容。

    两条判据满足其一即可：
    - 整句包含：归一化后一方是另一方的子串（双方都 ≥ AGENT_RESTATE_MIN_CHARS）；
      模型逐字复读注入句是真实存在的行为，这一档抓得最准；
    - 覆盖率：字符 3-gram 容器率 ≥ AGENT_RESTATE_CONTAIN（换措辞的整体复述）。
    """
    norm = _normalize(text)
    if len(norm) >= settings.AGENT_RESTATE_MIN_CHARS:
        for t in injected_texts or ():
            other = _normalize(t)
            if (len(other) >= settings.AGENT_RESTATE_MIN_CHARS
                    and (norm in other or other in norm)):
                return True
    return restatement_score(text, injected_texts) >= settings.AGENT_RESTATE_CONTAIN


# ---------- 轮次录入 ----------
def _to_working(store: SqliteStore, text: str, source: str) -> dict:
    """静默轮次的去向：只进会话工作记忆（L0），不过写入管线。

    规则打分（不调 LLM）只为填 working 的 salience 三要素——静默轮次是每轮都
    发生的路径，绝不能为它等一次可能 60s 的云端打分；情感同样走纯规则，与
    E8 绿色通道的效率红线同口径。
    """
    event = attention_gate(Event(content=text, source=source, type="observation"),
                           use_llm=False)
    event.valence, event.arousal = analyze_emotion(text, use_llm=False)
    store.working.add(text, importance=event.importance, arousal=event.arousal,
                      embedding=llm.embed(text))
    return {"action": "working_only", "signals": [], "importance": event.importance}


def _record_side(store: SqliteStore, text: str, source: str, injected_texts,
                 task_context: str, outcome: str, use_llm: bool,
                 own_recent=()) -> dict:
    """一侧（用户 / 助手）文本的录入决策。"""
    if not text or not text.strip():
        return {"action": "skipped", "reason": "empty", "signals": []}
    # 复述守卫只作用于助手侧。自增强回路是「注入 -> 模型复述 -> 复述入库 ->
    # 与原文相似 -> 下次两者都命中 -> 计数都涨」，环上只有模型自己的输出；
    # 用户是人、是记忆的源头——重复之前的问题、引用记忆原文都是真实输入，
    # 照常走信号判定（重复的显式声明会同键续证/取代，语义本来就对）。
    # 实测翻车：用户复问「你能忍受低效信息噪声吗？……请你分析下」因包含
    # 上一轮的原句被整条 restatement_skipped，追加的新内容随之丢失。
    # 助手侧的复述参照系 = 本轮注入 + 自己最近说过的几句话（own_recent）：
    # 只看注入会在多轮闲聊把注入锚点挤出工作记忆后漏判（实测：锚点被挤出
    # 后「已经完成，X」逃逸成经验写入），模型逐字复读自己刚说过的话同样是
    # 自我复制，不该重复沉淀。
    if (source == "assistant"
            and is_restatement(text, list(injected_texts) + list(own_recent))):
        store.log("agent", 0, "restatement_skipped", f"{source}: {text[:60]}")
        return {"action": "restatement_skipped", "signals": [],
                "score": round(restatement_score(text, injected_texts), 4)}

    signals = detect_signals(text, source)
    if not signals:
        return _to_working(store, text, source)

    sig = signals[0]
    result = ingest_event(store, text, source=source, type=sig.type,
                          task_context=task_context, outcome=outcome or sig.outcome,
                          use_llm=use_llm,
                          # P2-3 L2：纠正信号显式传参下发（pipeline 是 L4、本模块是 L5，
                          # 不许 pipeline import recorder——信号只经参数走）。指令/偏好/
                          # 身份/环境等其他信号不传（默认 False = 现状逐位不变），
                          # 纠正在 L1 显式声明通路之外只多一步 top-1 信念降权。
                          corrects=(sig.kind == "correction"))
    return {
        "action": "gated" if result["gated"] else "ingested",
        "signals": [s.kind for s in signals],
        "type": sig.type,
        "outcome": outcome or sig.outcome,
        "importance": result["importance"],
        "result": result,
    }


def record_turn(store: SqliteStore, user_text: str = "", assistant_text: str = "",
                injected_texts=(), task_context: str = "", outcome: str = "",
                use_llm: bool = False, own_recent=()) -> dict:
    """一轮对话的录入：两侧各自判定，返回结构化结果供 CLI 展示与评测断言。

    injected_texts 由 injector 给出（本轮真正进到模型眼前的记忆原文）。
    outcome 是调用方显式指定的结果（成功/失败），缺省时为空——成败词表已移除
    （P0-1），不再由助手侧散文判定，显式 outcome 是 experience 沉淀技能的前提。
    own_recent 是助手自己最近说过的几句话（loop 传 history 尾部）：助手侧复述
    识别的参照系 = 注入内容 + own_recent，用户侧不使用（用户不复述拦截）。
    """
    rec = {
        "user": _record_side(store, user_text, "user", injected_texts,
                             task_context, "", use_llm),
        "assistant": _record_side(store, assistant_text, "assistant", injected_texts,
                                  task_context, outcome, use_llm, own_recent=own_recent),
    }
    rec["restatement_skipped"] = sum(
        1 for side in ("user", "assistant")
        if rec[side]["action"] == "restatement_skipped")
    rec["signals"] = rec["user"].get("signals", []) + rec["assistant"].get("signals", [])
    store.log("agent", 0, "turn_record",
              f"user={rec['user']['action']} assistant={rec['assistant']['action']}")
    return rec
