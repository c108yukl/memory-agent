"""Agent 工具调用（V1.7.3）：记忆工具集 + 文本协议解析 + 执行器。

设计立场与 PLAN-V1.7 §0 一脉相承：自动注入仍是主读路径，工具是**补丁**不是
替代——注入漏了时模型可以主动点名查（memory_search），想留痕时点名写
（memory_remember，复用 V1.7 P2 的 remember()），维护类操作（睡眠/遗忘）向
模型开放但仅限用户要求时。全部工具都是既有稳定函数的薄封装，本模块不含任何
记忆机制；写入仍走 pipeline 唯一入口，复述拦截与绿色通道白名单原样生效。

协议选文本而非原生 function calling（用户定的方向：先注重实用和稳定）：
适配层 chat/chat_stream 只收发纯文本，免费网关与本地模型对原生 tools 参数的
支持参差不齐；文本协议对任何后端行为一致、零网络层改动、离线可测。解析器对
三种形态容错（规范标签 / ```tool_call 围栏 / 裸 JSON 对象），坏 JSON 不炸循环，
错误信息回填给模型让它自行纠正。

裁决权边界：冲突消解（resolve）不对模型开放——A 类冲突取代旧事实只能由
人工/证据裁决（D1），模型只有查看权（memory_conflicts）。
"""
from __future__ import annotations

import ast
import json
import re
import shlex
import subprocess
from dataclasses import dataclass, field

from memagent import settings
from memagent.adapters import llm
from memagent.consolidation import consolidate
from memagent.core.vectors import cosine
from memagent.forgetting import run_forgetting
from memagent.reports import build_health_report
from memagent.retrieval import build_context, retrieve
from memagent.retrieval.deep import deep_retrieve
from memagent.storage import SqliteStore

# ---------- 协议常量 ----------
TOOL_CALL_OPEN = "<tool_call>"
TOOL_CALL_CLOSE = "</tool_call>"

# 规范形态：<tool_call>…</tool_call>（容忍大小写与标签内空白）；围栏形态是
# 部分模型对「代码块」习惯的妥协——```tool_call … ``` 也接受。
_TAG_RE = re.compile(r"<\s*tool_call\s*>(.*?)<\s*/\s*tool_call\s*>", re.S | re.I)
_FENCE_RE = re.compile(r"```(?:tool_call|tool)\s*\n?(.*?)```", re.S | re.I)
# 清理最终回答里的残余标记（无论成对与否）：模型偶尔只写开标签不写闭标签
_TAG_STRIP_RE = re.compile(
    r"<\s*tool_call\s*>.*?(?:<\s*/\s*tool_call\s*>|$)|<\s*/\s*tool_call\s*>",
    re.S | re.I)
_FENCE_STRIP_RE = re.compile(r"```(?:tool_call|tool)\s*\n?.*?(?:```|$)", re.S | re.I)


@dataclass
class ToolCall:
    """解析出的一次调用意图：name 必填，args 缺省空 dict。"""

    name: str
    args: dict = field(default_factory=dict)


@dataclass
class ToolContext:
    """工具执行的上下文：store 必带；injected_texts 供 remember 的复述拦截用。"""

    store: SqliteStore
    injected_texts: tuple[str, ...] = ()


@dataclass
class ExecResult:
    """一次执行的结果：text 是回填给模型的文本（错误也用文本回填，不抛出）。"""

    ok: bool
    text: str


# ---------- 解析 ----------
def _extract_json_obj(block: str) -> dict:
    """从文本块中抠出第一个 JSON 对象：取首 { 到末 } 的子串解析。

    模型常在 JSON 前后加说明文字或围栏标记，首尾定位比整段解析稳；
    解析失败抛 ValueError，由 parse_tool_call 转成回填给模型的错误信息。
    json.loads 失败时回落 ast.literal_eval——小模型常写单引号 JSON
    （Python 字面量语法），literal_eval 只接受字面量、无代码执行风险。
    """
    start, end = block.find("{"), block.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("块内没有 JSON 对象")
    raw = block[start:end + 1]
    try:
        obj = json.loads(raw)
    except ValueError:
        try:
            obj = ast.literal_eval(raw)
        except (ValueError, SyntaxError, MemoryError, RecursionError):
            raise ValueError("JSON 解析失败（检查引号与逗号）")
    if not isinstance(obj, dict):
        raise ValueError("JSON 不是对象")
    return obj


def _normalize_call(obj: dict) -> ToolCall:
    """归一化模型的三种命名习惯：name/tool/tool_name；args/arguments/平铺参数。

    平铺形态 {"name": "memory_search", "query": "…"} 也接受（小模型常忘记包
    args 一层）：除 name/tool/tool_name 外的键全部当作参数。
    """
    name = obj.get("name") or obj.get("tool") or obj.get("tool_name")
    if not isinstance(name, str) or not name.strip():
        raise ValueError("缺少工具名 name")
    raw_args = obj.get("args", obj.get("arguments"))
    if raw_args is None:
        args = {k: v for k, v in obj.items()
                if k not in ("name", "tool", "tool_name")}
    elif isinstance(raw_args, dict):
        args = raw_args
    elif isinstance(raw_args, str) and raw_args.strip():
        parsed = _extract_json_obj(raw_args)
        args = parsed
    else:
        args = {}
    return ToolCall(name=name.strip(), args=args if isinstance(args, dict) else {})


def parse_tool_call(text: str) -> tuple[ToolCall | None, str | None]:
    """从模型输出解析工具调用，三态返回：

    - (ToolCall, None)：解析成功；
    - (None, None)：本轮没有工具调用（这就是最终回答）；
    - (None, 错误信息)：看起来想调用但格式无效——错误信息回填给模型自行纠正。

    容错顺序：规范标签 → ```tool_call 围栏 → 裸 JSON 对象（整段就是一个
    {"name": …} 且工具名可识别）。裸 JSON 要求工具名在注册表内才认，
    避免把恰好是 JSON 的正文误判成调用。
    """
    if not text:
        return None, None
    m = _TAG_RE.search(text) or _FENCE_RE.search(text)
    if m:
        try:
            return _normalize_call(_extract_json_obj(m.group(1))), None
        except ValueError as e:
            snippet = m.group(1).strip()[:120]
            return None, f"{e}（块内容: {snippet}）"
    stripped = text.strip()
    if stripped.startswith("{") and stripped.endswith("}"):
        try:
            obj = _extract_json_obj(stripped)
            name = obj.get("name") or obj.get("tool") or obj.get("tool_name")
            if isinstance(name, str) and name.strip() in TOOL_REGISTRY:
                return _normalize_call(obj), None
            if name:
                return None, f"未知工具 {name!r}"
        except ValueError as e:
            return None, f"{e}（原文: {stripped[:120]}）"
    return None, None


def strip_tool_call(text: str) -> str:
    """把最终回答里的残余工具标记清掉（含只写了一半的），正文原样保留。"""
    if not text:
        return text
    cleaned = _FENCE_STRIP_RE.sub("", _TAG_STRIP_RE.sub("", text))
    return cleaned.strip()


# ---------- 工具实现（每个都是既有函数的薄封装）----------
class ToolError(Exception):
    """工具参数/目标错误：执行未发生，execute_tool 以 ok=False 回填。

    与 Exception 分开接：参数错误是「调用没干该干的事」，必须让模型看到失败，
    而不是包着 ok=True 的错误文案（静默 no-op = 谎报成功的温床）。
    """


def _as_bool(value) -> bool:
    """工具布尔参数防呆（P1-4 memory_search deep 用）：JSON 布尔之外，模型常传
    字符串 "true"/"false"——字符串按 bool() 恒为真值，"false" 会被误判开启，
    必须显式解析（与 execute_tool 未知参数拒绝同一立场：静默误读 = 谎报温床）。
    """
    if isinstance(value, str):
        return value.strip().lower() in ("true", "1", "yes", "y", "on")
    return bool(value)


def _t_search(ctx: ToolContext, args: dict) -> str:
    """memory_search：主动检索。与 CLI retrieve 同口径（B3 门槛不在此——
    显式查询理应看到原始检索结果，E7 不确定标注照常带出）。
    deep=true 时走深搜（P1-4：LLM 展开查询多路召回；只读探索 boost_access=False，
    深搜是「查」不是「复习」；LLM 不可用/失败自动静默降级快搜）。"""
    query = str(args.get("query", "")).strip()
    if not query:
        raise ToolError("参数错误：query 不能为空。")
    query = query[:200]  # 检索词截断：FTS 与嵌入都不需要长文本，防灌水
    try:
        top_k = max(1, min(10, int(args.get("top_k", settings.AGENT_INJECT_TOP_K))))
    except (TypeError, ValueError):
        top_k = settings.AGENT_INJECT_TOP_K
    deep = _as_bool(args.get("deep", False))
    hits = (deep_retrieve(ctx.store, query, top_k=top_k) if deep
            else retrieve(ctx.store, query, top_k=top_k))
    if not hits:
        return "（没有检索到相关记忆）"
    out = build_context(hits, max_chars=settings.AGENT_TOOL_RESULT_MAX_CHARS,
                        store=ctx.store)
    if any(h.meta.get("uncertain") for h in hits if h.kind != "working"):
        # E7 口径一致：低置信检索显式示警，不静默
        out = "⚠ 检索置信度低，以下结果可能不相关\n" + out
    # 定位清单：检索命中的库内 id——memory_archive 按 id 归档全靠它（build_context
    # 渲染不含 id，模型自己造不出来）。working 命中是会话级瞬时内容，不进清单。
    manifest = [f"{h.kind} #{h.id}" for h in hits
                if h.kind in ("semantic", "episodic", "procedural")][:10]
    if manifest:
        out += "\n\n（定位清单·供 memory_archive 按 id 归档: " + "；".join(manifest) + "）"
    return out


def _t_remember(ctx: ToolContext, args: dict) -> str:
    """memory_remember：模型点名写入。复用 remember()——复述拦截、绿色通道
    白名单降级全部原样生效，这里只做渲染。"""
    from memagent.agent.tools import remember
    content = str(args.get("content", "")).strip()
    if not content:
        raise ToolError("参数错误：content 不能为空。")
    type_ = str(args.get("type") or "experience")
    outcome = str(args.get("outcome") or "")
    if outcome not in ("", "success", "failure"):
        outcome = ""
    r = remember(ctx.store, content, type=type_,
                 context=str(args.get("context") or ""), outcome=outcome,
                 injected_texts=ctx.injected_texts, use_llm=False)
    if r.get("rejected"):
        return "已拒绝：内容与刚注入/检索到的记忆重复（复述拦截），不要重复写入已有记忆。"
    if r["gated"]:
        note = "（type 不在绿色通道白名单，已降级走门控）" if r.get("degraded") else ""
        return f"未入长期记忆：门控判定价值不足，仅暂存会话工作记忆{note}。"
    lines = [f"已写入情景记忆 #{r['episodic_id']} (importance={r['importance']:.2f})"]
    for f in r["facts"]:
        action = {"created": "新建", "renewed": "复证",
                  "superseded": "取代旧版"}.get(f["action"], f["action"])
        lines.append(f"语义事实{action}: [{f['entity']}] {f['relation']} = {f['value']}")
    for sk in r["skills"]:
        lines.append(("技能已复用: " if sk["reused"] else "技能已沉淀: ") + sk["name"])
    return "\n".join(lines)


# ---------- 受控命令沙箱（P2-4 Grounded Loop，安全边界 D5-a 最严档）----------
# 动机（修整方案外部断裂 6）：此前记录的「经验」全是纸上谈兵——「备份失败」这条
# 经验是好是坏、是否配置问题，系统无法验证。run_command 给模型一个受控只读沙箱：
# 白名单内检查类命令的退出码与输出经 remember() 沉淀为结构化经验（success/failure
# 由退出码判定，非文本正则——这就是 outcome 信号的转世：P0-1 停掉的是「从散文里
# 猜成败」，这里是从退出码里读成败），技能成功率从此有了真实数据源。
# 安全边界（settings.SANDBOX_* 注释有完整论证）：白名单 + argvec 直传（绝不经
# shell 拼接）+ 元字符/换行双关卡 + 超时强杀 + 输出截断 + 全程审计（拒绝也记）。
_SANDBOX_METACHARS = ";|&`$><"


def _find_metachar_token(tokens: list[str]) -> str | None:
    """元字符扫描（纯函数）：任一 token 含 shell 元字符即返回该 token，否则 None。

    放在 shlex 切分**之后**做 token 级扫描：POSIX 引号剥离会把引号包裹的注入
    （git commit -m "a; b" → token `a; b`）裸露出来照样拦住——引号不是通行证。
    换行不在此扫：shlex 把它当空白吃掉，token 里留不住，raw 层在 _t_run_command
    第 0 步单独拦。
    """
    for tok in tokens:
        for ch in _SANDBOX_METACHARS:
            if ch in tok:
                return tok
    return None


def _allowlisted(tokens: list[str]) -> bool:
    """白名单匹配（纯函数）：(首token, 次token) 对比 settings.SANDBOX_ALLOW_PAIRS。

    - 首/次 token 一律小写归一（"GIT STATUS" 与 "git status" 同权，命令名无大小写
      语义，归一防「换大小写绕过白名单」的假象——绕不绕都无所谓，反正后面元字符
      与参数向量执行兜底，但匹配要确定性）；
    - 长度 2 的对匹配前两个 token，其余 token 是自由参数（git log --oneline -5）；
    - 长度 1 的对 = 只允许该单词单独成命令（("ls",) 不放行 `ls -la`）；
    - 空 token 列表一律 False（防御式，调用方理论上已拦）。
    """
    if not tokens:
        return False
    first = tokens[0].lower()
    second = tokens[1].lower() if len(tokens) > 1 else ""
    for pair in settings.SANDBOX_ALLOW_PAIRS:
        if pair[0].lower() != first:
            continue
        if len(pair) == 1:
            if len(tokens) == 1:
                return True
        elif second == pair[1].lower():
            return True
    return False


# 沙箱经验文本的 outcome 标记（构造见 _t_run_command 的 exp_text）：
# 防抖时从既有事实的 value 里反读 outcome 用——语义事实表没有 outcome 列，
# 但沙箱产出的经验文本自带「退出码 N（成功/失败）」标记，可确定性反解。
# 注意入库时 value 经 resolve_fact 的 NFKC 归一（全角括号变半角），两种形态都认。
_SBX_OUTCOME_RE = re.compile(r"退出码 \d+[（(](成功|失败)[）)]")


def _sandbox_outcome_of(value: str) -> str | None:
    """从经验文本反读 outcome：'成功'→success / '失败'→failure / 无标记→None。

    None 意味着这条事实不是沙箱产出（或格式已变）——outcome 无法确证时**不**
    参与防抖判定（宁多入一行，不错吞真教训，与 B1 「漏并 > 误并」同一取向）。
    """
    m = _SBX_OUTCOME_RE.search(value)
    if m is None:
        return None
    return "success" if m.group(1) == "成功" else "failure"


def _dup_success_experience(store: SqliteStore, domain: str, text: str) -> bool:
    """防抖判定（红线 4）：同任务域是否已有 cosine ≥ DEDUP_ABSORB_SIM 的同 outcome
    active 经验事实。只对成功结果调用（失败必入库，教训优先）。

    查法取 conflict_resolver._find_absorb_keeper 的保守思路、按「同键」收窄：
    经验事实的确定性键是 (任务域, lesson)，先 fetch active 再按 entity/relation
    过滤、逐条比嵌入——天然只认 active（pending 是试用期/待裁事实，不配当防抖
    依据），也不碰全库 cosine_search 的候选池放大。entity 用项目自己的
    normalize_entity（含别名/代词映射）归一后比对——管线入库时就是这么归一的，
    防抖口径必须与之一致，否则别名域永远查不重。嵌入为空/维度不齐时 cosine
    返回 0.0，天然落在阈值之下（保守：漏跳过只是多一行，错跳过会吞掉真教训）。
    """
    from memagent.encoding.entity_resolver import normalize_entity
    vec = llm.embed(text)
    if not vec:
        return False
    key = normalize_entity(domain, store.aliases.as_map())
    for fact in store.semantic.fetch(status="active", limit=10 ** 9):
        if fact.entity != key or fact.relation != "lesson" or not fact.embedding:
            continue
        if cosine(vec, fact.embedding) < settings.DEDUP_ABSORB_SIM:
            continue
        if _sandbox_outcome_of(fact.value) == "success":
            return True
    return False


def _confined_path(raw: str) -> str:
    """把用户给的相对路径 confinement 到工作区根内（原生读取工具的路径守门员）。

    - 空串 = 工作区根本身；相对路径基于 SANDBOX_CWD 解析；
    - 解析后必须仍落在工作区内（normcase 比较，Windows 盘符大小写不误伤），
      越界（..、绝对路径出区、跨盘）一律 ToolError——只读也不给看工作区外的东西；
    - commonpath 对跨盘参数会抛 ValueError，一并转成拒绝。
    """
    import os
    base = os.path.normcase(os.path.abspath(settings.SANDBOX_CWD))
    target = os.path.normcase(os.path.abspath(
        os.path.join(settings.SANDBOX_CWD, str(raw or "").strip())))
    try:
        inside = os.path.commonpath([base, target]) == base
    except ValueError:            # 不同盘符：commonpath 直接炸，视为越界
        inside = False
    if not inside:
        raise ToolError(f"已拒绝：路径越出工作区（{settings.SANDBOX_CWD}）。"
                        "原生读取工具只允许访问项目目录内的文件。")
    return target


def _t_list_dir(ctx: ToolContext, args: dict) -> str:
    """list_dir：列目录（原生实现，纯 os.listdir——替代 ls/dir 这类 shell
    内置命令，Windows 裸进程跑不了它们，实测 WinError 2）。只读、零进程、
    路径 confinement 到工作区（见 _confined_path）。"""
    import os
    target = _confined_path(str(args.get("path", "")))
    try:
        entries = sorted(os.listdir(target))
    except FileNotFoundError:
        raise ToolError(f"目录不存在: {target}")
    except NotADirectoryError:
        raise ToolError(f"不是目录: {target}")
    if len(entries) > settings.LIST_DIR_MAX_ENTRIES:
        entries = entries[:settings.LIST_DIR_MAX_ENTRIES]
        entries.append(f"…（超出 {settings.LIST_DIR_MAX_ENTRIES} 项已截断）")
    lines = []
    for name in entries:
        full = os.path.join(target, name)
        tag = "目录" if os.path.isdir(full) else "文件"
        size = ""
        if os.path.isfile(full):
            size = f" {os.path.getsize(full)}B"
        lines.append(f"  [{tag}] {name}{size}")
    if not lines:
        return f"目录 {target} 为空"
    return f"目录 {target}（{len(lines)} 项）：" + chr(10) + chr(10).join(lines)


def _t_read_file(ctx: ToolContext, args: dict) -> str:
    """read_file：读文本文件（原生实现，utf-8 errors=replace + 二进制探测 +
    截断到 READ_FILE_MAX_CHARS）。只读、零进程、路径 confinement 到工作区。"""
    import os
    target = _confined_path(str(args.get("path", "")))
    if not os.path.isfile(target):
        raise ToolError(f"文件不存在: {target}")
    with open(target, "rb") as f:
        head = f.read(1024)
        if bytes([0]) in head:
            raise ToolError("已拒绝：疑似二进制文件（含 NUL 字节），不做文本读取。")
    with open(target, "r", encoding="utf-8", errors="replace") as f:
        text = f.read(settings.READ_FILE_MAX_CHARS + 1)
    note = ""
    if len(text) > settings.READ_FILE_MAX_CHARS:
        text = text[:settings.READ_FILE_MAX_CHARS]
        note = (chr(10) + f"…（已截断，单次上限 {settings.READ_FILE_MAX_CHARS} 字符）")
    return (f"文件 {target}（前 {len(text)} 字符）：" + chr(10)
            + text + note)


def _t_run_command(ctx: ToolContext, args: dict) -> str:
    """run_command：受控只读命令沙箱。校验链顺序固定，全部拒绝路径先写审计
    sandbox_denied 再以 ToolError 回填（ok=False，绝不执行）：

      0. 换行预拦——shlex 把换行当空白吃掉，token 级扫描拦不住，raw 层先拦；
      1. shlex.split POSIX 切分（解析失败/空 → 拒绝）；
      2. token 级元字符扫描（; | & ` $ > <）；
      3. (首token, 次token) 白名单匹配（SANDBOX_ALLOW_PAIRS，小写归一）。

    通过后 subprocess.run(argvec, shell=False) 执行——参数向量直传进程，不经
    shell 拼接，白名单与元字符两关之外再叠一层结构防线；超时强杀、输出合并
    截断。拿到退出码后判 outcome，经验文本经 remember() 走 E8 通道入库：
    复述拦截/门控保底/(任务域, lesson) 确定性键/技能统计与 policy 进化全部
    原样复用。防抖（红线 4）：失败必入库；成功仅当同任务域无近重复同 outcome
    经验才入库（例行成功不刷版本链，跳过记审计 sandbox_dedup_skip）——防抖
    只挡「沉淀」，不挡「执行」，命令照常跑、结果照常回填模型。
    """
    from memagent.agent.tools import remember
    command = str(args.get("command", "")).strip()
    if "\n" in command or "\r" in command:
        ctx.store.log("tool", 0, "sandbox_denied", f"命令含换行: {command[:80]!r}")
        raise ToolError("已拒绝：命令含换行符，未执行。沙箱只接受单行命令。")
    try:
        tokens = shlex.split(command)
    except ValueError as e:
        ctx.store.log("tool", 0, "sandbox_denied", f"shlex 解析失败 ({e}): {command[:80]!r}")
        raise ToolError(f"已拒绝：命令解析失败（{e}），未执行。")
    if not tokens:
        ctx.store.log("tool", 0, "sandbox_denied", f"空命令: {command[:80]!r}")
        raise ToolError("已拒绝：空命令，未执行。")
    bad = _find_metachar_token(tokens)
    if bad is not None:
        ctx.store.log("tool", 0, "sandbox_denied",
                      f"token {bad!r} 含 shell 元字符: {command[:80]!r}")
        raise ToolError(f"已拒绝：token {bad!r} 含 shell 元字符，未执行。"
                        "沙箱只接受白名单内的只读检查命令。")
    if not _allowlisted(tokens):
        pair = (tokens[0].lower(), tokens[1].lower() if len(tokens) > 1 else "")
        ctx.store.log("tool", 0, "sandbox_denied",
                      f"{pair} 不在白名单: {command[:80]!r}")
        raise ToolError("已拒绝：命令不在只读白名单（SANDBOX_ALLOW_PAIRS），未执行。"
                        "可用示例: git status / git log / python --version；列目录/读文件用 list_dir / read_file 工具。")

    # ---- 校验全过，真正执行（此后命令已实际启动，审计走 sandbox_exec）----
    try:
        proc = subprocess.run(tokens, shell=False, capture_output=True, text=True,
                              timeout=settings.SANDBOX_TIMEOUT, cwd=settings.SANDBOX_CWD)
    except subprocess.TimeoutExpired:
        ctx.store.log("tool", 0, "sandbox_exec",
                      f"{command} -> 超时强杀(>{settings.SANDBOX_TIMEOUT}s)")
        raise ToolError(f"命令超过 {settings.SANDBOX_TIMEOUT} 秒已被强杀，"
                        "未取得退出码，本次不沉淀经验。")
    except (FileNotFoundError, OSError) as e:
        ctx.store.log("tool", 0, "sandbox_exec", f"{command} -> 启动失败 {e!r}")
        raise ToolError(f"命令启动失败：{e}（可执行文件不存在或无法访问）。"
                        "本次不沉淀经验。")
    except ValueError as e:  # text=True 严格解码失败等（Windows GBK 输出的边角）
        ctx.store.log("tool", 0, "sandbox_exec", f"{command} -> 输出解码失败 {e!r}")
        raise ToolError(f"命令输出解码失败：{e}。本次不沉淀经验。")

    code = proc.returncode
    ctx.store.log("tool", 0, "sandbox_exec", f"{command} -> exit {code}")
    output = ((proc.stdout or "") + (proc.stderr or "")).strip()
    shown = output[:settings.SANDBOX_OUTPUT_MAX_CHARS]
    if len(output) > len(shown):
        shown += f"…（已截断，完整 {len(output)} 字符）"
    summary = next((ln.strip() for ln in output.splitlines() if ln.strip()), "")
    if len(summary) > 120:
        # 入库摘要封顶（入库克制）：经验文本进情景记忆原文与 (任务域, lesson) 事实
        # 的 value，巨型首行会灌水记忆与嵌入——首行前 120 字足以标识「这次跑了
        # 什么、结果如何」，完整输出已回填给模型、审计里也有退出码可回溯
        summary = summary[:120] + "…"
    ok_word = "成功" if code == 0 else "失败"
    outcome = "success" if code == 0 else "failure"
    # 任务域：显式 context 优先（模型可给语义标签），缺省取命令首 token（小写
    # 归一，确定性键）——P0-2 守卫依赖它非空，首 token 已由拒绝链保证非空。
    domain = str(args.get("context") or "").strip() or tokens[0].lower()
    lines = [f"命令 {command}", f"退出码 {code}（{ok_word}）",
             "输出:", shown or "（无输出）"]

    exp_text = f"命令 {command} 退出码 {code}（{ok_word}）：{summary or '（无输出）'}"
    if _dup_success_experience(ctx.store, domain, exp_text):
        ctx.store.log("tool", 0, "sandbox_dedup_skip",
                      f"{command} -> exit {code} 与任务域「{domain}」既有成功经验"
                      "近重复，跳过入库（执行照常，只是不沉淀）")
        lines.append("（经验未入库：同任务域已有近重复的同结果经验，例行成功不刷版本链）")
        return "\n".join(lines)
    r = remember(ctx.store, exp_text, type="experience", context=domain,
                 outcome=outcome, injected_texts=ctx.injected_texts, use_llm=False)
    if r.get("rejected"):
        # 命令输出不是复述注入内容，走到这里说明模型把工具回填原样抄了回来——
        # 拦截照常生效（无害），如实告知即可，执行本身不受影响。
        lines.append("（经验未入库：与刚注入/检索到的记忆重复，复述拦截）")
    elif r.get("gated"):
        lines.append("（经验未入长期记忆：门控判定价值不足，仅暂存会话工作记忆）")
    else:
        lines.append(f"经验已入库: 情景 #{r['episodic_id']}（任务域「{domain}」"
                     f"outcome={outcome}）")
        for sk in r["skills"]:
            lines.append(("技能已复用: " if sk["reused"] else "技能已沉淀: ") + sk["name"])
    return "\n".join(lines)


def _t_sleep(ctx: ToolContext, args: dict) -> str:
    """memory_sleep：睡眠巩固。NREM×3 + REM + SM-2 重演，走维护通道，可能较慢。"""
    report = consolidate(ctx.store)
    rem = report.get("rem_associations", [])
    lines = [f"睡眠巩固完成（NREM×{report.get('nrem_rounds', 0)} + REM）: "
             f"聚类={report['clusters']}, 蒸馏事实={report['distilled_facts']}, "
             f"摘要替代={report['summarized']}, REM联想={len(rem)}, "
             f"REM写入={report.get('rem_facts', 0)}"]
    for r in rem[:3]:
        lines.append(f"联想: [{r['entities'][0]}] × [{r['entities'][1]}] "
                     f"（激活 {r['strength']}）")
    due = report.get("due_reviews", [])
    if due:
        recalled = sum(1 for d in due if d["recalled"])
        lines.append(f"今日回忆清单: {len(due)} 条到期重演（唤回 {recalled}）")
    return "\n".join(lines)


def _t_forget(ctx: ToolContext, args: dict) -> str:
    """memory_forget：主动遗忘（强度重算 → 归档 → 摘要降级 → 硬删，保守链路）。"""
    report = run_forgetting(ctx.store)
    return (f"遗忘完成: 归档={report['archived']}, 硬删={report['deleted']}, "
            f"活跃情景={report['episodic_active']}")


def _t_archive(ctx: ToolContext, args: dict) -> str:
    """memory_archive：按 id 归档一条记忆（用户点名删除时用）。

    保守原则：只归档不硬删（三仓储的行都保留，人工可逆）；id 必须真实存在，
    找不到就明说「未做任何改动」——V1.7.3 实测翻车的反面教材：memory_forget
    静默忽略参数跑了全局清扫，模型却对着 ok=True 谎报「已删除」。这里宁可
    返回失败也不给含糊的成功。

    kind 必须带上（semantic / episodic / procedural）：三张表的 id 各自独立
    自增，同一个数字在多张表里都有行（实测 episodic#1 与 semantic#1 并存），
    不指定 kind 会归档错表。kind 与 id 都来自 memory_search 的定位清单。
    """
    kind = str(args.get("kind", "")).strip().lower()
    if kind not in ("semantic", "episodic", "procedural"):
        raise ToolError("参数错误：kind 必须是 semantic / episodic / procedural"
                        "（都来自 memory_search 定位清单）。")
    try:
        target_id = int(args.get("id"))
    except (TypeError, ValueError):
        raise ToolError("参数错误：id 必须是整数。先用 memory_search 检索，"
                        "从结果末尾的「定位清单」里取 id 和 kind。")
    s = ctx.store
    if kind == "semantic":
        fact = s.semantic.get(target_id)
        if fact is None:
            raise ToolError(f"semantic 表里没有 id={target_id} 的事实。"
                            f"未做任何改动——请用 memory_search 重新定位。")
        s.semantic.set_status(target_id, "archived")
        return (f"已归档语义事实 #{target_id} [{fact.entity}] {fact.relation} "
                f"= {fact.value[:60]}（行保留，可人工恢复）")
    if kind == "episodic":
        epi = s.episodic.get(target_id)
        if epi is None:
            raise ToolError(f"episodic 表里没有 id={target_id} 的情景记忆。"
                            f"未做任何改动——请用 memory_search 重新定位。")
        s.episodic.set_status(target_id, "archived")
        return f"已归档情景记忆 #{target_id}：{epi.summary[:60]}（行保留，可人工恢复）"
    skill = s.procedural.get(target_id)
    if skill is None:
        raise ToolError(f"procedural 表里没有 id={target_id} 的技能。"
                        f"未做任何改动——请用 memory_search 重新定位。")
    s.procedural.set_status(target_id, "archived")
    return (f"已归档程序记忆(技能) #{target_id}「{skill.name}」"
            f"（行保留，可人工恢复）")


def _t_history(ctx: ToolContext, args: dict) -> str:
    """memory_history：某实体的事实版本链。"""
    entity = str(args.get("entity", "")).strip()
    if not entity:
        raise ToolError("参数错误：entity 不能为空。")
    facts = ctx.store.semantic.fetch_history(entity[:60])
    if not facts:
        return f"（没有 [{entity}] 的记忆）"
    lines = [f"[{entity}] 版本链（{len(facts)} 条）:"]
    for f in facts[:20]:
        period = f"{f.valid_from} ~ {f.valid_to or '今'}"
        extra = f" <-被#{f.superseded_by}取代" if f.superseded_by else ""
        evidence = f" 证据x{f.evidence_count}" if f.evidence_count > 1 else ""
        lines.append(f"  #{f.id} [{f.status}] {f.relation} = {f.value} "
                     f"({period}){evidence}{extra}")
    if len(facts) > 20:
        lines.append(f"  …另有 {len(facts) - 20} 条")
    return "\n".join(lines)


def _t_conflicts(ctx: ToolContext, args: dict) -> str:
    """memory_conflicts：只读列出待裁决冲突。resolve 不对模型开放（D1：
    A 类冲突取代旧事实只能人工/证据裁决）。"""
    rows = ctx.store.conflicts.fetch_all(status="pending")
    if not rows:
        return "（没有待裁决冲突）"
    lines = [f"待裁决冲突 {len(rows)} 条（裁决由用户完成）:"]
    for r in rows[:10]:
        old = ctx.store.semantic.get(r["old_id"])
        new = ctx.store.semantic.get(r["new_id"])
        lines.append(f"  #{r['conflict_id']} 旧「{old.value if old else '?'}」"
                     f" vs 新「{new.value if new else '?'}」")
    if len(rows) > 10:
        lines.append(f"  …另有 {len(rows) - 10} 条")
    return "\n".join(lines)


def _t_status(ctx: ToolContext, args: dict) -> str:
    """memory_status：库况概览。"""
    s = ctx.store
    epi = len(s.episodic.fetch(status="active", limit=10 ** 9))
    sem = len(s.semantic.fetch(status="active", limit=10 ** 9))
    pend = len(s.semantic.fetch(status="pending", limit=10 ** 9))
    skills = len(s.procedural.fetch())
    return (f"情景记忆(活跃) {epi} 条，语义记忆(活跃) {sem} 条 / 待裁决 {pend} 条，"
            f"技能 {skills} 个，会话工作记忆 {len(s.working)} 条；"
            f"LLM: {'可用' if llm.llm_available() else '离线'} ({llm.active_provider()})")


def _t_report(ctx: ToolContext, args: dict) -> str:
    """memory_report：健康报告（状态分布/版本链/检索空缺等）。"""
    return build_health_report(ctx.store)


@dataclass(frozen=True)
class ToolSpec:
    """一个工具的注册项：提示词里的一句话说明 + 执行函数 + 合法参数名集合。

    args_keys 是参数防呆的依据（V1.7.4）：模型把 memory_forget 当成「按内容删除」
    传了 content 进来，旧版静默忽略参数跑全局清扫，模型对着 ok=True 谎报「已删除」
    ——静默丢参 = 谎报的温床。现在未知参数一律拒绝并明说「调用未执行」。
    """

    hint: str
    handler: object
    args_keys: frozenset = frozenset()


TOOL_REGISTRY: dict[str, ToolSpec] = {
    "memory_search": ToolSpec(
        "主动检索记忆。args: {\"query\": \"检索词\", \"top_k\": 5(可选,1~10), "
        "\"deep\": false(可选,深搜=true 时用 LLM 展开查询多路召回，慢且只读)}。"
        "自动注入已覆盖的问题不要再查。", _t_search,
        frozenset({"query", "top_k", "deep"})),
    "memory_remember": ToolSpec(
        "点名写入记忆（AI 经验/环境状态）。args: {\"content\": \"内容\", "
        "\"type\": \"experience或env_statement\", \"context\": \"任务域(可选)\", "
        "\"outcome\": \"success或failure(可选)\"}。"
        "用户明确说出的偏好/指令/身份由系统自动记录，不要转写（转写会生成垃圾技能）；"
        "本工具只用于你自己的工作经验与环境状态。", _t_remember,
        frozenset({"content", "type", "context", "outcome"})),
    "run_command": ToolSpec(
        "执行白名单内的只读检查命令并以事实核查经验/检查环境（如确认「备份失败」是"
        "不是配置问题）。args: {\"command\": \"命令\", \"context\": \"任务域(可选,"
        "缺省取命令首词)\"}。退出码与输出会自动沉淀为经验（exit 0=成功，非 0=失败"
        "，真实反馈喂给技能统计），不需要再调 memory_remember 转写。白名单外命令"
        "（写入/删除/安装/联网等）一律拒绝且不执行：仅 git status/log/diff/show、"
        "ipconfig、systeminfo、tasklist、where、whoami、hostname、docker ps/images、"
        "python --version、pip list/show/--version、node --version（注意：ls/dir 是 "
        "shell 内置命令跑不了，列目录/读文件请用 list_dir/read_file 工具）。命令含 "
        "shell 元字符（; | & ` $ > <）或换行同样拒绝；参数防呆机制与其它工具一致。", _t_run_command,
        frozenset({"command", "context"})),
    "list_dir": ToolSpec(
        "列出项目目录内容（原生只读，替代 ls/dir——它们是 shell 内置命令跑不了）。"
        "args: {\"path\": \"子目录(可选,缺省项目根,仅限项目目录内)\"}。", _t_list_dir,
        frozenset({"path"})),
    "read_file": ToolSpec(
        "读项目内文本文件（原生只读，utf-8，超长截断，二进制拒绝）。"
        "args: {\"path\": \"项目内相对路径\"}。", _t_read_file,
        frozenset({"path"})),
    "memory_archive": ToolSpec(
        "按 id 归档一条记忆（用户要求删除某条时用；只归档不硬删，可人工恢复）。"
        "args: {\"id\": 记忆id, \"kind\": \"semantic或episodic或procedural\"}"
        "——两者都来自 memory_search 结果末尾的「定位清单」，不要凭空编。", _t_archive,
        frozenset({"id", "kind"})),
    "memory_history": ToolSpec(
        "查某实体的事实版本链。args: {\"entity\": \"实体名\"}。", _t_history,
        frozenset({"entity"})),
    "memory_conflicts": ToolSpec(
        "列出待裁决冲突（你只能查看，裁决由用户完成）。args: {}。", _t_conflicts,
        frozenset()),
    "memory_status": ToolSpec("记忆库统计概览。args: {}。", _t_status,
                              frozenset()),
    "memory_report": ToolSpec("记忆健康报告。args: {}。", _t_report,
                              frozenset()),
    "memory_sleep": ToolSpec(
        "睡眠巩固（蒸馏+联想+复习重演），耗时较长，仅当用户要求整理记忆时调用。"
        "args: {}。", _t_sleep, frozenset()),
    "memory_forget": ToolSpec(
        "主动遗忘清理（全局清扫低强度记忆，不带参数）。删除指定的一条记忆不用本工具，"
        "用 memory_archive。args: {}。", _t_forget, frozenset()),
}


def build_tools_prompt() -> str:
    """生成提示词里的工具说明段：settings.AGENT_TOOLS_PROMPT_HEADER（协议头）
    + 注册表逐项清单。清单由注册表驱动——提示词承诺的工具与可执行的工具
    永远同源，不许漂移。"""
    lines = [settings.AGENT_TOOLS_PROMPT_HEADER.rstrip("\n")]
    for name, spec in TOOL_REGISTRY.items():
        lines.append(f"- {name}：{spec.hint}")
    return "\n".join(lines)


# ---------- 执行 ----------
def execute_tool(store: SqliteStore, call: ToolCall, ctx: ToolContext) -> ExecResult:
    """执行一次工具调用：未知工具、非法参数、异常都转成文本回填，绝不炸循环。

    参数防呆（V1.7.4）：args 里出现注册表不认识的键 → 拒绝执行并明说「调用未
    执行」。静默忽略参数会让模型拿着 ok=True 编造成功（实测：模型给 memory_forget
    传 content，工具跑了全局清扫归档 0 条，模型却回报「已删干净」）。

    每次调用与结果都落审计（meta 表 action=tool_call / tool_result），
    与 remember_rejected 等既有审计同源——模型侧动作必须可回溯。
    """
    spec = TOOL_REGISTRY.get(call.name)
    if spec is None:
        return ExecResult(False, f"未知工具 {call.name!r}。可用: "
                          f"{', '.join(TOOL_REGISTRY)}")
    store.log("agent", 0, "tool_call",
              f"{call.name} {json.dumps(call.args, ensure_ascii=False)[:200]}")
    unknown = sorted(k for k in call.args if k not in spec.args_keys)
    if unknown:
        accepted = (f"可用参数: {', '.join(sorted(spec.args_keys))}"
                    if spec.args_keys else "本工具不带任何参数")
        text = (f"参数无效: {', '.join(unknown)}。{accepted}。"
                f"调用未执行，库没有任何改动。")
        store.log("agent", 0, "tool_result", f"{call.name} ok=False {text[:200]}")
        return ExecResult(False, text)
    try:
        text = spec.handler(ctx, call.args)
        ok = True
    except ToolError as e:  # 参数/目标错误：调用没干该干的事，必须以失败回填
        ok, text = False, str(e)
    except Exception as e:  # 工具的任何失败都降级为一条错误消息
        ok, text = False, f"工具执行失败: {e!r}"
    store.log("agent", 0, "tool_result", f"{call.name} ok={ok} {text[:200]}")
    return ExecResult(ok, text)


def truncate_result(text: str) -> str:
    """回填提示词前的字符预算裁剪（AGENT_TOOL_RESULT_MAX_CHARS）。"""
    cap = settings.AGENT_TOOL_RESULT_MAX_CHARS
    if len(text) <= cap:
        return text
    return text[:cap] + "…（已截断）"


def render_exchange(index: int, call: ToolCall | None, err: str | None,
                    result_text: str) -> str:
    """把一次「调用 → 结果」渲染回提示词（[工具调用记录] 段的一节）。"""
    if err is not None:
        return (f"\n[工具调用记录·第{index}次]\n"
                f"你的工具调用格式无效: {err}\n"
                f"请输出合法的 <tool_call> JSON 块重试，或直接回答用户。")
    args_json = json.dumps(call.args, ensure_ascii=False)
    return (f"\n[工具调用记录·第{index}次]\n"
            f"你调用了 {call.name}({args_json})，结果:\n"
            f"{truncate_result(result_text)}\n"
            f"（系统已执行完毕；若结果已足够请直接回答用户，不要重复调用。）")


# ---------- 流式过滤器 ----------
class ToolStreamFilter:
    """流式轮的工具标记过滤器（answer 通道的透传/吞没）。

    契约：不含标记的增量**原样、同序、同分片**转发（test_agent_loop 的
    on_delta 同序转发合同不许破）；只有当 answer 里出现 <tool_call> 时，
    标记及之后的增量被吞没（工具调用文本不该出现在用户屏幕上）。

    holdback 只保留「可能是标记前缀」的尾部（如 "<"、" <too"），普通文本
    立即放行——标记被拆进多个增量也能识别（flush 兜底放行剩余）。
    thinking/reset 与吞没后的 answer 不经此处理：thinking 原样转发（模型
    思考流不含调用块），reset 清空缓冲区重新开始。
    """

    def __init__(self, sink):
        self._sink = sink
        self._buf = ""
        self.suppressed = False   # 是否已检测到标记（本轮剩余增量全部吞没）

    def __call__(self, kind: str, text: str) -> None:
        if kind == "reset":
            self._buf, self.suppressed = "", False
            self._sink(kind, text)
            return
        if kind != "answer":
            self._sink(kind, text)
            return
        if self.suppressed:
            return          # 标记已出现：本轮剩余 answer 全部吞没，不再进缓冲
        self._buf += text
        idx = self._buf.find(TOOL_CALL_OPEN)
        if idx >= 0:
            head = self._buf[:idx]
            if head:
                self._sink("answer", head)
            self._buf = ""
            self.suppressed = True
            return
        keep = self._holdback_len(self._buf)
        if len(self._buf) > keep:
            self._sink("answer", self._buf[:len(self._buf) - keep])
            self._buf = self._buf[len(self._buf) - keep:]

    def flush(self) -> None:
        """流结束后放行缓冲残余（没有标记时缓冲里是正文的最后几个字）。"""
        if self._buf and not self.suppressed:
            self._sink("answer", self._buf)
        self._buf = ""

    @staticmethod
    def _holdback_len(buf: str) -> int:
        """buf 尾部与 TOOL_CALL_OPEN 前缀重合的最长长度（0=可全部放行）。"""
        for k in range(min(len(buf), len(TOOL_CALL_OPEN) - 1), 0, -1):
            if TOOL_CALL_OPEN.startswith(buf[-k:]):
                return k
        return 0
