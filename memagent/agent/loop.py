"""记忆原生循环（V1.7 P2 注入→生成→录入；V1.7.3 增加工具调用轮）。

一轮 turn 的编排：注入 → 生成（含工具轮：解析 <tool_call> → 执行 → 结果
回填提示词 → 继续生成，最多 AGENT_TOOL_MAX_ROUNDS 次）→ 录入。

工具是自动注入的补丁不是替代（PLAN-V1.7 §0 的立场不变）：框架每轮照旧自动
取记忆拼进提示词，模型只在注入漏了/要点名写入/用户要求整理时才动手。协议、
工具集与安全边界见 agent/toolkit——本模块只做轮次编排，不解释协议。

生成这一步仍然只有 `adapters.llm.chat` / `chat_stream` 两个出口（项目铁律：
任何模块不得绕过适配层发 HTTP）；无可用模型时返回 settings.AGENT_OFFLINE_REPLY。
离线回复刻意**不**复述注入内容——否则每轮都在制造「复述」，把自增强回路的
验证变成噪音（要测复述识别，用显式传入的 assistant_text，见 run / turn）。

自增强防护随之扩展到工具通道：工具带回到模型眼前的记忆原文（tool_texts）
与注入内容一并作为复述识别的参照系——模型把刚检索到的记忆原样复述进回答，
同样不沉淀（工具通道不是自增强回路的旁路）。

P1-4 深搜（默认关闭）：构造参数 deep=True 或会话内输入 /deep 切换——开启后
每轮注入改走 retrieval.deep.deep_retrieve（LLM 展开查询多路召回；只读探索，
boost_access=False）。深搜是用户点名的重检索不是主读路径：注入走快搜、模型
/用户想加深召回时才点名，离线静默降级快搜（见 retrieval/deep.py 的断路器）。

Batch-A 用户中断：流式轮可被中断（interrupt_event 置位 → 增量边界抛
TurnInterrupted）——已生成的部分文本保留为本轮回答并照常录入（中断的回答
也是真实输出），Turn.interrupted 留痕；非流式轮没有增量边界，不可中断。
"""
from __future__ import annotations

import threading
from dataclasses import dataclass, field

from memagent import settings
from memagent.adapters import llm
from memagent.agent.injector import Injection, inject
from memagent.agent.recorder import record_turn
from memagent.agent.toolkit import (
    ToolContext,
    ToolStreamFilter,
    execute_tool,
    parse_tool_call,
    render_exchange,
    strip_tool_call,
)
from memagent.retrieval.deep import deep_retrieve
from memagent.storage import SqliteStore

DEEP_COMMAND = "/deep"   # 会话级深搜开关命令（_toggle_deep 翻转并回显）


class TurnInterrupted(Exception):
    """用户请求中断本轮生成（Batch-A：TUI 生成中双击 Esc 置位 interrupt_event）。

    只在流式路径的增量边界抛出（回调层抛出、经 chat_stream 调用栈自然向上
    传播——适配层合同「on_delta 的异常不捕获」，零改动）；非流式整段调用
    没有增量边界，无法中断。partial 携带截至中断点已生成的部分回答文本，
    由 _generate_with_tools 捕获后作为本轮回答收尾（照常录入）。
    """

    def __init__(self, partial: str = ""):
        super().__init__(partial)
        self.partial = partial


@dataclass
class Turn:
    """一轮对话的完整留痕：问了什么、注入了什么、答了什么、沉淀了什么。"""

    user_text: str = ""
    assistant_text: str = ""
    injection: Injection = field(default_factory=Injection)
    record: dict = field(default_factory=dict)
    thinking: str = ""   # 本轮流式产生的思考全文（chat_stream on_delta 的 thinking 增量留底）
    tool_calls: list = field(default_factory=list)  # 本轮执行的工具调用留痕（name/args/ok/result）
    interrupted: bool = False  # 本轮生成被用户中断（双击 Esc，Batch-A；部分回答已保留）

    @property
    def injected_texts(self) -> list[str]:
        return self.injection.injected_texts


def build_prompt(system_prompt: str, memory_context: str, user_text: str,
                 history: list[Turn] | None = None, tool_notes: str = "") -> str:
    """拼装本轮提示词：系统指令 + 注入记忆 + 最近对话 + 工具记录 + 当前输入。

    四笔上下文预算互相独立：记忆片段（AGENT_INJECT_MAX_CHARS）、最近对话
    （AGENT_HISTORY_TURNS 轮）、工具记录（每次结果 AGENT_TOOL_RESULT_MAX_CHARS）、
    当前输入（不裁剪）。工具记录段紧贴当前输入——模型接着上次调用继续，视线
    距离最短。
    """
    parts = [f"[系统]\n{system_prompt}",
             "[记忆片段]\n" + (memory_context or "（无相关记忆）")]
    if history:
        recent = history[-settings.AGENT_HISTORY_TURNS:]
        lines = []
        for t in recent:
            lines.append(f"用户：{t.user_text}")
            if t.assistant_text:
                lines.append(f"助手：{t.assistant_text}")
        if lines:
            parts.append("[最近对话]\n" + "\n".join(lines))
    if tool_notes:
        parts.append(tool_notes.strip())
    parts.append(f"用户：{user_text}\n助手：")
    return "\n\n".join(parts)


class AgentLoop:
    """一轮一轮跑的对话循环；store 由调用方注入（组合根在入口层，不在这里）。"""

    def __init__(self, store: SqliteStore, system_prompt: str | None = None,
                 top_k: int | None = None, max_chars: int | None = None,
                 task_context: str = "", use_llm: bool = True, responder=None,
                 on_delta=None, enable_tools: bool | None = None,
                 deep: bool = False, tool_max_rounds: int | None = None,
                 interrupt_event: threading.Event | None = None):
        self.store = store
        self.system_prompt = system_prompt or settings.AGENT_SYSTEM_PROMPT
        self.top_k = top_k
        self.max_chars = max_chars
        self.task_context = task_context
        # 工具轮上限（P2-4 深度使用反馈）：None=用 settings（.env/TUI 可配），
        # 显式传参优先；负数钳 0（0=禁用工具轮，等价 enable_tools=False 的效果
        # 但保留提示词差异——模型仍知道工具存在只是不被允许多轮）
        if tool_max_rounds is None:
            self.tool_max_rounds = settings.AGENT_TOOL_MAX_ROUNDS
        else:
            self.tool_max_rounds = max(0, int(tool_max_rounds))
        self.use_llm = use_llm
        self.responder = responder          # 测试/评测注入的确定性应答器
        self.on_delta = on_delta            # 流式增量回调（None=不流式，走 chat 整段）
        # 用户中断信号（Batch-A 双击 Esc）：流式增量边界检查，置位即在下一个
        # delta 抛 TurnInterrupted；构造参数可选传入（TUI 与键位处理共享同一事件），
        # 缺省自建。非流式路径无增量边界，不受它影响
        self.interrupt_event = (interrupt_event if interrupt_event is not None
                                else threading.Event())
        self._turn_interrupted = False      # 本轮是否被中断（_generate_with_tools 置位）
        # P1-4 会话级深搜开关（默认关）：开启后每轮注入走 deep_retrieve（LLM
        # 展开查询多路召回，只读不写回）；会话内 /deep 随时翻转（_toggle_deep）
        self.deep = deep
        # 工具调用开关（V1.7.3）：开启时系统提示追加工具协议段，生成走工具轮循环
        self.enable_tools = (settings.AGENT_TOOLS_ENABLED if enable_tools is None
                             else enable_tools)
        if self.enable_tools:
            from memagent.agent.toolkit import build_tools_prompt
            self.system_prompt += build_tools_prompt()
        self._turn_thinking = ""            # 本轮流次思考缓冲（_capture 累积）
        self._delta_sink = None             # 本轮用户回调（_capture 的转发目标）
        self.history: list[Turn] = []
        self.stats = {"turns": 0, "restatement_skipped": 0, "ingested": 0,
                      "working_only": 0, "offline": 0, "tools": 0}

    # ---------- 一轮 ----------
    def turn(self, user_text: str, assistant_text: str | None = None,
             task_context: str | None = None, outcome: str = "",
             on_delta=None) -> Turn:
        """跑一轮；assistant_text 给出时跳过生成（脚本化回放，评测的确定性来源，
        也不触发工具——工具只由模型在生成路径上发起）。

        on_delta 给出时仅本轮流式（覆盖实例级 self.on_delta，缺省回落到它）。
        录入/复述识别/信号检测只看最终 assistant_text，与是否流式无关。
        会话命令（/deep）在本方法入口拦截：翻转开关并回显，不当作对话内容。
        interrupt_event 置位时流式轮在增量边界中断（Batch-A）：以已生成的部分
        回答收尾并照常录入，Turn.interrupted 标记中断事实。
        """
        if user_text.strip() == DEEP_COMMAND:
            return self._toggle_deep()
        # P1-4：深搜开启时注入走 deep_retrieve（只读多路召回）；关闭时不传
        # retriever_fn——inject 默认 retrieve，快搜轮的调用形态与从前逐位一致
        if self.deep:
            injection = inject(self.store, user_text, top_k=self.top_k,
                               max_chars=self.max_chars, retriever_fn=deep_retrieve)
        else:
            injection = inject(self.store, user_text, top_k=self.top_k,
                               max_chars=self.max_chars)
        generated = assistant_text is None
        tool_calls: list[dict] = []
        tool_texts: list[str] = []
        self._turn_interrupted = False   # 中断标记是每轮独立的（_generate_with_tools 置位）
        if generated:
            assistant_text, tool_calls, tool_texts = self._generate_with_tools(
                user_text, injection, on_delta=on_delta if on_delta is not None
                else self.on_delta)
        record = record_turn(self.store, user_text=user_text,
                             assistant_text=assistant_text or "",
                             injected_texts=list(injection.injected_texts) + tool_texts,
                             task_context=self.task_context if task_context is None
                             else task_context,
                             outcome=outcome, use_llm=self.use_llm,
                             own_recent=[t.assistant_text for t in self.history[-3:]
                                         if t.assistant_text])
        turn = Turn(user_text=user_text, assistant_text=assistant_text or "",
                    injection=injection, record=record,
                    thinking=self._turn_thinking if generated else "",
                    tool_calls=tool_calls, interrupted=self._turn_interrupted)
        self.history.append(turn)
        self.stats["tools"] += len(tool_calls)
        self._bump(record)
        return turn

    def run(self, turns: list[str]) -> list[Turn]:
        """批量跑（脚本化评测 / 非交互使用）。"""
        return [self.turn(t) for t in turns]

    # ---------- 会话命令 ----------
    def _toggle_deep(self) -> Turn:
        """/deep：翻转会话级深搜开关并回显状态（P1-4）。

        命令轮不是对话内容：不注入、不录入、不计轮次、不进 history——避免
        「/deep」出现在后续提示词的最近对话段里。开启后每轮注入改走
        deep_retrieve（LLM 展开查询多路召回；LLM 不可用时自动静默降级快搜）。
        """
        self.deep = not self.deep
        if self.deep:
            text = ("深搜记忆检索已开启：每轮注入改用 LLM 展开查询的多路召回"
                    "（较慢，多一次对话调用；只读探索，不写回记忆）。再输 /deep 可切回。")
        else:
            text = "深搜记忆检索已关闭：注入恢复默认快搜。"
        return Turn(user_text=DEEP_COMMAND, assistant_text=text)

    # ---------- 生成（含工具轮） ----------
    def _generate_with_tools(self, user_text: str, injection: Injection,
                             on_delta=None) -> tuple[str, list[dict], list[str]]:
        """生成 + 工具轮循环，返回 (最终回答, 调用留痕, 回填给模型的工具结果文本)。

        每轮生成后解析 <tool_call>：没有 → 这就是最终回答；有 → 执行工具、
        结果渲染回提示词再生成（最多 AGENT_TOOL_MAX_ROUNDS 次）。最后一轮提示
        词明示「直接回答」，模型仍输出调用时只取标记前的可见文本作答。
        解析失败不炸循环：错误信息回填给模型自行纠正（也占一轮额度）。

        用户中断（Batch-A）：流式增量边界（_generate 的守卫包装）与工具反馈
        回调（on_delta("tool"/"tool_result") 直调用户回调）都可能抛
        TurnInterrupted——捕获后保留已生成的部分文本作答（中断的回答也是
        真实输出，照常录入），本轮以部分回答收尾。
        """
        tool_notes = ""
        calls: list[dict] = []
        texts: list[str] = []
        text = ""
        interrupted = False
        try:
            for round_idx in range(self.tool_max_rounds + 1):
                last = round_idx == self.tool_max_rounds
                if last and calls:
                    tool_notes += "\n（工具调用轮次已达上限，本轮请直接回答用户，不要再调用工具）"
                text = self._generate(user_text, injection, on_delta=on_delta,
                                      tool_notes=tool_notes)
                if not self.enable_tools:
                    break
                call, err = parse_tool_call(text)
                if call is None and err is None:
                    break                       # 没有工具调用：本轮即最终回答
                if last:
                    break                       # 上限轮的调用不再执行：没有下一轮消费结果
                result_text = ""
                ok = False
                if err is not None:
                    result_text = err           # 解析失败：错误信息即回填内容
                else:
                    ctx = ToolContext(store=self.store,
                                      injected_texts=tuple(injection.injected_texts))
                    res = execute_tool(self.store, call, ctx)
                    ok, result_text = res.ok, res.text
                    if on_delta is not None:
                        on_delta("tool", f"{call.name} {result_text.splitlines()[0][:80]}")
                calls.append({"name": call.name if call else "(格式无效)",
                              "args": call.args if call else {},
                              "ok": ok, "result": result_text[:200]})
                texts.append(result_text)
                tool_notes += render_exchange(round_idx + 1, call, err, result_text)
                if on_delta is not None:
                    on_delta("tool_result", "结果已回填")
                if last:
                    break
        except TurnInterrupted as e:
            # 用户中断：partial 是本轮已收到的 answer 增量（守卫抛出时）；
            # 工具反馈回调抛出时 partial 为空、退回最近一轮已生成文本（残余
            # 调用标记随后被 strip_tool_call 清掉）。tool_notes 在此之后不再
            # 被消费（循环即止），中断留痕改随回答文本落盘（见下方 marker）
            text, interrupted = (e.partial or text), True
            tool_notes += "（已被用户中断）"
        self._turn_interrupted = interrupted
        # 关闭工具时输出原样保留（协议文本只是普通文本）；开启时把残余标记清掉
        final = strip_tool_call(text) if self.enable_tools else text
        if interrupted:
            # 中断标记随回答留痕：转录可见、录入如实（部分回答 + 中断事实）
            final = f"{final}（已被用户中断）" if final else "（已被用户中断）"
        return final, calls, texts

    def _generate(self, user_text: str, injection: Injection, on_delta=None,
                  tool_notes: str = "") -> str:
        """单轮生成。流式路径（on_delta 给出）在每个增量边界检查 interrupt_event：
        置位即抛 TurnInterrupted（携带已收到的 answer 增量拼成的部分文本）——
        异常从回调层抛出、经适配层自然向上传播（其合同不捕获回调异常，零改动）；
        非流式路径（chat 整段）没有增量边界，无法中途打断，保持原样。
        """
        prompt = build_prompt(self.system_prompt, injection.context, user_text,
                              self.history, tool_notes=tool_notes)
        self._turn_thinking = ""     # 本轮流次思考缓冲：新一轮从零开始
        self._delta_sink = on_delta  # _capture 的转发目标（本轮有效回调）
        out = None
        if self.use_llm:
            if on_delta is not None:
                # 流式路径：增量先过工具标记过滤器（无标记时逐字透传），再经
                # _capture 留底/转发；返回值仍是完整 answer
                filt = ToolStreamFilter(self._capture)
                partial: list[str] = []   # 中断时的部分文本 = 已下发的 answer 增量

                def _guarded(kind: str, t: str):
                    # 中断守卫（Batch-A）：检查先于下发——事件置位即抛，当前增量
                    # 不再进 UI/缓冲；适配层只看到回调抛异常，感知不到守卫存在
                    if self.interrupt_event.is_set():
                        raise TurnInterrupted("".join(partial))
                    if kind == "answer":
                        partial.append(t)
                    filt(kind, t)

                out = llm.chat_stream(prompt, system=self.system_prompt,
                                      temperature=0.2, on_delta=_guarded)
                filt.flush()
            else:
                out = llm.chat(prompt, system=self.system_prompt)
        if out is None and self.responder is not None:
            out = self.responder(prompt)
        if out:
            return out
        self.stats["offline"] += 1
        return settings.AGENT_OFFLINE_REPLY

    def _capture(self, kind: str, text: str):
        """chat_stream 的内部 on_delta 适配：thinking 增量累积到本轮流次缓冲，
        reset（此前增量作废）清空缓冲；全部增量原样转发给用户回调，返回值透传。"""
        if kind == "thinking":
            self._turn_thinking += text
        elif kind == "reset":
            self._turn_thinking = ""
        return self._delta_sink(kind, text) if self._delta_sink is not None else None

    # ---------- 统计 ----------
    def _bump(self, record: dict) -> None:
        self.stats["turns"] += 1
        self.stats["restatement_skipped"] += record.get("restatement_skipped", 0)
        for side in ("user", "assistant"):
            action = record.get(side, {}).get("action")
            if action == "ingested":
                self.stats["ingested"] += 1
            elif action == "working_only":
                self.stats["working_only"] += 1
