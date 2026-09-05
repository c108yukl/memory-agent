"""TUI：记忆系统的交互式终端界面（纯标准库，零第三方依赖）。

入口：python main.py tui（或 python -m memagent.tui）

实现要点：
- 全屏重绘 + 帧比较：内容不变不写屏，无闪烁
- 宽度感知排版：全角字符按 2 列计算，截断/填充不错位；
  结构字符只用 ASCII（| - >），避免"宽度歧义字符"在不同终端错位
- 键盘跨平台：Windows 走 msvcrt，POSIX 走 termios + select
- 慢操作（LLM 写入/巩固）先绘制"处理中"帧再阻塞执行；
  LLM 可用性探测放后台线程，避免启动卡顿
- 界面只做展示与调度，业务全部复用 pipeline/检索/巩固等现有模块
- 命令注册表（_COMMANDS）单表驱动各屏 footer 与 ? //help 帮助浮层——
  键位提示与真实按键分支漂移在结构上不可能再发生；
- 生成中双击 Esc 中断本轮（TurnInterrupted，部分回答保留照常录入）；
  flash 单槽 toast 5 秒自动消失；退出时打印会话统计 epilogue
"""
from __future__ import annotations

import os
import shutil
import sys
import threading
import time
import unicodedata

from memagent import settings
from memagent.adapters import llm
from memagent.adapters.llm import diag as llm_diag
from memagent.consolidation import consolidate
from memagent.consolidation.conflict_resolver import resolve_conflict
from memagent.forgetting import run_forgetting
from memagent.learning import SpacedRepetition
from memagent.maintenance import rebuild_fts
from memagent.pipeline import ingest_event
from memagent.reports import build_health_report, write_health_report
from memagent.retrieval import inject_provenance, provenance_suffix, retrieve
from memagent.storage import SqliteStore


# ---------------------------------------------------------------------------
# 宽度感知排版
# ---------------------------------------------------------------------------

def wlen(s: str) -> int:
    """可视宽度：全角(F/W)按 2 列，组合记号 0 列，其余 1 列。"""
    n = 0
    for ch in s:
        if unicodedata.combining(ch):
            continue
        n += 2 if unicodedata.east_asian_width(ch) in "FW" else 1
    return n


def wtrunc(s: str, width: int) -> str:
    """按可视宽度截断（保证不超出终端列数导致换行）。"""
    out, n = [], 0
    for ch in s:
        cw = 0 if unicodedata.combining(ch) else (2 if unicodedata.east_asian_width(ch) in "FW" else 1)
        if n + cw > width:
            break
        out.append(ch)
        n += cw
    return "".join(out)


def wpad(s: str, width: int) -> str:
    """按可视宽度右侧补空格到恰好 width 列。"""
    return s + " " * max(0, width - wlen(s))


class Style:
    """ANSI 颜色开关（非 TTY 或 NO_COLOR 时全部退化为空串）。"""

    def __init__(self):
        self.enabled = False

    def _c(self, code: str) -> str:
        return code if self.enabled else ""

    @property
    def reset(self): return self._c("\x1b[0m")
    @property
    def dim(self): return self._c("\x1b[2m")
    @property
    def rev(self): return self._c("\x1b[7m")
    @property
    def red(self): return self._c("\x1b[91m")
    @property
    def green(self): return self._c("\x1b[92m")
    @property
    def yellow(self): return self._c("\x1b[93m")
    @property
    def cyan(self): return self._c("\x1b[96m")


S = Style()


def seg(text: str, width: int, style: str = "") -> str:
    """截断 + 补齐到恰好 width 列，可整体套样式。"""
    t = wpad(wtrunc(text, width), width)
    return f"{style}{t}{S.reset}" if style else t


# ---------------------------------------------------------------------------
# V1.5 检索命中渲染（纯函数，供单测直测）
# ---------------------------------------------------------------------------

TAG_W = 20  # 标签列固定宽：恰容最宽组合 "[低置信][试用期 2/3]"（8+12 列）


def hit_tags(hit) -> str:
    """命中的标签：working -> [当下]；不确定 -> [低置信]；试用期 -> [试用期 n/N]；
    联想 -> [联想]；技能 -> [技能]。

    低置信与试用期可叠加（试用期直接命中常因分数打折而低置信），联想/技能不与
    试用期同现（试用期只由直接命中计数，联想带出走的是独立元数据）。working 是
    「看见的」而非「想起的」，豁免自信心标注，只有 [当下]。E8 技能命中豁免
    低置信（带出跟随触发词而非相关度），只有 [技能]。V1.7 P1：试用期命中带
    转正进度（hit_count 在检索写回前取样，即「这是它第 n 次被想起之前」的值）。
    """
    if hit.kind == "working":
        return "[当下]"
    parts = []
    if hit.meta.get("uncertain"):
        parts.append("[低置信]")
    if hit.meta.get("probation"):
        parts.append(f"[试用期 {hit.meta.get('hit_count', 0)}/{settings.PROMOTE_MIN_HITS}]")
    if hit.meta.get("associated"):
        parts.append("[联想]")
    if hit.meta.get("skill"):
        parts.append("[技能]")
    return "".join(parts)


def has_uncertain(hits) -> bool:
    """是否存在应示警的低置信命中（working 豁免，与 CLI 口径一致）。"""
    return any(h.kind != "working" and h.meta.get("uncertain") for h in hits or [])


def fmt_consolidate_report(r: dict, secs: float) -> list[str]:
    """巩固结果行（纯函数，与 CLI cmd_consolidate 同源数据不同排版）：
    NREM 轮数 / 蒸馏与摘要数 / REM 联想明细 / 今日回忆清单（唤回/遗忘 + 下次间隔）。
    结构字符只用 ASCII 与窄字符（✓✗），遵守 TUI 的宽度歧义字符纪律。
    """
    rem = r.get("rem_associations", [])
    lines = [f"睡眠巩固完成 (NREMx{r.get('nrem_rounds', 0)} + REM, {secs:.1f}s): "
             f"聚类={r['clusters']}, 蒸馏事实={r['distilled_facts']}, "
             f"摘要替代={r['summarized']}, REM联想={len(rem)}, REM写入={r.get('rem_facts', 0)}"]
    for p in rem[:5]:
        lines.append(f"  [REM] [{p['entities'][0]}] x [{p['entities'][1]}]"
                     f" (事实 #{p['facts'][0]} ~ #{p['facts'][1]}, 激活 {p['strength']})")
    if len(rem) > 5:
        lines.append(f"  ...另有 {len(rem) - 5} 条 REM 联想")
    due = r.get("due_reviews", [])
    if not due:
        lines.append("今日回忆清单: 无到期条目")
        return lines
    lines.append(f"今日回忆清单 (SM-2 重演): {len(due)} 条")
    for d in due[:10]:
        mark = "✓唤回" if d["recalled"] else "✗遗忘"
        lines.append(f"  [{mark}] #{d['fact_id']} [{d['entity']}] {d['relation']} = {d['value']}"
                     f" (下次间隔 {d['interval_days']} 天)")
    if len(due) > 10:
        lines.append(f"  ...另有 {len(due) - 10} 条到期事实")
    return lines


def _clamp_scroll(sel: int, scroll: int, view_h: int) -> int:
    if sel < scroll:
        scroll = sel
    elif sel >= scroll + view_h:
        scroll = sel - view_h + 1
    return max(0, scroll)


def _epilogue(stats: dict | None, secs: float) -> str:
    """退出 epilogue（纯函数，便于单测）：会话统计打到主屏回滚区。

    文案与 CLI cmd_agent 的退出统计逐字同款（main.py），追加本次运行秒数；
    stats 取 agent_loop.stats（TUI 会话从未对话过则空表按 0 计）。
    """
    s = stats or {}
    return (f"\n本轮会话: {s.get('turns', 0)} 轮，长期写入 {s.get('ingested', 0)} 次，"
            f"仅工作记忆 {s.get('working_only', 0)} 次，"
            f"复述拦截 {s.get('restatement_skipped', 0)} 次，"
            f"工具调用 {s.get('tools', 0)} 次，运行 {secs:.1f}s")


# ---------------------------------------------------------------------------
# 终端与键盘
# ---------------------------------------------------------------------------

_SPECIAL_KEYS = {"\r": "enter", "\n": "enter", "\t": "tab", "\x7f": "backspace",
                 "\x08": "backspace", "\x1b": "esc", "\x03": "ctrl-c", "\x11": "ctrl-q"}

_WIN_SCAN = {"H": "up", "P": "down", "K": "left", "M": "right", "G": "home", "O": "end",
             "I": "pgup", "Q": "pgdn", "S": "delete", "R": "insert"}
_CSI_FINAL = {"A": "up", "B": "down", "C": "right", "D": "left", "H": "home", "F": "end",
              "Z": "backtab"}
_CSI_TILDE = {"1": "home", "2": "insert", "3": "delete", "4": "end",
              "5": "pgup", "6": "pgdn", "7": "home", "8": "end"}


def _norm_key(ch: str) -> str:
    return _SPECIAL_KEYS.get(ch, ch)


class Terminal:
    """备用屏 + 光标管理的统一生命周期（离开时无条件恢复原始终端）。"""

    def __init__(self):
        self.is_win = os.name == "nt"
        self._old_termios = None
        self._fd = -1

    def __enter__(self):
        S.enabled = sys.stdout.isatty() and os.environ.get("NO_COLOR") is None
        if self.is_win:
            self._vt_enable()
        else:
            import termios
            import tty
            self._fd = sys.stdin.fileno()
            self._old_termios = termios.tcgetattr(self._fd)
            tty.setcbreak(self._fd)
        self._write("\x1b[?1049h\x1b[?25l")
        return self

    def __exit__(self, *exc):
        try:
            self._write("\x1b[?25h\x1b[?1049l\x1b[0m")
        finally:
            if not self.is_win and self._old_termios is not None:
                import termios
                termios.tcsetattr(self._fd, termios.TCSADRAIN, self._old_termios)
        return False

    def cooked_read(self, echo: bool = True) -> str | None:
        """行编辑模式读一行（P2-4 深度实测的输入修复）。

        为什么必须换读法：TUI 的逐字符裸读（getwch）与中文输入法组词天然冲突——
        组词期的拼音字母会原样漏进输入（实测「wa'd'w」式泄漏）、Del/数字编辑在
        组词态行为错乱且无回显。行模式（ENABLE_LINE_INPUT|ECHO|PROCESSED）把
        编辑交给控制台托管：输入法组词、Del/方向键/Backspace、候选框定位全部
        原生正确——cmd/PowerShell 的中文输入体验即来源于此。

        - Windows：保存当前输入模式 → 置行编辑三标志（关 VT 输入让控制台托管）→
          sys.stdin.readline()（回车返回；Ctrl+C 中断视为取消返回 None）→ 恢复原模式；
        - POSIX：恢复进入 TUI 前保存的 cooked termios → readline → 回到 cbreak。
        调用方负责光标已在输入行（draw 的 cursor 定位）与重绘（dirty）。
        """
        if self.is_win:
            import ctypes
            import msvcrt
            k32 = ctypes.windll.kernel32
            hin = k32.GetStdHandle(-10)
            mode = ctypes.c_uint32()
            saved = None
            if k32.GetConsoleMode(hin, ctypes.byref(mode)):
                saved = mode.value
                flags = 0x0001 | 0x0004 | (0x0002 if echo else 0)
                k32.SetConsoleMode(hin, flags)  # 行输入|回显(echo 可关)|已处理
            try:
                sys.stdout.flush()
                line = sys.stdin.readline()
                return None if line == "" else line.rstrip("\r\n")
            except KeyboardInterrupt:
                return None
            finally:
                if saved is not None:
                    k32.SetConsoleMode(hin, saved)
                msvcrt.getwch if False else None  # noqa: B018（保持 msvcrt 导入语义清晰）
        else:
            import termios
            if self._old_termios is None:
                return sys.stdin.readline()
            attrs = termios.tcgetattr(self._fd)
            if not echo:
                attrs[3] &= ~termios.ECHO
                termios.tcsetattr(self._fd, termios.TCSADRAIN, attrs)
            try:
                sys.stdout.flush()
                line = sys.stdin.readline()
                return None if line == "" else line.rstrip("\r\n")
            except KeyboardInterrupt:
                return None
            finally:
                import tty
                tty.setcbreak(self._fd)

    @staticmethod
    def _write(s: str) -> None:
        sys.stdout.write(s)
        sys.stdout.flush()

    @staticmethod
    def _vt_enable() -> None:
        """Windows 10+ 启用 VT 处理（失败则退化渲染，仍可运行）。"""
        try:
            import ctypes
            k32 = ctypes.windll.kernel32
            handle = k32.GetStdHandle(-11)
            mode = ctypes.c_uint32()
            if k32.GetConsoleMode(handle, ctypes.byref(mode)):
                k32.SetConsoleMode(handle, mode.value | 0x0004)
        except Exception:
            pass


class Keys:
    """非阻塞读键（带超时），统一归一化为键名或单字符。"""

    def __init__(self, is_win: bool):
        self.is_win = is_win
        if not is_win:
            self._fd = sys.stdin.fileno()

    def get(self, timeout: float) -> str | None:
        return self._get_win(timeout) if self.is_win else self._get_posix(timeout)

    def _get_win(self, timeout: float) -> str | None:
        import msvcrt
        deadline = time.monotonic() + timeout
        while True:
            if msvcrt.kbhit():
                ch = msvcrt.getwch()
                if ch in ("\x00", "\xe0"):  # 功能键/方向键两段式前缀
                    return _WIN_SCAN.get(msvcrt.getwch())
                return _norm_key(ch)
            if time.monotonic() >= deadline:
                return None
            time.sleep(0.02)

    def _get_posix(self, timeout: float) -> str | None:
        import select
        ready, _, _ = select.select([self._fd], [], [], timeout)
        if not ready:
            return None
        ch = sys.stdin.read(1)
        if ch != "\x1b":
            return _norm_key(ch)
        ready, _, _ = select.select([self._fd], [], [], 0.05)
        if not ready:  # 裸 Esc
            return "esc"
        c2 = sys.stdin.read(1)
        if c2 == "[":
            c3 = sys.stdin.read(1)
            if c3 in _CSI_TILDE:
                sys.stdin.read(1)  # 吃掉结尾 '~'
                return _CSI_TILDE[c3]
            return _CSI_FINAL.get(c3, "?")
        if c2 == "O":
            return _CSI_FINAL.get(sys.stdin.read(1), "?")
        return "esc"


class InputState:
    """单行文本编辑（全角字符与光标移动均可）。"""

    def __init__(self, text: str = ""):
        self.text = text
        self.pos = len(text)

    def insert(self, ch: str) -> None:
        self.text = self.text[:self.pos] + ch + self.text[self.pos:]
        self.pos += 1

    def backspace(self) -> None:
        if self.pos > 0:
            self.text = self.text[:self.pos - 1] + self.text[self.pos:]
            self.pos -= 1

    def delete(self) -> None:
        if self.pos < len(self.text):
            self.text = self.text[:self.pos] + self.text[self.pos + 1:]

    def left(self) -> None:
        self.pos = max(0, self.pos - 1)

    def right(self) -> None:
        self.pos = min(len(self.text), self.pos + 1)

    def home(self) -> None:
        self.pos = 0

    def end(self) -> None:
        self.pos = len(self.text)

    def clear(self) -> None:
        self.text, self.pos = "", 0

    def feed(self, key: str) -> bool:
        """输入态按键分发；返回 False 表示不是编辑键（由调用方继续处理）。"""
        if key == "left":
            self.left()
        elif key == "right":
            self.right()
        elif key == "home":
            self.home()
        elif key == "end":
            self.end()
        elif key == "backspace":
            self.backspace()
        elif key == "delete":
            self.delete()
        elif len(key) == 1 and key.isprintable():
            self.insert(key)
        else:
            return False
        return True


def input_line(inp: InputState, prompt: str, width: int) -> tuple[str, int]:
    """渲染输入行；超宽时以光标为锚点滚动窗口。返回 (行, 光标列偏移)。"""
    avail = width - wlen(prompt) - 1
    if avail < 4:
        return seg(prompt + inp.text, width), wlen(prompt)
    start = 0
    while wlen(inp.text[start:inp.pos]) > avail - 1 and start < inp.pos:
        start += 1
    shown = wtrunc(inp.text[start:], avail)
    col = wlen(prompt) + wlen(inp.text[start:inp.pos])
    return seg(prompt, wlen(prompt)) + seg(shown, avail), col


class _Quit(Exception):
    pass


# ---------------------------------------------------------------------------
# 应用主体
# ---------------------------------------------------------------------------

# 屏序 V1.8：Agent 第一（启动即对话）、设置第二（t 连通测试）、总览第三；
# 数字键 1-9 对应前九屏，"0" 补第 10 屏（报告）
SCR_AGENT, SCR_SETTINGS, SCR_OVERVIEW, SCR_ADD, SCR_RETRIEVE, SCR_MEM, \
    SCR_CONFLICTS, SCR_REVIEW, SCR_MAINTAIN, SCR_REPORT = range(10)

SCREENS = ["Agent", "设置", "总览", "写入", "检索", "记忆", "冲突", "复习", "整理", "报告"]

# 单槽 toast 与双击 Esc 的两个时间常量（秒）：前者由主循环轮询清空，
# 后者是「再按一次 Esc 中断生成」的确认窗口
FLASH_TTL = 5.0
_GUIDE_MSG = "Tab/数字键切页，q 或 Ctrl+C 退出"   # 消息行常驻引导语（flash 过期回落）
ESC_WINDOW = 5.0

# ---------------------------------------------------------------------------
# 命令注册表：各屏 footer 与 ? / /help 帮助浮层的单一真相源（Batch-A P0-1）
# ---------------------------------------------------------------------------
# key 与 handle_key 收到的键名一致（如 "a"/"enter"/"esc"/"t"，文本命令带 "/"）；
# desc 中文描述；hidden=True 不进 footer、只在帮助浮层出现（文本命令/帮助键）。
# check（可选）：静态断言用的源码字面量（缺省按 "/" 拆 key 逐个核对）——单测
# 用它核对注册表每一项在对应 _key_* 处理器源码中确有分支，footer 与按键分支
# 的漂移（如旧版冲突页 footer 漏 "b 误报共存"）在结构上不可能再发生。
_COMMANDS: dict[int, list[dict]] = {
    SCR_AGENT: [
        {"key": "enter", "desc": "发送"},
        {"key": "esc", "desc": "清空/双击中断"},
        {"key": "up/down", "desc": "滚动转录"},
        {"key": "tab", "desc": "切页"},
        {"key": "/debug", "desc": "展开/收起上一轮注入", "hidden": True},
        {"key": "/deep", "desc": "切换深搜检索", "hidden": True, "check": ("DEEP_COMMAND",)},
        {"key": "/help", "desc": "命令帮助", "hidden": True},
    ],
    SCR_SETTINGS: [
        {"key": "up/down", "desc": "选行"},
        {"key": "enter", "desc": "编辑/切换"},
        {"key": "a", "desc": "应用设置"},
        {"key": "s", "desc": "存 .env"},
        {"key": "r", "desc": "探测"},
        {"key": "t", "desc": "连通测试"},
        {"key": "esc", "desc": "取消编辑"},
        {"key": "tab", "desc": "切页"},
        {"key": "?", "desc": "帮助"},
    ],
    SCR_OVERVIEW: [
        {"key": "r", "desc": "刷新/探测"},
        {"key": "tab", "desc": "切页"},
        {"key": "q", "desc": "退出"},
        {"key": "?", "desc": "帮助"},
    ],
    SCR_ADD: [
        {"key": "enter", "desc": "写入"},
        {"key": "t", "desc": "切类型"},
        {"key": "l", "desc": "LLM 开关"},
        {"key": "up/down", "desc": "切输入框"},
        {"key": "esc", "desc": "清空"},
        {"key": "tab", "desc": "切页"},
    ],
    SCR_RETRIEVE: [
        {"key": "enter", "desc": "检索"},
        {"key": "+/-", "desc": "条数"},
        {"key": "up/down", "desc": "滚动"},
        {"key": "esc", "desc": "清空"},
        {"key": "tab", "desc": "切页"},
    ],
    SCR_MEM: [
        {"key": "up/down", "desc": "选择"},
        {"key": "t", "desc": "换仓储"},
        {"key": "s", "desc": "换状态"},
        {"key": "enter", "desc": "详情"},
        {"key": "r", "desc": "刷新"},
        {"key": "?", "desc": "帮助"},
    ],
    SCR_CONFLICTS: [
        {"key": "up/down", "desc": "选择"},
        {"key": "a", "desc": "采纳新版"},
        {"key": "k", "desc": "保留旧版"},
        {"key": "b", "desc": "误报共存"},
        {"key": "r", "desc": "刷新"},
        {"key": "?", "desc": "帮助"},
    ],
    SCR_REVIEW: [
        {"key": "up/down", "desc": "选择"},
        {"key": "0-5", "desc": "打分复习", "check": ("isdigit",)},
        {"key": "r", "desc": "刷新"},
        {"key": "?", "desc": "帮助"},
    ],
    SCR_MAINTAIN: [
        {"key": "up/down", "desc": "选择"},
        {"key": "enter", "desc": "执行"},
        {"key": "esc", "desc": "取消确认"},
        {"key": "?", "desc": "帮助"},
    ],
    SCR_REPORT: [
        {"key": "up/down", "desc": "滚动"},
        {"key": "r", "desc": "重新生成"},
        {"key": "w", "desc": "保存文件"},
        {"key": "?", "desc": "帮助"},
    ],
}

_ADD_TYPES = [
    ("observation", "普通事件"),
    ("preference_statement", "偏好声明（保底入库，可自动取代旧偏好）"),
    ("instruction", "指令（保底入库）"),
    ("identity_statement", "身份声明（保底入库）"),
    ("experience", "AI/任务经验（绿色通道保底, LFU 短半衰 7 天）"),
    ("env_statement", "环境状态（绿色通道, 最新即正确取代旧认知）"),
]
# V1.6 E8 绿色通道类型：事实键由 task_context 确定性构造（entity=任务域），
# 同域新写入自动取代旧版——所以写入页要多收一个「任务域」输入框
_ADD_GREEN = {"experience", "env_statement"}

_MAINTAIN_ITEMS = [
    ("睡眠巩固", "聚类摘要 + 语义蒸馏 + 冲突处理，完成后自动生成健康报告"),
    ("主动遗忘", "强度重算 -> 归档 -> 摘要降级 -> 硬删; 试用期超 30 天无人问津一并归档（需二次确认）"),
    ("保存健康报告", "写入 data/reports/health-<时间>.md"),
    ("重建 FTS 索引（预览）", "dry-run 统计条数；执行请用 CLI: rebuild-fts --apply"),
]

_EP_STATUSES = ["active", "summarized", "archived"]
_SE_STATUSES = ["active", "pending", "superseded"]

# 设置屏字段表：(标签, settings 属性名, 是否文本编辑, 是否机密)。
# 机密字段（云端 Key）任何形态都不明文显示：未编辑显示掩码，编辑态星号回显；
# 协议行不进文本编辑（Enter 直接在 openai/anthropic 间切换）。
_SET_FIELDS = [
    ("本地 URL", "LM_STUDIO_URL", True, False),
    ("云端协议", "CLOUD_PROTOCOL", False, False),
    ("云端 URL", "CLOUD_URL", True, False),
    ("云端 Key", "CLOUD_API_KEY", True, True),
    ("云端模型", "CLOUD_MODEL", True, False),
    ("备用模型", "CLOUD_FALLBACK_MODELS", True, False),
    ("嵌入 URL", "CLOUD_EMBED_URL", True, False),
    ("嵌入模型", "CLOUD_EMBED_MODEL", True, False),
    ("工具轮上限", "AGENT_TOOL_MAX_ROUNDS", True, False),
]
# 设置屏写 .env 用的环境变量名（与 settings.py 的 os.environ.get 键一一对应）
_SET_ENV_KEYS = {
    "LM_STUDIO_URL": "MEMAGENT_LLM_URL",
    "CLOUD_PROTOCOL": "MEMAGENT_CLOUD_PROTOCOL",
    "CLOUD_URL": "MEMAGENT_CLOUD_URL",
    "CLOUD_API_KEY": "MEMAGENT_CLOUD_API_KEY",
    "CLOUD_MODEL": "MEMAGENT_CLOUD_MODEL",
    "CLOUD_FALLBACK_MODELS": "MEMAGENT_CLOUD_FALLBACK_MODELS",
    "CLOUD_EMBED_URL": "MEMAGENT_CLOUD_EMBED_URL",
    "CLOUD_EMBED_MODEL": "MEMAGENT_CLOUD_EMBED_MODEL",
    "AGENT_TOOL_MAX_ROUNDS": "MEMAGENT_TOOL_MAX_ROUNDS",
}


def _mask_secret(key: str) -> str:
    """机密值掩码：sk-***abcd 式（头 3 尾 4）；过短或为空不泄露任何内容。"""
    if not key:
        return "(未配置)"
    if len(key) <= 8:
        return "***"
    return key[:3] + "***" + key[-4:]


class TuiApp:
    SIDEBAR_W = 12

    def __init__(self, store: SqliteStore, auto_probe: bool = True):
        self.store = store
        self.screen = SCR_AGENT
        self.dirty = True
        self.busy: str | None = None
        self.msg: tuple[str, str] = ("info", _GUIDE_MSG)
        self._draw_lock = threading.Lock()   # 帧写串行化（Batch-A 防撕裂）
        # flash 单槽 toast：记时间戳由主循环轮询过期清空（None=常驻引导语不消失）；
        # clock 可注入替身（单测确定性），toast 过期与双击 Esc 窗口共用同一时钟
        self.msg_at: float | None = None
        self.clock = time.monotonic
        self._last_frame = ""
        self._last_size: tuple[int, int] = (0, 0)
        self.op_secs = 0.0
        self.probe: dict = {"state": "idle", "available": False, "provider": ""}
        # 连通性诊断（设置屏 t 触发，后台线程跑 run_diag，绝不阻塞 UI）
        self.diag: dict = {"state": "idle", "results": []}
        # ? / /help 帮助浮层（模态：打开时吃掉除 esc/? 外的全部按键）
        self.help_open = False

        # 写入
        self.add_input = InputState()
        self.add_ctx = InputState()          # 绿色通道类型的任务域（事实键）
        self.add_focus = 0                   # 0=内容 1=任务域（上/下箭头切换）
        self.add_type_idx = 0
        self.add_use_llm = True
        self.add_result: list[str] = []
        # 检索
        self.ret_input = InputState()
        self.ret_topk = settings.RETRIEVE_TOP_K
        self.ret_hits = None
        self.ret_scroll = 0
        # 记忆浏览
        self.mem_tab = 0
        self.ep_status_idx = 0
        self.se_status_idx = 0
        self.mem_sel = 0
        self.mem_scroll = 0
        self.mem_detail = False
        self.mem_ep: list = []
        self.mem_se: list = []
        self.mem_sk: list = []
        # 冲突 / 复习 / 整理 / 报告
        self.conf_rows: list = []
        self.conf_sel = 0
        self.conf_scroll = 0
        self.rep_due: list = []
        self.rep_sel = 0
        self.rep_scroll = 0
        self.rep_plans: list = []
        self.mt_sel = 0
        self.mt_confirm = False
        self.mt_result: list[str] = []
        self.report_lines: list[str] | None = None
        self.report_scroll = 0
        # 设置（编辑缓冲与 _SET_FIELDS 一一对应；A 应用才写回 settings 模块属性）
        self.set_sel = 0
        self.set_editing = False
        self.set_buf: list[str] = [getattr(settings, f[1]) for f in _SET_FIELDS]
        self.set_input = InputState()
        # Agent 对话（循环惰性创建，见 _agent_loop；输入行/转录滚动/注入详情展开）
        self.agent_loop = None
        self.agent_input = InputState()
        self.agent_scroll = 0          # 转录从底部往回看的行数（0=跟随最新一轮）
        self.agent_debug = False       # d 展开/收起上一轮注入片段
        self.agent_busy = False        # 流式生成中（后台线程跑 loop.turn，Enter 守卫）
        # 进行中流式轮的增量缓冲（worker 线程经 on_delta 写入，主循环轮询重绘）
        self.agent_stream = {"thinking": "", "answer": ""}
        # 双击 Esc 中断生成（Batch-A）：共享事件在流式增量边界触发 TurnInterrupted；
        # _esc_at 是首按 Esc 的时刻（0=无待确认的首按），窗口内再按才真正置位
        self.agent_interrupt = threading.Event()
        self._esc_at = 0.0

        self.reload_all()
        if auto_probe:
            self.start_probe()

    # ---- 数据加载 ----

    def reload_all(self) -> None:
        self.reload_memories()
        self.reload_conflicts()
        self.reload_review()

    def reload_memories(self) -> None:
        self.mem_ep = self.store.episodic.fetch(status=_EP_STATUSES[self.ep_status_idx], limit=500)
        self.mem_se = self.store.semantic.fetch(status=_SE_STATUSES[self.se_status_idx], limit=2000)
        self.mem_sk = self.store.procedural.fetch()
        self.mem_sel = min(self.mem_sel, max(0, self._mem_len() - 1))

    def reload_conflicts(self) -> None:
        self.conf_rows = self.store.conflicts.fetch_all(status="pending")
        self.conf_sel = min(self.conf_sel, max(0, len(self.conf_rows) - 1))

    def reload_review(self) -> None:
        sr = SpacedRepetition(self.store.conn)
        rows = []
        for mem_id in sr.due(limit=100):
            fact = self.store.semantic.get(mem_id)
            text = f"[{fact.entity}] {fact.relation} = {fact.value}" if fact else "(记忆已删除)"
            rows.append((mem_id, text))
        self.rep_due = rows
        self.rep_plans = sr.status()
        self.rep_sel = min(self.rep_sel, max(0, len(self.rep_due) - 1))

    def start_probe(self) -> None:
        """后台线程探测 LLM 三通道（探测有超时，不阻塞界面）。"""
        if self.probe["state"] == "probing":
            return

        def worker():
            try:
                avail = llm.llm_available()
                provider = llm.active_provider() if avail else "offline(规则打分 + 哈希嵌入)"
                self.probe = {"state": "ok", "available": avail, "provider": provider}
            except Exception as e:  # 探测失败按离线处理，不影响使用
                self.probe = {"state": "ok", "available": False,
                              "provider": f"探测失败: {e}"}
            self.dirty = True

        self.probe = {"state": "probing", "available": False, "provider": ""}
        threading.Thread(target=worker, daemon=True).start()

    def start_diag(self) -> None:
        """后台线程跑五项通道连通性诊断（与 CLI `main.py test` 同一套 run_diag）。

        诊断含最长 20s 的网络请求，绝不阻塞 UI：运行态只显示提示行，结果一次到齐
        后整帧刷新；运行中重复按 t 忽略（防并发线程重复撞网），完成后可重跑。
        """
        if self.diag["state"] == "running":
            return

        def worker():
            try:
                results = llm_diag.run_diag()
            except Exception as e:  # run_diag 契约上不抛，此处仅兜底不让线程静默死掉
                results = [{"name": "诊断", "ok": False, "level": "error",
                            "detail": f"诊断失败: {e}", "latency_ms": 0.0}]
            self.diag = {"state": "done", "results": results}
            self.dirty = True

        self.diag = {"state": "running", "results": []}
        threading.Thread(target=worker, daemon=True).start()

    def _stats(self) -> dict:
        try:
            ep = dict(self.store.conn.execute(
                "SELECT status, COUNT(*) FROM episodic GROUP BY status").fetchall())
            se = dict(self.store.conn.execute(
                "SELECT status, COUNT(*) FROM semantic GROUP BY status").fetchall())
            pending = len(self.store.conflicts.fetch_all(status="pending"))
            skills = self.store.conn.execute("SELECT COUNT(*) FROM procedural").fetchone()[0]
            due = len(SpacedRepetition(self.store.conn).due(limit=10 ** 6))
            # 工作记忆是随本 store 实例存活的会话级 scratchpad——TUI 单进程会话里
            # 它是真实存在的（CLI 逐命令进程则每次为空），头部如实报数
            working = len(self.store.working)
            return {"ep": ep, "se": se, "pending": pending, "skills": skills,
                    "due": due, "working": working}
        except Exception as e:
            return {"error": str(e)}

    # ---- 反馈与阻塞执行 ----

    def flash(self, text: str, kind: str = "info") -> None:
        self.msg = (kind, text)
        self.msg_at = self.clock()   # 单槽 toast：同一槽新消息顶掉旧消息并重新计时
        self.dirty = True

    def _expire_flash(self) -> None:
        """flash 自动消失（Batch-A）：消息存在且超过 FLASH_TTL 秒即清空置脏。

        主循环每轮轮询调用（keys.get 的 0.4s 间隙）；渲染不变——清空后下一帧
        message_line 自然变空白。msg_at=None 的启动引导语永不过期。
        """
        if self.msg_at is not None and self.clock() - self.msg_at >= FLASH_TTL:
            self.msg = ("info", _GUIDE_MSG)   # 过期回落常驻引导语（单槽 toast 语义）
            self.msg_at = None
            self.dirty = True

    def run_blocking(self, label: str, fn):
        """慢操作：先画一帧"处理中"再阻塞执行（LLM 调用可达数秒）。"""
        self.busy = label
        self.force_draw()
        t0 = time.monotonic()
        try:
            return fn()
        except Exception as e:
            self.flash(f"操作失败: {e}", "err")
            return None
        finally:
            self.op_secs = time.monotonic() - t0
            self.busy = None
            self.dirty = True

    def force_draw(self) -> None:
        self._last_frame = ""
        self.draw(self._last_size if self._last_size != (0, 0) else shutil.get_terminal_size())
        self.dirty = False

    # ---- 主循环 ----

    def run(self) -> int:
        t0 = self.clock()
        try:
            with Terminal() as term:
                self.term = term
                keys = Keys(term.is_win)
                self._last_size = shutil.get_terminal_size()
                try:
                    while True:
                        size = shutil.get_terminal_size()
                        if self.dirty or size != self._last_size:
                            self.draw(size)
                            self._last_size = size
                            self.dirty = False
                        key = keys.get(0.4)
                        self._expire_flash()   # 轮询间隙检查 toast 是否到期
                        if key is not None:
                            self.handle_key(key)
                except (_Quit, KeyboardInterrupt):
                    pass
        finally:
            # 退出 epilogue（Batch-A）：已离开备用屏，统计打印到主屏回滚区，
            # 文案与 CLI cmd_agent 的退出统计同款（数据源同为本侧 agent_loop.stats）
            stats = self.agent_loop.stats if self.agent_loop is not None else {}
            print(_epilogue(stats, self.clock() - t0))
        return 0

    # ---- 按键分发 ----

    def handle_key(self, key: str) -> None:
        self.dirty = True
        if key in ("ctrl-c", "ctrl-q"):
            raise _Quit()
        if self.help_open:
            # 帮助浮层是模态（Batch-A）：esc/? 关闭，其余键一律忽略——含 tab/q，
            # 防浮层底下误切屏误退出；Ctrl+C 仍可退出（上面已处理）
            if key in ("esc", "?"):
                self.help_open = False
            return
        if key == "tab":
            self._switch((self.screen + 1) % len(SCREENS))
            return
        if key == "backtab":
            self._switch((self.screen - 1) % len(SCREENS))
            return
        scr = self.screen
        # 输入屏：所有可打印字符进输入框，先于全局快捷键
        if scr == SCR_ADD:
            self._key_add(key)
            return
        if scr == SCR_RETRIEVE:
            self._key_retrieve(key)
            return
        if scr == SCR_AGENT:
            self._key_agent(key)
            return
        if scr == SCR_SETTINGS and self.set_editing:
            self._key_settings(key)  # 编辑态同样先吃全部按键（否则编辑时 q/数字误触全局键）
            return
        # ? 帮助浮层：仅非输入屏（写入/检索/Agent 的可打印字符必须进输入框——
        # 逐字符兼容线；Agent 屏 ? 会进输入框，改走 /help 文本命令）
        if key == "?":
            self.help_open = True
            return
        # 非输入屏：数字键快速切页（复习页数字用于打分）；"0" 补第 10 屏（报告）
        if len(key) == 1 and key.isdigit() and scr != SCR_REVIEW:
            idx = len(SCREENS) - 1 if key == "0" else int(key) - 1
            if 0 <= idx < len(SCREENS):
                self._switch(idx)
            return
        if key == "q":
            raise _Quit()
        {SCR_OVERVIEW: self._key_overview,
         SCR_MEM: self._key_mem,
         SCR_CONFLICTS: self._key_conflicts,
         SCR_REVIEW: self._key_review,
         SCR_MAINTAIN: self._key_maintain,
         SCR_REPORT: self._key_report,
         SCR_SETTINGS: self._key_settings}[scr](key)

    def _switch(self, idx: int) -> None:
        self.screen = idx
        self.mt_confirm = False
        if idx == SCR_MEM:
            self.reload_memories()
        elif idx == SCR_CONFLICTS:
            self.reload_conflicts()
        elif idx == SCR_REVIEW:
            self.reload_review()
        elif idx == SCR_REPORT:
            self.report_lines = None  # 进入即重新生成
            self.report_scroll = 0
        elif idx == SCR_SETTINGS:
            self.set_editing = False
            self._settings_sync_buf()  # 每次进屏从 settings 重新同步编辑缓冲

    @staticmethod
    def _scroll_step(key: str, sel: int, n: int, view_h: int) -> tuple[int, bool]:
        """通用列表移动；返回 (新 sel, 是否处理)。"""
        if key == "up":
            return max(0, sel - 1), True
        if key == "down":
            return min(max(0, n - 1), sel + 1), True
        if key == "pgup":
            return max(0, sel - view_h), True
        if key == "pgdn":
            return min(max(0, n - 1), sel + view_h), True
        if key == "home":
            return 0, True
        if key == "end":
            return max(0, n - 1), True
        return sel, False

    # ---- 各屏按键 ----

    def _key_overview(self, key: str) -> None:
        if key == "r":
            self.start_probe()
            self.flash("正在后台探测 LLM 三通道...")

    def _add_green(self) -> bool:
        """当前写入类型是否走绿色通道（需要任务域输入框）。"""
        return _ADD_TYPES[self.add_type_idx][0] in _ADD_GREEN

    def _key_add(self, key: str) -> None:
        line = self._maybe_line_edit(key)
        if line is not None:
            self.add_input.text = line
            self.add_input.pos = len(line)
            self.handle_key("enter")   # 行编辑完成（回车结束）→ 直接送入写入
            return
        if key == "enter":
            content = self.add_input.text.strip()
            if not content:
                self.flash("内容为空", "warn")
                return
            type_name = _ADD_TYPES[self.add_type_idx][0]
            task_context = self.add_ctx.text.strip() if self._add_green() else ""

            def op():
                return ingest_event(self.store, content, source="user",
                                    type=type_name, task_context=task_context,
                                    use_llm=self.add_use_llm)

            mode = "LLM" if self.add_use_llm else "离线规则"
            r = self.run_blocking(f"写入中 ({mode})...", op)
            if r is None:
                return
            self.add_result = self._fmt_ingest(r)
            self.add_input.clear()
            self.add_ctx.clear()
            self.add_focus = 0  # 写完回内容框，连续录入不用再按上箭头
            self.flash(f"已处理 ({self.op_secs:.1f}s): {wtrunc(content, 24)}", "ok")
            self.reload_memories()
        elif key == "esc":
            self.add_input.clear()
            self.add_ctx.clear()
            self.add_result = []
            self.flash("输入已清空")
        elif key == "t":
            self.add_type_idx = (self.add_type_idx + 1) % len(_ADD_TYPES)
            if not self._add_green():
                self.add_focus = 0
        elif key == "l":
            self.add_use_llm = not self.add_use_llm
        elif key == "down" and self._add_green() and self.add_focus == 0:
            self.add_focus = 1
        elif key == "up" and self.add_focus == 1:
            self.add_focus = 0
        else:
            (self.add_ctx if self.add_focus else self.add_input).feed(key)

    def _key_retrieve(self, key: str) -> None:
        line = self._maybe_line_edit(key)
        if line is not None:
            self.ret_input.text = line
            self.ret_input.pos = len(line)
            self.handle_key("enter")   # 行编辑完成 → 直接检索
            return
        if key == "enter":
            query = self.ret_input.text.strip()
            if not query:
                self.flash("请输入问题", "warn")
                return
            hits = self.run_blocking("检索中...", lambda: retrieve(
                self.store, query, top_k=self.ret_topk))
            if hits is None:
                return
            self.ret_hits = hits
            self.ret_scroll = 0
            self.flash(f"命中 {len(hits)} 条 ({self.op_secs:.1f}s)", "ok")
        elif key == "esc":
            self.ret_input.clear()
            self.ret_hits = None
        elif key in ("+", "="):
            self.ret_topk = min(50, self.ret_topk + 5)
        elif key == "-":
            self.ret_topk = max(1, self.ret_topk - 5)
        elif key in ("up", "down", "pgup", "pgdn", "home", "end"):
            n = len(self.ret_hits or [])
            step = {"up": -1, "down": 1, "pgup": -8, "pgdn": 8, "home": -10 ** 9, "end": 10 ** 9}[key]
            self.ret_scroll = max(0, min(max(0, n - 8), self.ret_scroll + step))
        else:
            self.ret_input.feed(key)

    def _key_mem(self, key: str) -> None:
        view_h = 10  # 与绘制时的列表可视行数保持一致即可近似翻页
        if key == "t":
            self.mem_tab = (self.mem_tab + 1) % 3
            self.mem_sel = self.mem_scroll = 0
            self.mem_detail = False
            self.reload_memories()
        elif key == "s":
            if self.mem_tab == 0:
                self.ep_status_idx = (self.ep_status_idx + 1) % len(_EP_STATUSES)
            elif self.mem_tab == 1:
                self.se_status_idx = (self.se_status_idx + 1) % len(_SE_STATUSES)
            else:
                return
            self.mem_sel = self.mem_scroll = 0
            self.reload_memories()
        elif key == "r":
            self.reload_memories()
            self.flash("已刷新")
        elif key == "enter":
            if self._mem_len():
                self.mem_detail = not self.mem_detail
        else:
            sel, handled = self._scroll_step(key, self.mem_sel, self._mem_len(), view_h)
            if handled:
                self.mem_sel = sel

    def _key_conflicts(self, key: str) -> None:
        if key == "r":
            self.reload_conflicts()
            self.flash("已刷新")
        elif key in ("a", "A"):
            self._resolve_conflict("accept-new")
        elif key in ("k", "K"):
            self._resolve_conflict("keep-old")
        elif key in ("b", "B"):
            self._resolve_conflict("both")
        else:
            sel, handled = self._scroll_step(key, self.conf_sel, len(self.conf_rows), 4)
            if handled:
                self.conf_sel = sel

    def _resolve_conflict(self, resolution: str) -> None:
        if not self.conf_rows:
            return
        row = self.conf_rows[self.conf_sel]

        def op():
            return resolve_conflict(self.store, row["conflict_id"], resolution)

        res = self.run_blocking("裁决中...", op)
        if res is None:
            self.flash(f"冲突 #{row['conflict_id']} 不存在或已裁决", "warn")
            return
        self.flash(f"已裁决 #{row['conflict_id']}: {resolution}", "ok")
        self.reload_conflicts()
        self.reload_memories()

    def _key_review(self, key: str) -> None:
        if key == "r":
            self.reload_review()
            self.flash("已刷新")
        elif len(key) == 1 and key.isdigit():
            self._do_review(int(key))
        else:
            sel, handled = self._scroll_step(key, self.rep_sel, len(self.rep_due), 8)
            if handled:
                self.rep_sel = sel

    def _do_review(self, quality: int) -> None:
        if not self.rep_due:
            self.flash("今日没有到期条目", "warn")
            return
        mem_id = self.rep_due[self.rep_sel][0]

        def op():
            return SpacedRepetition(self.store.conn).review(mem_id, quality)

        info = self.run_blocking("记录复习...", op)
        if info is None:
            return
        self.flash(f"#{mem_id} 复习完成: 下次 {info['due_at']} "
                   f"(间隔 {info['interval_days']} 天)", "ok")
        self.reload_review()

    def _key_maintain(self, key: str) -> None:
        if key == "esc":
            self.mt_confirm = False
            return
        sel, handled = self._scroll_step(key, self.mt_sel, len(_MAINTAIN_ITEMS), 6)
        if handled:
            self.mt_sel = sel
            self.mt_confirm = False
            return
        if key == "enter":
            if self.mt_sel == 1 and not self.mt_confirm:  # 主动遗忘需二次确认
                self.mt_confirm = True
                return
            self._run_maintain()

    def _run_maintain(self) -> None:
        idx = self.mt_sel
        self.mt_confirm = False
        if idx == 0:
            def op():
                return consolidate(self.store)

            r = self.run_blocking("睡眠巩固中 (LLM 梳理可达数十秒)...", op)
            if r is None:
                return
            # V1.5：NREM 轮数 / 蒸馏与摘要数 / REM 联想明细 / 今日回忆清单
            self.mt_result = fmt_consolidate_report(r, self.op_secs)
            try:  # 巩固后自动生成健康报告（应用层副作用，失败不阻断）
                self.mt_result.append(f"健康报告: {write_health_report(self.store)}")
            except Exception as e:
                self.mt_result.append(f"健康报告生成失败: {e!r}")
        elif idx == 1:
            def op():
                return run_forgetting(self.store)

            r = self.run_blocking("主动遗忘中...", op)
            if r is None:
                return
            self.mt_result = [f"遗忘完成: 归档={r['archived']}, 硬删={r['deleted']},",
                              f"活跃={r['episodic_active']}"]
        elif idx == 2:
            def op():
                return write_health_report(self.store)

            path = self.run_blocking("生成健康报告...", op)
            if path is None:
                return
            self.mt_result = [f"健康报告已保存: {path}"]
        else:
            def op():
                return rebuild_fts(self.store, dry_run=True)

            r = self.run_blocking("统计 FTS 条数...", op)
            if r is None:
                return
            self.mt_result = [f"[预览] 将重建 FTS 索引: 情景 {r['episodic']} / 语义 {r['semantic']} 条",
                              "执行请在 CLI 运行: python main.py rebuild-fts --apply"]
            self.flash("预览完成（未写库）")
            return
        self.flash(f"完成 ({self.op_secs:.1f}s)", "ok")
        self.reload_all()
        self.report_lines = None

    def _key_report(self, key: str) -> None:
        if key == "r":
            self.report_lines = None
            self.flash("重新生成中...")
            self.force_draw()
            self.report_lines = build_health_report(self.store).splitlines()
            self.flash("健康报告已重新生成", "ok")
            return
        if key == "w":
            path = self.run_blocking("保存健康报告...", lambda: write_health_report(self.store))
            if path:
                self.flash(f"已保存: {path}", "ok")
            return
        n = len(self.report_lines or [])
        sel, handled = self._scroll_step(key, self.report_scroll, max(n, 1), 12)
        if handled:
            self.report_scroll = sel

    # 设置 ------------------------------------------------------------------

    def _settings_sync_buf(self) -> None:
        """编辑缓冲 <- 当前 settings 模块属性（进屏时刷新，缓冲是唯一编辑草稿）。

        缓冲恒为字符串：settings 属性可能是 int（工具轮上限），而缓冲是文本编辑
        草稿、进编辑态要喂 InputState（len()）——int 不 str 化在编辑入口必炸
        （实测翻车：应用后数值字段进编辑态 TypeError）。"""
        self.set_buf = [str(getattr(settings, f[1])) for f in _SET_FIELDS]

    def _key_settings(self, key: str) -> None:
        if self.set_editing:
            if key == "enter":
                f = _SET_FIELDS[self.set_sel]
                text = self.set_input.text.strip()
                # 机密行空输入确认=保持原值（防误清 Key；换 Key 直接输入新值）
                if not (f[3] and not text):
                    self.set_buf[self.set_sel] = text
                self.set_editing = False
                self.flash("已修改，按 A 应用后运行时生效")
            elif key == "esc":
                self.set_editing = False  # 取消编辑：缓冲保持编辑前的值
                self.flash("已取消编辑")
            else:
                self.set_input.feed(key)
            return
        if key == "enter":
            f = _SET_FIELDS[self.set_sel]
            term = getattr(self, "term", None)
            if f[2] and term is not None and term.is_win:
                # Windows 实况：行编辑读值（输入法/Del 由控制台托管；机密行关回显）。
                # 关键顺序：先以编辑态重绘把真实光标定位到字段行，再交控制台行读——
                # 否则回显打在屏幕上一次光标停留的位置（实测翻车：按 Enter 只见换行）
                self.set_editing = True
                self.set_input = InputState("")
                self.force_draw()
                line = term.cooked_read(echo=not f[3])
                self.set_editing = False
                self.dirty = True
                if line is None:
                    self.flash("已取消编辑")
                    return
                if not line.strip() and not f[3]:
                    self.flash("空输入 = 保留原值")
                    return
                self.set_buf[self.set_sel] = line.strip() if not f[3] else line.rstrip()
                if f[1] == "AGENT_TOOL_MAX_ROUNDS":
                    try:
                        n = max(0, min(20, int(self.set_buf[self.set_sel] or 3)))
                        self.set_buf[self.set_sel] = str(n)
                    except ValueError:
                        self.flash(f"「{line.strip()}」不是 0~20 的整数，未修改", "err")
                        return
                self.flash("已修改，按 A 应用后运行时生效")
                return
            if not f[2]:  # 协议行：Enter 直接在 openai/anthropic 间切换，不进文本编辑
                self.set_buf[self.set_sel] = ("anthropic"
                                              if self.set_buf[self.set_sel] == "openai"
                                              else "openai")
            else:
                self.set_editing = True
                # 机密行进编辑时输入框清空（不明文回显旧 Key）；其余行带入当前值
                self.set_input = InputState(
                    "" if f[3] else str(self.set_buf[self.set_sel]))
        elif key in ("a", "A"):
            # 应用：编辑缓冲写回 settings 模块属性并强制重探——运行时立即生效
            for i, (_, attr, _, _) in enumerate(_SET_FIELDS):
                val = self.set_buf[i]
                if attr == "AGENT_TOOL_MAX_ROUNDS":
                    # 数值字段（P2-4 深度使用反馈）：坏输入拒绝整批应用
                    # （宁拒绝不静默回默认——静默会让「改了没生效」无处排查）
                    try:
                        val = max(0, min(20, int(str(val).strip() or 3)))
                    except (TypeError, ValueError):
                        self.flash(f"工具轮上限「{val}」不是 0~20 的整数，整批未应用",
                                   "err")
                        return
                    val = int(val)
                    self.set_buf[i] = str(val)   # 缓冲恒为字符串（编辑态喂 InputState）
                    # 已惰性创建的对话循环同步生效（不用重进 Agent 屏）
                    if self.agent_loop is not None:
                        self.agent_loop.tool_max_rounds = val
                setattr(settings, attr, val)
            llm.reset_probes()
            self.flash("设置已应用（运行时生效），r 重新探测可验证连通", "ok")
        elif key in ("s", "S"):
            self._settings_save_env()
        elif key == "r":
            self.start_probe()
            self.flash("正在后台探测 LLM 三通道...")
        elif key == "t":
            # 与 CLI `main.py test` 共用同一套诊断（后台线程，见 start_diag）
            self.start_diag()
            self.flash("正在后台跑连通性测试（五项），完成后显示在本页")
        else:
            sel, handled = self._scroll_step(key, self.set_sel, len(_SET_FIELDS), 10)
            if handled:
                self.set_sel = sel

    def _settings_save_env(self) -> None:
        """合并保存到项目根 .env（settings._load_dotenv 启动时加载的正是该路径）。

        流程：先把原 .env 复制为 .env.bak 备份，再逐行合并——本屏管理的
        MEMAGENT_* 键原位替换值（没有则追加到末尾），注释、空行与其他变量
        一律原样保留。空值键写成 KEY=（忠实回存运行时状态：空值经
        os.environ.get 得空串，云端通道因无 Key 自动禁用，语义一致）。
        .env 只在下次进程启动时生效（已存在的环境变量优先）；本进程立即生效靠 A 应用。
        """
        path = os.path.join(settings.BASE_DIR, ".env")
        try:
            old: list[str] = []
            if os.path.exists(path):
                shutil.copyfile(path, path + ".bak")  # 写前备份
                with open(path, "r", encoding="utf-8") as f:
                    old = f.read().splitlines()
            # str() 包一层：数值字段（工具轮上限）应用后缓冲/模块属性可能是 int，
            # .env 是纯文本格式，写入前统一字符串化（P2-4 工具轮设置引入）
            values = {_SET_ENV_KEYS[f[1]]: str(self.set_buf[i]).strip()
                      for i, f in enumerate(_SET_FIELDS)}
            seen: set[str] = set()
            out: list[str] = []
            for ln in old:
                stripped = ln.strip()
                name = stripped.split("=", 1)[0].strip() if "=" in stripped else ""
                if name in values and not stripped.startswith("#"):
                    out.append(f"{name}={values[name]}")
                    seen.add(name)
                else:
                    out.append(ln)
            out += [f"{k}={v}" for k, v in values.items() if k not in seen]
            with open(path, "w", encoding="utf-8", newline="\n") as f:
                f.write("\n".join(out) + ("\n" if out else ""))
            self.flash("已保存到 .env（原文件备份为 .env.bak），重启进程后加载", "ok")
        except OSError as e:
            self.flash(f"保存 .env 失败: {e}", "err")

    # Agent 对话 --------------------------------------------------------------

    def _agent_loop(self):
        """惰性创建内置对话循环（与 main.py cmd_agent 同款：局部导入防循环依赖）。"""
        if self.agent_loop is None:
            from memagent.agent import AgentLoop
            self.agent_loop = AgentLoop(self.store, task_context="tui-agent",
                                        use_llm=True)
        # 双击 Esc 中断（Batch-A）：统一指向 TUI 侧的共享事件——键位处理置位、
        # worker 线程收尾复位，同一信号源（loop 自身默认事件被此处覆盖）
        self.agent_loop.interrupt_event = self.agent_interrupt
        return self.agent_loop

    def _key_agent(self, key: str) -> None:
        loop = self._agent_loop()
        if key == "enter":
            self._submit_agent(self.agent_input.text.strip())
            return
        # Windows 实况：可打印字符进控制台行编辑（输入法组词/Del/数字编辑由控制台
        # 原生托管——逐字符裸读与输入法组词冲突是实测翻车主因）。种子字符先
        # ungetwch 回队列，行编辑第一次读取即含它。POSIX/测试（无 term）保持逐字符。
        if key == "esc":
            if self.agent_busy:
                # 生成中 Esc 双击中断（Batch-A）：首按只提示，ESC_WINDOW 秒内再按
                # 才置位中断事件（防误触清空/打断）；超时重新计第一次。worker 在
                # 流式增量边界抛 TurnInterrupted，本轮以部分回答收尾（busy 复位）。
                now = self.clock()
                if self._esc_at and now - self._esc_at <= ESC_WINDOW:
                    self.agent_interrupt.set()
                    self._esc_at = 0.0
                    self.flash("正在中断生成...", "warn")
                else:
                    self._esc_at = now
                    self.flash("再按一次 Esc 中断生成", "warn")
                return
            self.agent_input.clear()
            return
        if key in ("up", "down", "pgup", "pgdn", "home", "end"):
            # agent_scroll = 从底部往回看的行数；可视高度在 _body_agent 里钳制
            step = {"up": 1, "down": -1, "pgup": 8, "pgdn": -8,
                    "home": 10 ** 9, "end": -10 ** 9}[key]
            self.agent_scroll = max(0, self.agent_scroll + step)
            return
        line = self._maybe_line_edit(key)
        if line is not None:
            self._submit_agent(line.strip())
            return
        self.agent_input.feed(key)

    def _maybe_line_edit(self, key: str) -> str | None:
        """可打印字符 + Windows 实况（self.term 在场）→ 控制台行编辑读一行。

        返回读到的行（调用方把它当作「用户已敲完整行并回车」处理）；
        None = 不是行编辑场景（非可打印/POSIX/测试无 term），调用方走旧按键路径。
        输入法组词、Del/数字编辑由控制台原生托管（cmd/PowerShell 同源体验）。"""
        term = getattr(self, "term", None)
        if term is None or not term.is_win:
            return None
        if len(key) != 1 or not key.isprintable():
            return None
        import msvcrt
        msvcrt.ungetwch(key)
        line = self._focus_line_read()
        return line if line and line.strip() else None

    def _focus_line_read(self) -> str | None:
        """聚焦当前输入行做控制台行编辑：重绘定位光标 → 行模式读一行 → 标脏。
        返回 None = 用户取消（Ctrl+C）或空行。读完后全帧重绘恢复 TUI 视图。"""
        term = getattr(self, "term", None)
        if term is None:
            return None
        self.force_draw()      # 光标定位到输入行（draw 对 cursor 行显示硬件光标）
        try:
            line = term.cooked_read()
            if line is not None:
                time.sleep(0.05)   # 回车回显落定 + 输入法收尾，再交全帧重绘
            return line
        finally:
            self.dirty = True  # 读完整帧重绘，恢复 TUI 视图

    def _submit_agent(self, text: str) -> None:
        loop = self._agent_loop()
        if not text:
            self.flash("请输入内容", "warn")
            return
        if self.agent_busy:
            self.flash("上一轮还在生成，请稍候", "warn")
            return
        if text == "/debug":
            # 显示命令（TUI 本地处理，不进 loop）：等价旧 d 键的注入详情开关
            self.agent_debug = not self.agent_debug
            self.agent_input.clear()
            self.flash(f"注入详情显示: {'开' if self.agent_debug else '关'}", "ok")
            return
        if text == "/help":
            # 帮助浮层（TUI 本地处理，不进 loop，Batch-A）：Agent 屏 ? 是打印字符
            # 会进输入框——文本命令是帮助浮层在本屏的唯一入口，浮层同款
            self.help_open = True
            self.agent_input.clear()
            self.dirty = True
            return
        self.agent_busy = True
        self.agent_interrupt.clear()   # 新一轮从干净信号开始（上轮收尾已清，这里兜底）
        self.agent_stream = {"thinking": "", "answer": ""}
        self.agent_input.clear()
        self.agent_scroll = 0   # 新一轮回到底部，跟随最新转录
        self.agent_debug = False
        if getattr(self, "term", None) is not None:
            # 提交瞬间的输入法收尾（候选框拆除/组词串清理）与紧随其后的全帧重绘
            # 在 conPTY 上会交错成叠帧（实测录屏：新旧两帧同行叠印）——让终端先
            # 完成收尾再重绘，50ms 用户无感
            time.sleep(0.05)
        if loop.use_llm:
            # 真实 LLM 生成可达数秒：放后台线程流式推进（仿 start_probe：
            # daemon 线程 + dirty 轮询重绘）。回调只改缓冲与 dirty，不碰 UI。
            self.force_draw()
            threading.Thread(target=self._agent_turn_worker,
                             args=(loop, text), daemon=True).start()
        else:
            # 离线确定性路径（use_llm=False，测试/评测同款）：毫秒级完成，
            # 同步跑完保证转录即时可见（单元测试由此获得确定性）。
            self._agent_turn_worker(loop, text)

    def _agent_on_delta(self, kind: str, text: str) -> None:
        """流式增量回调（worker 线程调用）：只改缓冲与 dirty，主循环轮询重绘。

        tool/tool_result 是 V1.7.3 工具调用的实时反馈：拼进 live 视图的 answer
        缓冲即可（转录数据源是 loop.history，turn 完成后只显示干净的
        assistant_text，工具行只在生成过程中可见）。
        Batch-A 双击 Esc 中断：事件置位即从回调层抛 TurnInterrupted（loop 模块
        异常，经 chat_stream 调用栈向上传播到 loop 的工具轮捕获——适配层零改动，
        on_delta 异常本就不被适配层捕获）。loop 层的增量守卫先行拦截，这里是
        第二道防线（守卫与用户回调之间理论同线程无窗，防直调场景漏拦）。
        """
        if self.agent_interrupt.is_set():
            from memagent.agent.loop import TurnInterrupted
            raise TurnInterrupted()
        if kind == "reset":        # 此前增量作废：清空缓冲
            self.agent_stream = {"thinking": "", "answer": ""}
        elif kind == "tool":
            self.agent_stream["answer"] += f"\n[工具] {text}"
        elif kind == "tool_result":
            self.agent_stream["answer"] += "（结果已回填）"
        elif kind in ("thinking", "answer"):
            self.agent_stream[kind] += text
        self.dirty = True

    def _agent_turn_worker(self, loop, text: str) -> None:
        """一轮对话的线程体：真实 LLM 走后台线程，离线路径同步调用（_key_agent）。

        收尾（含异常）复位 busy、清空流式缓冲（完成的 Turn 已由 turn() 归档进
        loop.history——转录唯一数据源），并置 dirty 触发主循环重绘。
        Batch-A：接好中断事件后跑 turn——用户双击 Esc 置位事件，流式增量边界
        抛 TurnInterrupted，由 loop 捕获并以部分回答照常录入（turn.interrupted
        标记）；事件是一次性信号，收尾必须 clear，防影响下一轮。
        """
        loop.interrupt_event = self.agent_interrupt   # 接线（测试可绕过 _agent_loop 直建 loop）
        t0 = time.monotonic()
        try:
            turn = loop.turn(text, on_delta=self._agent_on_delta)
            if turn.interrupted:
                self.flash(f"已中断，已生成的部分回答已保留 "
                           f"({time.monotonic() - t0:.1f}s)", "warn")
            else:
                self.flash(f"第 {loop.stats['turns']} 轮完成 "
                           f"({time.monotonic() - t0:.1f}s)，"
                           f"注入 {len(turn.injection.injected_texts)} 条"
                           + (f"，工具 {len(turn.tool_calls)} 次" if turn.tool_calls else ""),
                           "ok")
        except Exception as e:
            self.flash(f"生成失败: {e}", "err")
        finally:
            self.agent_interrupt.clear()
            self._esc_at = 0.0    # 双击窗口一并复位：中断后 Esc 回到「清空输入」语义
            self.agent_busy = False
            self.agent_stream = {"thinking": "", "answer": ""}
            self.dirty = True

    # ---- 绘制 ----

    def draw(self, size: tuple[int, int]) -> None:
        cols, rows = size
        if cols < 40 or rows < 10:
            sys.stdout.write(f"\x1b[2J\x1b[1;1H{wtrunc('终端太小，请至少 40x10', cols)}")
            sys.stdout.flush()
            return
        self.stats = self._stats()
        sb_w = self.SIDEBAR_W
        body_w = cols - sb_w - 1
        body_h = rows - 3

        lines = [self._header(cols)]
        sidebar = self._sidebar(sb_w, body_h)
        body, cursor = self._body(body_w, body_h)
        for i in range(body_h):
            sb = sidebar[i] if i < len(sidebar) else " " * sb_w
            bl = body[i] if i < len(body) else " " * body_w
            lines.append(sb + S.dim + "|" + S.reset + bl)
        lines.append(self._message_line(cols))
        lines.append(self._footer(cols))

        if self.busy:
            mid = len(lines) // 2
            text = f">> {self.busy} <<"
            lines[mid] = S.rev + wpad(" " * max(0, (cols - wlen(text)) // 2) + text, cols) + S.reset

        out = [f"\x1b[{r};1H{txt}\x1b[0m\x1b[K" for r, txt in enumerate(lines, 1)]
        out.append("\x1b[J")
        if cursor:
            out.append(f"\x1b[{cursor[0] + 2};{sb_w + 2 + cursor[1]}H\x1b[?25h")
        else:
            out.append("\x1b[?25l")
        frame = "".join(out)
        if frame != self._last_frame:
            with self._draw_lock:   # 帧写串行化：杜绝任何路径的并发写撕裂
                sys.stdout.write(frame)
                sys.stdout.flush()
                self._last_frame = frame

    def _header(self, w: int) -> str:
        left = " memory-agent TUI"
        c = getattr(self, "stats", {})
        if "error" in c:
            right = " 数据库异常"
        else:
            if self.probe["state"] != "ok":
                llm_txt = "LLM 探测中..."
            else:
                llm_txt = "LLM 可用" if self.probe["available"] else "LLM 离线"
            right = (f" {llm_txt} | 情景{c['ep'].get('active', 0)}"
                     f" 语义{c['se'].get('active', 0)} 试用{c['se'].get('pending', 0)}"
                     f" 冲突{c['pending']}"
                     f" 技能{c['skills']} 到期{c['due']} 工作{c['working']}")
        pad = max(1, w - wlen(left) - wlen(right))
        return S.rev + wpad(left + " " * pad + right, w) + S.reset

    def _sidebar(self, w: int, h: int) -> list[str]:
        lines = []
        for i, name in enumerate(SCREENS):
            badge = ""
            c = getattr(self, "stats", {})
            if "error" not in c:
                if i == SCR_CONFLICTS and c["pending"]:
                    badge = f" {c['pending']}"
                elif i == SCR_REVIEW and c["due"]:
                    badge = f" {c['due']}"
            text = f" {i + 1} {name}{badge}"
            if i == self.screen:
                lines.append(S.rev + wpad(wtrunc(text, w), w) + S.reset)
            else:
                lines.append(wpad(wtrunc(text, w), w))
        lines.append(wpad("", w))  # 与正文首行对齐的空行
        return lines

    def _message_line(self, w: int) -> str:
        kind, text = self.msg
        style, prefix = {"info": ("", " "), "ok": (S.green, " OK "),
                         "warn": (S.yellow, " ! "), "err": (S.red, " X ")}[kind]
        padded = wpad(wtrunc(prefix + text, w), w)
        return f"{style}{padded}{S.reset}" if style else padded

    def _footer(self, w: int) -> str:
        # footer 由命令注册表生成（单一真相源，Batch-A P0-1）：键 描述 用 · 串联，
        # hidden 项跳过（只进帮助浮层）——手写拼接的键位漂移从此在结构上不可能
        items = [f"{c['key']} {c['desc']}" for c in _COMMANDS.get(self.screen, [])
                 if not c.get("hidden")]
        left = " " + " · ".join(items)
        right = "Ctrl+C 退出 "
        pad = max(1, w - wlen(left) - wlen(right))
        return S.dim + wpad(wtrunc(left + " " * pad + right, w), w) + S.reset

    def _body(self, w: int, h: int) -> tuple[list[str], tuple[int, int] | None]:
        if self.help_open:   # 帮助浮层：整屏覆盖当前屏（Batch-A）
            lines, cursor = self._body_help(w, h)
        else:
            builder = {SCR_OVERVIEW: self._body_overview, SCR_ADD: self._body_add,
                       SCR_RETRIEVE: self._body_retrieve, SCR_MEM: self._body_mem,
                       SCR_CONFLICTS: self._body_conflicts, SCR_REVIEW: self._body_review,
                       SCR_MAINTAIN: self._body_maintain, SCR_REPORT: self._body_report,
                       SCR_SETTINGS: self._body_settings, SCR_AGENT: self._body_agent}[self.screen]
            lines, cursor = builder(w, h)
        lines = [ln if ln else " " * w for ln in lines[:h]]
        while len(lines) < h:
            lines.append(" " * w)
        return lines, cursor

    @staticmethod
    def _rule(w: int) -> str:
        return seg("  " + "-" * max(0, w - 4), w, S.dim)

    @staticmethod
    def _kv_rows(pairs: list[tuple[str, str]], w: int, label_w: int = 14) -> list[str]:
        out = []
        for label, value in pairs:
            out.append(seg(f" {label}", label_w) + seg(str(value), w - label_w))
        return out

    # 帮助浮层 ---------------------------------------------------------------

    def _body_help(self, w: int, h: int) -> tuple[list[str], tuple[int, int] | None]:
        """? / /help 帮助浮层（Batch-A）：当前屏命令表 + 关闭提示，整屏覆盖。

        数据源唯一：_COMMANDS 注册表（含 hidden 项——文本命令与帮助键只在
        浮层出现，footer 不显示）；键名列按可视宽度对齐，正文交给 seg 截断。
        """
        cmds = _COMMANDS.get(self.screen, [])
        key_w = max((wlen(c["key"]) for c in cmds), default=0)
        lines = [seg(f"  帮助 · {SCREENS[self.screen]}屏命令", w, S.cyan),
                 self._rule(w)]
        for c in cmds:
            lines.append(seg("  " + wpad(c["key"], key_w) + "  " + c["desc"], w))
        lines.append("")
        lines.append(seg("  esc 或 ? 关闭帮助", w, S.dim))
        return lines, None

    # 总览 ------------------------------------------------------------------

    def _body_overview(self, w: int, h: int) -> tuple[list[str], tuple[int, int] | None]:
        from collections import Counter

        lines = [""]
        st = self.probe
        if st["state"] != "ok":
            lines += self._kv_rows([("LLM", "探测中...")], w)
        else:
            lines += self._kv_rows(
                [("LLM", "可用" if st["available"] else "不可用 (规则打分 + 哈希嵌入)")], w)
            lines += self._kv_rows([("服务商", st["provider"])], w)
        c = getattr(self, "stats", None) or self._stats()  # 直调 body（测试）时兜底
        if "error" in c:
            lines += self._kv_rows([("数据库", f"查询失败: {c['error']}")], w)
        else:
            ep, se = c["ep"], c["se"]
            lines += self._kv_rows([
                ("情景记忆", f"活跃 {ep.get('active', 0)} | 摘要替代 {ep.get('summarized', 0)}"
                            f" | 归档 {ep.get('archived', 0)}"),
                ("语义记忆", f"活跃 {se.get('active', 0)} | 试用期+待裁 {se.get('pending', 0)}"
                            f" | 已取代 {se.get('superseded', 0)}"),
                ("程序记忆", f"{c['skills']} 条"),
                ("工作记忆", f"{c['working']} 条 (会话级, 本会话内可检索, 退出蒸发)"),
                ("待裁决冲突", f"{c['pending']} 条 (到冲突页裁决)"),
                ("今日到期复习", f"{c['due']} 条 (到复习页打分)"),
            ], w)
            if se.get("pending", 0):  # V1.7 P1: 试用期语义（D1 拆两类，转正规则不同）
                lines += self._kv_rows([
                    ("试用期转正", f"{se.get('pending', 0)} 条 pending: B 类(无冲突)命中达 "
                                  f"{settings.PROMOTE_MIN_HITS} 次自动转正; "
                                  f"A 类(冲突待裁)只能裁决, 到冲突页处理"),
                ], w)
        lines += self._kv_rows([("数据文件", settings.DB_PATH)], w)
        try:
            actions = [r["action"].split("->")[0].split("-")[0]
                       for r in self.store.conn.execute(
                           "SELECT action FROM meta ORDER BY id DESC LIMIT 100").fetchall()]
            top = ", ".join(f"{a} x{n}" for a, n in Counter(actions).most_common(6)) or "无"
            lines += self._kv_rows([("最近审计", top)], w)
        except Exception:
            pass
        lines.append("")
        lines.append(seg("  1 Agent(/deep 深搜) | 2 设置(模型/工具轮/t 测试) | 3 总览 | 0 报告 | Tab 循环切页",
                         w - 2, S.dim))
        lines.append(seg("  数字 1-9/0 直达各屏；启动默认在 Agent 页对话，r 重新探测 LLM 并刷新",
                         w - 2, S.dim))
        return lines, None

    # 写入 ------------------------------------------------------------------

    def _body_add(self, w: int, h: int) -> tuple[list[str], tuple[int, int] | None]:
        type_name, type_desc = _ADD_TYPES[self.add_type_idx]
        green = self._add_green()
        lines = [
            seg(f"  类型 [{self.add_type_idx + 1}/{len(_ADD_TYPES)}] {type_name}  "
                f"({type_desc})  [t 切换]", w, S.cyan),
            seg(f"  LLM 打分: {'开 (云端/本地)' if self.add_use_llm else '关 (纯规则, 离线)'}"
                f"  [l 切换]", w, S.yellow if not self.add_use_llm else ""),
            self._rule(w),
        ]
        cursor = None
        content_line, content_col = input_line(self.add_input, "  写入内容> ", w)
        if not green:
            lines.append(content_line)
            cursor = (len(lines) - 1, content_col)
        else:
            ctx_line, ctx_col = input_line(self.add_ctx, "  任务域/键> ", w)
            if self.add_focus == 0:
                lines.append(content_line)
                cursor = (len(lines) - 1, content_col)
                lines.append(ctx_line)
            else:
                lines.append(content_line)
                lines.append(ctx_line)
                cursor = (len(lines) - 1, ctx_col)
            lines.append(seg("  绿色通道: 任务域即事实键, 同域新经验自动取代旧版"
                             "  [上/下 切输入框]", w, S.dim))
        lines.append(self._rule(w))
        if self.add_result:
            lines.append(seg("  最近一次写入管线:", w, S.dim))
            lines += [seg(f"  {t}", w) for t in self.add_result[: h - len(lines) - 1]]
        else:
            lines.append(seg("  回车写入；例: \"以后给我的回答请保持正式、结构化。\"", w, S.dim))
            lines.append(seg("  显式声明类型 (偏好/指令/身份) 保底入库且可自动取代旧偏好", w, S.dim))
            lines.append(seg("  绿色类型 (经验/环境) 需填任务域; 低置信新事实进试用期,"
                             " 检索命中够数自动转正", w, S.dim))
        return lines, cursor

    # 检索 ------------------------------------------------------------------

    def _body_retrieve(self, w: int, h: int) -> tuple[list[str], tuple[int, int] | None]:
        lines = [""]
        line, col = input_line(self.ret_input, "  检索问题> ", w)
        lines.append(line)
        cursor = (len(lines) - 1, col)
        lines.append(seg(f"  返回条数: {self.ret_topk}  [+/- 调整]   (FTS 关键词 + 向量余弦混合)",
                         w, S.dim))
        lines.append(self._rule(w))
        if self.ret_hits is None:
            lines.append(seg("  输入问题后回车检索，例: \"用户喜欢什么回答风格？\"", w, S.dim))
        elif not self.ret_hits:
            lines.append(seg("  未检索到相关记忆", w, S.yellow))
        else:
            lines.append(seg(f"  命中 {len(self.ret_hits)} 条 (按相关度排序):", w, S.dim))
            if has_uncertain(self.ret_hits):  # E7 低置信表面化：与 CLI 同口径示警
                lines.append(seg("  ⚠ 检索置信度低，以下结果可能不相关 (feeling-of-knowing)",
                                 w, S.yellow))
            for hit in self.ret_hits[self.ret_scroll:]:
                inject_provenance(hit, self.store)  # 与 build_context 同一数据源
                suffix = provenance_suffix(hit)
                text = f"{hit.text} {suffix}" if suffix else hit.text
                lines.append(seg(f"  {hit_tags(hit)}", 2 + TAG_W)
                             + seg(f"({hit.kind}:{hit.score:.2f}) ", 22)
                             + seg(text, w - 22 - 2 - TAG_W))
        return lines, cursor

    # 记忆浏览 -----------------------------------------------------------------

    def _mem_len(self) -> int:
        return [len(self.mem_ep), len(self.mem_se), len(self.mem_sk)][self.mem_tab]

    def _mem_items(self) -> list:
        return [self.mem_ep, self.mem_se, self.mem_sk][self.mem_tab]

    def _body_mem(self, w: int, h: int) -> tuple[list[str], tuple[int, int] | None]:
        tab_names = [f"情景 {_EP_STATUSES[self.ep_status_idx]} {len(self.mem_ep)}",
                     f"语义 {_SE_STATUSES[self.se_status_idx]} {len(self.mem_se)}",
                     f"程序 {len(self.mem_sk)}"]
        cells = [(S.rev if i == self.mem_tab else S.dim) + f" {name} " + S.reset
                 for i, name in enumerate(tab_names)]
        head = "  " + " | ".join(cells)
        vis = wlen("  " + " | ".join(f" {n} " for n in tab_names))
        hint = "   [t 换仓储, s 换状态]"
        if vis + wlen(hint) <= w:
            head += S.dim + hint + S.reset
            vis += wlen(hint)
        head += " " * max(0, w - vis)
        lines = [head]
        lines.append(self._rule(w))

        detail_h = min(11, max(0, (h - 4) // 2)) if self.mem_detail else 0
        list_h = h - 3 - detail_h
        items = self._mem_items()
        self.mem_scroll = _clamp_scroll(self.mem_sel, self.mem_scroll, max(1, list_h))
        for i in range(self.mem_scroll, min(len(items), self.mem_scroll + list_h)):
            item = items[i]
            text = self._fmt_mem_row(item)
            if i == self.mem_sel:
                lines.append(seg("> " + text, w, S.rev))
            else:
                lines.append(seg("  " + text, w))
        if not items:
            lines.append(seg("  (空)", w, S.dim))

        if self.mem_detail and items and self.mem_sel < len(items):
            lines.append(seg("  详情 [Enter 收起]", w, S.cyan))
            lines += self._detail_rows(items[self.mem_sel], w, detail_h)
        return lines, None

    def _fmt_mem_row(self, item) -> str:
        if self.mem_tab == 0:
            m = item
            tag = f"  <摘要自{m.source_ids}>" if m.is_summary else ""
            return f"#{m.id} [{m.strength:.2f}] {m.summary} (访问:{m.access_count}){tag}"
        if self.mem_tab == 1:
            f = item
            status = f" ({f.status})" if self.se_status_idx != 0 else ""
            probation = (f" (试用期 {f.hit_count}/{settings.PROMOTE_MIN_HITS})"
                         if f.status == "pending" else "")
            return f"#{f.id} [{f.entity}] {f.relation} = {f.value}" \
                   f" (conf={f.confidence:.2f}){status}{probation}"
        s = item
        return f"#{s.id} {s.name}  触发:{s.trigger}  策略:{s.policy}" \
               f"  用:{s.usage_count} 成功率:{s.success_rate:.0%}"

    def _detail_rows(self, item, w: int, h: int) -> list[str]:
        if self.mem_tab == 0:
            m = item
            pairs = [("id / 状态", f"#{m.id} / {m.status}"),
                     ("强度 / 重要度", f"{m.strength:.2f} / {m.importance:.2f}")]
            if m.arousal > 0:  # E2 情感痕迹：闪光灯记忆的唤醒度（0 = 平静事件不展示）
                pairs.append(("情感", f"{m.arousal:.2f}"))
            pairs += [("创建 / 最近访问", f"{m.created_at} / {m.last_access_at}"),
                      ("访问次数", str(m.access_count)),
                      ("上下文", m.context or "-"),
                      ("动作 / 结果", f"{m.action or '-'} / {m.outcome or '-'}"),
                      ("溯源", f"摘要自 {m.source_ids}，由 #{m.summarized_by} 生成"
                               if m.is_summary else "-")]
        elif self.mem_tab == 1:
            f = item
            pairs = [("id / 状态", f"#{f.id} / {f.status}"),
                     ("主体", f"[{f.entity}] {f.relation}"),
                     ("值", f.value),
                     ("置信度", f"{f.confidence:.2f}"),
                     ("证据次数", f"x{f.evidence_count} (真实观测续证, 检索不累积)"),
                     ("命中计数", f"x{f.hit_count} (被想起次数; 试用期达 "
                                 f"{settings.PROMOTE_MIN_HITS} 次自动转正, 与证据分账)"),
                     ("生效期", f"{f.valid_from} ~ {f.valid_to or '今'}"),
                     ("被取代", f"#{f.superseded_by}" if f.superseded_by else "-")]
        else:
            s = item
            pairs = [("id", f"#{s.id}"),
                     ("技能名", s.name),
                     ("触发", s.trigger),
                     ("策略", s.policy),
                     ("使用 / 成功率", f"{s.usage_count} / {s.success_rate:.0%}")]
        return [ln for ln in self._kv_rows(pairs, w)][:h]

    # 冲突 ------------------------------------------------------------------

    def _body_conflicts(self, w: int, h: int) -> tuple[list[str], tuple[int, int] | None]:
        lines = [seg(f"  待裁决冲突 ({len(self.conf_rows)} 条)   "
                     f"[a 采纳新版 | k 保留旧版 | b 误报共存 | r 刷新]", w)]
        if not self.conf_rows:
            lines.append(seg("  没有待裁决冲突。低置信冲突写入时会自动挂起在此等待人工裁决。", w, S.dim))
            return lines, None
        flat: list[tuple[int, str, bool]] = []  # (冲突下标, 文本, 是否标题行)
        for i, r in enumerate(self.conf_rows):
            old = self.store.semantic.get(r["old_id"])
            new = self.store.semantic.get(r["new_id"])
            flat.append((i, f"#{r['conflict_id']}  {r['created_at']}", True))
            flat.append((i, f"旧 #{r['old_id']}: {old.value if old else '?'}", False))
            flat.append((i, f"新 #{r['new_id']}: {new.value if new else '?'}", False))
        view_h = h - 2
        scroll = _clamp_scroll(self.conf_sel * 3, self.conf_scroll, max(1, view_h - 2))
        self.conf_scroll = scroll
        shown = flat[scroll: scroll + view_h]
        for idx, text, is_head in shown:
            selected = idx == self.conf_sel
            if is_head:
                prefix = "> " if selected else "  "
                style = S.rev if selected else S.cyan
                lines.append(seg(prefix + text, w, style))
            else:
                lines.append(seg("    " + text, w))
        return lines, None

    # 复习 ------------------------------------------------------------------

    def _body_review(self, w: int, h: int) -> tuple[list[str], tuple[int, int] | None]:
        lines = [seg(f"  今日到期 ({len(self.rep_due)} 条)   [选中后按 0-5 打分 | r 刷新]", w)]
        if not self.rep_due:
            lines.append(seg("  今日没有到期复习 (SM-2 间隔重复)", w, S.dim))
        else:
            view_h = min(len(self.rep_due), max(1, (h - 2) // 2))
            self.rep_scroll = _clamp_scroll(self.rep_sel, self.rep_scroll, view_h)
            for i in range(self.rep_scroll, min(len(self.rep_due), self.rep_scroll + view_h)):
                mem_id, text = self.rep_due[i]
                row = f"  #{mem_id} {text}"
                lines.append(seg(row, w, S.rev if i == self.rep_sel else ""))
        lines.append(seg(f"  复习计划 ({len(self.rep_plans)} 条):", w, S.dim))
        for row in self.rep_plans[: max(1, h - len(lines) - 1)]:
            lines.append(seg(f"  #{row['memory_id']}  间隔={row['interval_days']}天"
                             f"  下次={row['due_at']}", w))
        return lines, None

    # 整理 ------------------------------------------------------------------

    def _body_maintain(self, w: int, h: int) -> tuple[list[str], tuple[int, int] | None]:
        lines = [seg("  记忆整理（均与 CLI 行为一致，可放心执行）", w, S.dim)]
        for i, (title, desc) in enumerate(_MAINTAIN_ITEMS):
            if i == self.mt_sel:
                lines.append(seg(f"> {title}", w, S.rev))
            else:
                lines.append(seg(f"  {title}", w))
            lines.append(seg(f"    {desc}", w, S.dim))
        if self.mt_confirm:
            lines.append(seg("  !! 再次 Enter 确认执行主动遗忘（归档/硬删），Esc 取消", w, S.red))
        if self.mt_result:
            lines.append(self._rule(w))
            lines += [seg(f"  {t}", w) for t in self.mt_result[: h - len(lines) - 1]]
        return lines, None

    # 报告 ------------------------------------------------------------------

    def _body_report(self, w: int, h: int) -> tuple[list[str], tuple[int, int] | None]:
        if self.report_lines is None:
            self.report_lines = build_health_report(self.store).splitlines()
        n = len(self.report_lines)
        self.report_scroll = min(self.report_scroll, max(0, n - 1))
        lines = [seg(f"  记忆健康报告 ({n} 行)   [r 重新生成 | w 保存 | 上/下 滚动]", w)]
        for ln in self.report_lines[self.report_scroll: self.report_scroll + h - 1]:
            lines.append(seg(" " + ln, w))
        return lines, None

    # 设置 ------------------------------------------------------------------

    def _body_settings(self, w: int, h: int) -> tuple[list[str], tuple[int, int] | None]:
        lines = [seg("  模型配置   [A 应用=运行时生效 | S 保存到 .env | r 重新探测]", w, S.dim)]
        lines.append(self._rule(w))
        cursor = None
        # 分组标题（仅渲染层：行选择索引不受影响）——9 个字段一线平铺被实测
        # 反馈为「乱、不清晰」，按职责分四组一眼可辨
        groups = {1: "—— 云端对话通道（协议/地址/密钥/模型）——",
                  6: "—— 嵌入通道（向量化）——",
                  8: "—— 行为（对话循环）——"}
        for i, (label, attr, editable, secret) in enumerate(_SET_FIELDS):
            if i in groups:
                lines.append(seg(f"  {groups[i]}", w, S.dim))
            if i == self.set_sel and self.set_editing:
                prompt = f"  {label}> "
                if secret:  # 机密行：星号回显，任何形态不明文
                    lines.append(seg(prompt + "*" * len(self.set_input.text), w, S.cyan))
                    col = min(wlen(prompt) + self.set_input.pos, w - 1)
                    cursor = (len(lines) - 1, col)
                else:
                    line, col = input_line(self.set_input, prompt, w)
                    lines.append(line)
                    cursor = (len(lines) - 1, col)
            else:
                val = self.set_buf[i]
                shown = _mask_secret(val) if secret else (val if val else "(空)")
                mark = "  *已改(A 生效)" if val != getattr(settings, attr) else ""
                extra = "" if editable else "  [Enter 切换]"
                prefix = "> " if i == self.set_sel else "  "
                lines.append(seg(f"{prefix}{label}: {shown}{extra}{mark}", w,
                                 S.rev if i == self.set_sel else ""))
        lines.append(self._rule(w))
        st = self.probe
        if st["state"] == "ok":
            prov = st["provider"] if st["available"] else "离线 (规则打分 + 哈希嵌入)"
        else:
            prov = "探测中..." if st["state"] == "probing" else "未探测 (r 探测)"
        lines.append(seg(f"  当前探测: {prov}", w, S.dim))
        # 连通性诊断结果（t 触发，后台线程跑 run_diag）：逐行 ✓/!/✗，超出 h 预算截断
        d = self.diag
        budget = max(0, h - len(lines) - 1)  # 保底留 1 行给底部按键提示
        if d["state"] == "running":
            if budget:
                lines.append(seg("  连通性测试中... (网络请求最长 20s/项)", w, S.dim))
        elif d["state"] == "done":
            if budget:
                lines.append(seg("  连通性测试 (t 重跑):", w, S.dim))
                budget -= 1
            for r in d["results"][:budget]:
                mark, style = {"info": ("✓", S.green), "warn": ("!", S.yellow),
                               "error": ("✗", S.red)}[r["level"]]
                lines.append(seg(f"  {mark} {r['name']} ({r['latency_ms'] / 1000:.1f}s): "
                                 f"{r['detail']}", w, style))
        lines.append(seg("  上/下 选行 | Enter 编辑/切换 | Esc 取消 | 带 * 行按 A 应用后生效"
                         "（S 只写 .env，重启进程才加载）", w, S.dim))
        return lines, cursor

    # Agent 对话 --------------------------------------------------------------

    def _body_agent(self, w: int, h: int) -> tuple[list[str], tuple[int, int] | None]:
        loop = self._agent_loop()
        lines = [seg("  Agent 记忆原生对话 (task=tui-agent)   [/debug 展开/收起上一轮注入，/deep 切深搜]", w, S.dim)]
        lines.append(self._rule(w))
        inp_line, col = input_line(self.agent_input, "  对 Agent 说> ", w)
        s = loop.stats
        bottom = [self._rule(w)]
        if self.agent_busy:
            bottom.append(seg("  生成中...（思考 dim 弱化；完成后折叠为「思考 N 字」）", w, S.dim))
        bottom += [inp_line,
                   # B5-d 深搜状态指示：状态区常驻显示当前开关（P1-4 的 /deep 在 loop
                   # 层拦截早已可用，但界面无回显——用户不知道开没开）。放行首而非
                   # 行尾：窄终端（60 列）下行尾会被 seg 截掉，行首永远可见；开启时
                   # 整行换 cyan 提亮——非默认态要能被一眼看出（NO_COLOR/非 TTY 时
                   # 样式退化为空串，文本仍在）。
                   seg(f"  深搜:{'开' if loop.deep else '关'} | 轮次 {s['turns']} | "
                       f"长期写入 {s['ingested']} | 仅工作记忆 {s['working_only']} | "
                       f"复述拦截 {s['restatement_skipped']} | 离线 {s['offline']}",
                       w, S.cyan if loop.deep else S.dim)]
        view_h = max(1, h - len(lines) - len(bottom))
        trans: list[str] = []   # 转录平面化：每轮 3~4 行（你/助手/调试摘要/思考折叠）
        n_history = len(loop.history)
        for i, t in enumerate(loop.history):
            trans.append(seg(f"你: {t.user_text}", w))
            trans.append(seg(f"助手: {t.assistant_text}", w))
            ua = t.record.get("user", {}).get("action", "?")
            aa = t.record.get("assistant", {}).get("action", "?")
            trans.append(seg(f"[注入 {len(t.injection.injected_texts)} 条 | "
                             f"用户:{ua} 助手:{aa}]", w, S.dim))
            if t.thinking:
                trans.append(seg(f"思考 {len(t.thinking)} 字", w, S.dim))
            if self.agent_debug:
                if t.thinking:      # 与注入全文同区：展开时显示全部轮次的思考全文
                    for cl in (t.thinking.splitlines() or [""]):
                        trans.append(seg("  | " + cl, w, S.dim))
                ctx = t.injection.context or "（无相关记忆）"
                for cl in (ctx.splitlines() or [""]):
                    trans.append(seg("  | " + cl, w, S.dim))
        if self.agent_busy:
            # 进行中轮：转录尾部实时显示流式增量。思维链不再只给尾部 200 单行
            # （用户实测「展示不完全」）——显示尾部 4 行，每行交给 seg 按宽截断；
            # 完整思维链在轮次完成后经 /debug 展开
            th = self.agent_stream["thinking"]
            if th:
                tl = [t for t in th.splitlines() if t.strip()] or [th]
                trans.append(seg(f"思考中（尾部 {min(4, len(tl))} 行；完成后 /debug 展开全文）:",
                                 w, S.dim))
                for cl in tl[-4:]:
                    trans.append(seg("  | " + cl, w, S.dim))
            ans = self.agent_stream["answer"]
            if ans:
                trans.append(seg(f"助手: {ans}", w))
        self.agent_scroll = min(self.agent_scroll, max(0, len(trans) - view_h))
        start = max(0, len(trans) - view_h - self.agent_scroll)
        shown = trans[start: start + view_h]
        lines += shown if shown else [
            seg("  还没有对话。在下方输入并回车；记忆每轮自动注入，离线时自动降级应答。", w, S.dim)]
        lines += bottom
        return lines, (len(lines) - 2, col)

    # ---- 写入结果格式化（与 CLI cmd_add 输出一致）----

    def _fmt_ingest(self, r: dict) -> list[str]:
        lines = [f"门控打分: importance={r['importance']:.3f}"]
        if r["gated"]:
            # E3 双写后两档事件都在会话工作记忆里（本会话可检索, 退出蒸发），
            # 差别只在长期记忆是否入库——文案与 CLI cmd_add 同语义
            lines.append("低价值，不入长期记忆 (同在本会话工作记忆中，会话结束蒸发)"
                         if r["reason"] == "dropped"
                         else "中等价值，已暂存会话工作记忆 (本会话内可检索，会话结束蒸发)")
            return lines
        mem = self.store.episodic.get(r["episodic_id"])
        lines.append(f"情景记忆已写入: #{r['episodic_id']} [{mem.summary if mem else ''}]")
        for s_ in r["skipped_facts"]:
            lines.append(f"同键低置信已跳过: {s_['value']} (conf={s_['confidence']})")
        for f in r["facts"]:
            label = f"#{f['fact_id']} [{f['entity']}] {f['relation']} = {f['value']}"
            if f["action"] == "created":
                lines.append(f"语义记忆新建: {label} (conf={f['confidence']})")
            elif f["action"] == "renewed":
                lines.append(f"语义记忆复证: {label} (证据+1)")
            elif f["action"] == "superseded":
                lines.append(f"语义记忆变更: {label}，取代旧版 {f['superseded']}")
            else:
                lines.append(f"语义记忆待裁: {label}，冲突 {f['conflict_ids']} (到冲突页裁决)")
        for sk in r["skills"]:
            if sk["reused"]:
                lines.append(f"技能已复用: {sk['name']} (使用次数+1)")
            else:
                lines.append(f"技能已沉淀: {sk['name']} <- {sk['policy']}")
        return lines


def run(store: SqliteStore) -> int:
    return TuiApp(store).run()


def main() -> int:
    os.makedirs(settings.DATA_DIR, exist_ok=True)
    store = SqliteStore()
    try:
        return run(store)
    finally:
        store.close()


if __name__ == "__main__":
    sys.exit(main())
