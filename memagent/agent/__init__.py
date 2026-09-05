"""记忆原生 Agent 循环（V1.7 P2；V1.7.3 增加工具调用，入口层）。

职责边界：本包**只做编排**——把既有的读路径（retrieval）与写路径（pipeline）
按一轮对话的节奏串起来，不含任何记忆机制。分层理由见 ARCHITECTURE.md §1：
与 CLI / TUI / eval 平级，只自上而下调用 pipeline / retrieval / memory /
consolidation / forgetting，禁止被下层反向 import（test_architecture 静态守护）。

一轮 turn 的四件事：
  注入（injector）→ 由框架自动取记忆、裁剪、拼提示词，模型不需要主动查；
  生成（loop）    → 调 adapters.llm（唯一 LLM 出口），无模型时离线兜底；
  工具（toolkit） → 模型可经文本协议点名调用记忆工具（查/写/睡眠/遗忘…），
                    全部是对既有函数的薄封装，工具只是自动注入的补丁；
  录入（recorder）→ 静默优先：默认只进工作记忆，检测到信号才走写入管线；
                    复述刚注入/刚检索到的内容不沉淀（自增强回路的闸）。

模型主动写入入口 tools.remember() 与工具通道共享：只接受 GREEN_TYPES
（experience / env_statement），滥用由 E8 遗忘侧的「宽进严出」兜底。
"""
from memagent.agent.injector import Injection, inject
from memagent.agent.loop import AgentLoop, Turn, build_prompt
from memagent.agent.recorder import (
    Signal,
    detect_signals,
    is_restatement,
    record_turn,
    restatement_score,
)
from memagent.agent.toolkit import (
    TOOL_REGISTRY,
    ToolCall,
    ToolContext,
    build_tools_prompt,
    execute_tool,
    parse_tool_call,
    strip_tool_call,
)
from memagent.agent.tools import remember

__all__ = ["AgentLoop", "Turn", "build_prompt", "Injection", "inject",
           "record_turn", "detect_signals", "is_restatement", "restatement_score",
           "Signal", "remember", "TOOL_REGISTRY", "ToolCall", "ToolContext",
           "build_tools_prompt", "execute_tool", "parse_tool_call", "strip_tool_call"]
