"""评测层：机制是否有用的度量尺（顶层，可依赖一切业务模块）。

mini：16 场景迷你评测（eval-mini）；harness：场景 DSL + 四项指标完整评测（eval）。
评测固定走离线降级路径（规则打分 + 哈希嵌入），保证可复现。
"""
from memagent.eval.mini import run_mini

__all__ = ["run_mini"]
