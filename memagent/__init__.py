"""memagent：类比人脑的持续学习 Agent 记忆框架（V0.4 评估体系版）。

分层（依赖只能自上而下，详见 ARCHITECTURE.md）：

    入口/评测   main.py  memagent.tui  memagent.eval        ← 只编排，不含机制
    应用编排    pipeline / maintenance / reports / storage(组合根)
    记忆机制    attention / encoding / retrieval / consolidation / forgetting
    记忆支撑    memory(仓储) / learning(强度与间隔重复)
    基础设施    adapters.llm(LLM 三通道) / storage.base(仓储抽象)
    纯核心      core(领域对象+纯函数) / settings(全局配置)

依赖方向由 tests/unit/test_architecture.py 守护。
"""

__version__ = "0.4.0"
