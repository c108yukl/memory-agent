"""适配层：外部服务的统一入口。当前只有 LLM（adapters/llm，三通道）。

全项目的 LLM 调用必须经 memagent.adapters.llm 的模块级函数
（chat / maintenance_chat / embed / llm_available），
测试 mock 这几个函数即可获得确定性输出，不允许绕过直达客户端。
"""
