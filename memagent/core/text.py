"""文本处理：中文 2-gram 分词（让 FTS5 对中文真正可用）。

FTS5 默认按空白切 token，中文整句会变成单个 token，关键词检索完全失效。
把文本切成字符二元组空格连接后建索引/查询，"部署" 就能命中 "项目部署文档"。
"""
from __future__ import annotations


def bigram(text: str) -> str:
    """字符二元组分词：保留原文整词 + 全部 2-gram，空格连接。"""
    text = " ".join(text.split())
    if not text:
        return ""
    grams = [text[i:i + 2] for i in range(len(text) - 1)] + [text]
    return " ".join(g for g in grams if g.strip())
