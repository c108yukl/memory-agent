"""向量工具：自写余弦相似度与本地哈希嵌入（无第三方依赖）。"""
from __future__ import annotations

import hashlib
import math


def cosine(a: list[float], b: list[float]) -> float:
    """自写余弦相似度：避免引入向量库，顺便复习线代。"""
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = 0.0
    na = 0.0
    nb = 0.0
    for x, y in zip(a, b):
        dot += x * y
        na += x * x
        nb += y * y
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (math.sqrt(na) * math.sqrt(nb))


def hash_embed(text: str, dim: int = 64) -> list[float]:
    """本地哈希嵌入：字符二元组哈希到向量桶（有符号），无模型也可用。"""
    vec = [0.0] * dim
    text = text.lower()
    grams = [text[i:i + 2] for i in range(len(text) - 1)] + [text]
    for gram in grams:
        h = int(hashlib.md5(gram.encode("utf-8")).hexdigest(), 16)
        idx = h % dim
        sign = 1.0 if (h >> 16) % 2 == 0 else -1.0
        vec[idx] += sign
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]
