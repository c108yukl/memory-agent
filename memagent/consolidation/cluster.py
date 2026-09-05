"""巩固聚类（E5）：并查集连通分量——相似度 ≥ 阈值即连边，同传递闭包归一簇。

E5 前是贪心锚点聚类：锚点与其相似者成簇，传递相似不传递归属（A~B、B~C 而 A≁C
时 C 会落单）。连通分量按传递闭包归簇，语义更稳；代价是簇可能变大——超
CONSOLIDATE_MAX_CLUSTER 时退回锚点切分（锚点 + 与锚点相似者成组、组员数封顶），
防传递链把整夜记忆并成一簇。两两余弦仍需 O(n²) 次预计算（数据量小可接受），
但每次查/并集近似 O(1)，且划分不依赖锚点选取顺序。
"""
from __future__ import annotations

from memagent import settings
from memagent.adapters import llm
from memagent.core.domain import EpisodicMemory
from memagent.core.vectors import cosine
from memagent.storage import SqliteStore


class _DSU:
    """并查集：路径减半 + 按大小合并，均摊近 O(1) 单次查并。"""

    def __init__(self, n: int):
        self.parent = list(range(n))
        self.size = [1] * n

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]  # 路径减半
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.size[ra] < self.size[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        self.size[ra] += self.size[rb]


def _similar(a: EpisodicMemory, b: EpisodicMemory) -> bool:
    if not a.embedding or not b.embedding:
        return False
    return cosine(a.embedding, b.embedding) >= settings.CONSOLIDATE_MIN_SIMILARITY


def _split_oversized(members: list[EpisodicMemory]) -> list[list[EpisodicMemory]]:
    """连通分量超规模时的保守切分：锚点 + 与锚点相似者成组（退回 E5 前的锚点语义），
    组员数封顶 CONSOLIDATE_MAX_CLUSTER——传递闭包不应把整夜记忆并成一簇。"""
    groups: list[list[EpisodicMemory]] = []
    used: set[int] = set()
    for anchor in members:
        if anchor.id in used:
            continue
        group = [anchor]
        used.add(anchor.id)
        for other in members:
            if other.id in used or len(group) >= settings.CONSOLIDATE_MAX_CLUSTER:
                continue
            if _similar(anchor, other):
                group.append(other)
                used.add(other.id)
        groups.append(group)
    return groups


def cluster_episodes(episodes: list[EpisodicMemory], store: SqliteStore) -> list[list[EpisodicMemory]]:
    """连通分量聚类：相似度 ≥ CONSOLIDATE_MIN_SIMILARITY 连边，同分量归一簇。

    簇内成员保持传入顺序（调用方 job 已按唤醒度降序稳定排序——高唤醒者优先
    成为簇代表被保留原文）；无嵌入者不连边、自成单簇。"""
    n = len(episodes)
    dsu = _DSU(n)
    for i in range(n):
        for j in range(i + 1, n):
            if _similar(episodes[i], episodes[j]):
                dsu.union(i, j)

    components: dict[int, list[EpisodicMemory]] = {}
    for i, ep in enumerate(episodes):
        components.setdefault(dsu.find(i), []).append(ep)

    clusters: list[list[EpisodicMemory]] = []
    for members in components.values():
        groups = (_split_oversized(members)
                  if len(members) > settings.CONSOLIDATE_MAX_CLUSTER else [members])
        for group in groups:
            if len(group) >= settings.CONSOLIDATE_MIN_CLUSTER or llm.maintenance_available():
                clusters.append(group)
    return clusters
