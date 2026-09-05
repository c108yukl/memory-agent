"""E4 联想激活：实体图构建 + ACT-R 式两跳扩散。

关键洞察：SemanticFact(entity, relation, value) 三元组本身就是图的边——
实体为节点，一条事实连接其 entity 与「值中精确出现的已知实体」。
加 entity_alias 别名表与 evidence_count，图在数据层已 80% 物化，
本模块只负责现建图与扩散（每次检索重建，数据量小不做缓存）。

激活数学（ACT-R 式）：激活 = 基础强度 + Σ(源激活 × 衰减系数)。
扩散只传播查询带来的流入激活——节点 base 是固有属性、不参与再传播，
否则 hub 节点的静态强度会劫持信号（二跳反超一跳，衰减失去意义）。

对外接口（E5 巩固的 REM 阶段将复用，保持只读、无副作用）：
  build_entity_graph(store) -> EntityGraph   只读 memory 仓储建图
  graph.activate(seeds)                     纯函数扩散，不碰 store
  spread(store, seed_entities)              建图 + 扩散一步到位
"""
from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass

from memagent import settings
from memagent.core.domain import SemanticFact
from memagent.storage import SqliteStore

# 每份多余证据（evidence_count 超 1 的部分）给 base 的微弱加成（E6 改 log 饱和）：
# 重复观测过的实体更「实」，但加成随证据数对数增长——线性无界的证据项会把
# 被反复观测/检索的实体在图上无限抬高（与 strength.frequency_component 同一取舍）
_EVIDENCE_BASE_BONUS = 0.05


@dataclass(frozen=True)
class EntityGraph:
    """实体联想图（只读快照）。

    base             实体基础强度：1 - exp(-active事实数) 饱和 + 证据对数加成，封顶 1.0
    edges            邻接表（按无向使用：联想没有方向语义）
    facts_by_entity  每个实体挂靠的 active 事实（联想命中的候选池，E5 REM 重组原料）
    aliases          别名 -> 规范实体（种子实体名入图前先归一）
    """

    base: Mapping[str, float]
    edges: Mapping[str, frozenset[str]]
    facts_by_entity: Mapping[str, tuple[SemanticFact, ...]]
    aliases: Mapping[str, str]

    def activate(self, seeds: Mapping[str, float]) -> dict[str, float]:
        """ACT-R 式两跳扩散（纯函数）：激活 = base + Σ(源激活 × 衰减)。

        种子实体保留其种子激活值（直接命中本身即激活源，不与 base 相加）；
        每跳 frontier 只携带上一跳获得的流入激活；同一实体收到多条路径的
        流入时取最大——激活衡量「够到了多强」，不累加并行噪声。
        """
        normalized = {self.aliases.get(e, e): float(v) for e, v in seeds.items()}
        result: dict[str, float] = dict(normalized)
        frontier = dict(normalized)
        for _ in range(settings.ACTIVATION_MAX_HOPS):
            inflow: dict[str, float] = {}
            for entity, act in frontier.items():
                for neighbor in self.edges.get(entity, ()):
                    if neighbor in normalized:
                        continue  # 不回流种子：种子激活由直接命中给定
                    got = act * settings.ACTIVATION_DECAY
                    if got > inflow.get(neighbor, 0.0):
                        inflow[neighbor] = got
            if not inflow:
                break
            for entity, got in inflow.items():
                result[entity] = max(result.get(entity, 0.0),
                                     self.base.get(entity, 0.0) + got)
            frontier = inflow
        return result

    def facts_of(self, entity: str) -> tuple[SemanticFact, ...]:
        """某实体挂靠的 active 事实（别名自动归一到规范实体）。"""
        return self.facts_by_entity.get(self.aliases.get(entity, entity), ())


def build_entity_graph(store: SqliteStore) -> EntityGraph:
    """从 active 语义事实现建实体图（只读仓储，无写回）。

    实体先经别名表映射到规范形式再入图；连边用精确子串匹配值中出现的
    已知实体名（含别名）——不做模糊匹配，宁可少连不可误伤。
    """
    alias_map = store.aliases.as_map()
    facts = store.semantic.fetch(status="active")

    facts_by: dict[str, list[SemanticFact]] = {}
    for fact in facts:
        entity = alias_map.get(fact.entity, fact.entity)
        facts_by.setdefault(entity, []).append(fact)

    base: dict[str, float] = {}
    for entity, group in facts_by.items():
        extra_evidence = sum(max(0, f.evidence_count - 1) for f in group)
        base[entity] = min(1.0, 1.0 - math.exp(-len(group))
                           + _EVIDENCE_BASE_BONUS * math.log1p(extra_evidence))

    known: dict[str, str] = {e: e for e in facts_by}  # 已知实体名 -> 规范实体
    for alias, canonical in alias_map.items():
        if canonical in facts_by:
            known.setdefault(alias, canonical)

    edges: dict[str, set[str]] = {e: set() for e in facts_by}
    for fact in facts:
        entity = alias_map.get(fact.entity, fact.entity)
        for name, canonical in known.items():
            if name and canonical != entity and name in fact.value:
                edges[entity].add(canonical)
                edges[canonical].add(entity)

    return EntityGraph(base=base,
                       edges={e: frozenset(n) for e, n in edges.items()},
                       facts_by_entity={e: tuple(g) for e, g in facts_by.items()},
                       aliases=alias_map)


def spread(store: SqliteStore,
           seed_entities: Mapping[str, float] | Iterable[str]) -> dict[str, float]:
    """建图 + 扩散一步到位（E5 巩固 REM 阶段复用的入口）。

    seed_entities 接受「实体 -> 激活值」映射，或实体集合（缺省激活 1.0）；
    种子为空返回空 dict——无直接命中不做联想，避免无中生有的噪声。
    """
    if isinstance(seed_entities, Mapping):
        seeds = {e: float(v) for e, v in seed_entities.items()}
    else:
        seeds = {e: 1.0 for e in seed_entities}
    if not seeds:
        return {}
    return build_entity_graph(store).activate(seeds)
