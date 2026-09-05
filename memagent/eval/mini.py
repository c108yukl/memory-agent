"""mini golden 评测（B6）：15~20 个确定性场景，量化 Phase B 的记忆质量。

离线模式运行（规则打分 + 哈希嵌入 + 规则抽取）——评测对象是管线逻辑
（门控/冲突消解/版本链/检索），不引入云端模型的随机性，结果可复现。
"""
from __future__ import annotations

import contextlib
import json
import os
import tempfile
from unittest import mock

from memagent import settings
from memagent.adapters import llm
from memagent.attention import attention_gate
from memagent.consolidation.conflict_resolver import (
    dedupe_same_key,
    store_fact_with_conflict_check,
)
from memagent.core import Event
from memagent.core.vectors import hash_embed
from memagent.encoding import encode_episodic, extract_semantic_facts
from memagent.encoding.entity_resolver import normalize_value, resolve_fact
from memagent.retrieval import retrieve
from memagent.storage import SqliteStore

BASELINE = {"preference_hit_rate": 0.85, "conflict_resolution_accuracy": 0.80,
            "history_integrity": 1.00}

_EXPLICIT_TYPES = ("instruction", "preference_statement")


@contextlib.contextmanager
def offline_mode():
    """强制离线确定性：LLM 不可用，嵌入退回本地哈希（后端档同报 "hash"，
    E7 置信阈值取 0.30 兜底档——与测试包级打桩口径一致）。"""
    with mock.patch.object(llm, "llm_available", return_value=False), \
         mock.patch.object(llm, "maintenance_available", return_value=False), \
         mock.patch.object(llm, "chat", return_value=None), \
         mock.patch.object(llm, "maintenance_chat", return_value=None), \
         mock.patch.object(llm, "local_chat", return_value=None), \
         mock.patch.object(llm, "embed_backend", return_value="hash"), \
         mock.patch.object(llm, "embed",
                           side_effect=lambda t: hash_embed(t, settings.EMBED_FALLBACK_DIM)):
        yield


def run_scenario(case: dict) -> dict:
    """跑一个场景，返回三项指标（None 表示该项不适用）。"""
    tmp = tempfile.mkdtemp()
    store = SqliteStore(os.path.join(tmp, "eval.db"))
    try:
        for ev_spec in case["events"]:
            event = Event(content=ev_spec["content"], type=ev_spec.get("type", "observation"))
            event = attention_gate(event, use_llm=False)
            if event.importance < settings.WRITE_THRESHOLD:
                continue
            mem = encode_episodic(event)
            mem_id = store.episodic.add(mem)
            facts, _skipped = dedupe_same_key(extract_semantic_facts(event))
            for fact in facts:
                resolve_fact(fact, store.aliases.as_map())
                fact.source_event_ids = [mem_id]
                source = "explicit" if event.type in _EXPLICIT_TYPES else "model"
                store_fact_with_conflict_check(store, fact, source=source)

        expected = case.get("expected_active_value")
        result = {"hit": None, "resolution": None, "history": None}

        if expected is not None:
            expected_norm = normalize_value(expected)
            active_values = [f.value for f in store.semantic.fetch(status="active")]
            result["resolution"] = expected_norm in active_values

            if case.get("query"):
                hits = retrieve(store, case["query"], top_k=5)
                result["hit"] = any(expected_norm in normalize_value(h.text) for h in hits)

        expected_count = case.get("expected_history_count")
        if expected_count is not None:
            total = len(store.conn.execute("SELECT id FROM semantic").fetchall())
            result["history"] = (total == expected_count)
        return result
    finally:
        store.close()


def run_mini(path: str | None = None) -> dict:
    path = path or os.path.join(settings.BASE_DIR, "evals", "mini_golden.jsonl")
    with open(path, encoding="utf-8") as f:
        cases = [json.loads(line) for line in f if line.strip()]

    with offline_mode():
        results = {c["case_id"]: run_scenario(c) for c in cases}

    def rate(key: str) -> float:
        vals = [r[key] for r in results.values() if r[key] is not None]
        return round(sum(vals) / len(vals), 4) if vals else 1.0

    failures = {cid: r for cid, r in results.items()
                if any(v is False for v in r.values())}
    return {
        "cases": len(cases),
        "preference_hit_rate": rate("hit"),
        "conflict_resolution_accuracy": rate("resolution"),
        "history_integrity": rate("history"),
        "failures": failures,
        "passed": (
            rate("hit") >= BASELINE["preference_hit_rate"]
            and rate("resolution") >= BASELINE["conflict_resolution_accuracy"]
            and rate("history") >= BASELINE["history_integrity"]
        ),
    }
