"""存储抽象层：仓储基类。未来 Postgres/向量库仓储实现同一接口即可替换。"""
from __future__ import annotations

from typing import Callable

import sqlite3


class BaseRepo:
    """每个仓储共享同一个连接与审计回调。

    audit 签名: audit(memory_type: str, memory_id: int, action: str, detail: str = "")
    """

    def __init__(self, conn: sqlite3.Connection, audit: Callable[..., None]):
        self.conn = conn
        self.audit = audit
