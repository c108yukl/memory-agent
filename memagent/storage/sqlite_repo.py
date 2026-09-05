"""SQLite 存储入口：建表、管理连接，组合三类记忆仓储、工作记忆与审计日志。

表设计对应类脑映射：
- episodic   情景记忆（具体事件）
- semantic   语义记忆（事实/偏好三元组）
- procedural 程序记忆（技能模板）
- meta       元记忆（审计/访问统计）
- working    工作记忆（E3）不建表——纯内存 self.working，随实例生死（易失即设计）
"""
from __future__ import annotations

import os
import sqlite3
from datetime import datetime, timezone

from memagent import settings
from memagent.core.clock import now_iso
from memagent.memory import (
    AuditLog,
    ConflictRepo,
    EpisodicMemoryRepo,
    EntityAliasRepo,
    ProceduralSkillRepo,
    SemanticMemoryRepo,
    WorkingMemory,
)

SCHEMA = """
CREATE TABLE IF NOT EXISTS episodic (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    summary TEXT NOT NULL,
    context TEXT DEFAULT '',
    action TEXT DEFAULT '',
    outcome TEXT DEFAULT '',
    importance REAL DEFAULT 0,
    created_at TEXT NOT NULL,
    access_count INTEGER DEFAULT 0,
    last_access_at TEXT DEFAULT '',
    strength REAL DEFAULT 0,
    status TEXT DEFAULT 'active',
    embedding TEXT DEFAULT '[]',
    source_ids TEXT DEFAULT '[]',
    summarized_by INTEGER DEFAULT 0,
    is_summary INTEGER DEFAULT 0,
    arousal REAL DEFAULT 0,
    category TEXT DEFAULT '',
    source TEXT NOT NULL DEFAULT 'unknown'
);
CREATE TABLE IF NOT EXISTS semantic (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    entity TEXT NOT NULL,
    relation TEXT NOT NULL,
    value TEXT NOT NULL,
    confidence REAL DEFAULT 0,
    valid_from TEXT NOT NULL,
    valid_to TEXT DEFAULT '',
    conflict_note TEXT DEFAULT '',
    embedding TEXT DEFAULT '[]',
    status TEXT DEFAULT 'active',
    raw_entity TEXT DEFAULT '',
    raw_value TEXT DEFAULT '',
    superseded_by INTEGER DEFAULT 0,
    evidence_count INTEGER DEFAULT 1,
    source_event_ids TEXT DEFAULT '[]',
    hit_count INTEGER DEFAULT 0,
    validity_context TEXT NOT NULL DEFAULT ''
);
CREATE TABLE IF NOT EXISTS procedural (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    trigger TEXT DEFAULT '',
    policy TEXT DEFAULT '',
    success_count INTEGER DEFAULT 0,
    failure_count INTEGER DEFAULT 0,
    usage_count INTEGER DEFAULT 0,
    success_rate REAL DEFAULT 1.0,
    status TEXT DEFAULT 'active'
);
CREATE TABLE IF NOT EXISTS meta (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    memory_type TEXT NOT NULL,
    memory_id INTEGER NOT NULL,
    action TEXT NOT NULL,
    detail TEXT DEFAULT '',
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS migration_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    started_at TEXT NOT NULL,
    finished_at TEXT DEFAULT '',
    status TEXT DEFAULT 'ok',
    affected_rows INTEGER DEFAULT 0,
    backup_path TEXT DEFAULT '',
    detail TEXT DEFAULT ''
);
CREATE TABLE IF NOT EXISTS entity_alias (
    alias TEXT PRIMARY KEY,
    canonical TEXT NOT NULL,
    created_at TEXT NOT NULL,
    source TEXT DEFAULT 'manual'
);
CREATE TABLE IF NOT EXISTS memory_conflict (
    conflict_id INTEGER PRIMARY KEY AUTOINCREMENT,
    old_id INTEGER NOT NULL,
    new_id INTEGER NOT NULL,
    conflict_type TEXT DEFAULT 'value_conflict',
    status TEXT DEFAULT 'pending',
    reason TEXT DEFAULT '',
    created_at TEXT NOT NULL,
    resolved_at TEXT DEFAULT '',
    resolution TEXT DEFAULT ''
);
CREATE VIRTUAL TABLE IF NOT EXISTS episodic_fts USING fts5(summary, content='episodic', content_rowid='id');
CREATE VIRTUAL TABLE IF NOT EXISTS semantic_fts USING fts5(value, content='semantic', content_rowid='id');
"""

# FTS 索引表 DDL：集中一处，供建库与 maintenance.rebuild_fts 复用。
# 重建必须走 DROP + CREATE，不能用「DELETE FROM <fts>」清空后重插——实测后者在主表
# 存在「没有对应索引条目」的行时会损坏索引（DatabaseError: database disk image is
# malformed，external content 表特性）；DROP/CREATE 是 DDL，重复执行也安全。
FTS_TABLE_DDL = {
    "episodic_fts": ("CREATE VIRTUAL TABLE episodic_fts "
                     "USING fts5(summary, content='episodic', content_rowid='id')"),
    "semantic_fts": ("CREATE VIRTUAL TABLE semantic_fts "
                     "USING fts5(value, content='semantic', content_rowid='id')"),
}

# 旧库升级：加列声明（PRAGMA 检测后 ALTER TABLE，幂等）
_COLUMN_MIGRATIONS = {
    "semantic": {
        "raw_entity": "TEXT DEFAULT ''",
        "raw_value": "TEXT DEFAULT ''",
        "superseded_by": "INTEGER DEFAULT 0",
        "evidence_count": "INTEGER DEFAULT 1",
        "source_event_ids": "TEXT DEFAULT '[]'",
        "hit_count": "INTEGER DEFAULT 0",   # P1 转正通道：试用期被检索命中的累计次数
        "validity_context": "TEXT NOT NULL DEFAULT ''",   # P2-2 条件上下文（认识论限定）：
        # JSON {"preconditions":[...],"environment":[...],"expires_if":str}；空=无条件永久有效。
        # 存量行为空串 = 事实无条件成立，语义与升级前完全一致（零回归）
    },
    "episodic": {
        "source_ids": "TEXT DEFAULT '[]'",
        "summarized_by": "INTEGER DEFAULT 0",
        "is_summary": "INTEGER DEFAULT 0",
        "arousal": "REAL DEFAULT 0",
        "category": "TEXT DEFAULT ''",   # E8 记忆分层：experience 经验层（LFU 策略标记）
        "source": "TEXT NOT NULL DEFAULT 'unknown'",   # P0-4 来源：user|assistant|system|
        # tool|feedback（编码时从 Event.source 透传）。存量行无法追溯真实来源，诚实标
        # 'unknown' 比空串可查询性好；检索/注入暂不消费，P1-5 启用
    },
}


class SqliteStore:
    """组合根：一个连接 + 三仓储 + 审计。调用方通过 store.episodic / store.semantic / store.procedural 访问。"""

    def __init__(self, db_path: str = settings.DB_PATH):
        d = os.path.dirname(db_path)
        if d:
            os.makedirs(d, exist_ok=True)
        self.db_path = db_path
        # check_same_thread=False：TUI 流式输出由工作线程调 loop.turn 写库（working/
        # 审计直写）。本机 CPython sqlite3 threadsafety=3（编译级 serialized，实测
        # sqlite3.threadsafety==3），同一连接的跨线程共享由 SQLite 串行化保证。
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)
        self.conn.commit()
        self._ensure_columns()

        self.audit_log = AuditLog(self.conn)
        audit = self.audit_log.log
        self.episodic = EpisodicMemoryRepo(self.conn, audit)
        self.semantic = SemanticMemoryRepo(self.conn, audit)
        self.procedural = ProceduralSkillRepo(self.conn, audit)
        self.aliases = EntityAliasRepo(self.conn, audit)
        self.conflicts = ConflictRepo(self.conn, audit)
        # E3 工作记忆：纯内存、随本实例生死（会话 = 存储实例生命周期）。
        # 纯属性组合，无表无迁移——进程退出即蒸发是设计目标。
        self.working = WorkingMemory()

    def _ensure_columns(self) -> None:
        """旧库无损升级：缺失的列补上（CREATE TABLE IF NOT EXISTS 不覆盖已有表结构）。"""
        for table, columns in _COLUMN_MIGRATIONS.items():
            existing = {r["name"] for r in self.conn.execute(f"PRAGMA table_info({table})")}
            for col, decl in columns.items():
                if col not in existing:
                    self.conn.execute(f"ALTER TABLE {table} ADD COLUMN {col} {decl}")
        self.conn.commit()

    def backup(self, tag: str) -> str:
        """热备份到 data/backups/，返回备份路径。迁移类命令 --apply 前必须调用。"""
        backup_dir = os.path.join(os.path.dirname(os.path.abspath(self.db_path)), "backups")
        os.makedirs(backup_dir, exist_ok=True)
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        path = os.path.join(backup_dir, f"{os.path.basename(self.db_path)}.{tag}.{ts}.bak")
        dest = sqlite3.connect(path)
        try:
            self.conn.backup(dest)
        finally:
            dest.close()
        return path

    def log_migration(self, name: str, status: str = "ok", affected_rows: int = 0,
                      backup_path: str = "", detail: str = "") -> None:
        now = now_iso()
        self.conn.execute(
            "INSERT INTO migration_log (name, started_at, finished_at, status, affected_rows, backup_path, detail) "
            "VALUES (?,?,?,?,?,?,?)",
            (name, now, now, status, affected_rows, backup_path, detail))
        self.conn.commit()

    def log(self, memory_type: str, memory_id: int, action: str, detail: str = "") -> None:
        self.audit_log.log(memory_type, memory_id, action, detail)

    def close(self) -> None:
        self.conn.close()
