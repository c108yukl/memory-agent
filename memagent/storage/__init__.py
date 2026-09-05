"""存储层：SQLite 组合根（sqlite_repo 建库 + 装配全部仓储）与仓储抽象基类（base）。

组合根是全项目唯一允许"知道所有仓储"的位置：机制层模块只从这里导入
SqliteStore 类型与仓储实例，不自行开连接。依赖方向见 ARCHITECTURE.md。
"""
from memagent.storage.sqlite_repo import SqliteStore

__all__ = ["SqliteStore"]
