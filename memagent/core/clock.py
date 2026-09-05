"""时钟工具：统一的 ISO 时间与时间戳转换。

set_offset/clear_offset 仅供评测 harness 做确定性时间旅行（保留曲线、SM-2
间隔都需要推进时间），生产代码不得调用。偏移作用于全部三个取时函数，
任何模块经 clock 取到的"现在"保持一致。
"""
from __future__ import annotations

import time
from datetime import datetime, timedelta, timezone

_OFFSET: timedelta | None = None


def set_offset(days: float = 0.0, hours: float = 0.0) -> None:
    """整体平移系统时钟（绝对偏移，非累加）。仅评测使用。"""
    global _OFFSET
    _OFFSET = timedelta(days=days, hours=hours)


def clear_offset() -> None:
    global _OFFSET
    _OFFSET = None


def now_dt() -> datetime:
    now = datetime.now(timezone.utc)
    return now + _OFFSET if _OFFSET else now


def now_iso() -> str:
    return now_dt().isoformat(timespec="seconds")


def now_ts() -> float:
    if _OFFSET is not None:
        return time.time() + _OFFSET.total_seconds()
    return time.time()


def iso_to_ts(iso: str) -> float:
    if not iso:
        return 0.0
    try:
        return datetime.fromisoformat(iso).timestamp()
    except ValueError:
        return 0.0
