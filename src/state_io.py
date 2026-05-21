"""Cross-process-safe helpers for persistent JSON state files.

Uses fcntl.flock (POSIX) on a sibling .lock file so that concurrent
pipeline runs (daily + shorts, daily + breaking) never corrupt shared
state files via simultaneous read-modify-write cycles.

Falls back to a no-op on non-POSIX platforms (Windows / CI mock).
"""
from __future__ import annotations

import contextlib
import json
from pathlib import Path
from typing import Any, Iterator

from loguru import logger


@contextlib.contextmanager
def json_lock(path: Path) -> Iterator[None]:
    """Hold an exclusive POSIX file lock while the caller modifies *path*.

    Uses a sibling ``<stem>.lock`` file so the lock survives the atomic
    ``tmp.replace(path)`` rename used by writers.
    """
    lock_path = path.with_suffix(".lock")
    try:
        import fcntl  # noqa: PLC0415 — stdlib, POSIX only
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        with open(lock_path, "w") as lf:
            fcntl.flock(lf, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(lf, fcntl.LOCK_UN)
    except ImportError:
        yield  # non-POSIX: proceed without locking


def atomic_json_write(path: Path, data: Any) -> None:
    """Write *data* as JSON to *path* atomically under an exclusive lock.

    Write order: acquire lock → write to .tmp → rename over target.
    On any failure the .tmp is cleaned up and the exception is re-raised.
    """
    with json_lock(path):
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        try:
            tmp.write_text(json.dumps(data, indent=2, default=str))
            tmp.replace(path)
        except Exception as exc:
            logger.warning(f"atomic_json_write({path.name}): {exc}")
            try:
                tmp.unlink(missing_ok=True)
            except OSError:
                pass
            raise
