"""Versioned helpers for append-only JSONL persistent stores.

Each store type carries its own SCHEMA_VERSION constant so stores can
evolve independently.  stamp() injects _schema_version into every write;
all from_dict() implementations already filter to known dataclass fields,
so old records without the field are read back transparently.

Typical usage inside a store class::

    from src.jsonl_store import append_record, SHORTS_SCHEMA_V

    def _append(self, path: Path, record: dict[str, Any]) -> None:
        try:
            append_record(path, record, SHORTS_SCHEMA_V)
        except Exception as exc:
            logger.warning(f"write failed: {exc}")
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator

# ── Per-store schema versions ─────────────────────────────────────────────────
# Bump the relevant constant when the record shape changes in a breaking way.

PERF_SCHEMA_V:   int = 2   # data/performance.json
SHORTS_SCHEMA_V: int = 2   # data/shorts_learning_v2.jsonl / shorts_scripts_v2.jsonl
EXP_SCHEMA_V:    int = 2   # data/shorts_experiments.jsonl / shorts_results.jsonl


def stamp(record: dict[str, Any], version: int) -> dict[str, Any]:
    """Return a shallow copy of *record* with ``_schema_version`` set."""
    out = dict(record)
    out["_schema_version"] = version
    return out


def append_record(path: Path, record: dict[str, Any], version: int) -> None:
    """Append one JSON-line to *path*, stamped with *version*.

    Creates parent directories on first write.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as fh:
        fh.write(json.dumps(stamp(record, version)) + "\n")


def iter_records(path: Path) -> Iterator[dict[str, Any]]:
    """Yield parsed dicts from a JSONL file; skip blank and malformed lines."""
    if not path.exists():
        return
    for raw in path.read_text().splitlines():
        raw = raw.strip()
        if not raw:
            continue
        try:
            yield json.loads(raw)
        except json.JSONDecodeError:
            pass


def read_records(path: Path) -> list[dict[str, Any]]:
    """Return all records from a JSONL file as a list."""
    return list(iter_records(path))


def rewrite_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    """Atomically overwrite *path* with *records* (one JSON line each)."""
    tmp = path.with_suffix(".tmp")
    try:
        tmp.write_text("\n".join(json.dumps(r) for r in records) + "\n")
        tmp.replace(path)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise
