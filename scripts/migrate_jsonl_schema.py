#!/usr/bin/env python3
"""Migrate persistent stores to the current _schema_version.

Adds ``_schema_version`` to every record that is missing it (version 1 →
current).  Safe to re-run: records that already carry the field are left
unchanged.  A timestamped backup is written before any file is touched.

Usage
-----
    # Migrate all stores (default)
    python scripts/migrate_jsonl_schema.py

    # Preview without writing
    python scripts/migrate_jsonl_schema.py --dry-run

    # One store only
    python scripts/migrate_jsonl_schema.py --store performance
    python scripts/migrate_jsonl_schema.py --store shorts_learning
    python scripts/migrate_jsonl_schema.py --store shorts_experiments

Stores
------
    performance       data/performance.json          (JSON array)
    shorts_learning   data/shorts_learning_v2.jsonl  (JSONL)
    shorts_scripts    data/shorts_scripts_v2.jsonl   (JSONL)
    shorts_experiments data/shorts_experiments.jsonl (JSONL)
    shorts_results    data/shorts_results.jsonl      (JSONL)
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

# ── project root on path so config/src are importable ────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.jsonl_store import (
    EXP_SCHEMA_V,
    PERF_SCHEMA_V,
    SHORTS_SCHEMA_V,
    rewrite_jsonl,
    stamp,
)

# ── store descriptors ─────────────────────────────────────────────────────────

_STORES: dict[str, tuple[str, int, str]] = {
    # key: (relative path, target version, format)
    "performance":        ("data/performance.json",            PERF_SCHEMA_V,   "json_array"),
    "shorts_learning":    ("data/shorts_learning_v2.jsonl",    SHORTS_SCHEMA_V, "jsonl"),
    "shorts_scripts":     ("data/shorts_scripts_v2.jsonl",     SHORTS_SCHEMA_V, "jsonl"),
    "shorts_experiments": ("data/shorts_experiments.jsonl",    EXP_SCHEMA_V,    "jsonl"),
    "shorts_results":     ("data/shorts_results.jsonl",        EXP_SCHEMA_V,    "jsonl"),
}


# ── helpers ───────────────────────────────────────────────────────────────────

def _backup(path: Path) -> Path:
    ts  = datetime.now().strftime("%Y%m%d_%H%M%S")
    bak = path.with_name(f"{path.stem}.bak_{ts}{path.suffix}")
    shutil.copy2(path, bak)
    return bak


def _load_json_array(path: Path) -> list[dict[str, Any]]:
    try:
        data = json.loads(path.read_text())
        return data if isinstance(data, list) else []
    except Exception as exc:
        print(f"  ERROR reading {path.name}: {exc}", file=sys.stderr)
        return []


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for lineno, raw in enumerate(path.read_text().splitlines(), 1):
        raw = raw.strip()
        if not raw:
            continue
        try:
            records.append(json.loads(raw))
        except json.JSONDecodeError as exc:
            print(f"  WARN line {lineno} malformed, skipping: {exc}", file=sys.stderr)
    return records


def _migrate_records(
    records: list[dict[str, Any]],
    version: int,
) -> tuple[list[dict[str, Any]], int]:
    """Return (migrated_records, count_changed)."""
    migrated = []
    changed  = 0
    for r in records:
        if r.get("_schema_version") != version:
            migrated.append(stamp(r, version))
            changed += 1
        else:
            migrated.append(r)
    return migrated, changed


def migrate_store(key: str, root: Path, *, dry_run: bool) -> None:
    rel_path, version, fmt = _STORES[key]
    path = root / rel_path

    print(f"\n[{key}]  {rel_path}  (target v{version})")

    if not path.exists():
        print(f"  SKIP — file does not exist yet")
        return

    if fmt == "json_array":
        records = _load_json_array(path)
    else:
        records = _load_jsonl(path)

    total = len(records)
    migrated, changed = _migrate_records(records, version)

    if changed == 0:
        print(f"  OK   — {total} records already at v{version}, nothing to do")
        return

    print(f"  MIGRATE {changed}/{total} records: v(missing) → v{version}")

    if dry_run:
        print(f"  DRY-RUN — no files written")
        return

    bak = _backup(path)
    print(f"  BACKUP  → {bak.name}")

    if fmt == "json_array":
        tmp = path.with_suffix(".tmp")
        try:
            tmp.write_text(json.dumps(migrated, indent=2, default=str))
            tmp.replace(path)
        except Exception:
            tmp.unlink(missing_ok=True)
            raise
    else:
        rewrite_jsonl(path, migrated)

    print(f"  DONE    — {path.name} updated")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Migrate JSONL/JSON stores to current _schema_version",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Stores: " + ", ".join(_STORES),
    )
    ap.add_argument(
        "--store",
        choices=[*_STORES, "all"],
        default="all",
        help="Which store to migrate (default: all)",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview changes without writing anything",
    )
    ap.add_argument(
        "--data-dir",
        default=str(ROOT),
        help="Project root directory (default: repo root)",
    )
    args = ap.parse_args()

    root = Path(args.data_dir)
    keys = list(_STORES) if args.store == "all" else [args.store]

    if args.dry_run:
        print("=== DRY RUN — no files will be modified ===")

    for key in keys:
        migrate_store(key, root, dry_run=args.dry_run)

    print("\nDone.")


if __name__ == "__main__":
    main()
