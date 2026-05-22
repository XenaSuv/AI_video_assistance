"""Tests for src/jsonl_store.py and scripts/migrate_jsonl_schema.py."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.jsonl_store import (
    EXP_SCHEMA_V,
    PERF_SCHEMA_V,
    SHORTS_SCHEMA_V,
    append_record,
    iter_records,
    read_records,
    rewrite_jsonl,
    stamp,
)


# ── stamp ─────────────────────────────────────────────────────────────────────

class TestStamp:
    def test_adds_version(self) -> None:
        out = stamp({"a": 1}, 2)
        assert out["_schema_version"] == 2

    def test_returns_copy(self) -> None:
        original = {"a": 1}
        out = stamp(original, 2)
        assert original is not out
        assert "_schema_version" not in original

    def test_preserves_all_fields(self) -> None:
        out = stamp({"x": 1, "y": "hello"}, 2)
        assert out["x"] == 1
        assert out["y"] == "hello"

    def test_overwrites_existing_version(self) -> None:
        out = stamp({"_schema_version": 1, "a": 1}, 2)
        assert out["_schema_version"] == 2


# ── append_record ─────────────────────────────────────────────────────────────

class TestAppendRecord:
    def test_creates_file(self, tmp_path: Path) -> None:
        p = tmp_path / "store.jsonl"
        append_record(p, {"x": 1}, 2)
        assert p.exists()

    def test_writes_valid_json_line(self, tmp_path: Path) -> None:
        p = tmp_path / "store.jsonl"
        append_record(p, {"key": "val"}, 2)
        data = json.loads(p.read_text().strip())
        assert data["key"] == "val"
        assert data["_schema_version"] == 2

    def test_appends_multiple_records(self, tmp_path: Path) -> None:
        p = tmp_path / "store.jsonl"
        for i in range(3):
            append_record(p, {"i": i}, 2)
        lines = [l for l in p.read_text().splitlines() if l.strip()]
        assert len(lines) == 3

    def test_creates_parent_dirs(self, tmp_path: Path) -> None:
        p = tmp_path / "a" / "b" / "store.jsonl"
        append_record(p, {}, 2)
        assert p.exists()

    def test_version_constants_are_int(self) -> None:
        assert isinstance(PERF_SCHEMA_V, int)
        assert isinstance(SHORTS_SCHEMA_V, int)
        assert isinstance(EXP_SCHEMA_V, int)
        assert PERF_SCHEMA_V >= 2
        assert SHORTS_SCHEMA_V >= 2
        assert EXP_SCHEMA_V >= 2


# ── iter_records / read_records ───────────────────────────────────────────────

class TestReadRecords:
    def test_returns_empty_for_missing_file(self, tmp_path: Path) -> None:
        assert read_records(tmp_path / "missing.jsonl") == []

    def test_reads_multiple_records(self, tmp_path: Path) -> None:
        p = tmp_path / "store.jsonl"
        for i in range(4):
            append_record(p, {"i": i}, 2)
        records = read_records(p)
        assert len(records) == 4
        assert [r["i"] for r in records] == [0, 1, 2, 3]

    def test_skips_blank_lines(self, tmp_path: Path) -> None:
        p = tmp_path / "store.jsonl"
        p.write_text('{"a":1}\n\n{"b":2}\n')
        records = read_records(p)
        assert len(records) == 2

    def test_skips_malformed_lines(self, tmp_path: Path) -> None:
        p = tmp_path / "store.jsonl"
        p.write_text('{"a":1}\nNOT JSON\n{"b":2}\n')
        records = read_records(p)
        assert len(records) == 2
        assert records[0]["a"] == 1
        assert records[1]["b"] == 2

    def test_iter_records_is_generator(self, tmp_path: Path) -> None:
        p = tmp_path / "store.jsonl"
        append_record(p, {"x": 42}, 2)
        import types
        assert isinstance(iter_records(p), types.GeneratorType)


# ── rewrite_jsonl ─────────────────────────────────────────────────────────────

class TestRewriteJsonl:
    def test_overwrites_existing(self, tmp_path: Path) -> None:
        p = tmp_path / "store.jsonl"
        p.write_text('{"old": true}\n')
        rewrite_jsonl(p, [{"new": True}])
        records = read_records(p)
        assert len(records) == 1
        assert records[0]["new"] is True

    def test_no_tmp_after_success(self, tmp_path: Path) -> None:
        p = tmp_path / "store.jsonl"
        rewrite_jsonl(p, [{"a": 1}])
        assert not p.with_suffix(".tmp").exists()

    def test_empty_list_writes_empty_file(self, tmp_path: Path) -> None:
        p = tmp_path / "store.jsonl"
        rewrite_jsonl(p, [])
        assert p.read_text().strip() == ""


# ── migration script ──────────────────────────────────────────────────────────

class TestMigrateScript:
    """Integration tests for scripts/migrate_jsonl_schema.py."""

    @staticmethod
    def _run(args: list[str], root: Path) -> tuple[int, str]:
        import subprocess, sys
        script = Path(__file__).resolve().parent.parent / "scripts" / "migrate_jsonl_schema.py"
        result = subprocess.run(
            [sys.executable, str(script)] + args + ["--data-dir", str(root)],
            capture_output=True, text=True,
        )
        return result.returncode, result.stdout + result.stderr

    def test_dry_run_does_not_write(self, tmp_path: Path) -> None:
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        store = data_dir / "shorts_learning_v2.jsonl"
        store.write_text('{"short_id":"a","hook_type":"conflict","style":"fast_cut","persona":"direct","topic":"AI","retention_3s":0.5}\n')

        rc, out = self._run(["--store", "shorts_learning", "--dry-run"], tmp_path)
        assert rc == 0
        assert "DRY-RUN" in out
        # File unchanged — no _schema_version added
        data = json.loads(store.read_text().strip())
        assert "_schema_version" not in data

    def test_migrates_jsonl_records(self, tmp_path: Path) -> None:
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        store = data_dir / "shorts_learning_v2.jsonl"
        store.write_text(
            '{"short_id":"a","retention_3s":0.5}\n'
            '{"short_id":"b","retention_3s":0.7,"_schema_version":2}\n'
        )

        rc, _ = self._run(["--store", "shorts_learning"], tmp_path)
        assert rc == 0

        records = read_records(store)
        assert all(r["_schema_version"] == SHORTS_SCHEMA_V for r in records)

    def test_migrates_json_array(self, tmp_path: Path) -> None:
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        store = data_dir / "performance.json"
        store.write_text(json.dumps([
            {"video_id": "v1", "views": 0},
            {"video_id": "v2", "views": 100, "_schema_version": 2},
        ]))

        rc, _ = self._run(["--store", "performance"], tmp_path)
        assert rc == 0

        migrated = json.loads(store.read_text())
        assert all(r["_schema_version"] == PERF_SCHEMA_V for r in migrated)

    def test_creates_backup(self, tmp_path: Path) -> None:
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        store = data_dir / "shorts_results.jsonl"
        store.write_text('{"experiment_id":"x","video_id":"v"}\n')

        self._run(["--store", "shorts_results"], tmp_path)

        backups = list(data_dir.glob("shorts_results.bak_*.jsonl"))
        assert len(backups) == 1

    def test_idempotent_second_run(self, tmp_path: Path) -> None:
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        store = data_dir / "shorts_experiments.jsonl"
        store.write_text('{"experiment_id":"x"}\n')

        self._run(["--store", "shorts_experiments"], tmp_path)
        content_after_first = store.read_text()

        self._run(["--store", "shorts_experiments"], tmp_path)
        content_after_second = store.read_text()

        assert content_after_first == content_after_second

    def test_skip_missing_store(self, tmp_path: Path) -> None:
        rc, out = self._run(["--store", "shorts_scripts"], tmp_path)
        assert rc == 0
        assert "SKIP" in out

    def test_all_stores_runs_without_error(self, tmp_path: Path) -> None:
        rc, _ = self._run(["--dry-run"], tmp_path)
        assert rc == 0
