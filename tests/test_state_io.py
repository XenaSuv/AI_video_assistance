"""Tests for src/state_io.py — cross-process-safe JSON state helpers."""
from __future__ import annotations

import json
import threading
from pathlib import Path

import pytest

from src.state_io import atomic_json_write, json_lock


# ── json_lock ──────────────────────────────────────────────────────────────────

class TestJsonLock:
    def test_creates_lock_file(self, tmp_path: Path) -> None:
        target = tmp_path / "state.json"
        target.write_text("{}")
        lock_file = target.with_suffix(".lock")
        with json_lock(target):
            assert lock_file.exists()

    def test_lock_file_persists_after_exit(self, tmp_path: Path) -> None:
        target = tmp_path / "state.json"
        target.write_text("{}")
        lock_file = target.with_suffix(".lock")
        with json_lock(target):
            pass
        # .lock file is kept (it's cheap, avoids recreate race)
        assert lock_file.exists()

    def test_yields_normally_on_success(self, tmp_path: Path) -> None:
        target = tmp_path / "state.json"
        executed = False
        with json_lock(target):
            executed = True
        assert executed

    def test_propagates_exception(self, tmp_path: Path) -> None:
        target = tmp_path / "state.json"
        with pytest.raises(ValueError, match="boom"):
            with json_lock(target):
                raise ValueError("boom")

    def test_concurrent_writes_are_serialized(self, tmp_path: Path) -> None:
        """Two threads must not interleave their writes."""
        target = tmp_path / "counter.json"
        target.write_text("0")
        errors: list[Exception] = []

        def increment(n: int) -> None:
            try:
                for _ in range(n):
                    with json_lock(target):
                        val = int(target.read_text())
                        target.write_text(str(val + 1))
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=increment, args=(20,)) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        assert int(target.read_text()) == 80

    def test_lock_directory_created_if_missing(self, tmp_path: Path) -> None:
        target = tmp_path / "nested" / "dir" / "state.json"
        with json_lock(target):
            pass
        assert (target.parent / "state.lock").exists()


# ── atomic_json_write ──────────────────────────────────────────────────────────

class TestAtomicJsonWrite:
    def test_writes_valid_json(self, tmp_path: Path) -> None:
        target = tmp_path / "data.json"
        atomic_json_write(target, {"key": "value", "num": 42})
        assert json.loads(target.read_text()) == {"key": "value", "num": 42}

    def test_writes_list(self, tmp_path: Path) -> None:
        target = tmp_path / "list.json"
        atomic_json_write(target, [1, 2, 3])
        assert json.loads(target.read_text()) == [1, 2, 3]

    def test_overwrites_existing(self, tmp_path: Path) -> None:
        target = tmp_path / "data.json"
        target.write_text('{"old": true}')
        atomic_json_write(target, {"new": True})
        assert json.loads(target.read_text()) == {"new": True}

    def test_no_tmp_file_after_success(self, tmp_path: Path) -> None:
        target = tmp_path / "data.json"
        atomic_json_write(target, {"x": 1})
        assert not target.with_suffix(".tmp").exists()

    def test_creates_parent_dirs(self, tmp_path: Path) -> None:
        target = tmp_path / "a" / "b" / "c.json"
        atomic_json_write(target, {})
        assert target.exists()

    def test_non_serialisable_uses_str_default(self, tmp_path: Path) -> None:
        from pathlib import PurePosixPath
        target = tmp_path / "data.json"
        atomic_json_write(target, {"p": PurePosixPath("/tmp/x")})
        data = json.loads(target.read_text())
        assert data["p"] == "/tmp/x"

    def test_concurrent_writes_last_write_wins(self, tmp_path: Path) -> None:
        """All threads complete without corruption; result is valid JSON."""
        target = tmp_path / "shared.json"
        errors: list[Exception] = []

        def writer(val: int) -> None:
            try:
                atomic_json_write(target, {"v": val})
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=writer, args=(i,)) for i in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        data = json.loads(target.read_text())
        assert "v" in data
        assert isinstance(data["v"], int)
