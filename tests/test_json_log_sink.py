"""Tests for src/json_log_sink.py — JsonSink callable sink for loguru."""
from __future__ import annotations

import json
import threading
from datetime import datetime
from pathlib import Path

import pytest
from loguru import logger

from src.json_log_sink import JsonSink, _safe


# ── helpers ───────────────────────────────────────────────────────────────────

def _add(path: Path, **kwargs) -> int:
    """Register JsonSink with loguru; return sink ID for cleanup."""
    return logger.add(JsonSink(path), level="DEBUG", **kwargs)


def _lines(path: Path) -> list[dict]:
    """Parse every line of a JSONL file and return list of dicts."""
    return [json.loads(line) for line in path.read_text().splitlines() if line]


# ── file creation ─────────────────────────────────────────────────────────────

class TestFileCreation:
    def test_creates_file_on_first_write(self, tmp_path):
        p = tmp_path / "run.jsonl"
        sid = _add(p)
        try:
            logger.debug("hello")
        finally:
            logger.remove(sid)
        assert p.exists()

    def test_one_line_per_record(self, tmp_path):
        p = tmp_path / "run.jsonl"
        sid = _add(p)
        try:
            logger.debug("a")
            logger.info("b")
            logger.warning("c")
        finally:
            logger.remove(sid)
        assert len(_lines(p)) == 3

    def test_each_line_is_valid_json(self, tmp_path):
        p = tmp_path / "run.jsonl"
        sid = _add(p)
        try:
            for i in range(5):
                logger.debug(f"msg {i}")
        finally:
            logger.remove(sid)
        for line in p.read_text().splitlines():
            json.loads(line)  # must not raise

    def test_appends_across_multiple_calls(self, tmp_path):
        p = tmp_path / "run.jsonl"
        sink = JsonSink(p)
        sid1 = logger.add(sink, level="DEBUG")
        logger.debug("first")
        logger.remove(sid1)

        sid2 = logger.add(sink, level="DEBUG")
        logger.debug("second")
        logger.remove(sid2)

        assert len(_lines(p)) == 2


# ── required fields ───────────────────────────────────────────────────────────

class TestRequiredFields:
    def _record(self, tmp_path: Path) -> dict:
        p = tmp_path / "run.jsonl"
        sid = _add(p)
        try:
            logger.info("field check")
        finally:
            logger.remove(sid)
        return _lines(p)[0]

    def test_all_required_fields_present(self, tmp_path):
        r = self._record(tmp_path)
        for field in ("ts", "level", "msg", "module", "function", "line"):
            assert field in r, f"Missing field: {field}"

    def test_msg_matches_logged_text(self, tmp_path):
        p = tmp_path / "run.jsonl"
        sid = _add(p)
        try:
            logger.info("exact content here")
        finally:
            logger.remove(sid)
        assert _lines(p)[0]["msg"] == "exact content here"

    def test_level_debug(self, tmp_path):
        p = tmp_path / "run.jsonl"
        sid = _add(p)
        try:
            logger.debug("dbg")
        finally:
            logger.remove(sid)
        assert _lines(p)[0]["level"] == "DEBUG"

    def test_level_warning(self, tmp_path):
        p = tmp_path / "run.jsonl"
        sid = _add(p)
        try:
            logger.warning("warn")
        finally:
            logger.remove(sid)
        assert _lines(p)[0]["level"] == "WARNING"

    def test_level_error(self, tmp_path):
        p = tmp_path / "run.jsonl"
        sid = _add(p)
        try:
            logger.error("err")
        finally:
            logger.remove(sid)
        assert _lines(p)[0]["level"] == "ERROR"

    def test_ts_is_parseable_iso8601(self, tmp_path):
        r = self._record(tmp_path)
        datetime.fromisoformat(r["ts"])  # must not raise

    def test_ts_has_millisecond_precision(self, tmp_path):
        r = self._record(tmp_path)
        # isoformat with timespec="milliseconds" always has "." in time part
        time_part = r["ts"].split("T")[1]
        assert "." in time_part

    def test_line_is_integer(self, tmp_path):
        r = self._record(tmp_path)
        assert isinstance(r["line"], int)
        assert r["line"] > 0

    def test_module_is_string(self, tmp_path):
        r = self._record(tmp_path)
        assert isinstance(r["module"], str)
        assert len(r["module"]) > 0

    def test_function_is_string(self, tmp_path):
        r = self._record(tmp_path)
        assert isinstance(r["function"], str)


# ── extra / bind ──────────────────────────────────────────────────────────────

class TestExtraField:
    def test_extra_present_when_bind_used(self, tmp_path):
        p = tmp_path / "run.jsonl"
        sid = _add(p)
        try:
            logger.bind(pipeline="daily").info("bound")
        finally:
            logger.remove(sid)
        r = _lines(p)[0]
        assert "extra" in r
        assert r["extra"]["pipeline"] == "daily"

    def test_extra_absent_without_bind(self, tmp_path):
        p = tmp_path / "run.jsonl"
        sid = _add(p)
        try:
            logger.info("no bind")
        finally:
            logger.remove(sid)
        r = _lines(p)[0]
        # either absent or explicitly empty — both acceptable
        assert "extra" not in r or not r["extra"]

    def test_extra_multiple_bind_keys(self, tmp_path):
        p = tmp_path / "run.jsonl"
        sid = _add(p)
        try:
            logger.bind(pipeline="digest", run_id="abc123").info("multi bind")
        finally:
            logger.remove(sid)
        r = _lines(p)[0]
        assert r["extra"]["pipeline"] == "digest"
        assert r["extra"]["run_id"] == "abc123"

    def test_non_scalar_extra_coerced_to_str(self, tmp_path):
        p = tmp_path / "run.jsonl"
        sid = _add(p)
        try:
            logger.bind(obj={"nested": True}).info("object extra")
        finally:
            logger.remove(sid)
        r = _lines(p)[0]
        # value must be a JSON string, not a dict
        assert isinstance(r["extra"]["obj"], str)


# ── exception capture ─────────────────────────────────────────────────────────

class TestExceptionField:
    def test_exc_field_present_on_exception(self, tmp_path):
        p = tmp_path / "run.jsonl"
        sid = _add(p)
        try:
            try:
                raise ValueError("boom")
            except ValueError:
                logger.exception("caught it")
        finally:
            logger.remove(sid)
        r = _lines(p)[0]
        assert "exc" in r
        assert "ValueError" in r["exc"]
        assert "boom" in r["exc"]

    def test_exc_field_absent_without_exception(self, tmp_path):
        p = tmp_path / "run.jsonl"
        sid = _add(p)
        try:
            logger.info("clean")
        finally:
            logger.remove(sid)
        assert "exc" not in _lines(p)[0]

    def test_exc_contains_traceback(self, tmp_path):
        p = tmp_path / "run.jsonl"
        sid = _add(p)
        try:
            try:
                1 / 0
            except ZeroDivisionError:
                logger.exception("div by zero")
        finally:
            logger.remove(sid)
        r = _lines(p)[0]
        assert "Traceback" in r["exc"]
        assert "ZeroDivisionError" in r["exc"]


# ── rotation ──────────────────────────────────────────────────────────────────

class TestRotation:
    def test_rotates_when_file_exceeds_max_bytes(self, tmp_path):
        path = tmp_path / "run.jsonl"
        sink = JsonSink(path, max_bytes=100)
        sid = logger.add(sink, level="DEBUG")
        try:
            for _ in range(20):
                logger.debug("long enough message to trigger rotation now")
        finally:
            logger.remove(sid)
        backup = tmp_path / "run.1.jsonl"
        assert backup.exists()

    def test_fresh_file_continues_after_rotation(self, tmp_path):
        path = tmp_path / "run.jsonl"
        sink = JsonSink(path, max_bytes=100)
        sid = logger.add(sink, level="DEBUG")
        try:
            for _ in range(20):
                logger.debug("trigger rotation with this long message content")
        finally:
            logger.remove(sid)
        # Current file must exist and be valid JSONL
        assert path.exists()
        for line in path.read_text().splitlines():
            json.loads(line)

    def test_previous_backup_overwritten(self, tmp_path):
        path = tmp_path / "run.jsonl"
        backup = tmp_path / "run.1.jsonl"
        backup.write_text("old backup content\n")
        # Pre-fill path above max_bytes so the very first write triggers rotation
        path.write_text("x" * 20)
        sink = JsonSink(path, max_bytes=10)
        sid = logger.add(sink, level="DEBUG")
        try:
            logger.debug("trigger rotation")
        finally:
            logger.remove(sid)
        assert backup.exists()
        # Backup now holds the old path content, not the original "old backup" text
        assert "old backup content" not in backup.read_text()

    def test_no_rotation_when_under_limit(self, tmp_path):
        path = tmp_path / "run.jsonl"
        backup = tmp_path / "run.1.jsonl"
        sink = JsonSink(path, max_bytes=10 * 1024 * 1024)  # 10 MB — won't trigger
        sid = logger.add(sink, level="DEBUG")
        try:
            logger.debug("small msg")
        finally:
            logger.remove(sid)
        assert not backup.exists()


# ── thread safety ─────────────────────────────────────────────────────────────

class TestThreadSafety:
    def test_concurrent_writes_produce_valid_jsonl(self, tmp_path):
        path = tmp_path / "run.jsonl"
        sink = JsonSink(path)
        sid = logger.add(sink, level="DEBUG")
        errors: list[Exception] = []

        def _write(n: int) -> None:
            try:
                for i in range(20):
                    logger.debug(f"thread {n} message {i}")
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=_write, args=(i,)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        logger.remove(sid)

        assert not errors, f"Thread errors: {errors}"
        lines = path.read_text().splitlines()
        assert len(lines) == 100, f"Expected 100 lines, got {len(lines)}"
        for line in lines:
            json.loads(line)  # every line must be valid JSON


# ── _safe helper ──────────────────────────────────────────────────────────────

class TestSafeHelper:
    @pytest.mark.parametrize("v", [None, True, False, 0, 1, 3.14, "hello"])
    def test_scalars_returned_unchanged(self, v):
        assert _safe(v) is v

    def test_dict_coerced_to_str(self):
        assert isinstance(_safe({"a": 1}), str)

    def test_list_coerced_to_str(self):
        assert isinstance(_safe([1, 2, 3]), str)

    def test_object_coerced_to_str(self):
        class Obj:
            def __str__(self):
                return "myobj"
        assert _safe(Obj()) == "myobj"


# ── integration: setup_logging wires the sink ─────────────────────────────────

class TestSetupLoggingIntegration:
    def test_setup_logging_creates_jsonl_file(self, tmp_path):
        from unittest.mock import patch
        # Patch scrub to be a no-op filter
        with patch("src.pipeline_helpers.logger"):
            pass  # just ensure import works
        from src.pipeline_helpers import _setup_logging
        _setup_logging(tmp_path)
        logger.info("integration test message")
        # Give loguru a moment to flush (it's synchronous, so immediate)
        assert (tmp_path / "run.jsonl").exists()
        lines = _lines(tmp_path / "run.jsonl")
        assert any(r["msg"] == "integration test message" for r in lines)
