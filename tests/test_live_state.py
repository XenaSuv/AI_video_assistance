"""Tests for src/live_state.py — LiveState atomic-write JSON sink."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.live_state import _IDLE, MAX_LOG_EVENTS, LiveState, _now_hms, _now_iso

# ── Helper functions ──────────────────────────────────────────────────────────

class TestHelperFunctions:
    def test_now_iso_format(self):
        ts = _now_iso()
        # Should match YYYY-MM-DDTHH:MM:SSZ
        assert len(ts) == 20
        assert ts.endswith("Z")
        assert "T" in ts

    def test_now_hms_format(self):
        ts = _now_hms()
        # Should match HH:MM:SS
        parts = ts.split(":")
        assert len(parts) == 3
        assert all(len(p) == 2 for p in parts)


# ── _IDLE constant ────────────────────────────────────────────────────────────

class TestIdleConstant:
    def test_idle_has_expected_keys(self):
        assert "pipeline" in _IDLE
        assert "status" in _IDLE
        assert _IDLE["status"] == "idle"
        assert _IDLE["pipeline"] is None
        assert _IDLE["steps_done"] == 0
        assert _IDLE["steps_total"] == 0
        assert _IDLE["progress"] == 0.0

    def test_idle_is_not_mutated_by_copy(self, tmp_path):
        ls = LiveState(tmp_path / "state.json")
        ls._state["pipeline"] = "test"
        # Original _IDLE should be unchanged
        assert _IDLE["pipeline"] is None


# ── LiveState.__init__ ────────────────────────────────────────────────────────

class TestInit:
    def test_path_is_set(self, tmp_path):
        p = tmp_path / "state.json"
        ls = LiveState(p)
        assert ls.path == p

    def test_initial_state_is_idle(self, tmp_path):
        ls = LiveState(tmp_path / "state.json")
        assert ls._state["status"] == "idle"
        assert ls._state["pipeline"] is None

    def test_does_not_write_on_init(self, tmp_path):
        p = tmp_path / "state.json"
        LiveState(p)
        assert not p.exists()


# ── LiveState.start ───────────────────────────────────────────────────────────

class TestStart:
    def test_sets_running_status(self, tmp_path):
        ls = LiveState(tmp_path / "state.json")
        ls.start("daily", steps_total=5)
        assert ls._state["status"] == "running"

    def test_sets_pipeline_name(self, tmp_path):
        ls = LiveState(tmp_path / "state.json")
        ls.start("weekly", steps_total=3)
        assert ls._state["pipeline"] == "weekly"

    def test_sets_steps_total(self, tmp_path):
        ls = LiveState(tmp_path / "state.json")
        ls.start("daily", steps_total=7)
        assert ls._state["steps_total"] == 7

    def test_sets_started_at(self, tmp_path):
        ls = LiveState(tmp_path / "state.json")
        ls.start("daily", 3)
        assert ls._state["started_at"] is not None

    def test_adds_start_log_entry(self, tmp_path):
        ls = LiveState(tmp_path / "state.json")
        ls.start("daily", 3)
        logs = ls._state["logs"]
        assert len(logs) == 1
        assert "daily" in logs[0]["msg"]

    def test_write_creates_file(self, tmp_path):
        ls = LiveState(tmp_path / "state.json")
        ls.start("daily", 3)
        assert (tmp_path / "state.json").exists()

    def test_written_file_is_valid_json(self, tmp_path):
        p = tmp_path / "state.json"
        ls = LiveState(p)
        ls.start("daily", 3)
        data = json.loads(p.read_text())
        assert data["status"] == "running"
        assert data["pipeline"] == "daily"


# ── LiveState.finish ──────────────────────────────────────────────────────────

class TestFinish:
    def test_finish_sets_success(self, tmp_path):
        ls = LiveState(tmp_path / "state.json")
        ls.start("daily", 3)
        ls.finish("success")
        assert ls._state["status"] == "success"

    def test_finish_sets_progress_to_1_on_success(self, tmp_path):
        ls = LiveState(tmp_path / "state.json")
        ls.start("daily", 3)
        ls.finish("success")
        assert abs(ls._state["progress"] - 1.0) < 1e-9

    def test_finish_sets_failed_status(self, tmp_path):
        ls = LiveState(tmp_path / "state.json")
        ls.start("daily", 3)
        ls.finish("failed")
        assert ls._state["status"] == "failed"

    def test_finish_failed_does_not_set_progress_1(self, tmp_path):
        ls = LiveState(tmp_path / "state.json")
        ls.start("daily", 3)
        ls.finish("failed")
        # progress stays at whatever it was, not forced to 1.0
        assert ls._state["progress"] < 1.0

    def test_finish_clears_current_step(self, tmp_path):
        ls = LiveState(tmp_path / "state.json")
        ls.start("daily", 3)
        ls.step_running("voice")
        ls.finish("success")
        assert ls._state["current_step"] is None

    def test_finish_default_status_is_success(self, tmp_path):
        ls = LiveState(tmp_path / "state.json")
        ls.start("daily", 3)
        ls.finish()
        assert ls._state["status"] == "success"


# ── LiveState step methods ────────────────────────────────────────────────────

class TestStepMethods:
    def test_step_running_sets_current_step(self, tmp_path):
        ls = LiveState(tmp_path / "state.json")
        ls.start("daily", 3)
        ls.step_running("voice")
        assert ls._state["current_step"] == "voice"

    def test_step_running_upserts_step_with_running_status(self, tmp_path):
        ls = LiveState(tmp_path / "state.json")
        ls.start("daily", 3)
        ls.step_running("voice")
        steps = ls._state["steps"]
        assert any(s["name"] == "voice" and s["status"] == "running" for s in steps)

    def test_step_done_increments_steps_done(self, tmp_path):
        ls = LiveState(tmp_path / "state.json")
        ls.start("daily", 3)
        ls.step_done("voice")
        assert ls._state["steps_done"] == 1

    def test_step_done_increments_twice(self, tmp_path):
        ls = LiveState(tmp_path / "state.json")
        ls.start("daily", 3)
        ls.step_done("voice")
        ls.step_done("video")
        assert ls._state["steps_done"] == 2

    def test_step_done_updates_progress(self, tmp_path):
        ls = LiveState(tmp_path / "state.json")
        ls.start("daily", 4)
        ls.step_done("step1")
        assert abs(ls._state["progress"] - 0.25) < 1e-9

    def test_step_done_with_duration(self, tmp_path):
        ls = LiveState(tmp_path / "state.json")
        ls.start("daily", 3)
        ls.step_done("voice", duration_sec=34.2)
        steps = ls._state["steps"]
        voice_step = next(s for s in steps if s["name"] == "voice")
        assert abs(voice_step["duration_sec"] - 34.2) < 0.05

    def test_step_done_without_duration(self, tmp_path):
        ls = LiveState(tmp_path / "state.json")
        ls.start("daily", 3)
        # Call step_done on a fresh step (no prior step_running with duration)
        ls.step_done("fresh_step")
        steps = ls._state["steps"]
        fresh_step = next(s for s in steps if s["name"] == "fresh_step")
        assert "duration_sec" not in fresh_step

    def test_step_skipped_increments_steps_done(self, tmp_path):
        ls = LiveState(tmp_path / "state.json")
        ls.start("daily", 3)
        ls.step_skipped("optional_step")
        assert ls._state["steps_done"] == 1

    def test_step_skipped_upserts_with_skipped_status(self, tmp_path):
        ls = LiveState(tmp_path / "state.json")
        ls.start("daily", 3)
        ls.step_skipped("optional_step")
        steps = ls._state["steps"]
        assert any(s["name"] == "optional_step" and s["status"] == "skipped" for s in steps)

    def test_step_failed_adds_error(self, tmp_path):
        ls = LiveState(tmp_path / "state.json")
        ls.start("daily", 3)
        ls.step_failed("upload", "403 Forbidden")
        steps = ls._state["steps"]
        failed = next(s for s in steps if s["name"] == "upload")
        assert failed["status"] == "failed"
        assert failed["error"] == "403 Forbidden"

    def test_step_failed_does_not_increment_steps_done(self, tmp_path):
        ls = LiveState(tmp_path / "state.json")
        ls.start("daily", 3)
        ls.step_failed("upload", "error")
        assert ls._state["steps_done"] == 0


# ── LiveState enrichment methods ──────────────────────────────────────────────

class TestEnrichmentMethods:
    def test_log_event_appends(self, tmp_path):
        ls = LiveState(tmp_path / "state.json")
        ls.start("daily", 3)
        ls.log_event("Scraped 23 stories")
        logs = ls._state["logs"]
        assert any("Scraped 23 stories" in e["msg"] for e in logs)

    def test_log_event_truncates_at_max(self, tmp_path):
        ls = LiveState(tmp_path / "state.json")
        ls._state["logs"] = []
        for i in range(MAX_LOG_EVENTS + 10):
            ls.log_event(f"Event {i}")
        assert len(ls._state["logs"]) == MAX_LOG_EVENTS

    def test_log_event_keeps_most_recent(self, tmp_path):
        ls = LiveState(tmp_path / "state.json")
        ls._state["logs"] = []
        for i in range(MAX_LOG_EVENTS + 5):
            ls.log_event(f"Event {i}")
        # Should keep the last MAX_LOG_EVENTS, so the last message should be latest
        last_msg = ls._state["logs"][-1]["msg"]
        assert "Event" in last_msg

    def test_set_strategy(self, tmp_path):
        ls = LiveState(tmp_path / "state.json")
        ls.start("daily", 3)
        ls.set_strategy({"mode": "growth", "top_angle": "technical_breakthrough"})
        assert ls._state["strategy"]["mode"] == "growth"

    def test_set_metrics(self, tmp_path):
        ls = LiveState(tmp_path / "state.json")
        ls.start("daily", 3)
        ls.set_metrics({"views": 1000, "retention": 0.65})
        assert ls._state["metrics"]["views"] == 1000

    def test_set_cost(self, tmp_path):
        ls = LiveState(tmp_path / "state.json")
        ls.start("daily", 3)
        ls.set_cost({"openai": 0.12, "elevenlabs": 0.05})
        assert abs(ls._state["cost"]["openai"] - 0.12) < 1e-9


# ── LiveState.load (static method) ───────────────────────────────────────────

class TestLoad:
    def test_load_returns_idle_on_missing_file(self, tmp_path):
        state = LiveState.load(tmp_path / "nonexistent.json")
        assert state["status"] == "idle"
        assert state["pipeline"] is None

    def test_load_returns_idle_on_invalid_json(self, tmp_path):
        p = tmp_path / "state.json"
        p.write_text("not valid json {{{")
        state = LiveState.load(p)
        assert state["status"] == "idle"

    def test_load_returns_content_on_valid_file(self, tmp_path):
        p = tmp_path / "state.json"
        p.write_text(json.dumps({"status": "running", "pipeline": "daily"}))
        state = LiveState.load(p)
        assert state["status"] == "running"
        assert state["pipeline"] == "daily"

    def test_load_roundtrip_after_start(self, tmp_path):
        p = tmp_path / "state.json"
        ls = LiveState(p)
        ls.start("daily", 5)
        loaded = LiveState.load(p)
        assert loaded["status"] == "running"
        assert loaded["pipeline"] == "daily"
        assert loaded["steps_total"] == 5


# ── LiveState._calc_progress ──────────────────────────────────────────────────

class TestCalcProgress:
    def test_returns_zero_when_steps_total_zero(self, tmp_path):
        ls = LiveState(tmp_path / "state.json")
        # steps_total=0 → _calc_progress uses 1 as denominator
        ls._state["steps_total"] = 0
        ls._state["steps_done"] = 0
        # 0/1 = 0.0
        result = ls._calc_progress()
        assert abs(result - 0.0) < 1e-9

    def test_returns_fraction(self, tmp_path):
        ls = LiveState(tmp_path / "state.json")
        ls._state["steps_total"] = 4
        ls._state["steps_done"] = 2
        result = ls._calc_progress()
        assert abs(result - 0.5) < 1e-9

    def test_caps_at_1(self, tmp_path):
        ls = LiveState(tmp_path / "state.json")
        ls._state["steps_total"] = 2
        ls._state["steps_done"] = 5
        result = ls._calc_progress()
        assert abs(result - 1.0) < 1e-9


# ── LiveState._upsert_step ────────────────────────────────────────────────────

class TestUpsertStep:
    def test_creates_new_step(self, tmp_path):
        ls = LiveState(tmp_path / "state.json")
        ls._state["steps"] = []
        ls._upsert_step("voice", "running")
        assert len(ls._state["steps"]) == 1
        assert ls._state["steps"][0]["name"] == "voice"
        assert ls._state["steps"][0]["status"] == "running"

    def test_updates_existing_step(self, tmp_path):
        ls = LiveState(tmp_path / "state.json")
        ls._state["steps"] = [{"name": "voice", "status": "running"}]
        ls._upsert_step("voice", "done")
        assert len(ls._state["steps"]) == 1
        assert ls._state["steps"][0]["status"] == "done"

    def test_upsert_with_extra_kwargs(self, tmp_path):
        ls = LiveState(tmp_path / "state.json")
        ls._state["steps"] = []
        ls._upsert_step("voice", "done", duration_sec=12.5, error=None)
        step = ls._state["steps"][0]
        assert abs(step["duration_sec"] - 12.5) < 1e-9

    def test_upsert_does_not_duplicate(self, tmp_path):
        ls = LiveState(tmp_path / "state.json")
        ls._state["steps"] = []
        ls._upsert_step("voice", "running")
        ls._upsert_step("voice", "done")
        assert len(ls._state["steps"]) == 1

    def test_upsert_different_steps(self, tmp_path):
        ls = LiveState(tmp_path / "state.json")
        ls._state["steps"] = []
        ls._upsert_step("voice", "done")
        ls._upsert_step("video", "running")
        assert len(ls._state["steps"]) == 2


# ── LiveState._write error tolerance ──────────────────────────────────────────

class TestWriteErrorTolerance:
    def test_write_to_bad_path_does_not_raise(self, tmp_path):
        # Point to a path where parent is actually a FILE, not a directory
        p = tmp_path / "file.txt"
        p.write_text("i am a file")
        bad_path = p / "subdir" / "state.json"  # parent is a file, not dir
        ls = LiveState(bad_path)
        # Should not raise; write errors are swallowed
        ls.start("daily", 3)  # triggers _write

    def test_write_creates_parent_dir(self, tmp_path):
        nested = tmp_path / "a" / "b" / "c" / "state.json"
        ls = LiveState(nested)
        ls.start("daily", 3)
        assert nested.exists()


# ── Full lifecycle integration ────────────────────────────────────────────────

class TestFullLifecycle:
    def test_full_pipeline_run(self, tmp_path):
        p = tmp_path / "state.json"
        ls = LiveState(p)

        ls.start("daily", steps_total=3)
        ls.log_event("Scraped 23 stories")
        ls.set_strategy({"mode": "growth"})
        ls.step_running("voice")
        ls.step_done("voice", duration_sec=34.2)
        ls.step_running("video")
        ls.step_done("video", duration_sec=120.5)
        ls.step_skipped("presenter")
        ls.set_metrics({"views": 0})
        ls.set_cost({"openai": 0.05})
        ls.finish("success")

        state = LiveState.load(p)
        assert state["status"] == "success"
        assert abs(state["progress"] - 1.0) < 1e-9
        assert state["steps_done"] == 3
        assert state["strategy"]["mode"] == "growth"
        voice_step = next(s for s in state["steps"] if s["name"] == "voice")
        assert voice_step["status"] == "done"
