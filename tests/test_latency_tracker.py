"""Tests for src/latency_tracker.py."""
from __future__ import annotations

import json

import pytest

from src.latency_tracker import STEP_THRESHOLDS, TOTAL_THRESHOLD, LatencyTracker, fmt_duration

# ── fmt_duration ──────────────────────────────────────────────────────────────

class TestFmtDuration:
    def test_seconds_only(self):
        assert fmt_duration(45) == "45s"

    def test_minutes_and_seconds(self):
        assert fmt_duration(252) == "4m 12s"

    def test_exact_minute(self):
        assert fmt_duration(60) == "1m 0s"

    def test_zero(self):
        assert fmt_duration(0) == "0s"

    def test_one_hour(self):
        assert fmt_duration(3600) == "60m 0s"


# ── LatencyTracker.record ─────────────────────────────────────────────────────

def _trace(steps: list[dict], total_sec: float = 100.0) -> dict:
    return {"steps": steps, "total_sec": total_sec, "status": "success"}


def _step(name: str, dur: float, status: str = "done") -> dict:
    return {"name": name, "duration_sec": dur, "status": status}


class TestRecord:
    def test_stores_step_duration(self, tmp_path):
        t = LatencyTracker(tmp_path)
        t.record(_trace([_step("scrape", 30.0)]))
        assert t._history["scrape"] == [30.0]

    def test_stores_total_duration(self, tmp_path):
        t = LatencyTracker(tmp_path)
        t.record(_trace([], total_sec=500.0))
        assert t._history["_total"] == [500.0]

    def test_multiple_runs_accumulate(self, tmp_path):
        t = LatencyTracker(tmp_path)
        t.record(_trace([_step("scrape", 30.0)]))
        t.record(_trace([_step("scrape", 45.0)]))
        assert t._history["scrape"] == [30.0, 45.0]

    def test_failed_step_is_recorded(self, tmp_path):
        t = LatencyTracker(tmp_path)
        t.record(_trace([_step("voice", 120.0, status="failed")]))
        assert t._history["voice"] == [120.0]

    def test_skipped_step_not_recorded(self, tmp_path):
        t = LatencyTracker(tmp_path)
        t.record(_trace([_step("video", 0.5, status="skipped")]))
        assert "video" not in t._history

    def test_caps_history_at_30_entries(self, tmp_path):
        t = LatencyTracker(tmp_path)
        for i in range(35):
            t.record(_trace([_step("scrape", float(i))]))
        assert len(t._history["scrape"]) == 30

    def test_persists_to_disk(self, tmp_path):
        t = LatencyTracker(tmp_path)
        t.record(_trace([_step("scrape", 42.0)]))
        raw = json.loads((tmp_path / "latency_history.json").read_text())
        assert raw["scrape"] == [42.0]

    def test_loaded_from_disk_on_init(self, tmp_path):
        t1 = LatencyTracker(tmp_path)
        t1.record(_trace([_step("scrape", 42.0)]))
        t2 = LatencyTracker(tmp_path)
        assert t2._history["scrape"] == [42.0]

    def test_step_with_none_duration_ignored(self, tmp_path):
        t = LatencyTracker(tmp_path)
        t.record(_trace([{"name": "scrape", "duration_sec": None, "status": "done"}]))
        assert "scrape" not in t._history


# ── LatencyTracker.p95 ────────────────────────────────────────────────────────

class TestP95:
    def test_returns_none_for_unknown_step(self, tmp_path):
        t = LatencyTracker(tmp_path)
        assert t.p95("nonexistent") is None

    def test_single_value_is_p95(self, tmp_path):
        t = LatencyTracker(tmp_path)
        t.record(_trace([_step("scrape", 50.0)]))
        assert t.p95("scrape") == 50.0

    def test_p95_of_sorted_values(self, tmp_path):
        t = LatencyTracker(tmp_path)
        for v in range(1, 21):  # 20 values: 1..20
            t.record(_trace([_step("scrape", float(v))]))
        # P95 index = int(20 * 0.95) = 19 → sorted[19] = 20.0
        assert t.p95("scrape") == 20.0

    def test_p95_does_not_exceed_max(self, tmp_path):
        t = LatencyTracker(tmp_path)
        for v in [10.0, 20.0, 30.0]:
            t.record(_trace([_step("script", v)]))
        assert t.p95("script") == 30.0


# ── LatencyTracker.slow_steps ─────────────────────────────────────────────────

class TestSlowSteps:
    def test_empty_when_all_fast(self, tmp_path):
        t = LatencyTracker(tmp_path)
        trace = _trace([_step("scrape", 30.0), _step("script", 60.0)])
        assert t.slow_steps(trace) == []

    def test_detects_slow_step(self, tmp_path):
        t = LatencyTracker(tmp_path)
        slow_dur = STEP_THRESHOLDS["voice"] + 1
        result = t.slow_steps(_trace([_step("voice", slow_dur)]))
        assert len(result) == 1
        assert result[0]["name"] == "voice"
        assert result[0]["duration_sec"] == slow_dur

    def test_detects_slow_total(self, tmp_path):
        t = LatencyTracker(tmp_path)
        result = t.slow_steps(_trace([], total_sec=TOTAL_THRESHOLD + 60))
        assert any(r["name"] == "_total" for r in result)

    def test_includes_threshold_in_result(self, tmp_path):
        t = LatencyTracker(tmp_path)
        slow_dur = STEP_THRESHOLDS["video"] + 100
        result = t.slow_steps(_trace([_step("video", slow_dur)]))
        assert result[0]["threshold_sec"] == STEP_THRESHOLDS["video"]

    def test_includes_p95_when_available(self, tmp_path):
        t = LatencyTracker(tmp_path)
        t.record(_trace([_step("video", 200.0)]))  # seed history
        slow_dur = STEP_THRESHOLDS["video"] + 100
        result = t.slow_steps(_trace([_step("video", slow_dur)]))
        assert result[0]["p95_sec"] is not None

    def test_step_at_exact_threshold_not_slow(self, tmp_path):
        t = LatencyTracker(tmp_path)
        result = t.slow_steps(_trace([_step("scrape", STEP_THRESHOLDS["scrape"])]))
        assert result == []

    def test_multiple_slow_steps(self, tmp_path):
        t = LatencyTracker(tmp_path)
        result = t.slow_steps(_trace([
            _step("voice",  STEP_THRESHOLDS["voice"]  + 1),
            _step("video",  STEP_THRESHOLDS["video"]  + 1),
        ]))
        names = {r["name"] for r in result}
        assert names == {"voice", "video"}


# ── LatencyTracker.step_summary ───────────────────────────────────────────────

class TestStepSummary:
    def test_returns_rows_for_completed_steps(self, tmp_path):
        t = LatencyTracker(tmp_path)
        rows = t.step_summary(_trace([
            _step("scrape", 45.0),
            _step("script", 180.0),
        ]))
        assert len(rows) == 2
        assert rows[0]["name"] == "scrape"
        assert rows[0]["duration_str"] == "45s"

    def test_slow_flag_set_correctly(self, tmp_path):
        t = LatencyTracker(tmp_path)
        rows = t.step_summary(_trace([
            _step("scrape", STEP_THRESHOLDS["scrape"] + 1),
            _step("script", 10.0),
        ]))
        slow_row   = next(r for r in rows if r["name"] == "scrape")
        normal_row = next(r for r in rows if r["name"] == "script")
        assert slow_row["slow"] is True
        assert normal_row["slow"] is False

    def test_unknown_step_not_flagged_slow(self, tmp_path):
        t = LatencyTracker(tmp_path)
        rows = t.step_summary(_trace([_step("custom_step", 9999.0)]))
        assert rows[0]["slow"] is False

    def test_none_duration_step_excluded(self, tmp_path):
        t = LatencyTracker(tmp_path)
        rows = t.step_summary(_trace([
            {"name": "scrape", "duration_sec": None, "status": "done"},
            _step("script", 60.0),
        ]))
        assert len(rows) == 1
        assert rows[0]["name"] == "script"
