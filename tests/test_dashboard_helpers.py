"""Tests for pure helper functions in src/dashboard.py.

dashboard.py is a Streamlit script with module-level st.* calls, so we
stub the entire streamlit module with a MagicMock before importing it.
The _StreamlitMock subclass makes st.columns / st.tabs return the right
number of items so tuple-unpacking in render() doesn't raise ValueError.
"""
from __future__ import annotations

import json
import sys
import time as _time
from pathlib import Path
from unittest.mock import MagicMock, patch

# ── stub streamlit before any import of src.dashboard ─────────────────────────

class _StreamlitMock(MagicMock):
    """MagicMock that returns N items from columns() / tabs() correctly."""

    def columns(self, spec, **kwargs):  # type: ignore[override]
        n = len(spec) if isinstance(spec, (list, tuple)) else int(spec)
        return tuple(MagicMock() for _ in range(n))

    def tabs(self, labels):  # type: ignore[override]
        return tuple(MagicMock() for _ in labels)


if "streamlit" not in sys.modules:
    sys.modules["streamlit"] = _StreamlitMock()

# Import dashboard with time.sleep suppressed (it sleeps REFRESH_SEC at module level)
_real_sleep = _time.sleep
_time.sleep = lambda _: None  # no-op during import
try:
    if "src.dashboard" not in sys.modules:
        import src.dashboard  # noqa: F401 — side-effect import needed for coverage
finally:
    _time.sleep = _real_sleep

from src.dashboard import (  # noqa: E402
    _fmt_duration,
    _load_performance_records,
    _load_scene_retention,
    _load_state,
    _load_weekly_costs,
    _status_badge,
    _steps_map,
    render_ctr_trend,
    render_hook_trend,
    render_scene_retention,
    render_top_videos,
    render_weekly_cost,
)

# ── _load_state ───────────────────────────────────────────────────────────────

class TestLoadState:
    def test_returns_default_when_file_missing(self, tmp_path, monkeypatch):
        monkeypatch.setattr("src.dashboard.STATE_FILE", tmp_path / "nope.json")
        s = _load_state()
        assert s["status"] == "idle"
        assert s["steps"] == []

    def test_reads_existing_file(self, tmp_path, monkeypatch):
        p = tmp_path / "live_state.json"
        p.write_text(json.dumps({"status": "running", "steps": []}))
        monkeypatch.setattr("src.dashboard.STATE_FILE", p)
        s = _load_state()
        assert s["status"] == "running"

    def test_returns_default_on_corrupt_json(self, tmp_path, monkeypatch):
        p = tmp_path / "live_state.json"
        p.write_text("not-json")
        monkeypatch.setattr("src.dashboard.STATE_FILE", p)
        s = _load_state()
        assert s["status"] == "idle"


# ── _status_badge ─────────────────────────────────────────────────────────────

class TestStatusBadge:
    def test_returns_html_span(self):
        html = _status_badge("idle")
        assert "<span" in html
        assert "IDLE" in html

    def test_running_status(self):
        assert "RUNNING" in _status_badge("running")

    def test_unknown_status_uses_fallback_color(self):
        html = _status_badge("unknown_xyz")
        assert "<span" in html

    def test_underscores_replaced_in_label(self):
        assert "_" not in _status_badge("quality_gate_failed").split(">")[1]


# ── _fmt_duration ─────────────────────────────────────────────────────────────

class TestFmtDuration:
    def test_none_returns_dash(self):
        assert _fmt_duration(None) == "—"

    def test_under_60_seconds(self):
        assert "s" in _fmt_duration(45)
        assert "m" not in _fmt_duration(45)

    def test_over_60_seconds(self):
        assert "m" in _fmt_duration(90)

    def test_exactly_0(self):
        assert "s" in _fmt_duration(0)


# ── _steps_map ────────────────────────────────────────────────────────────────

class TestStepsMap:
    def test_all_standard_steps_present(self):
        m = _steps_map({})
        from src.dashboard import STEP_ORDER
        for name in STEP_ORDER:
            assert name in m

    def test_pending_status_for_missing_steps(self):
        m = _steps_map({})
        assert all(v["status"] == "pending" for v in m.values())

    def test_recorded_steps_override_pending(self):
        state = {"steps": [{"name": "scrape", "status": "done"}]}
        m = _steps_map(state)
        assert m["scrape"]["status"] == "done"

    def test_extra_steps_appended(self):
        state = {"steps": [{"name": "custom_step", "status": "done"}]}
        m = _steps_map(state)
        assert "custom_step" in m


# ── _load_performance_records ─────────────────────────────────────────────────

class TestLoadPerformanceRecords:
    def test_returns_empty_list_when_file_missing(self, tmp_path, monkeypatch):
        monkeypatch.setattr("src.dashboard.DATA_DIR", tmp_path)
        assert _load_performance_records() == []

    def test_parses_json_records(self, tmp_path, monkeypatch):
        monkeypatch.setattr("src.dashboard.DATA_DIR", tmp_path)
        (tmp_path / "performance.json").write_text(json.dumps([{"video_id": "abc"}]))
        recs = _load_performance_records()
        assert recs[0]["video_id"] == "abc"

    def test_returns_empty_on_corrupt_json(self, tmp_path, monkeypatch):
        monkeypatch.setattr("src.dashboard.DATA_DIR", tmp_path)
        (tmp_path / "performance.json").write_text("!!bad")
        assert _load_performance_records() == []


# ── _load_weekly_costs ────────────────────────────────────────────────────────

class TestLoadWeeklyCosts:
    def test_returns_empty_when_no_output_dir(self, tmp_path, monkeypatch):
        monkeypatch.setattr("src.dashboard.OUTPUT_DIR", tmp_path / "nonexistent")
        assert _load_weekly_costs() == {}

    def test_picks_up_cost_reports(self, tmp_path, monkeypatch):
        monkeypatch.setattr("src.dashboard.OUTPUT_DIR", tmp_path)
        import datetime as dt
        today = dt.date.today().isoformat()
        run_dir = tmp_path / today
        run_dir.mkdir()
        (run_dir / "cost_report.json").write_text(json.dumps({"total_usd": 1.23}))
        costs = _load_weekly_costs()
        assert today in costs
        assert abs(costs[today] - 1.23) < 1e-9

    def test_ignores_non_date_directories(self, tmp_path, monkeypatch):
        monkeypatch.setattr("src.dashboard.OUTPUT_DIR", tmp_path)
        d = tmp_path / "not-a-date"
        d.mkdir()
        (d / "cost_report.json").write_text(json.dumps({"total_usd": 99.0}))
        assert _load_weekly_costs() == {}

    def test_sums_multiple_pipelines_same_day(self, tmp_path, monkeypatch):
        monkeypatch.setattr("src.dashboard.OUTPUT_DIR", tmp_path)
        import datetime as dt
        today = dt.date.today().isoformat()
        for sub in ("daily", "digest"):
            d = tmp_path / sub / today
            d.mkdir(parents=True)
            (d / "cost_report.json").write_text(json.dumps({"total_usd": 1.0}))
        costs = _load_weekly_costs()
        assert abs(costs[today] - 2.0) < 1e-9

    def test_ignores_old_reports(self, tmp_path, monkeypatch):
        monkeypatch.setattr("src.dashboard.OUTPUT_DIR", tmp_path)
        old = "2000-01-01"
        d = tmp_path / old
        d.mkdir()
        (d / "cost_report.json").write_text(json.dumps({"total_usd": 5.0}))
        assert _load_weekly_costs() == {}


# ── _load_scene_retention ─────────────────────────────────────────────────────

class TestLoadSceneRetention:
    def test_returns_empty_when_file_missing(self, tmp_path, monkeypatch):
        monkeypatch.setattr("src.dashboard.DATA_DIR", tmp_path)
        assert _load_scene_retention() == {}

    def test_parses_scene_scores(self, tmp_path, monkeypatch):
        monkeypatch.setattr("src.dashboard.DATA_DIR", tmp_path)
        (tmp_path / "scene_performance.json").write_text(
            json.dumps({"image": 0.55, "avatar": 0.70})
        )
        data = _load_scene_retention()
        assert abs(data["avatar"] - 0.70) < 1e-9


# ── render functions (streamlit calls are mocked) ─────────────────────────────

class TestRenderHookTrend:
    def test_no_data_does_not_raise(self):
        render_hook_trend([])  # calls st.caption(...)

    def test_with_data_does_not_raise(self):
        records = [
            {"timestamp": "2026-05-17T10:00:00", "avg_view_percentage": 0.65},
            {"timestamp": "2026-05-16T10:00:00", "avg_view_percentage": 0.70},
        ]
        render_hook_trend(records)  # calls st.line_chart(...)

    def test_skips_records_without_retention(self):
        render_hook_trend([{"timestamp": "2026-05-17T10:00:00"}])


class TestRenderCtrTrend:
    def test_no_data_does_not_raise(self):
        render_ctr_trend([])

    def test_with_ctr_data(self):
        records = [{"timestamp": "2026-05-17T10:00:00", "ctr": 0.05}]
        render_ctr_trend(records)


class TestRenderWeeklyCost:
    def test_no_data_does_not_raise(self):
        render_weekly_cost({})

    def test_with_cost_data(self):
        render_weekly_cost({"2026-05-17": 1.23, "2026-05-16": 0.45})


class TestRenderSceneRetention:
    def test_no_data_does_not_raise(self):
        render_scene_retention({})

    def test_with_scene_data(self):
        render_scene_retention({"image": 0.55, "avatar": 0.70, "text_overlay": 0.60})


class TestRenderTopVideos:
    def test_no_records_does_not_raise(self):
        render_top_videos([])

    def test_with_records_no_views_shows_caption(self):
        # records with 0 views → st.caption() path
        render_top_videos([{"title": "T", "views": 0, "avg_view_percentage": 0.5}])

    def test_with_scored_records(self):
        records = [
            {
                "title": "Great video",
                "timestamp": "2026-05-17T10:00:00",
                "content_type": "daily",
                "views": 1000,
                "ctr": 0.05,
                "avg_view_percentage": 0.65,
            }
        ]
        render_top_videos(records)
