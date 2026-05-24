"""Tests for src/watchdog.py — find_last_success, check, check_all, heartbeat."""
from __future__ import annotations

import datetime as dt
import json
import math
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pytest

from src.watchdog import (
    PipelineHealth,
    _PIPELINE_THRESHOLDS,
    check,
    check_all,
    find_last_success,
    heartbeat,
)

UTC = dt.timezone.utc
NOW = dt.datetime(2026, 5, 16, 12, 0, 0, tzinfo=UTC)


# ── helpers ────────────────────────────────────────────────────────────────────

def _write_trace(
    run_dir: Path,
    pipeline: str = "daily",
    status: str = "success",
    finished_at: str = "2026-05-16T10:00:00Z",
) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "pipeline_trace.json").write_text(
        json.dumps({
            "pipeline":    pipeline,
            "started_at":  "2026-05-16T08:00:00Z",
            "finished_at": finished_at,
            "total_sec":   7200.0,
            "status":      status,
            "steps":       [],
        })
    )


# ── find_last_success ──────────────────────────────────────────────────────────

class TestFindLastSuccess:
    def test_returns_none_when_empty_dir(self, tmp_path):
        assert find_last_success(tmp_path) is None

    def test_returns_none_when_no_success(self, tmp_path):
        _write_trace(tmp_path / "2026-05-16", status="failed")
        assert find_last_success(tmp_path) is None

    def test_returns_none_for_wrong_pipeline(self, tmp_path):
        _write_trace(tmp_path / "2026-05-16", pipeline="weekly")
        assert find_last_success(tmp_path, pipeline="daily") is None

    def test_returns_timestamp_for_single_success(self, tmp_path):
        _write_trace(tmp_path / "2026-05-16", finished_at="2026-05-16T10:00:00Z")
        result = find_last_success(tmp_path)
        assert result is not None
        assert result.year == 2026 and result.hour == 10

    def test_returns_most_recent_when_multiple(self, tmp_path):
        _write_trace(tmp_path / "2026-05-14", finished_at="2026-05-14T10:00:00Z")
        _write_trace(tmp_path / "2026-05-15", finished_at="2026-05-15T10:00:00Z")
        _write_trace(tmp_path / "2026-05-16", finished_at="2026-05-16T10:00:00Z")
        result = find_last_success(tmp_path)
        assert result is not None
        assert result.day == 16

    def test_ignores_failed_run_from_today(self, tmp_path):
        _write_trace(tmp_path / "2026-05-15", finished_at="2026-05-15T10:00:00Z")
        _write_trace(tmp_path / "2026-05-16", status="failed", finished_at="2026-05-16T10:00:00Z")
        result = find_last_success(tmp_path)
        assert result is not None
        assert result.day == 15

    def test_skips_missing_finished_at(self, tmp_path):
        run_dir = tmp_path / "2026-05-16"
        run_dir.mkdir(parents=True)
        (run_dir / "pipeline_trace.json").write_text(
            json.dumps({"pipeline": "daily", "status": "success", "finished_at": None, "steps": []})
        )
        assert find_last_success(tmp_path) is None

    def test_skips_invalid_json(self, tmp_path):
        run_dir = tmp_path / "2026-05-16"
        run_dir.mkdir(parents=True)
        (run_dir / "pipeline_trace.json").write_text("not { json")
        assert find_last_success(tmp_path) is None

    def test_skips_malformed_timestamp(self, tmp_path):
        run_dir = tmp_path / "2026-05-16"
        run_dir.mkdir(parents=True)
        (run_dir / "pipeline_trace.json").write_text(
            json.dumps({"pipeline": "daily", "status": "success", "finished_at": "not-a-date", "steps": []})
        )
        assert find_last_success(tmp_path) is None

    def test_result_is_utc_aware(self, tmp_path):
        _write_trace(tmp_path / "2026-05-16", finished_at="2026-05-16T10:00:00Z")
        result = find_last_success(tmp_path)
        assert result is not None
        assert result.tzinfo is not None


# ── check ──────────────────────────────────────────────────────────────────────

class TestCheck:
    def _run_check(self, tmp_path, max_hours=36.0, **trace_kwargs):
        with patch("src.watchdog.notify_watchdog_alert") as mock_alert:
            alerted = check(
                pipeline="daily",
                max_hours=max_hours,
                output_dir=tmp_path,
                now=NOW,
            )
        return alerted, mock_alert

    def test_returns_false_when_recent(self, tmp_path):
        # 2h ago — well within 36h threshold
        _write_trace(tmp_path / "2026-05-16", finished_at="2026-05-16T10:00:00Z")
        alerted, mock_alert = self._run_check(tmp_path)
        assert not alerted
        mock_alert.assert_not_called()

    def test_returns_true_and_alerts_when_stale(self, tmp_path):
        # 50h ago — exceeds 36h threshold
        _write_trace(tmp_path / "2026-05-14", finished_at="2026-05-14T10:00:00Z")
        alerted, mock_alert = self._run_check(tmp_path)
        assert alerted
        mock_alert.assert_called_once()

    def test_returns_true_and_alerts_when_no_success_found(self, tmp_path):
        alerted, mock_alert = self._run_check(tmp_path)
        assert alerted
        mock_alert.assert_called_once()
        kwargs = mock_alert.call_args[1]
        assert kwargs["last_run_at"] is None

    def test_alert_called_with_correct_pipeline(self, tmp_path):
        alerted, mock_alert = self._run_check(tmp_path)
        kwargs = mock_alert.call_args[1]
        assert kwargs["pipeline"] == "daily"

    def test_alert_hours_since_infinite_when_no_success(self, tmp_path):
        _, mock_alert = self._run_check(tmp_path)
        kwargs = mock_alert.call_args[1]
        assert kwargs["hours_since"] == float("inf")

    def test_alert_hours_since_accurate(self, tmp_path):
        # 50h ago
        _write_trace(tmp_path / "2026-05-14", finished_at="2026-05-14T10:00:00Z")
        _, mock_alert = self._run_check(tmp_path)
        kwargs = mock_alert.call_args[1]
        assert 49.0 < kwargs["hours_since"] < 51.0

    def test_exactly_at_threshold_does_not_alert(self, tmp_path):
        # exactly 36h ago
        ts = (NOW - dt.timedelta(hours=36)).strftime("%Y-%m-%dT%H:%M:%SZ")
        _write_trace(tmp_path / "2026-05-15", finished_at=ts)
        alerted, mock_alert = self._run_check(tmp_path, max_hours=36.0)
        assert not alerted
        mock_alert.assert_not_called()

    def test_custom_max_hours_respected(self, tmp_path):
        # 10h ago — fine under 36h but over 6h
        _write_trace(tmp_path / "2026-05-16", finished_at="2026-05-16T02:00:00Z")
        alerted, mock_alert = self._run_check(tmp_path, max_hours=6.0)
        assert alerted
        mock_alert.assert_called_once()


# ── notify_watchdog_alert (integration check via slack_notifier) ───────────────

class TestNotifyWatchdogAlert:
    def _run(self, last_run_at, hours_since):
        from src.slack_notifier import notify_watchdog_alert
        with patch("src.slack_notifier._http_post") as mock_post:
            with patch("src.slack_notifier.settings") as mock_settings:
                mock_settings.slack_webhook_url = "https://hooks.slack.com/test"
                notify_watchdog_alert(
                    last_run_at=last_run_at,
                    hours_since=hours_since,
                    pipeline="daily",
                )
        return mock_post

    def test_posts_when_webhook_set(self):
        mock_post = self._run(last_run_at=None, hours_since=float("inf"))
        mock_post.assert_called_once()

    def test_color_is_red(self):
        mock_post = self._run(last_run_at=None, hours_since=float("inf"))
        color = mock_post.call_args[1]["json"]["attachments"][0]["color"]
        assert color == "#e01e5a"

    def test_message_contains_pipeline_name(self):
        mock_post = self._run(last_run_at=None, hours_since=float("inf"))
        text = mock_post.call_args[1]["json"]["attachments"][0]["text"]
        assert "Daily" in text or "daily" in text

    def test_message_contains_no_run_found_when_none(self):
        mock_post = self._run(last_run_at=None, hours_since=float("inf"))
        text = mock_post.call_args[1]["json"]["attachments"][0]["text"]
        assert "No successful run" in text

    def test_message_contains_last_run_time_when_known(self):
        last = dt.datetime(2026, 5, 14, 10, 0, 0, tzinfo=UTC)
        mock_post = self._run(last_run_at=last, hours_since=50.0)
        text = mock_post.call_args[1]["json"]["attachments"][0]["text"]
        assert "2026-05-14" in text

    def test_no_post_when_no_webhook(self):
        from src.slack_notifier import notify_watchdog_alert
        with patch("src.slack_notifier._http_post") as mock_post:
            with patch("src.slack_notifier.settings") as mock_settings:
                mock_settings.slack_webhook_url = ""
                notify_watchdog_alert(last_run_at=None, hours_since=999.0, pipeline="daily")
        mock_post.assert_not_called()


# ── PipelineHealth ─────────────────────────────────────────────────────────────

class TestPipelineHealth:
    def _health(self, last_run_at, hours_since, threshold_hours, is_stale):
        return PipelineHealth(
            pipeline="daily",
            last_run_at=last_run_at,
            hours_since=hours_since,
            threshold_hours=threshold_hours,
            is_stale=is_stale,
        )

    def test_emoji_ok_when_recent(self):
        h = self._health(NOW, 5.0, 50.0, False)
        assert h.status_emoji == "✅"

    def test_emoji_warning_when_approaching_threshold(self):
        # 80% of 50h = 40h threshold; hours_since=41 => warning
        h = self._health(NOW, 41.0, 50.0, False)
        assert h.status_emoji == "⚠️"

    def test_emoji_red_when_stale(self):
        h = self._health(NOW, 60.0, 50.0, True)
        assert h.status_emoji == "🔴"

    def test_emoji_black_when_no_run(self):
        h = self._health(None, float("inf"), 50.0, True)
        assert h.status_emoji == "⚫"

    def test_warning_boundary_exact_75pct(self):
        # exactly 75% → hours_since > threshold * 0.75 is False when equal, so still ✅
        h = self._health(NOW, 37.5, 50.0, False)
        assert h.status_emoji == "✅"

    def test_warning_boundary_just_above_75pct(self):
        h = self._health(NOW, 37.6, 50.0, False)
        assert h.status_emoji == "⚠️"


# ── check_all ──────────────────────────────────────────────────────────────────

class TestCheckAll:
    _THRESHOLDS = {"daily": 50.0, "weekly": 80.0}

    def test_returns_one_entry_per_pipeline(self, tmp_path):
        result = check_all(output_dir=tmp_path, now=NOW, thresholds=self._THRESHOLDS)
        assert len(result) == 2
        pipelines = {r.pipeline for r in result}
        assert pipelines == {"daily", "weekly"}

    def test_healthy_when_recent(self, tmp_path):
        _write_trace(tmp_path / "2026-05-16", pipeline="daily", finished_at="2026-05-16T10:00:00Z")
        result = check_all(output_dir=tmp_path, now=NOW, thresholds=self._THRESHOLDS)
        daily = next(r for r in result if r.pipeline == "daily")
        assert not daily.is_stale
        assert daily.last_run_at is not None

    def test_stale_when_old(self, tmp_path):
        # 51h ago — exceeds 50h threshold
        _write_trace(tmp_path / "2026-05-14", pipeline="daily", finished_at="2026-05-14T09:00:00Z")
        result = check_all(output_dir=tmp_path, now=NOW, thresholds=self._THRESHOLDS)
        daily = next(r for r in result if r.pipeline == "daily")
        assert daily.is_stale

    def test_stale_when_no_run(self, tmp_path):
        result = check_all(output_dir=tmp_path, now=NOW, thresholds=self._THRESHOLDS)
        for r in result:
            assert r.is_stale
            assert r.last_run_at is None
            assert r.hours_since == float("inf")

    def test_threshold_stored_on_health(self, tmp_path):
        result = check_all(output_dir=tmp_path, now=NOW, thresholds=self._THRESHOLDS)
        weekly = next(r for r in result if r.pipeline == "weekly")
        assert weekly.threshold_hours == 80.0

    def test_hours_since_accurate(self, tmp_path):
        _write_trace(tmp_path / "2026-05-16", pipeline="daily", finished_at="2026-05-16T10:00:00Z")
        result = check_all(output_dir=tmp_path, now=NOW, thresholds=self._THRESHOLDS)
        daily = next(r for r in result if r.pipeline == "daily")
        assert 1.9 < daily.hours_since < 2.1

    def test_uses_default_thresholds_when_none_passed(self, tmp_path):
        result = check_all(output_dir=tmp_path, now=NOW)
        pipelines = {r.pipeline for r in result}
        assert pipelines == set(_PIPELINE_THRESHOLDS.keys())


# ── heartbeat ──────────────────────────────────────────────────────────────────

class TestHeartbeat:
    _THRESHOLDS = {"daily": 50.0, "weekly": 80.0}

    def _run(self, tmp_path, **trace_kwargs):
        with patch("src.watchdog.notify_heartbeat") as mock_board:
            with patch("src.watchdog.notify_watchdog_alert") as mock_alert:
                result = heartbeat(
                    output_dir=tmp_path,
                    now=NOW,
                    thresholds=self._THRESHOLDS,
                )
        return result, mock_board, mock_alert

    def test_returns_pipeline_health_list(self, tmp_path):
        result, _, _ = self._run(tmp_path)
        assert len(result) == 2
        assert all(isinstance(r, PipelineHealth) for r in result)

    def test_notify_heartbeat_called_once(self, tmp_path):
        _, mock_board, _ = self._run(tmp_path)
        mock_board.assert_called_once()

    def test_notify_heartbeat_receives_all_statuses(self, tmp_path):
        result, mock_board, _ = self._run(tmp_path)
        passed = mock_board.call_args[0][0]
        assert len(passed) == len(result)

    def test_alert_sent_for_each_stale(self, tmp_path):
        # no traces → both stale
        _, _, mock_alert = self._run(tmp_path)
        assert mock_alert.call_count == 2

    def test_alert_not_sent_for_healthy(self, tmp_path):
        _write_trace(tmp_path / "2026-05-16", pipeline="daily", finished_at="2026-05-16T10:00:00Z")
        _write_trace(tmp_path / "2026-05-16b", pipeline="weekly", finished_at="2026-05-16T10:00:00Z")
        _, _, mock_alert = self._run(tmp_path)
        mock_alert.assert_not_called()

    def test_partial_stale_alerts_only_stale(self, tmp_path):
        # daily is recent, weekly has no run
        _write_trace(tmp_path / "2026-05-16", pipeline="daily", finished_at="2026-05-16T10:00:00Z")
        _, _, mock_alert = self._run(tmp_path)
        assert mock_alert.call_count == 1
        kwargs = mock_alert.call_args[1]
        assert kwargs["pipeline"] == "weekly"

    def test_alert_kwargs_match_health(self, tmp_path):
        _, _, mock_alert = self._run(tmp_path)
        calls_by_pipeline = {c[1]["pipeline"]: c[1] for c in mock_alert.call_args_list}
        assert "daily" in calls_by_pipeline
        assert calls_by_pipeline["daily"]["last_run_at"] is None
        assert calls_by_pipeline["daily"]["hours_since"] == float("inf")


# ── notify_heartbeat (integration check via slack_notifier) ───────────────────

class TestNotifyHeartbeat:
    def _make_status(self, pipeline, last_run_at, hours_since, threshold, is_stale):
        return PipelineHealth(
            pipeline=pipeline,
            last_run_at=last_run_at,
            hours_since=hours_since,
            threshold_hours=threshold,
            is_stale=is_stale,
        )

    def _run(self, statuses):
        from src.slack_notifier import notify_heartbeat
        with patch("src.slack_notifier._http_post") as mock_post:
            with patch("src.slack_notifier.settings") as mock_settings:
                mock_settings.slack_webhook_url = "https://hooks.slack.com/test"
                notify_heartbeat(statuses)
        return mock_post

    def test_posts_once(self):
        s = self._make_status("daily", NOW, 2.0, 50.0, False)
        mock_post = self._run([s])
        mock_post.assert_called_once()

    def test_color_green_when_all_healthy(self):
        s = self._make_status("daily", NOW, 2.0, 50.0, False)
        mock_post = self._run([s])
        color = mock_post.call_args[1]["json"]["attachments"][0]["color"]
        assert color == "#2eb886"

    def test_color_red_when_any_stale(self):
        s = self._make_status("daily", None, float("inf"), 50.0, True)
        mock_post = self._run([s])
        color = mock_post.call_args[1]["json"]["attachments"][0]["color"]
        assert color == "#e01e5a"

    def test_text_contains_pipeline_name(self):
        s = self._make_status("daily", NOW, 2.0, 50.0, False)
        mock_post = self._run([s])
        text = mock_post.call_args[1]["json"]["attachments"][0]["text"]
        assert "daily" in text

    def test_text_contains_no_run_found_when_none(self):
        s = self._make_status("daily", None, float("inf"), 50.0, True)
        mock_post = self._run([s])
        text = mock_post.call_args[1]["json"]["attachments"][0]["text"]
        assert "no run found" in text

    def test_text_contains_hours_when_run_known(self):
        s = self._make_status("daily", NOW, 2.0, 50.0, False)
        mock_post = self._run([s])
        text = mock_post.call_args[1]["json"]["attachments"][0]["text"]
        assert "2.0h" in text

    def test_no_post_when_no_webhook(self):
        from src.slack_notifier import notify_heartbeat
        s = self._make_status("daily", None, float("inf"), 50.0, True)
        with patch("src.slack_notifier._http_post") as mock_post:
            with patch("src.slack_notifier.settings") as mock_settings:
                mock_settings.slack_webhook_url = ""
                notify_heartbeat([s])
        mock_post.assert_not_called()

    def test_multiple_pipelines_all_appear_in_text(self):
        statuses = [
            self._make_status("daily",  NOW, 2.0, 50.0, False),
            self._make_status("weekly", NOW, 5.0, 80.0, False),
        ]
        mock_post = self._run(statuses)
        text = mock_post.call_args[1]["json"]["attachments"][0]["text"]
        assert "daily" in text
        assert "weekly" in text
