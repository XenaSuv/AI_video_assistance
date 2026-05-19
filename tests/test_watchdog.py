"""Tests for src/watchdog.py — find_last_success and check."""
from __future__ import annotations

import datetime as dt
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.watchdog import check, find_last_success

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
