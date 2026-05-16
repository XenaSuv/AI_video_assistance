"""Tests for slack_notifier — pure helpers and mocked HTTP calls."""
from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch, call

import pytest

# Stub requests so import succeeds without installing
sys.modules.setdefault("requests", MagicMock())

from src.slack_notifier import _duration_str, _yt_url, notify_success, notify_failure, notify_slow_steps


# ── Pure helpers ──────────────────────────────────────────────────────────────

class TestDurationStr:
    def test_seconds_only(self):
        assert _duration_str(45) == "45s"

    def test_one_minute(self):
        assert _duration_str(60) == "1m 0s"

    def test_minutes_and_seconds(self):
        assert _duration_str(125) == "2m 5s"

    def test_zero_seconds(self):
        assert _duration_str(0) == "0s"

    def test_exact_hour(self):
        assert _duration_str(3600) == "60m 0s"


class TestYtUrl:
    def test_returns_url_for_valid_id(self):
        assert _yt_url("abc123") == "https://youtu.be/abc123"

    def test_returns_none_for_none(self):
        assert _yt_url(None) is None

    def test_returns_none_for_empty_string(self):
        assert _yt_url("") is None


# ── Notify functions (mocked HTTP) ────────────────────────────────────────────

class TestNotifySuccess:
    def _run(self, summary: dict, pipeline: str = "daily"):
        with patch("src.slack_notifier._http_post") as mock_post:
            with patch("src.slack_notifier.settings") as mock_settings:
                mock_settings.slack_webhook_url = "https://hooks.slack.com/test"
                notify_success(summary, pipeline)
        return mock_post

    def test_posts_once_on_success(self):
        mock_post = self._run({"date": "2026-01-01", "title": "Test video"})
        mock_post.assert_called_once()

    def test_no_post_when_webhook_not_configured(self):
        with patch("src.slack_notifier._http_post") as mock_post:
            with patch("src.slack_notifier.settings") as mock_settings:
                mock_settings.slack_webhook_url = ""
                notify_success({"date": "2026-01-01"}, "daily")
        mock_post.assert_not_called()

    def test_payload_contains_pipeline_name(self):
        with patch("src.slack_notifier._http_post") as mock_post:
            with patch("src.slack_notifier.settings") as mock_settings:
                mock_settings.slack_webhook_url = "https://hooks.slack.com/test"
                notify_success({"title": "Episode"}, "weekly")
        payload = mock_post.call_args[1]["json"]
        text = payload["attachments"][0]["text"]
        assert "weekly" in text.lower() or "Weekly" in text

    def test_payload_color_is_green(self):
        with patch("src.slack_notifier._http_post") as mock_post:
            with patch("src.slack_notifier.settings") as mock_settings:
                mock_settings.slack_webhook_url = "https://hooks.slack.com/test"
                notify_success({}, "daily")
        payload = mock_post.call_args[1]["json"]
        assert payload["attachments"][0]["color"] == "#2eb886"

    def test_slack_failure_is_non_fatal(self):
        with patch("src.slack_notifier._http_post", side_effect=Exception("timeout")):
            with patch("src.slack_notifier.settings") as mock_settings:
                mock_settings.slack_webhook_url = "https://hooks.slack.com/test"
                # should not raise
                notify_success({}, "daily")

    def test_includes_video_id_link(self):
        with patch("src.slack_notifier._http_post") as mock_post:
            with patch("src.slack_notifier.settings") as mock_settings:
                mock_settings.slack_webhook_url = "https://hooks.slack.com/test"
                notify_success({"video_id": "vid123"}, "daily")
        payload = mock_post.call_args[1]["json"]
        text = payload["attachments"][0]["text"]
        assert "youtu.be/vid123" in text

    def test_includes_duration(self):
        with patch("src.slack_notifier._http_post") as mock_post:
            with patch("src.slack_notifier.settings") as mock_settings:
                mock_settings.slack_webhook_url = "https://hooks.slack.com/test"
                notify_success({"total_duration_sec": 125}, "daily")
        payload = mock_post.call_args[1]["json"]
        text = payload["attachments"][0]["text"]
        assert "2m 5s" in text


class TestNotifySuccessWithTimings:
    def _run(self, step_timings):
        with patch("src.slack_notifier._http_post") as mock_post:
            with patch("src.slack_notifier.settings") as mock_settings:
                mock_settings.slack_webhook_url = "https://hooks.slack.com/test"
                notify_success({"date": "2026-01-01"}, "daily", step_timings=step_timings)
        return mock_post.call_args[1]["json"]["attachments"][0]["text"]

    def test_step_timing_block_included(self):
        timings = [{"name": "scrape", "duration_str": "45s", "slow": False}]
        text = self._run(timings)
        assert "scrape" in text
        assert "45s" in text

    def test_slow_step_shows_warning_emoji(self):
        timings = [{"name": "video", "duration_str": "18m 0s", "slow": True}]
        text = self._run(timings)
        assert "⚠️" in text

    def test_fast_step_shows_ok_emoji(self):
        timings = [{"name": "scrape", "duration_str": "30s", "slow": False}]
        text = self._run(timings)
        assert "✅" in text

    def test_no_timing_block_when_none(self):
        with patch("src.slack_notifier._http_post") as mock_post:
            with patch("src.slack_notifier.settings") as mock_settings:
                mock_settings.slack_webhook_url = "https://hooks.slack.com/test"
                notify_success({"date": "2026-01-01"}, "daily", step_timings=None)
        text = mock_post.call_args[1]["json"]["attachments"][0]["text"]
        assert "Step timing" not in text


class TestNotifySlowSteps:
    def _run(self, slow):
        with patch("src.slack_notifier._http_post") as mock_post:
            with patch("src.slack_notifier.settings") as mock_settings:
                mock_settings.slack_webhook_url = "https://hooks.slack.com/test"
                notify_slow_steps(slow, pipeline="daily", date="2026-01-01")
        return mock_post

    def test_no_post_when_empty(self):
        mock_post = self._run([])
        mock_post.assert_not_called()

    def test_posts_when_slow_steps_present(self):
        slow = [{"name": "video", "duration_sec": 1200, "threshold_sec": 900, "p95_sec": None}]
        mock_post = self._run(slow)
        mock_post.assert_called_once()

    def test_payload_color_is_orange(self):
        slow = [{"name": "video", "duration_sec": 1200, "threshold_sec": 900, "p95_sec": None}]
        mock_post = self._run(slow)
        payload = mock_post.call_args[1]["json"]
        assert payload["attachments"][0]["color"] == "#ffa500"

    def test_step_name_in_message(self):
        slow = [{"name": "voice", "duration_sec": 700, "threshold_sec": 600, "p95_sec": None}]
        mock_post = self._run(slow)
        text = mock_post.call_args[1]["json"]["attachments"][0]["text"]
        assert "voice" in text

    def test_p95_included_when_available(self):
        slow = [{"name": "video", "duration_sec": 1200, "threshold_sec": 900, "p95_sec": 800.0}]
        mock_post = self._run(slow)
        text = mock_post.call_args[1]["json"]["attachments"][0]["text"]
        assert "P95" in text or "13m" in text  # 800s = 13m 20s


class TestNotifyFailure:
    def test_payload_color_is_red(self):
        with patch("src.slack_notifier._http_post") as mock_post:
            with patch("src.slack_notifier.settings") as mock_settings:
                mock_settings.slack_webhook_url = "https://hooks.slack.com/test"
                notify_failure(ValueError("boom"), "daily")
        payload = mock_post.call_args[1]["json"]
        assert payload["attachments"][0]["color"] == "#e01e5a"

    def test_error_message_in_payload(self):
        with patch("src.slack_notifier._http_post") as mock_post:
            with patch("src.slack_notifier.settings") as mock_settings:
                mock_settings.slack_webhook_url = "https://hooks.slack.com/test"
                notify_failure(RuntimeError("disk full"), "daily")
        payload = mock_post.call_args[1]["json"]
        text = payload["attachments"][0]["text"]
        assert "disk full" in text

    def test_no_post_when_no_webhook(self):
        with patch("src.slack_notifier._http_post") as mock_post:
            with patch("src.slack_notifier.settings") as mock_settings:
                mock_settings.slack_webhook_url = ""
                notify_failure(Exception("x"), "daily")
        mock_post.assert_not_called()
