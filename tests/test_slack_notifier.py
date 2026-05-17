"""Tests for slack_notifier — pure helpers and mocked HTTP calls."""
from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch, call

import pytest

# Stub requests so import succeeds without installing
sys.modules.setdefault("requests", MagicMock())

from src.slack_notifier import (
    _duration_str, _yt_url, _spend_bar,
    notify_success, notify_failure, notify_slow_steps,
    notify_budget_alert, notify_budget_summary,
)
from src.budget_guard import BudgetStatus


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


class TestSpendBar:
    def test_empty_when_no_limit(self):
        assert _spend_bar(100.0, 0) == ""

    def test_full_bar_at_100_pct(self):
        assert _spend_bar(50.0, 50.0) == "█" * 10

    def test_empty_bar_at_0_pct(self):
        assert _spend_bar(0.0, 50.0) == "░" * 10

    def test_half_bar_at_50_pct(self):
        bar = _spend_bar(25.0, 50.0)
        assert bar.count("█") == 5
        assert bar.count("░") == 5

    def test_caps_at_100_pct_when_over_budget(self):
        bar = _spend_bar(999.0, 50.0)
        assert bar == "█" * 10


def _budget_status(day_usd=5.0, month_usd=50.0, day_limit=50.0, month_limit=500.0):
    return BudgetStatus(
        date="2026-05-16",
        day_usd=day_usd, month_usd=month_usd,
        day_limit=day_limit, month_limit=month_limit,
    )


class TestNotifyBudgetAlert:
    def _run(self, status):
        with patch("src.slack_notifier._http_post") as mock_post:
            with patch("src.slack_notifier.settings") as mock_settings:
                mock_settings.slack_webhook_url = "https://hooks.slack.com/test"
                notify_budget_alert(status, pipeline="daily")
        return mock_post

    def test_posts_when_day_exceeded(self):
        mock_post = self._run(_budget_status(day_usd=60.0))
        mock_post.assert_called_once()

    def test_posts_when_month_exceeded(self):
        mock_post = self._run(_budget_status(month_usd=600.0))
        mock_post.assert_called_once()

    def test_color_is_red(self):
        mock_post = self._run(_budget_status(day_usd=60.0))
        color = mock_post.call_args[1]["json"]["attachments"][0]["color"]
        assert color == "#e01e5a"

    def test_day_spend_in_message(self):
        mock_post = self._run(_budget_status(day_usd=75.5))
        text = mock_post.call_args[1]["json"]["attachments"][0]["text"]
        assert "75.50" in text

    def test_month_spend_in_message_when_exceeded(self):
        mock_post = self._run(_budget_status(month_usd=550.0))
        text = mock_post.call_args[1]["json"]["attachments"][0]["text"]
        assert "550.00" in text

    def test_no_post_when_no_webhook(self):
        with patch("src.slack_notifier._http_post") as mock_post:
            with patch("src.slack_notifier.settings") as mock_settings:
                mock_settings.slack_webhook_url = ""
                notify_budget_alert(_budget_status(day_usd=99.0), "daily")
        mock_post.assert_not_called()


class TestNotifyBudgetSummary:
    def _run(self, status):
        with patch("src.slack_notifier._http_post") as mock_post:
            with patch("src.slack_notifier.settings") as mock_settings:
                mock_settings.slack_webhook_url = "https://hooks.slack.com/test"
                notify_budget_summary(status, pipeline="daily")
        return mock_post

    def test_always_posts_even_when_under_budget(self):
        mock_post = self._run(_budget_status(day_usd=5.0, month_usd=50.0))
        mock_post.assert_called_once()

    def test_green_when_under_budget(self):
        mock_post = self._run(_budget_status(day_usd=5.0))
        color = mock_post.call_args[1]["json"]["attachments"][0]["color"]
        assert color == "#2eb886"

    def test_red_when_exceeded(self):
        mock_post = self._run(_budget_status(day_usd=99.0))
        color = mock_post.call_args[1]["json"]["attachments"][0]["color"]
        assert color == "#e01e5a"

    def test_shows_day_and_month_spend(self):
        mock_post = self._run(_budget_status(day_usd=12.34, month_usd=123.45))
        text = mock_post.call_args[1]["json"]["attachments"][0]["text"]
        assert "12.34" in text
        assert "123.45" in text


class TestNotifyQuotaAlert:
    def _run(self, used: int, limit: int):
        from src.slack_notifier import notify_quota_alert
        with patch("src.slack_notifier._http_post") as mock_post:
            with patch("src.slack_notifier.settings") as mock_settings:
                mock_settings.slack_webhook_url = "https://hooks.slack.com/test"
                notify_quota_alert(used=used, limit=limit, pipeline="daily")
        return mock_post

    def test_posts_when_webhook_set(self):
        assert self._run(8500, 10_000).called

    def test_orange_when_near_limit(self):
        color = self._run(8500, 10_000).call_args[1]["json"]["attachments"][0]["color"]
        assert color == "#ffa500"

    def test_red_when_fully_exhausted(self):
        color = self._run(10_000, 10_000).call_args[1]["json"]["attachments"][0]["color"]
        assert color == "#e01e5a"

    def test_message_contains_usage(self):
        text = self._run(8500, 10_000).call_args[1]["json"]["attachments"][0]["text"]
        assert "8,500" in text
        assert "10,000" in text

    def test_no_post_when_no_webhook(self):
        from src.slack_notifier import notify_quota_alert
        with patch("src.slack_notifier._http_post") as mock_post:
            with patch("src.slack_notifier.settings") as mock_settings:
                mock_settings.slack_webhook_url = ""
                notify_quota_alert(used=9000, limit=10_000, pipeline="daily")
        mock_post.assert_not_called()


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


class TestNotifySuccessOptionalFields:
    def _run(self, summary: dict, pipeline: str = "daily"):
        with patch("src.slack_notifier._http_post") as mock_post:
            with patch("src.slack_notifier.settings") as mock_settings:
                mock_settings.slack_webhook_url = "https://hooks.slack.com/test"
                notify_success(summary, pipeline)
        return mock_post

    def _text(self, summary: dict) -> str:
        mock_post = self._run(summary)
        return mock_post.call_args[1]["json"]["attachments"][0]["text"]

    def test_includes_num_scenes(self):
        text = self._text({"num_scenes": 5})
        assert "5 scenes" in text

    def test_includes_thumbnail_style(self):
        text = self._text({"thumbnail_style": "cinematic"})
        assert "cinematic" in text

    def test_includes_tool_name(self):
        text = self._text({"tool": "claude"})
        assert "claude" in text

    def test_includes_short_id_link(self):
        text = self._text({"short_id": "short999"})
        assert "youtu.be/short999" in text

    def test_includes_tiktok_id(self):
        text = self._text({"tiktok_id": "tiktok777"})
        assert "tiktok777" in text

    def test_includes_ru_status(self):
        text = self._text({"ru": {"status": "ok", "title": "RU Title"}})
        assert "ok" in text
        assert "RU Title" in text

    def test_includes_ru_status_without_title(self):
        text = self._text({"ru": {"status": "failed"}})
        assert "failed" in text


class TestNotifyFailureTraceback:
    def test_includes_traceback_snippet(self):
        with patch("src.slack_notifier._http_post") as mock_post:
            with patch("src.slack_notifier.settings") as mock_settings:
                mock_settings.slack_webhook_url = "https://hooks.slack.com/test"
                notify_failure(
                    RuntimeError("boom"),
                    "daily",
                    traceback_str="Traceback (most recent call last):\n  File x.py line 10\n    raise RuntimeError('boom')\nRuntimeError: boom",
                )
        text = mock_post.call_args[1]["json"]["attachments"][0]["text"]
        assert "boom" in text
