"""Tests for src/scheduler.py — _run_with_retry, build_scheduler, job functions."""
from __future__ import annotations

import sys
from unittest.mock import MagicMock, call, patch

import pytest

# APScheduler is not installed — stub it before importing scheduler
_apscheduler_mock = MagicMock()
sys.modules.setdefault("apscheduler", _apscheduler_mock)
sys.modules.setdefault("apscheduler.schedulers", _apscheduler_mock)
sys.modules.setdefault("apscheduler.schedulers.blocking", _apscheduler_mock)
sys.modules.setdefault("apscheduler.triggers", _apscheduler_mock)
sys.modules.setdefault("apscheduler.triggers.cron", _apscheduler_mock)

# Stub deps pulled in by lazy imports inside job functions.
# NOTE: src.main and src.topic_main are NOT stubbed here at module level because
# doing so would permanently replace those modules in sys.modules for the entire
# test session, breaking test_topic_main.py tests that rely on the real
# src.topic_main module.  Instead, each test that invokes the job functions
# stubs those modules with patch.dict(sys.modules, ...) so the stub is
# automatically removed when the test finishes.
for _stub in (
    "elevenlabs",
    "elevenlabs.client",
    "google.oauth2.credentials",
    "google.auth.exceptions",
    "googleapiclient.http",
):
    sys.modules.setdefault(_stub, MagicMock())

from src.scheduler import (
    MAX_RETRIES,
    RETRY_DELAY_SEC,
    _run_with_retry,
    build_scheduler,
    job_daily_news,
    job_topic,
    main,
)


class TestRunWithRetry:
    def test_calls_fn_once_on_success(self):
        fn = MagicMock()
        _run_with_retry("test", fn)
        fn.assert_called_once()

    def test_retries_on_failure(self):
        fn = MagicMock(side_effect=[Exception("err"), Exception("err"), None])
        with patch("src.scheduler.time.sleep"):
            _run_with_retry("test", fn)
        assert fn.call_count == 3

    def test_stops_after_max_retries(self):
        fn = MagicMock(side_effect=Exception("always fails"))
        with patch("src.scheduler.time.sleep"):
            _run_with_retry("test", fn)
        assert fn.call_count == MAX_RETRIES

    def test_sleeps_between_retries(self):
        fn = MagicMock(side_effect=[Exception("fail"), None])
        with patch("src.scheduler.time.sleep") as mock_sleep:
            _run_with_retry("test", fn)
        mock_sleep.assert_called_once_with(RETRY_DELAY_SEC)

    def test_does_not_sleep_on_last_retry_failure(self):
        fn = MagicMock(side_effect=Exception("fail"))
        with patch("src.scheduler.time.sleep") as mock_sleep:
            _run_with_retry("test", fn)
        assert mock_sleep.call_count == MAX_RETRIES - 1

    def test_passes_kwargs_to_fn(self):
        fn = MagicMock()
        _run_with_retry("test", fn, key="value", num=42)
        fn.assert_called_once_with(key="value", num=42)

    def test_succeeds_on_second_attempt(self):
        fn = MagicMock(side_effect=[Exception("first"), "success"])
        with patch("src.scheduler.time.sleep"):
            _run_with_retry("test", fn)
        assert fn.call_count == 2


class TestBuildScheduler:
    def test_returns_scheduler(self):
        fake_scheduler = MagicMock()
        with patch("src.scheduler.BlockingScheduler", return_value=fake_scheduler):
            result = build_scheduler()
        assert result is fake_scheduler

    def test_adds_three_jobs(self):
        fake_scheduler = MagicMock()
        with patch("src.scheduler.BlockingScheduler", return_value=fake_scheduler):
            build_scheduler()
        assert fake_scheduler.add_job.call_count == 3

    def test_daily_news_job_registered(self):
        fake_scheduler = MagicMock()
        with patch("src.scheduler.BlockingScheduler", return_value=fake_scheduler):
            build_scheduler()
        ids = [c.kwargs.get("id") for c in fake_scheduler.add_job.call_args_list]
        assert "daily_news" in ids

    def test_topic_tuesday_job_registered(self):
        fake_scheduler = MagicMock()
        with patch("src.scheduler.BlockingScheduler", return_value=fake_scheduler):
            build_scheduler()
        ids = [c.kwargs.get("id") for c in fake_scheduler.add_job.call_args_list]
        assert "topic_tuesday" in ids

    def test_topic_thursday_job_registered(self):
        fake_scheduler = MagicMock()
        with patch("src.scheduler.BlockingScheduler", return_value=fake_scheduler):
            build_scheduler()
        ids = [c.kwargs.get("id") for c in fake_scheduler.add_job.call_args_list]
        assert "topic_thursday" in ids


class TestJobFunctions:
    def test_job_daily_news_calls_run_pipeline(self):
        # Stub src.main only for the duration of this test so the lazy import
        # inside job_daily_news() succeeds without affecting other test modules.
        with patch.dict(sys.modules, {"src.main": MagicMock()}):
            with patch("src.scheduler._run_with_retry") as mock_retry:
                job_daily_news()
        mock_retry.assert_called_once()
        assert mock_retry.call_args[0][0] == "daily_news"

    def test_job_topic_calls_run_topic_pipeline(self):
        # Stub src.topic_main only for the duration of this test so the lazy
        # import inside job_topic() succeeds without permanently replacing the
        # real src.topic_main module in sys.modules (which would break
        # test_topic_main.py tests that need the real run_topic_pipeline).
        with patch.dict(sys.modules, {"src.topic_main": MagicMock()}):
            with patch("src.scheduler._run_with_retry") as mock_retry:
                job_topic()
        mock_retry.assert_called_once()
        assert mock_retry.call_args[0][0] == "topic"


class TestMain:
    def test_list_flag_prints_jobs_and_exits(self, capsys):
        scheduler = MagicMock()
        fake_job = MagicMock()
        fake_job.id = "daily_news"
        fake_job.name = "Daily AI News Video"
        fake_job.next_run_time = None
        scheduler.get_jobs.return_value = [fake_job]

        with patch("src.scheduler.build_scheduler", return_value=scheduler):
            with patch("sys.argv", ["scheduler", "--list"]):
                main()

        captured = capsys.readouterr()
        assert "daily_news" in captured.out
        scheduler.shutdown.assert_called_once_with(wait=False)

    def test_no_args_starts_scheduler(self):
        scheduler = MagicMock()
        scheduler.start.side_effect = KeyboardInterrupt()

        with patch("src.scheduler.build_scheduler", return_value=scheduler):
            with patch("sys.argv", ["scheduler"]):
                main()

        scheduler.start.assert_called_once()
