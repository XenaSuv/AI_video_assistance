"""Tests for src/breaking_detector.py."""
from __future__ import annotations

import json
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.breaking_detector import (
    _COOLDOWN_H,
    _MAX_PER_DAY,
    _circuit_breaker_ok,
    _is_important,
    _is_recent,
    _load_seen,
    _save_seen,
    _seen_path,
    check_for_breaking,
    mark_publish_failed,
    mark_published,
)
from src.scraper import NewsItem


def _make_item(
    title: str = "OpenAI Releases GPT-6",
    source: str = "OpenAI",
    url: str = "https://openai.com/gpt6",
    published: str | None = None,
    score: float = 5.0,
) -> NewsItem:
    return NewsItem(
        source=source,
        title=title,
        url=url,
        summary="Major AI announcement.",
        authors=[],
        published=published or date.today().isoformat(),
        score=score,
    )


# ── _seen_path ─────────────────────────────────────────────────────────────────

class TestSeenPath:
    def test_returns_path_in_data_dir(self, tmp_path):
        result = _seen_path(tmp_path)
        assert result == tmp_path / "breaking_seen.json"


# ── _load_seen ─────────────────────────────────────────────────────────────────

class TestLoadSeen:
    def test_returns_empty_when_file_absent(self, tmp_path):
        result = _load_seen(tmp_path)
        assert result == {"seen_urls": [], "videos": []}

    def test_loads_existing_file(self, tmp_path):
        data = {"seen_urls": ["https://example.com"], "videos": []}
        _seen_path(tmp_path).write_text(json.dumps(data))
        result = _load_seen(tmp_path)
        assert result["seen_urls"] == ["https://example.com"]

    def test_returns_empty_on_invalid_json(self, tmp_path):
        _seen_path(tmp_path).write_text("not-json{")
        result = _load_seen(tmp_path)
        assert result == {"seen_urls": [], "videos": []}


# ── _save_seen ─────────────────────────────────────────────────────────────────

class TestSaveSeen:
    def test_creates_file(self, tmp_path):
        data = {"seen_urls": ["https://x.com"], "videos": []}
        _save_seen(tmp_path, data)
        assert _seen_path(tmp_path).exists()

    def test_creates_data_dir(self, tmp_path):
        subdir = tmp_path / "data"
        data = {"seen_urls": [], "videos": []}
        _save_seen(subdir, data)
        assert subdir.exists()

    def test_roundtrip(self, tmp_path):
        data = {"seen_urls": ["https://a.com", "https://b.com"], "videos": []}
        _save_seen(tmp_path, data)
        loaded = _load_seen(tmp_path)
        assert loaded == data


# ── _is_important ──────────────────────────────────────────────────────────────

class TestIsImportant:
    def test_true_for_launches_keyword(self):
        assert _is_important("OpenAI launches GPT-6 model") is True

    def test_true_for_releases_keyword(self):
        assert _is_important("Anthropic releases Claude 4") is True

    def test_true_for_announces(self):
        assert _is_important("Google announces new AI breakthrough") is True

    def test_true_for_model_name(self):
        assert _is_important("GPT-5 available now") is True

    def test_true_for_claude(self):
        assert _is_important("Claude update incoming") is True

    def test_false_for_generic_blog_post(self):
        assert _is_important("The future of AI in 2027") is False

    def test_false_for_empty_title(self):
        assert _is_important("") is False

    def test_case_insensitive(self):
        assert _is_important("OPENAI LAUNCHES NEW MODEL") is True

    def test_true_for_partnership(self):
        assert _is_important("Microsoft partnership with OpenAI") is True

    def test_true_for_open_source(self):
        assert _is_important("Meta releases open source LLM") is True


# ── _is_recent ─────────────────────────────────────────────────────────────────

class TestIsRecent:
    def test_today_is_recent(self):
        assert _is_recent(date.today().isoformat()) is True

    def test_yesterday_is_recent(self):
        yesterday = (date.today() - timedelta(days=1)).isoformat()
        assert _is_recent(yesterday) is True

    def test_two_days_ago_is_not_recent(self):
        old = (date.today() - timedelta(days=2)).isoformat()
        assert _is_recent(old) is False

    def test_invalid_date_returns_true(self):
        assert _is_recent("not-a-date") is True

    def test_none_returns_true(self):
        assert _is_recent(None) is True


# ── _circuit_breaker_ok ────────────────────────────────────────────────────────

class TestCircuitBreakerOk:
    def test_ok_when_no_videos(self):
        assert _circuit_breaker_ok({"seen_urls": [], "videos": []}) is True

    def test_blocked_when_max_per_day_reached(self):
        today = date.today().isoformat()
        seen = {
            "seen_urls": [],
            "videos": [
                {"published_at": f"{today}T10:00:00+00:00"},
                {"published_at": f"{today}T12:00:00+00:00"},
            ],
        }
        assert _circuit_breaker_ok(seen) is False

    def test_ok_when_under_max_per_day(self):
        # One video published 5+ hours ago → under daily limit AND past cooldown
        old = datetime.now(timezone.utc) - timedelta(hours=_COOLDOWN_H + 1)
        seen = {
            "seen_urls": [],
            "videos": [{"published_at": old.isoformat()}],
        }
        assert _circuit_breaker_ok(seen) is True

    def test_blocked_within_cooldown(self):
        recent = datetime.now(timezone.utc) - timedelta(hours=1)
        seen = {
            "seen_urls": [],
            "videos": [{"published_at": recent.isoformat()}],
        }
        assert _circuit_breaker_ok(seen) is False

    def test_ok_after_cooldown(self):
        old = datetime.now(timezone.utc) - timedelta(hours=_COOLDOWN_H + 1)
        seen = {
            "seen_urls": [],
            "videos": [{"published_at": old.isoformat()}],
        }
        assert _circuit_breaker_ok(seen) is True


# ── mark_published ─────────────────────────────────────────────────────────────

class TestMarkPublished:
    def test_records_video(self, tmp_path):
        item = _make_item()
        mark_published(tmp_path, item, "vid_123")
        seen = _load_seen(tmp_path)
        assert len(seen["videos"]) == 1
        assert seen["videos"][0]["video_id"] == "vid_123"

    def test_records_url_and_title(self, tmp_path):
        item = _make_item(title="GPT-6 Release", url="https://openai.com/gpt6")
        mark_published(tmp_path, item)
        seen = _load_seen(tmp_path)
        assert seen["videos"][0]["url"] == "https://openai.com/gpt6"
        assert seen["videos"][0]["title"] == "GPT-6 Release"

    def test_appends_to_existing_videos(self, tmp_path):
        item1 = _make_item(url="https://a.com")
        item2 = _make_item(url="https://b.com")
        mark_published(tmp_path, item1)
        mark_published(tmp_path, item2)
        seen = _load_seen(tmp_path)
        assert len(seen["videos"]) == 2


# ── mark_publish_failed ────────────────────────────────────────────────────────

class TestMarkPublishFailed:
    def test_records_failed_video(self, tmp_path):
        item = _make_item()
        mark_publish_failed(tmp_path, item, "API error")
        seen = _load_seen(tmp_path)
        assert len(seen["failed_videos"]) == 1
        assert seen["failed_videos"][0]["error"] == "API error"

    def test_does_not_increment_videos(self, tmp_path):
        item = _make_item()
        mark_publish_failed(tmp_path, item)
        seen = _load_seen(tmp_path)
        assert len(seen.get("videos", [])) == 0


# ── check_for_breaking ─────────────────────────────────────────────────────────

class TestCheckForBreaking:
    def test_returns_none_when_no_items(self, tmp_path):
        with patch("src.breaking_detector.scrape_official_sources", return_value=[]):
            result = check_for_breaking(tmp_path)
        assert result is None

    def test_returns_none_when_circuit_breaker_active(self, tmp_path):
        today = date.today().isoformat()
        data = {
            "seen_urls": [],
            "videos": [
                {"published_at": f"{today}T10:00:00+00:00"},
                {"published_at": f"{today}T12:00:00+00:00"},
            ],
        }
        _save_seen(tmp_path, data)
        with patch("src.breaking_detector.scrape_official_sources", return_value=[_make_item()]):
            result = check_for_breaking(tmp_path)
        assert result is None

    def test_returns_best_item(self, tmp_path):
        items = [
            _make_item(title="OpenAI launches GPT-6", url="https://a.com", score=5.0),
            _make_item(title="Google releases Gemini Ultra", url="https://b.com", score=4.8),
        ]
        with patch("src.breaking_detector.scrape_official_sources", return_value=items):
            result = check_for_breaking(tmp_path)
        assert result is not None
        assert result.score == 5.0

    def test_skips_already_seen_urls(self, tmp_path):
        item = _make_item(url="https://already-seen.com")
        _save_seen(tmp_path, {"seen_urls": ["https://already-seen.com"], "videos": []})
        with patch("src.breaking_detector.scrape_official_sources", return_value=[item]):
            result = check_for_breaking(tmp_path)
        assert result is None

    def test_skips_unimportant_items(self, tmp_path):
        item = _make_item(title="The future of AI")
        with patch("src.breaking_detector.scrape_official_sources", return_value=[item]):
            result = check_for_breaking(tmp_path)
        assert result is None

    def test_saves_seen_urls(self, tmp_path):
        item = _make_item(url="https://new.com")
        with patch("src.breaking_detector.scrape_official_sources", return_value=[item]):
            check_for_breaking(tmp_path)
        seen = _load_seen(tmp_path)
        assert "https://new.com" in seen["seen_urls"]
