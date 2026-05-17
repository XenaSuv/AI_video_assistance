"""Tests for src/analytics.py — all functions with mocked _load_db."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any
from unittest.mock import patch

import pytest

from src.analytics import (
    top_hooks,
    top_titles,
    hook_keywords,
    top_keywords,
    recent_records,
    get_trend_report,
    worst_hooks,
    boost_if_pattern_works,
    get_proven_hooks,
    get_proven_titles,
    get_engagement_stats,
    get_recommendations,
)


# ── helpers ────────────────────────────────────────────────────────────────────

def _record(
    views: int = 1000,
    likes: int = 100,
    comments: int = 20,
    hook: str = "But here is the problem",
    title: str = "GPT-5 Breaks Records",
    description: str = "AI news today",
    ctr: float = 0.05,
    timestamp: str | None = None,
) -> dict[str, Any]:
    if timestamp is None:
        timestamp = datetime.utcnow().isoformat()
    return {
        "views": views,
        "likes": likes,
        "comments": comments,
        "hook": hook,
        "title": title,
        "description": description,
        "ctr": ctr,
        "timestamp": timestamp,
    }


def _db(*records) -> list[dict]:
    return list(records)


# ═══════════════════════════════════════════════════════════════════════════════
# top_hooks
# ═══════════════════════════════════════════════════════════════════════════════

class TestTopHooks:
    def test_returns_empty_when_no_data(self):
        with patch("src.analytics._load_db", return_value=[]):
            assert top_hooks() == []

    def test_returns_hooks_sorted_by_engagement(self):
        high = _record(views=1000, likes=200, comments=50, hook="high hook")
        low  = _record(views=1000, likes=10,  comments=1,  hook="low hook")
        with patch("src.analytics._load_db", return_value=_db(low, high)):
            result = top_hooks(limit=2)
        assert result[0]["hook"] == "high hook"

    def test_respects_min_views_filter(self):
        with patch("src.analytics._load_db", return_value=_db(_record(views=100))):
            result = top_hooks(min_views=500)
        assert result == []

    def test_respects_limit(self):
        records = [_record(hook=f"hook {i}") for i in range(10)]
        with patch("src.analytics._load_db", return_value=records):
            result = top_hooks(limit=3)
        assert len(result) == 3

    def test_computes_engagement_rate(self):
        rec = _record(views=1000, likes=50, comments=25)
        with patch("src.analytics._load_db", return_value=_db(rec)):
            result = top_hooks()
        assert result[0]["engagement_rate"] == (50 + 25 * 2) / 1000

    def test_fallback_zero_ctr_when_missing(self):
        rec = _record(ctr=None)
        with patch("src.analytics._load_db", return_value=_db(rec)):
            result = top_hooks()
        assert result[0]["ctr"] == 0


# ═══════════════════════════════════════════════════════════════════════════════
# top_titles
# ═══════════════════════════════════════════════════════════════════════════════

class TestTopTitles:
    def test_sorted_by_ctr(self):
        high = _record(title="High CTR title", ctr=0.10)
        low  = _record(title="Low CTR title",  ctr=0.02)
        with patch("src.analytics._load_db", return_value=_db(low, high)):
            result = top_titles(limit=2)
        assert result[0]["title"] == "High CTR title"

    def test_min_views_filter(self):
        with patch("src.analytics._load_db", return_value=_db(_record(views=50))):
            result = top_titles(min_views=100)
        assert result == []

    def test_empty_db(self):
        with patch("src.analytics._load_db", return_value=[]):
            assert top_titles() == []


# ═══════════════════════════════════════════════════════════════════════════════
# hook_keywords
# ═══════════════════════════════════════════════════════════════════════════════

class TestHookKeywords:
    def test_counts_words_in_hooks(self):
        records = [_record(views=600, hook="robot beats human") for _ in range(3)]
        with patch("src.analytics._load_db", return_value=records):
            result = hook_keywords(min_views=500, top_n=5)
        words = [w for w, _ in result]
        assert "robot" in words or "beats" in words or "human" in words

    def test_filters_stop_words(self):
        records = [_record(views=600, hook="the a an and is this") for _ in range(3)]
        with patch("src.analytics._load_db", return_value=records):
            result = hook_keywords(min_views=500)
        words = [w for w, _ in result]
        for stop in ["the", "a", "an", "and", "is", "this"]:
            assert stop not in words

    def test_respects_min_views(self):
        with patch("src.analytics._load_db", return_value=_db(_record(views=100))):
            result = hook_keywords(min_views=500)
        assert result == []

    def test_respects_top_n(self):
        words_str = " ".join([f"word{i}" for i in range(20)])
        records = [_record(views=600, hook=words_str)] * 5
        with patch("src.analytics._load_db", return_value=records):
            result = hook_keywords(min_views=0, top_n=5)
        assert len(result) <= 5


# ═══════════════════════════════════════════════════════════════════════════════
# top_keywords
# ═══════════════════════════════════════════════════════════════════════════════

class TestTopKeywords:
    def test_returns_keywords_from_hook_title_desc(self):
        rec = _record(
            views=500,
            hook="neural network breakthrough",
            title="Neural Networks Explained",
            description="about neural networks and deep learning",
        )
        with patch("src.analytics._load_db", return_value=_db(rec)):
            result = top_keywords(min_views=300, top_n=10)
        words = [w for w, _ in result]
        assert any("neural" in w or "network" in w for w in words)

    def test_filters_stop_words_and_short_words(self):
        rec = _record(views=500, hook="to the and or a")
        with patch("src.analytics._load_db", return_value=_db(rec)):
            result = top_keywords(min_views=0)
        words = [w for w, _ in result]
        for stop in ["to", "the", "and", "or", "a"]:
            assert stop not in words

    def test_empty_db(self):
        with patch("src.analytics._load_db", return_value=[]):
            assert top_keywords() == []


# ═══════════════════════════════════════════════════════════════════════════════
# recent_records
# ═══════════════════════════════════════════════════════════════════════════════

class TestRecentRecords:
    def test_returns_records_within_days(self):
        now_ts = datetime.utcnow().isoformat()
        old_ts = (datetime.utcnow() - timedelta(days=60)).isoformat()
        recs = [_record(timestamp=now_ts), _record(timestamp=old_ts)]
        with patch("src.analytics._load_db", return_value=recs):
            result = recent_records(days=30)
        assert len(result) == 1

    def test_skips_records_without_timestamp(self):
        rec = _record()
        del rec["timestamp"]
        rec["timestamp"] = None
        with patch("src.analytics._load_db", return_value=_db(rec)):
            result = recent_records(days=30)
        assert result == []

    def test_skips_records_with_invalid_timestamp(self):
        rec = _record(timestamp="not-a-date")
        with patch("src.analytics._load_db", return_value=_db(rec)):
            result = recent_records(days=30)
        assert result == []

    def test_empty_db(self):
        with patch("src.analytics._load_db", return_value=[]):
            assert recent_records() == []


# ═══════════════════════════════════════════════════════════════════════════════
# get_trend_report
# ═══════════════════════════════════════════════════════════════════════════════

class TestGetTrendReport:
    def test_empty_when_no_recent_records(self):
        with patch("src.analytics._load_db", return_value=[]):
            result = get_trend_report()
        assert result["total_videos"] == 0

    def test_aggregates_recent_records(self):
        now_ts = datetime.utcnow().isoformat()
        recs = [_record(views=1000, likes=100, ctr=0.05, timestamp=now_ts)] * 3
        with patch("src.analytics._load_db", return_value=recs):
            result = get_trend_report(days=30)
        assert result["total_videos"] == 3
        assert result["total_views"] == 3000
        assert abs(result["avg_ctr"] - 0.05) < 0.001

    def test_includes_top_hooks_and_titles(self):
        now_ts = datetime.utcnow().isoformat()
        recs = [_record(views=200, timestamp=now_ts, hook="test hook", title="test title")]
        with patch("src.analytics._load_db", return_value=recs):
            result = get_trend_report(days=30)
        assert "top_hooks" in result
        assert "top_titles" in result


# ═══════════════════════════════════════════════════════════════════════════════
# worst_hooks
# ═══════════════════════════════════════════════════════════════════════════════

class TestWorstHooks:
    def test_returns_lowest_engagement(self):
        bad  = _record(views=1000, likes=1,   hook="terrible hook")
        good = _record(views=1000, likes=200, hook="great hook")
        with patch("src.analytics._load_db", return_value=_db(good, bad)):
            result = worst_hooks(limit=1)
        assert result[0]["hook"] == "terrible hook"

    def test_empty_db(self):
        with patch("src.analytics._load_db", return_value=[]):
            assert worst_hooks() == []

    def test_zero_views_treated_as_zero_engagement(self):
        rec = _record(views=0, likes=0)
        with patch("src.analytics._load_db", return_value=_db(rec)):
            result = worst_hooks()
        assert result[0]["engagement_rate"] == 0


# ═══════════════════════════════════════════════════════════════════════════════
# boost_if_pattern_works
# ═══════════════════════════════════════════════════════════════════════════════

class TestBoostIfPatternWorks:
    def test_returns_true_when_keyword_in_text(self):
        with patch("src.analytics.hook_keywords", return_value=[("robot", 5), ("neural", 3)]):
            assert boost_if_pattern_works("This is about robot arms") is True

    def test_returns_false_when_no_match(self):
        with patch("src.analytics.hook_keywords", return_value=[("robot", 5)]):
            assert boost_if_pattern_works("completely different content") is False

    def test_case_insensitive(self):
        with patch("src.analytics.hook_keywords", return_value=[("robot", 5)]):
            assert boost_if_pattern_works("ROBOT REVOLUTION") is True

    def test_returns_false_for_empty_keywords(self):
        with patch("src.analytics.hook_keywords", return_value=[]):
            assert boost_if_pattern_works("anything") is False


# ═══════════════════════════════════════════════════════════════════════════════
# get_proven_hooks / get_proven_titles
# ═══════════════════════════════════════════════════════════════════════════════

class TestGetProvenHooksAndTitles:
    def test_get_proven_hooks_returns_strings(self):
        hooks = [{"hook": f"hook {i}", "views": 200, "engagement_rate": 0.1} for i in range(5)]
        with patch("src.analytics.top_hooks", return_value=hooks):
            result = get_proven_hooks(limit=3)
        assert isinstance(result, list)
        assert all(isinstance(h, str) for h in result)

    def test_get_proven_titles_returns_strings(self):
        titles = [{"title": f"Title {i}", "ctr": 0.05, "views": 200} for i in range(5)]
        with patch("src.analytics.top_titles", return_value=titles):
            result = get_proven_titles(limit=3)
        assert isinstance(result, list)
        assert all(isinstance(t, str) for t in result)

    def test_proven_hooks_empty_when_no_data(self):
        with patch("src.analytics.top_hooks", return_value=[]):
            assert get_proven_hooks() == []


# ═══════════════════════════════════════════════════════════════════════════════
# get_engagement_stats
# ═══════════════════════════════════════════════════════════════════════════════

class TestGetEngagementStats:
    def test_empty_db(self):
        with patch("src.analytics._load_db", return_value=[]):
            result = get_engagement_stats()
        assert result["videos"] == 0
        assert result["avg_engagement"] == 0

    def test_computes_stats(self):
        recs = [_record(views=1000, likes=100, ctr=0.05)] * 2
        with patch("src.analytics._load_db", return_value=recs):
            result = get_engagement_stats()
        assert result["videos"] == 2
        assert result["total_views"] == 2000
        assert abs(result["avg_ctr"] - 0.05) < 0.001


# ═══════════════════════════════════════════════════════════════════════════════
# get_recommendations
# ═══════════════════════════════════════════════════════════════════════════════

class TestGetRecommendations:
    def test_returns_all_keys(self):
        with patch("src.analytics._load_db", return_value=[]):
            result = get_recommendations()
        assert "top_hooks" in result
        assert "top_titles" in result
        assert "proven_keywords" in result
        assert "hot_terms" in result
        assert "avoid_patterns" in result
        assert "stats" in result

    def test_aggregates_from_data(self):
        recs = [_record(views=500, hook="neural network beats human", title="AI Title")] * 3
        with patch("src.analytics._load_db", return_value=recs):
            result = get_recommendations()
        assert isinstance(result["top_hooks"], list)
        assert isinstance(result["avoid_patterns"], list)
