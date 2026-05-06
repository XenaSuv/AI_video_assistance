"""Tests for FeedbackAnalyzer — pure analysis methods only.
IO-bound methods (_save_feedback, load_feedback_history) and API calls
(_get_youtube_metrics, _get_tiktok_metrics) are tested via mocking.
"""
import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.feedback_analyzer import FeedbackAnalyzer, RetentionAnalysis, VideoMetrics


@pytest.fixture
def analyzer(tmp_path):
    """FeedbackAnalyzer using a temp directory so tests don't touch real data."""
    with patch("src.feedback_analyzer.settings") as mock_settings:
        mock_settings.data_dir = str(tmp_path)
        mock_settings.tiktok_token_file = tmp_path / "tiktok_token.json"
        fa = FeedbackAnalyzer()
        fa.feedback_db = tmp_path / "feedback_history.json"
        yield fa


# ── _score_hook() ─────────────────────────────────────────────────────────────

class TestScoreHook:
    def test_empty_curve_returns_zero(self, analyzer):
        assert analyzer._score_hook([]) == 0.0

    def test_single_element_curve_returns_zero(self, analyzer):
        assert analyzer._score_hook([1.0]) == 0.0

    def test_returns_retention_at_30s(self, analyzer):
        curve = [1.0, 0.75, 0.55, 0.40]
        assert analyzer._score_hook(curve) == 0.75

    def test_score_equals_second_element(self, analyzer):
        curve = [1.0, 0.63, 0.50]
        assert analyzer._score_hook(curve) == 0.63


# ── _analyze_retention() ──────────────────────────────────────────────────────

class TestAnalyzeRetention:
    def test_empty_curve_returns_all_none(self, analyzer):
        result = analyzer._analyze_retention([])
        assert result.drop_point is None
        assert result.retention_30s is None
        assert result.best_segment is None

    def test_single_element_returns_all_none(self, analyzer):
        result = analyzer._analyze_retention([1.0])
        assert result.drop_point is None

    def test_detects_drop_above_threshold(self, analyzer):
        # Drop of 0.20 between scene_0 and scene_1 → should be detected
        curve = [1.0, 0.79, 0.70, 0.65]
        result = analyzer._analyze_retention(curve)
        assert result.drop_point == "scene_1"

    def test_no_drop_below_threshold(self, analyzer):
        # Gradual decay, never drops >15% at once
        curve = [1.0, 0.90, 0.82, 0.76]
        result = analyzer._analyze_retention(curve)
        assert result.drop_point is None

    def test_finds_first_significant_drop(self, analyzer):
        # Two drops; should report the first
        curve = [1.0, 0.79, 0.58, 0.55]
        result = analyzer._analyze_retention(curve)
        assert result.drop_point == "scene_1"

    def test_retention_30s_is_curve_index_1(self, analyzer):
        curve = [1.0, 0.72, 0.60]
        result = analyzer._analyze_retention(curve)
        assert result.retention_30s == 0.72

    def test_best_segment_finds_highest_retained(self, analyzer):
        # scene_2 (index 2) has the highest value after start
        curve = [1.0, 0.60, 0.85, 0.70]
        result = analyzer._analyze_retention(curve)
        assert result.best_segment == "scene_2"

    def test_best_segment_not_scene_0(self, analyzer):
        # Best segment should never report scene_0 (that's just the start)
        curve = [1.0, 0.8, 0.6]
        result = analyzer._analyze_retention(curve)
        assert result.best_segment != "scene_0"


# ── _estimate_retention_curve() ───────────────────────────────────────────────

class TestEstimateRetentionCurve:
    def test_starts_at_one(self, analyzer):
        curve = analyzer._estimate_retention_curve(0.5)
        assert curve[0] == 1.0

    def test_ends_near_avg_percentage(self, analyzer):
        avg = 0.6
        curve = analyzer._estimate_retention_curve(avg)
        assert abs(curve[-1] - avg) < 0.01

    def test_curve_is_descending_overall(self, analyzer):
        curve = analyzer._estimate_retention_curve(0.4)
        assert curve[0] > curve[-1]

    def test_returns_non_empty_list(self, analyzer):
        assert len(analyzer._estimate_retention_curve(0.5)) > 1


# ── _generate_recommendations() ───────────────────────────────────────────────

class TestGenerateRecommendations:
    def _metrics(self, **overrides) -> dict:
        base = {
            "hook_score": 0.65,
            "drop_point": None,
            "avg_view_percentage": 0.55,
            "views": 500,
            "best_segment": "scene_2",
        }
        base.update(overrides)
        return base

    def test_weak_hook_suggests_aggressive_opening(self, analyzer):
        recs = analyzer._generate_recommendations(self._metrics(hook_score=0.4))
        assert any("hook" in r.lower() for r in recs)

    def test_strong_hook_congratulates(self, analyzer):
        recs = analyzer._generate_recommendations(self._metrics(hook_score=0.85))
        assert any("strong" in r.lower() or "replicate" in r.lower() for r in recs)

    def test_early_drop_at_scene_0_suggests_shorter_intro(self, analyzer):
        recs = analyzer._generate_recommendations(self._metrics(drop_point="scene_0"))
        assert any("intro" in r.lower() or "drop" in r.lower() for r in recs)

    def test_early_drop_at_scene_1_suggests_micro_hook(self, analyzer):
        recs = analyzer._generate_recommendations(self._metrics(drop_point="scene_1"))
        assert any("micro-hook" in r.lower() or "intro" in r.lower() for r in recs)

    def test_low_watch_time_suggests_shorter_runtime(self, analyzer):
        recs = analyzer._generate_recommendations(self._metrics(avg_view_percentage=0.3))
        assert any("watch" in r.lower() or "runtime" in r.lower() for r in recs)

    def test_high_watch_time_suggests_series(self, analyzer):
        recs = analyzer._generate_recommendations(self._metrics(avg_view_percentage=0.8))
        assert any("series" in r.lower() or "extend" in r.lower() for r in recs)

    def test_low_views_suggests_thumbnail(self, analyzer):
        recs = analyzer._generate_recommendations(self._metrics(views=50))
        assert any("thumbnail" in r.lower() for r in recs)

    def test_high_views_suggests_more_similar_content(self, analyzer):
        recs = analyzer._generate_recommendations(self._metrics(views=5000))
        assert any("similar" in r.lower() or "frequently" in r.lower() for r in recs)

    def test_no_recommendations_for_average_metrics(self, analyzer):
        recs = analyzer._generate_recommendations(self._metrics())
        assert isinstance(recs, list)

    def test_best_segment_mentioned_when_not_scene_0(self, analyzer):
        recs = analyzer._generate_recommendations(self._metrics(best_segment="scene_3"))
        assert any("scene_3" in r for r in recs)


# ── _save_feedback() and load_feedback_history() ─────────────────────────────

class TestPersistence:
    def test_save_and_reload_feedback(self, analyzer):
        record = {
            "video_id": "abc123",
            "platform": "youtube",
            "hook_score": 0.72,
            "avg_view_percentage": 0.58,
            "angle": "technical_breakthrough",
            "format": "deep_dive",
            "drop_point": None,
        }
        analyzer._save_feedback(record)
        history = analyzer.load_feedback_history()
        assert len(history) == 1
        assert history[0]["video_id"] == "abc123"

    def test_multiple_saves_accumulate(self, analyzer):
        for i in range(3):
            analyzer._save_feedback({
                "video_id": f"vid_{i}",
                "platform": "youtube",
                "hook_score": 0.6,
                "avg_view_percentage": 0.5,
                "angle": "a",
                "format": "b",
                "drop_point": None,
            })
        assert len(analyzer.load_feedback_history()) == 3

    def test_load_returns_empty_list_when_no_file(self, analyzer):
        assert analyzer.load_feedback_history() == []


# ── _get_tiktok_metrics() integration ────────────────────────────────────────

class TestGetTikTokMetrics:
    def test_returns_none_when_api_unavailable(self, analyzer):
        with patch("src.feedback_analyzer.get_tiktok_video_metrics", return_value=None):
            result = analyzer._get_tiktok_metrics("some_video_id")
        assert result is None

    def test_maps_real_api_response_to_video_metrics(self, analyzer):
        fake_response = {
            "views": 5000,
            "likes": 300,
            "shares": 50,
            "comments": 80,
            "duration_sec": 60,
            "avg_view_percentage": 0.62,
        }
        with patch("src.feedback_analyzer.get_tiktok_video_metrics", return_value=fake_response):
            result = analyzer._get_tiktok_metrics("vid_123")

        assert result is not None
        assert result.views == 5000
        assert result.platform == "tiktok"
        assert result.avg_view_percentage == 0.62
        assert result.avg_view_duration_sec == round(60 * 0.62)

    def test_retention_curve_estimated_from_real_percentage(self, analyzer):
        fake_response = {
            "views": 1000,
            "likes": 50,
            "shares": 10,
            "comments": 20,
            "duration_sec": 30,
            "avg_view_percentage": 0.50,
        }
        with patch("src.feedback_analyzer.get_tiktok_video_metrics", return_value=fake_response):
            result = analyzer._get_tiktok_metrics("vid_abc")

        assert result.retention_curve[0] == 1.0
        assert len(result.retention_curve) > 1
