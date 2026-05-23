"""Tests for FeedbackAnalyzer — pure analysis methods only.
IO-bound methods (_save_feedback, load_feedback_history) and API calls
(_get_youtube_metrics, _get_tiktok_metrics) are tested via mocking.
"""
import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

import datetime

from src.feedback_analyzer import (
    FeedbackAnalyzer,
    RetentionAnalysis,
    VideoMetrics,
    _compute_window_stats,
    _parse_ts,
)
from src.shared_types import WindowStats


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


# ── analyze() ────────────────────────────────────────────────────────────────

class TestAnalyze:
    def test_analyze_youtube_returns_result(self, analyzer):
        with patch("src.feedback_analyzer.get_video_metrics", return_value={
            "views": 1000, "avg_view_duration": 120, "avg_view_percentage": 0.6
        }):
            with patch("src.feedback_analyzer.get_retention_curve", return_value=[1.0, 0.7, 0.5, 0.4]):
                result = analyzer.analyze("vid123", "youtube", {"angle": "technical_breakthrough", "format": "deep_dive"})
        assert result["video_id"] == "vid123"
        assert result["platform"] == "youtube"
        assert result["views"] == 1000

    def test_analyze_youtube_stores_angle_and_format(self, analyzer):
        with patch("src.feedback_analyzer.get_video_metrics", return_value={
            "views": 500, "avg_view_duration": 90, "avg_view_percentage": 0.55
        }):
            with patch("src.feedback_analyzer.get_retention_curve", return_value=[1.0, 0.65, 0.5]):
                result = analyzer.analyze("vid999", "youtube", {"angle": "industry_impact", "format": "quick_hit"})
        assert result["angle"] == "industry_impact"
        assert result["format"] == "quick_hit"

    def test_analyze_tiktok_returns_result(self, analyzer):
        fake_response = {
            "views": 2000, "likes": 100, "shares": 20, "comments": 30,
            "duration_sec": 45, "avg_view_percentage": 0.55,
        }
        with patch("src.feedback_analyzer.get_tiktok_video_metrics", return_value=fake_response):
            result = analyzer.analyze("tiktok_vid", "tiktok", {"angle": "hot_take", "format": "short"})
        assert result["video_id"] == "tiktok_vid"
        assert result["platform"] == "tiktok"

    def test_analyze_unsupported_platform_returns_empty(self, analyzer):
        result = analyzer.analyze("vid_x", "instagram", {"angle": "a", "format": "b"})
        assert result == {}

    def test_analyze_no_metrics_returns_empty(self, analyzer):
        with patch("src.feedback_analyzer.get_video_metrics", return_value=None):
            with patch("src.feedback_analyzer.get_retention_curve", return_value=None):
                result = analyzer.analyze("vid_none", "youtube", {"angle": "a", "format": "b"})
        assert result == {}

    def test_analyze_exception_returns_empty(self, analyzer):
        with patch("src.feedback_analyzer.get_video_metrics", side_effect=RuntimeError("API down")):
            result = analyzer.analyze("vid_err", "youtube", {"angle": "a", "format": "b"})
        assert result == {}

    def test_analyze_saves_feedback_to_history(self, analyzer):
        with patch("src.feedback_analyzer.get_video_metrics", return_value={
            "views": 300, "avg_view_duration": 60, "avg_view_percentage": 0.5
        }):
            with patch("src.feedback_analyzer.get_retention_curve", return_value=[1.0, 0.6, 0.45]):
                analyzer.analyze("vid_save", "youtube", {"angle": "threat_to_jobs", "format": "deep_dive"})
        history = analyzer.load_feedback_history()
        assert any(h["video_id"] == "vid_save" for h in history)

    def test_analyze_with_published_at(self, analyzer):
        from datetime import datetime as dt
        ts = dt(2026, 5, 1, 10, 0, 0)
        with patch("src.feedback_analyzer.get_video_metrics", return_value={
            "views": 800, "avg_view_duration": 100, "avg_view_percentage": 0.6
        }):
            with patch("src.feedback_analyzer.get_retention_curve", return_value=[1.0, 0.7, 0.55]):
                result = analyzer.analyze("vid_ts", "youtube", {}, published_at=ts)
        assert result["published_at"] == ts.isoformat()


# ── _get_youtube_metrics() ───────────────────────────────────────────────────

class TestGetYouTubeMetrics:
    def test_returns_video_metrics_object(self, analyzer):
        with patch("src.feedback_analyzer.get_video_metrics", return_value={
            "views": 1500, "avg_view_duration": 200, "avg_view_percentage": 0.65
        }):
            with patch("src.feedback_analyzer.get_retention_curve", return_value=[1.0, 0.8, 0.6, 0.5]):
                result = analyzer._get_youtube_metrics("vid_yt")
        assert result is not None
        assert result.video_id == "vid_yt"
        assert result.platform == "youtube"
        assert result.views == 1500
        assert result.retention_curve == [1.0, 0.8, 0.6, 0.5]

    def test_falls_back_to_estimated_curve_when_none(self, analyzer):
        with patch("src.feedback_analyzer.get_video_metrics", return_value={
            "views": 700, "avg_view_duration": 80, "avg_view_percentage": 0.5
        }):
            with patch("src.feedback_analyzer.get_retention_curve", return_value=None):
                result = analyzer._get_youtube_metrics("vid_fallback")
        assert result is not None
        assert result.retention_curve[0] == 1.0
        assert len(result.retention_curve) > 1

    def test_returns_none_when_no_basic_metrics(self, analyzer):
        with patch("src.feedback_analyzer.get_video_metrics", return_value=None):
            result = analyzer._get_youtube_metrics("vid_missing")
        assert result is None

    def test_returns_none_on_exception(self, analyzer):
        with patch("src.feedback_analyzer.get_video_metrics", side_effect=Exception("network error")):
            result = analyzer._get_youtube_metrics("vid_exc")
        assert result is None


# ── _generate_recommendations() — avg_view_percentage branches ───────────────

class TestGenerateRecommendationsWatchTime:
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

    def test_below_50_percent_suggests_shorter_runtime(self, analyzer):
        recs = analyzer._generate_recommendations(self._metrics(avg_view_percentage=0.3))
        assert any("50%" in r or "runtime" in r.lower() or "watch" in r.lower() for r in recs)

    def test_above_75_percent_suggests_series(self, analyzer):
        recs = analyzer._generate_recommendations(self._metrics(avg_view_percentage=0.8))
        assert any("series" in r.lower() or "extend" in r.lower() or "sequel" in r.lower() for r in recs)

    def test_mid_range_no_watch_time_recommendation(self, analyzer):
        recs = analyzer._generate_recommendations(self._metrics(avg_view_percentage=0.6))
        # Neither branch should fire for a value between 0.5 and 0.75
        assert not any("50%" in r for r in recs)
        assert not any("series" in r.lower() for r in recs)


# ── _save_feedback() error path ───────────────────────────────────────────────

class TestSaveFeedbackErrorPath:
    def test_save_feedback_handles_write_error_gracefully(self, analyzer, tmp_path):
        # Point feedback_db to a path whose parent doesn't exist so write_text fails
        bad_path = tmp_path / "nonexistent_dir" / "feedback.json"
        analyzer.feedback_db = bad_path
        record = {
            "video_id": "vid_err",
            "platform": "youtube",
            "hook_score": 0.5,
            "avg_view_percentage": 0.5,
            "angle": "a",
            "format": "b",
            "drop_point": None,
        }
        # Should not raise; exception is swallowed and logged as warning
        analyzer._save_feedback(record)


# ── load_feedback_history() error path ───────────────────────────────────────

class TestLoadFeedbackHistoryErrorPath:
    def test_load_returns_empty_list_on_read_error(self, analyzer, tmp_path):
        # Write invalid JSON so json.loads raises
        bad_file = tmp_path / "feedback_history.json"
        bad_file.write_text("NOT VALID JSON {{{")
        analyzer.feedback_db = bad_file
        result = analyzer.load_feedback_history()
        assert result == []


# ── get_angle_performance() ───────────────────────────────────────────────────

class TestGetAnglePerformance:
    def _write_history(self, analyzer, records):
        import json as _json
        analyzer.feedback_db.write_text(_json.dumps(records))

    def test_returns_performance_stats_for_known_angle(self, analyzer):
        records = [
            {"video_id": "v1", "hook_score": 0.7, "avg_view_percentage": 0.6, "angle": "technical_breakthrough", "format": "deep_dive", "drop_point": None},
            {"video_id": "v2", "hook_score": 0.8, "avg_view_percentage": 0.65, "angle": "technical_breakthrough", "format": "quick_hit", "drop_point": None},
        ]
        self._write_history(analyzer, records)
        stats = analyzer.get_angle_performance("technical_breakthrough")
        assert stats is not None
        assert stats.category == "technical_breakthrough"
        assert stats.sample_size == 2
        assert abs(stats.avg_hook_score - 0.75) < 1e-9
        assert stats.success_rate == 1.0

    def test_returns_none_for_empty_history(self, analyzer):
        # No feedback_db file exists
        stats = analyzer.get_angle_performance("technical_breakthrough")
        assert stats is None

    def test_returns_none_for_no_matching_angle(self, analyzer):
        records = [
            {"video_id": "v1", "hook_score": 0.7, "avg_view_percentage": 0.6, "angle": "industry_impact", "format": "deep_dive", "drop_point": None},
        ]
        self._write_history(analyzer, records)
        stats = analyzer.get_angle_performance("technical_breakthrough")
        assert stats is None

    def test_handles_exception_gracefully(self, analyzer, tmp_path):
        bad_file = tmp_path / "feedback_history.json"
        bad_file.write_text("INVALID JSON")
        analyzer.feedback_db = bad_file
        stats = analyzer.get_angle_performance("some_angle")
        assert stats is None

    def test_success_rate_computed_correctly(self, analyzer):
        records = [
            {"video_id": "v1", "hook_score": 0.7, "avg_view_percentage": 0.5, "angle": "threat_to_jobs", "format": "hot_take", "drop_point": None},
            {"video_id": "v2", "hook_score": 0.4, "avg_view_percentage": 0.4, "angle": "threat_to_jobs", "format": "hot_take", "drop_point": None},
        ]
        self._write_history(analyzer, records)
        stats = analyzer.get_angle_performance("threat_to_jobs")
        assert stats is not None
        # Only v1 has hook_score > 0.6
        assert abs(stats.success_rate - 0.5) < 1e-9


# ── get_format_performance() ──────────────────────────────────────────────────

class TestGetFormatPerformance:
    def _write_history(self, analyzer, records):
        import json as _json
        analyzer.feedback_db.write_text(_json.dumps(records))

    def test_returns_performance_stats_for_known_format(self, analyzer):
        records = [
            {"video_id": "v1", "hook_score": 0.75, "avg_view_percentage": 0.7, "angle": "industry_impact", "format": "deep_dive", "drop_point": None},
            {"video_id": "v2", "hook_score": 0.65, "avg_view_percentage": 0.6, "angle": "hot_take", "format": "deep_dive", "drop_point": None},
        ]
        self._write_history(analyzer, records)
        stats = analyzer.get_format_performance("deep_dive")
        assert stats is not None
        assert stats.category == "deep_dive"
        assert stats.sample_size == 2

    def test_returns_none_for_empty_history(self, analyzer):
        stats = analyzer.get_format_performance("deep_dive")
        assert stats is None

    def test_returns_none_for_no_matching_format(self, analyzer):
        records = [
            {"video_id": "v1", "hook_score": 0.7, "avg_view_percentage": 0.6, "angle": "industry_impact", "format": "quick_hit", "drop_point": None},
        ]
        self._write_history(analyzer, records)
        stats = analyzer.get_format_performance("deep_dive")
        assert stats is None

    def test_handles_exception_gracefully(self, analyzer, tmp_path):
        bad_file = tmp_path / "feedback_history.json"
        bad_file.write_text("INVALID JSON {{")
        analyzer.feedback_db = bad_file
        stats = analyzer.get_format_performance("some_format")
        assert stats is None

    def test_avg_hook_score_computed_correctly(self, analyzer):
        records = [
            {"video_id": "v1", "hook_score": 0.6, "avg_view_percentage": 0.5, "angle": "a", "format": "hot_take", "drop_point": None},
            {"video_id": "v2", "hook_score": 0.8, "avg_view_percentage": 0.7, "angle": "b", "format": "hot_take", "drop_point": None},
        ]
        self._write_history(analyzer, records)
        stats = analyzer.get_format_performance("hot_take")
        assert stats is not None
        assert abs(stats.avg_hook_score - 0.7) < 1e-9


# ── _parse_ts ─────────────────────────────────────────────────────────────────

class TestParsTs:
    def test_valid_iso_string(self):
        ts = "2024-01-15T10:30:00"
        result = _parse_ts(ts)
        assert result == datetime.datetime(2024, 1, 15, 10, 30, 0)

    def test_none_returns_datetime_min(self):
        assert _parse_ts(None) == datetime.datetime.min

    def test_empty_string_returns_datetime_min(self):
        assert _parse_ts("") == datetime.datetime.min

    def test_malformed_string_returns_datetime_min(self):
        assert _parse_ts("not-a-date") == datetime.datetime.min


# ── _compute_window_stats ─────────────────────────────────────────────────────

class TestComputeWindowStats:
    def test_empty_list_returns_none(self):
        assert _compute_window_stats([]) is None

    def test_single_record(self):
        records = [{"hook_score": 0.8, "avg_view_percentage": 0.65}]
        ws = _compute_window_stats(records)
        assert isinstance(ws, WindowStats)
        assert ws.sample_size == 1
        assert ws.avg_hook_score == 0.8
        assert ws.avg_watch_time == 0.65

    def test_success_rate_threshold_is_0_6(self):
        records = [
            {"hook_score": 0.7, "avg_view_percentage": 0.5},  # above 0.6 → success
            {"hook_score": 0.5, "avg_view_percentage": 0.4},  # at or below → not success
        ]
        ws = _compute_window_stats(records)
        assert ws is not None
        assert abs(ws.success_rate - 0.5) < 1e-9

    def test_averages_computed_correctly(self):
        records = [
            {"hook_score": 0.6, "avg_view_percentage": 0.4},
            {"hook_score": 0.8, "avg_view_percentage": 0.6},
        ]
        ws = _compute_window_stats(records)
        assert ws is not None
        assert abs(ws.avg_hook_score - 0.7) < 1e-9
        assert abs(ws.avg_watch_time - 0.5) < 1e-9


# ── Rolling window — get_angle_performance ────────────────────────────────────

def _ts(days_ago: float) -> str:
    """ISO timestamp N days in the past."""
    dt = datetime.datetime.now() - datetime.timedelta(days=days_ago)
    return dt.isoformat()


class TestRollingWindowAngle:
    def _write(self, analyzer: FeedbackAnalyzer, records: list[dict]) -> None:
        analyzer.feedback_db.write_text(json.dumps(records))

    def test_lifetime_populated(self, analyzer):
        records = [
            {"video_id": "v1", "hook_score": 0.7, "avg_view_percentage": 0.6,
             "angle": "tech", "format": "f", "drop_point": None, "timestamp": _ts(30)},
        ]
        self._write(analyzer, records)
        stats = analyzer.get_angle_performance("tech")
        assert stats is not None
        assert stats.lifetime is not None
        assert stats.lifetime.sample_size == 1

    def test_recent_7d_none_when_all_records_are_old(self, analyzer):
        records = [
            {"video_id": "v1", "hook_score": 0.7, "avg_view_percentage": 0.6,
             "angle": "tech", "format": "f", "drop_point": None, "timestamp": _ts(10)},
        ]
        self._write(analyzer, records)
        stats = analyzer.get_angle_performance("tech")
        assert stats is not None
        assert stats.recent_7d is None

    def test_recent_7d_populated_for_fresh_records(self, analyzer):
        records = [
            {"video_id": "v1", "hook_score": 0.7, "avg_view_percentage": 0.6,
             "angle": "tech", "format": "f", "drop_point": None, "timestamp": _ts(2)},
        ]
        self._write(analyzer, records)
        stats = analyzer.get_angle_performance("tech")
        assert stats is not None
        assert stats.recent_7d is not None
        assert stats.recent_7d.sample_size == 1

    def test_recent_7d_only_counts_records_within_window(self, analyzer):
        records = [
            {"video_id": "v1", "hook_score": 0.9, "avg_view_percentage": 0.8,
             "angle": "tech", "format": "f", "drop_point": None, "timestamp": _ts(2)},
            {"video_id": "v2", "hook_score": 0.3, "avg_view_percentage": 0.2,
             "angle": "tech", "format": "f", "drop_point": None, "timestamp": _ts(20)},
        ]
        self._write(analyzer, records)
        stats = analyzer.get_angle_performance("tech")
        assert stats is not None
        # lifetime sees both
        assert stats.lifetime is not None
        assert stats.lifetime.sample_size == 2
        # 7d sees only v1
        assert stats.recent_7d is not None
        assert stats.recent_7d.sample_size == 1
        assert stats.recent_7d.avg_hook_score == 0.9

    def test_records_without_timestamp_excluded_from_7d(self, analyzer):
        records = [
            {"video_id": "v1", "hook_score": 0.8, "avg_view_percentage": 0.6,
             "angle": "tech", "format": "f", "drop_point": None},  # no timestamp
        ]
        self._write(analyzer, records)
        stats = analyzer.get_angle_performance("tech")
        assert stats is not None
        assert stats.lifetime is not None
        assert stats.recent_7d is None

    def test_flat_fields_mirror_lifetime(self, analyzer):
        records = [
            {"video_id": "v1", "hook_score": 0.75, "avg_view_percentage": 0.6,
             "angle": "tech", "format": "f", "drop_point": None, "timestamp": _ts(30)},
        ]
        self._write(analyzer, records)
        stats = analyzer.get_angle_performance("tech")
        assert stats is not None
        assert stats.avg_hook_score == stats.lifetime.avg_hook_score  # type: ignore[union-attr]
        assert stats.sample_size == stats.lifetime.sample_size         # type: ignore[union-attr]


# ── Rolling window — get_format_performance ───────────────────────────────────

class TestRollingWindowFormat:
    def _write(self, analyzer: FeedbackAnalyzer, records: list[dict]) -> None:
        analyzer.feedback_db.write_text(json.dumps(records))

    def test_recent_7d_higher_score_than_lifetime(self, analyzer):
        """Scenario: recent videos improved; 7d score > lifetime score."""
        records = [
            {"video_id": "v_old", "hook_score": 0.4, "avg_view_percentage": 0.4,
             "angle": "a", "format": "deep_dive", "drop_point": None, "timestamp": _ts(30)},
            {"video_id": "v_new", "hook_score": 0.9, "avg_view_percentage": 0.8,
             "angle": "a", "format": "deep_dive", "drop_point": None, "timestamp": _ts(1)},
        ]
        self._write(analyzer, records)
        stats = analyzer.get_format_performance("deep_dive")
        assert stats is not None
        assert stats.recent_7d is not None
        assert stats.lifetime is not None
        assert stats.recent_7d.avg_hook_score > stats.lifetime.avg_hook_score

    def test_lifetime_none_never_happens_when_records_exist(self, analyzer):
        records = [
            {"video_id": "v1", "hook_score": 0.7, "avg_view_percentage": 0.6,
             "angle": "a", "format": "hot_take", "drop_point": None, "timestamp": _ts(5)},
        ]
        self._write(analyzer, records)
        stats = analyzer.get_format_performance("hot_take")
        assert stats is not None
        assert stats.lifetime is not None
