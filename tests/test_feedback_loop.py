"""Tests for deferred feedback collection loop."""
import json
import pytest
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch


# ── helpers ───────────────────────────────────────────────────────────────────

def _ts(hours_ago: float) -> str:
    return (datetime.utcnow() - timedelta(hours=hours_ago)).isoformat()


def _write_perf_db(tmp_path: Path, records: list[dict]) -> Path:
    db = tmp_path / "performance.json"
    db.write_text(json.dumps(records))
    return db


# ── performance_tracker: save_result ──────────────────────────────────────────

class TestSaveResult:
    def test_saves_angle_format_platform(self, tmp_path, monkeypatch):
        from src import performance_tracker as pt
        monkeypatch.setattr(pt, "_get_db_path", lambda: tmp_path / "performance.json")

        pt.save_result(
            "vid1", "hook text", "title", "desc",
            angle="technical_breakthrough", format="deep_dive", platform="youtube",
        )

        records = json.loads((tmp_path / "performance.json").read_text())
        assert records[0]["angle"] == "technical_breakthrough"
        assert records[0]["format"] == "deep_dive"
        assert records[0]["platform"] == "youtube"
        assert records[0]["analyzed"] is False

    def test_analyzed_defaults_to_false(self, tmp_path, monkeypatch):
        from src import performance_tracker as pt
        monkeypatch.setattr(pt, "_get_db_path", lambda: tmp_path / "performance.json")

        pt.save_result("vid2", "hook", "title")
        records = json.loads((tmp_path / "performance.json").read_text())
        assert records[0]["analyzed"] is False


class TestMarkAnalyzed:
    def test_sets_analyzed_true(self, tmp_path, monkeypatch):
        from src import performance_tracker as pt
        monkeypatch.setattr(pt, "_get_db_path", lambda: tmp_path / "performance.json")

        _write_perf_db(tmp_path, [{"video_id": "v1", "analyzed": False, "timestamp": _ts(25)}])
        pt.mark_analyzed("v1")

        records = json.loads((tmp_path / "performance.json").read_text())
        assert records[0]["analyzed"] is True

    def test_does_not_crash_on_unknown_id(self, tmp_path, monkeypatch):
        from src import performance_tracker as pt
        monkeypatch.setattr(pt, "_get_db_path", lambda: tmp_path / "performance.json")

        _write_perf_db(tmp_path, [])
        pt.mark_analyzed("nonexistent")  # should log warning, not raise


class TestGetUnanalyzed:
    def test_returns_old_unanalyzed(self, tmp_path, monkeypatch):
        from src import performance_tracker as pt
        monkeypatch.setattr(pt, "_get_db_path", lambda: tmp_path / "performance.json")

        _write_perf_db(tmp_path, [
            {"video_id": "old", "analyzed": False, "timestamp": _ts(30)},
        ])
        result = pt.get_unanalyzed(min_age_hours=24)
        assert len(result) == 1
        assert result[0]["video_id"] == "old"

    def test_skips_recent_records(self, tmp_path, monkeypatch):
        from src import performance_tracker as pt
        monkeypatch.setattr(pt, "_get_db_path", lambda: tmp_path / "performance.json")

        _write_perf_db(tmp_path, [
            {"video_id": "new", "analyzed": False, "timestamp": _ts(2)},
        ])
        assert pt.get_unanalyzed(min_age_hours=24) == []

    def test_skips_already_analyzed(self, tmp_path, monkeypatch):
        from src import performance_tracker as pt
        monkeypatch.setattr(pt, "_get_db_path", lambda: tmp_path / "performance.json")

        _write_perf_db(tmp_path, [
            {"video_id": "done", "analyzed": True, "timestamp": _ts(48)},
        ])
        assert pt.get_unanalyzed(min_age_hours=24) == []

    def test_returns_multiple_pending(self, tmp_path, monkeypatch):
        from src import performance_tracker as pt
        monkeypatch.setattr(pt, "_get_db_path", lambda: tmp_path / "performance.json")

        _write_perf_db(tmp_path, [
            {"video_id": "a", "analyzed": False, "timestamp": _ts(48)},
            {"video_id": "b", "analyzed": False, "timestamp": _ts(72)},
            {"video_id": "c", "analyzed": True,  "timestamp": _ts(48)},
        ])
        result = pt.get_unanalyzed(min_age_hours=24)
        ids = [r["video_id"] for r in result]
        assert sorted(ids) == ["a", "b"]


# ── FeedbackAnalyzer.collect_deferred_feedback ────────────────────────────────

class TestCollectDeferredFeedback:
    def _make_analyzer(self, tmp_path):
        from src.feedback_analyzer import FeedbackAnalyzer
        fa = FeedbackAnalyzer.__new__(FeedbackAnalyzer)
        fa.youtube = None
        fa.feedback_db = tmp_path / "feedback_history.json"
        fa.feedback_db.parent.mkdir(parents=True, exist_ok=True)
        return fa

    def test_returns_zero_when_nothing_pending(self, tmp_path, monkeypatch):
        import src.performance_tracker as pt
        monkeypatch.setattr(pt, "_get_db_path", lambda: tmp_path / "performance.json")
        _write_perf_db(tmp_path, [])

        fa = self._make_analyzer(tmp_path)
        assert fa.collect_deferred_feedback() == 0

    def test_calls_analyze_for_each_pending(self, tmp_path, monkeypatch):
        import src.performance_tracker as pt
        monkeypatch.setattr(pt, "_get_db_path", lambda: tmp_path / "performance.json")
        _write_perf_db(tmp_path, [
            {"video_id": "v1", "analyzed": False, "timestamp": _ts(30),
             "angle": "hot_take", "format": "quick_hit", "platform": "youtube"},
        ])

        fa = self._make_analyzer(tmp_path)
        fake_result = {"hook_score": 0.7, "avg_view_percentage": 0.6}
        fa.analyze = MagicMock(return_value=fake_result)

        with patch("src.feedback_analyzer.mark_analyzed") as mock_mark:
            count = fa.collect_deferred_feedback(min_age_hours=24)

        assert count == 1
        fa.analyze.assert_called_once_with(
            "v1", "youtube", {"angle": "hot_take", "format": "quick_hit"}
        )
        mock_mark.assert_called_once_with("v1")

    def test_skips_failed_analyze(self, tmp_path, monkeypatch):
        import src.performance_tracker as pt
        monkeypatch.setattr(pt, "_get_db_path", lambda: tmp_path / "performance.json")
        _write_perf_db(tmp_path, [
            {"video_id": "v1", "analyzed": False, "timestamp": _ts(30),
             "platform": "youtube"},
        ])

        fa = self._make_analyzer(tmp_path)
        fa.analyze = MagicMock(return_value={})  # empty = failure

        with patch("src.feedback_analyzer.mark_analyzed") as mock_mark:
            count = fa.collect_deferred_feedback(min_age_hours=24)

        assert count == 0
        mock_mark.assert_not_called()

    def test_processes_multiple_videos(self, tmp_path, monkeypatch):
        import src.performance_tracker as pt
        monkeypatch.setattr(pt, "_get_db_path", lambda: tmp_path / "performance.json")
        _write_perf_db(tmp_path, [
            {"video_id": "v1", "analyzed": False, "timestamp": _ts(26), "platform": "youtube"},
            {"video_id": "v2", "analyzed": False, "timestamp": _ts(50), "platform": "youtube"},
        ])

        fa = self._make_analyzer(tmp_path)
        fa.analyze = MagicMock(return_value={"hook_score": 0.5, "avg_view_percentage": 0.5})

        with patch("src.feedback_analyzer.mark_analyzed"):
            count = fa.collect_deferred_feedback(min_age_hours=24)

        assert count == 2
        assert fa.analyze.call_count == 2
