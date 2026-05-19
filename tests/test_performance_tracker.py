"""Tests for performance_tracker — save_result, update_metrics, mark_analyzed,
get_unanalyzed, get_top_hooks, get_top_titles, get_stats_by_keyword, get_summary,
list_records, get_record, and persistence helpers _load_db/_save_db.

Strategy: patch _get_db_path in each test (or use a shared tmp_db fixture)
so all reads/writes go to a per-test temp file. Settings is a frozen dataclass
and cannot be monkeypatched directly.
"""
from __future__ import annotations

import json
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

import src.performance_tracker as pt

# ── Fixture: redirect DB path to a tmp file ───────────────────────────────────

@pytest.fixture
def db_path(tmp_path):
    return tmp_path / "performance.json"


@pytest.fixture(autouse=True)
def isolate_db(db_path):
    """Patch _get_db_path so every test gets its own clean DB."""
    with patch("src.performance_tracker._get_db_path", return_value=db_path):
        yield db_path


# ── _load_db / _save_db ───────────────────────────────────────────────────────

class TestPersistenceHelpers:
    def test_load_db_returns_empty_list_when_file_missing(self, db_path):
        assert not db_path.exists()
        result = pt._load_db()
        assert result == []

    def test_save_and_load_roundtrip(self):
        records = [{"video_id": "v1", "title": "Test"}]
        pt._save_db(records)
        loaded = pt._load_db()
        assert loaded == records

    def test_load_db_handles_corrupt_json_gracefully(self, db_path):
        db_path.write_text("NOT VALID JSON", encoding="utf-8")
        result = pt._load_db()
        assert result == []

    def test_save_db_handles_write_error_gracefully(self, db_path):
        # Replace the file with a directory to make open() fail
        db_path.mkdir()
        pt._save_db([{"video_id": "v1"}])  # should not raise

    def test_load_db_returns_empty_on_null_json(self, db_path):
        db_path.write_text("null", encoding="utf-8")
        result = pt._load_db()
        assert result == []


# ── save_result ────────────────────────────────────────────────────────────────

class TestSaveResult:
    def test_creates_record(self):
        pt.save_result("vid_001", "AI is changing everything", "Test Title")
        records = pt._load_db()
        assert len(records) == 1
        r = records[0]
        assert r["video_id"] == "vid_001"
        assert r["hook"] == "AI is changing everything"
        assert r["title"] == "Test Title"

    def test_defaults_are_correct(self):
        pt.save_result("v2", "hook text", "Title")
        r = pt._load_db()[0]
        assert r["platform"] == "youtube"
        assert r["content_type"] == "daily"
        assert r["analyzed"] is False
        assert r["views"] == 0
        assert r["likes"] == 0
        assert r["ctr"] is None
        assert r["avg_view_percentage"] is None

    def test_custom_platform_and_content_type(self):
        pt.save_result(
            "v3", "hook", "Title",
            content_type="weekly",
            platform="tiktok",
            angle="technical_breakthrough",
            format="deep_dive",
        )
        r = pt._load_db()[0]
        assert r["platform"] == "tiktok"
        assert r["content_type"] == "weekly"
        assert r["angle"] == "technical_breakthrough"
        assert r["format"] == "deep_dive"

    def test_strategy_and_auto_actions_stored(self):
        pt.save_result(
            "v4", "hook", "Title",
            strategy={"angle": "impact"},
            auto_actions=[{"type": "pin"}],
        )
        r = pt._load_db()[0]
        assert r["strategy"] == {"angle": "impact"}
        assert r["auto_actions"] == [{"type": "pin"}]

    def test_appends_multiple_records(self):
        pt.save_result("v1", "h1", "T1")
        pt.save_result("v2", "h2", "T2")
        records = pt._load_db()
        assert len(records) == 2
        assert records[0]["video_id"] == "v1"
        assert records[1]["video_id"] == "v2"

    def test_timestamp_is_parseable_iso(self):
        pt.save_result("v1", "h", "T")
        r = pt._load_db()[0]
        assert "timestamp" in r
        datetime.fromisoformat(r["timestamp"])  # should not raise

    def test_description_stored(self):
        pt.save_result("v1", "h", "T", description="Some description text")
        r = pt._load_db()[0]
        assert r["description"] == "Some description text"

    def test_none_strategy_defaults_to_empty_dict(self):
        pt.save_result("v1", "h", "T", strategy=None)
        r = pt._load_db()[0]
        assert r["strategy"] == {}

    def test_none_auto_actions_defaults_to_empty_list(self):
        pt.save_result("v1", "h", "T", auto_actions=None)
        r = pt._load_db()[0]
        assert r["auto_actions"] == []


# ── update_metrics ─────────────────────────────────────────────────────────────

class TestUpdateMetrics:
    def _seed(self, video_id="v1"):
        pt.save_result(video_id, "hook", "Title")

    def test_updates_views(self):
        self._seed()
        pt.update_metrics("v1", views=10000)
        assert pt._load_db()[0]["views"] == 10000

    def test_updates_all_fields(self):
        self._seed()
        pt.update_metrics(
            "v1",
            views=5000,
            likes=300,
            comments=50,
            shares=10,
            ctr=0.06,
            watch_time_sec=12000.0,
            avg_view_duration_sec=240.0,
            avg_view_percentage=0.55,
        )
        r = pt._load_db()[0]
        assert r["views"] == 5000
        assert r["likes"] == 300
        assert r["comments"] == 50
        assert r["shares"] == 10
        assert abs(r["ctr"] - 0.06) < 1e-9
        assert abs(r["watch_time_sec"] - 12000.0) < 1e-9
        assert abs(r["avg_view_duration_sec"] - 240.0) < 1e-9
        assert abs(r["avg_view_percentage"] - 0.55) < 1e-9

    def test_missing_video_does_not_raise(self):
        # Empty DB — should log warning, not raise
        pt.update_metrics("nonexistent", views=100)

    def test_only_specified_fields_are_updated(self):
        self._seed()
        pt.update_metrics("v1", views=999)
        r = pt._load_db()[0]
        assert r["views"] == 999
        assert r["likes"] == 0  # default, unchanged

    def test_updates_the_updated_timestamp(self):
        self._seed()
        pt.update_metrics("v1", views=100)
        r = pt._load_db()[0]
        assert "updated" in r
        datetime.fromisoformat(r["updated"])  # parseable


# ── mark_analyzed ──────────────────────────────────────────────────────────────

class TestMarkAnalyzed:
    def test_sets_analyzed_flag_to_true(self):
        pt.save_result("v1", "h", "T")
        assert pt._load_db()[0]["analyzed"] is False
        pt.mark_analyzed("v1")
        assert pt._load_db()[0]["analyzed"] is True

    def test_missing_video_does_not_raise(self):
        pt.mark_analyzed("nonexistent")  # Should log warning, not raise


# ── get_unanalyzed ─────────────────────────────────────────────────────────────

class TestGetUnanalyzed:
    def _write_record(self, db_path, video_id, analyzed=False, hours_ago=48):
        ts = (datetime.utcnow() - timedelta(hours=hours_ago)).isoformat()
        records = []
        if db_path.exists():
            records = json.loads(db_path.read_text())
        records.append({
            "video_id": video_id,
            "analyzed": analyzed,
            "timestamp": ts,
            "views": 0,
        })
        db_path.write_text(json.dumps(records))

    def test_returns_old_unanalyzed_records(self, db_path):
        self._write_record(db_path, "v1", analyzed=False, hours_ago=48)
        result = pt.get_unanalyzed(min_age_hours=24)
        assert len(result) == 1
        assert result[0]["video_id"] == "v1"

    def test_excludes_already_analyzed(self, db_path):
        self._write_record(db_path, "v1", analyzed=True, hours_ago=48)
        result = pt.get_unanalyzed(min_age_hours=24)
        assert result == []

    def test_excludes_too_recent(self, db_path):
        self._write_record(db_path, "v1", analyzed=False, hours_ago=1)
        result = pt.get_unanalyzed(min_age_hours=24)
        assert result == []

    def test_handles_bad_timestamp_gracefully(self, db_path):
        db_path.write_text(json.dumps([
            {"video_id": "v1", "analyzed": False, "timestamp": "not-a-date"}
        ]))
        result = pt.get_unanalyzed(min_age_hours=24)
        assert result == []

    def test_mixes_analyzed_and_unanalyzed(self, db_path):
        self._write_record(db_path, "v_old", analyzed=False, hours_ago=72)
        self._write_record(db_path, "v_done", analyzed=True, hours_ago=72)
        result = pt.get_unanalyzed(min_age_hours=24)
        assert len(result) == 1
        assert result[0]["video_id"] == "v_old"


# ── get_top_hooks ──────────────────────────────────────────────────────────────

class TestGetTopHooks:
    def _make_record(self, hook, views, likes, comments=0, shares=0):
        return {
            "hook": hook,
            "views": views,
            "likes": likes,
            "comments": comments,
            "shares": shares,
            "ctr": None,
            "avg_view_percentage": None,
        }

    def test_sorted_by_engagement(self, db_path):
        db_path.write_text(json.dumps([
            self._make_record("hook_low", 1000, 10),
            self._make_record("hook_high", 1000, 100),
        ]))
        top = pt.get_top_hooks(limit=10)
        assert top[0]["hook"] == "hook_high"

    def test_excludes_zero_view_records(self, db_path):
        db_path.write_text(json.dumps([self._make_record("no_views", 0, 0)]))
        assert pt.get_top_hooks() == []

    def test_limit_respected(self, db_path):
        records = [self._make_record(f"hook_{i}", 1000, i * 10) for i in range(20)]
        db_path.write_text(json.dumps(records))
        assert len(pt.get_top_hooks(limit=5)) == 5

    def test_engagement_weights_comments_and_shares(self, db_path):
        # hook_b: (10 + 30*2 + 20*3)/1000 = 0.13  > hook_a: 50/1000 = 0.05
        db_path.write_text(json.dumps([
            self._make_record("hook_a", 1000, 50, comments=0, shares=0),
            self._make_record("hook_b", 1000, 10, comments=30, shares=20),
        ]))
        top = pt.get_top_hooks()
        assert top[0]["hook"] == "hook_b"

    def test_empty_db_returns_empty_list(self):
        assert pt.get_top_hooks() == []


# ── get_top_titles ─────────────────────────────────────────────────────────────

class TestGetTopTitles:
    def _make_record(self, title, views, ctr=None):
        return {"title": title, "views": views, "likes": 0, "ctr": ctr, "avg_view_percentage": None}

    def test_sorted_by_ctr(self, db_path):
        db_path.write_text(json.dumps([
            self._make_record("Title A", 1000, ctr=0.05),
            self._make_record("Title B", 1000, ctr=0.10),
        ]))
        top = pt.get_top_titles()
        assert top[0]["title"] == "Title B"

    def test_excludes_zero_view_records(self, db_path):
        db_path.write_text(json.dumps([self._make_record("No Views", 0, ctr=0.08)]))
        assert pt.get_top_titles() == []

    def test_limit_respected(self, db_path):
        records = [self._make_record(f"T{i}", 1000, ctr=i * 0.01) for i in range(20)]
        db_path.write_text(json.dumps(records))
        assert len(pt.get_top_titles(limit=3)) == 3

    def test_null_ctr_treated_as_zero(self, db_path):
        db_path.write_text(json.dumps([
            self._make_record("Has CTR", 1000, ctr=0.05),
            self._make_record("No CTR", 1000, ctr=None),
        ]))
        top = pt.get_top_titles()
        assert top[0]["title"] == "Has CTR"


# ── get_stats_by_keyword ───────────────────────────────────────────────────────

class TestGetStatsByKeyword:
    def _base(self, hook, views=100, likes=5, ctr=0.05):
        return {
            "hook": hook,
            "title": hook,
            "description": "",
            "views": views,
            "likes": likes,
            "comments": 0,
            "shares": 0,
            "ctr": ctr,
            "avg_view_percentage": 0.5,
        }

    def test_no_match_returns_zero_count(self, db_path):
        db_path.write_text(json.dumps([self._base("totally unrelated")]))
        stats = pt.get_stats_by_keyword("artificial intelligence")
        assert stats["count"] == 0

    def test_returns_aggregates_for_matches(self, db_path):
        db_path.write_text(json.dumps([
            self._base("GPT-4 is amazing", views=1000, likes=50),
            self._base("GPT-5 just launched", views=2000, likes=100),
        ]))
        stats = pt.get_stats_by_keyword("gpt")
        assert stats["count"] == 2
        assert stats["total_views"] == 3000
        assert stats["total_likes"] == 150

    def test_case_insensitive(self, db_path):
        db_path.write_text(json.dumps([self._base("OpenAI releases GPT-5")]))
        stats = pt.get_stats_by_keyword("openai")
        assert stats["count"] == 1

    def test_can_search_title_field(self, db_path):
        record = self._base("unrelated hook")
        record["title"] = "Neural Network Breakthrough"
        db_path.write_text(json.dumps([record]))
        stats = pt.get_stats_by_keyword("neural", field="title")
        assert stats["count"] == 1

    def test_avg_engagement_computed(self, db_path):
        # 100 likes / 1000 views = 10%
        db_path.write_text(json.dumps([self._base("AI model", views=1000, likes=100)]))
        stats = pt.get_stats_by_keyword("AI")
        assert abs(stats["avg_engagement"] - 10.0) < 1e-9


# ── get_summary ────────────────────────────────────────────────────────────────

class TestGetSummary:
    def test_empty_db(self):
        summary = pt.get_summary()
        assert summary["total_videos"] == 0
        assert summary["total_views"] == 0

    def test_aggregates_across_records(self, db_path):
        db_path.write_text(json.dumps([
            {"views": 1000, "likes": 50, "ctr": 0.05, "avg_view_percentage": 0.4},
            {"views": 2000, "likes": 100, "ctr": 0.08, "avg_view_percentage": 0.5},
        ]))
        summary = pt.get_summary()
        assert summary["total_videos"] == 2
        assert summary["total_views"] == 3000
        assert summary["total_likes"] == 150
        assert summary["videos_with_metrics"] == 2
        assert abs(summary["avg_ctr"] - 0.065) < 1e-9
        assert abs(summary["avg_view_percentage"] - 0.45) < 1e-9

    def test_engagement_percent(self, db_path):
        db_path.write_text(json.dumps([
            {"views": 1000, "likes": 100, "ctr": None, "avg_view_percentage": None}
        ]))
        summary = pt.get_summary()
        assert abs(summary["avg_engagement_percent"] - 10.0) < 1e-9

    def test_videos_with_metrics_only_counts_nonzero_views(self, db_path):
        db_path.write_text(json.dumps([
            {"views": 500, "likes": 10, "ctr": 0.04, "avg_view_percentage": 0.3},
            {"views": 0,   "likes": 0,  "ctr": None, "avg_view_percentage": None},
        ]))
        summary = pt.get_summary()
        assert summary["videos_with_metrics"] == 1


# ── list_records / get_record ──────────────────────────────────────────────────

class TestListAndGetRecord:
    def test_list_records_returns_all(self):
        pt.save_result("v1", "h1", "T1")
        pt.save_result("v2", "h2", "T2")
        records = pt.list_records()
        assert len(records) == 2

    def test_get_record_returns_correct_entry(self):
        pt.save_result("v1", "h1", "T1")
        pt.save_result("v2", "h2", "T2")
        record = pt.get_record("v2")
        assert record is not None
        assert record["title"] == "T2"

    def test_get_record_returns_none_for_missing(self):
        record = pt.get_record("does_not_exist")
        assert record is None

    def test_list_records_empty_when_db_missing(self):
        assert pt.list_records() == []
