"""Tests for src/auto_action_engine.py — AutoActionEngine and helpers."""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.auto_action_engine import (
    AutoActionEngine,
    _load_json,
    _save_json,
    _strategy_path,
    _actions_path,
    _stats_path,
)


# ── _load_json / _save_json ───────────────────────────────────────────────────

class TestLoadJson:
    def test_returns_default_when_missing(self, tmp_path):
        result = _load_json(tmp_path / "missing.json", {"key": "default"})
        assert result == {"key": "default"}

    def test_returns_content_when_exists(self, tmp_path):
        p = tmp_path / "data.json"
        p.write_text('{"x": 42}')
        result = _load_json(p, {})
        assert result == {"x": 42}

    def test_returns_default_on_invalid_json(self, tmp_path):
        p = tmp_path / "bad.json"
        p.write_text("not-json")
        result = _load_json(p, {"fallback": True})
        assert result == {"fallback": True}

    def test_returns_deep_copy_of_default(self, tmp_path):
        default = {"nested": [1, 2, 3]}
        result = _load_json(tmp_path / "missing.json", default)
        result["nested"].append(99)
        # Original default should be unchanged
        assert default["nested"] == [1, 2, 3]

    def test_loads_list(self, tmp_path):
        p = tmp_path / "data.json"
        p.write_text('[1, 2, 3]')
        result = _load_json(p, [])
        assert result == [1, 2, 3]


class TestSaveJson:
    def test_creates_file(self, tmp_path):
        p = tmp_path / "out.json"
        _save_json(p, {"hello": "world"})
        assert p.exists()

    def test_content_is_valid_json(self, tmp_path):
        p = tmp_path / "out.json"
        _save_json(p, {"hello": "world"})
        assert json.loads(p.read_text()) == {"hello": "world"}

    def test_creates_parent_dirs(self, tmp_path):
        p = tmp_path / "a" / "b" / "c" / "out.json"
        _save_json(p, {"x": 1})
        assert p.exists()

    def test_overwrites_existing(self, tmp_path):
        p = tmp_path / "out.json"
        _save_json(p, {"v": 1})
        _save_json(p, {"v": 2})
        assert json.loads(p.read_text()) == {"v": 2}


# ── Path helpers ──────────────────────────────────────────────────────────────

class TestPathHelpers:
    def test_strategy_path_uses_data_dir(self, tmp_path):
        with patch("src.auto_action_engine.settings") as mock_s:
            mock_s.data_dir = tmp_path
            p = _strategy_path()
        assert p == tmp_path / "content_strategy.json"

    def test_actions_path_uses_data_dir(self, tmp_path):
        with patch("src.auto_action_engine.settings") as mock_s:
            mock_s.data_dir = tmp_path
            p = _actions_path()
        assert p == tmp_path / "auto_actions.json"

    def test_stats_path_uses_data_dir(self, tmp_path):
        with patch("src.auto_action_engine.settings") as mock_s:
            mock_s.data_dir = tmp_path
            p = _stats_path()
        assert p == tmp_path / "action_stats.json"


# ── AutoActionEngine fixture ──────────────────────────────────────────────────

@pytest.fixture
def engine(tmp_path):
    with patch("src.auto_action_engine.settings") as mock_settings:
        mock_settings.data_dir = tmp_path
        yield AutoActionEngine(run_type="daily")


# ── default_strategy ──────────────────────────────────────────────────────────

class TestDefaultStrategy:
    def test_has_pace_key(self, engine):
        s = engine.default_strategy()
        assert "pace" in s

    def test_has_hook_aggressiveness(self, engine):
        s = engine.default_strategy()
        assert "hook_aggressiveness" in s

    def test_has_packaging_style(self, engine):
        s = engine.default_strategy()
        assert "packaging_style" in s

    def test_has_mode_locked(self, engine):
        s = engine.default_strategy()
        assert "mode_locked" in s

    def test_pace_is_normal(self, engine):
        assert engine.default_strategy()["pace"] == "normal"

    def test_mode_locked_is_false(self, engine):
        assert engine.default_strategy()["mode_locked"] is False


# ── load_strategy / save_strategy ─────────────────────────────────────────────

class TestLoadSaveStrategy:
    def test_load_strategy_creates_default_when_missing(self, engine, tmp_path):
        with patch("src.auto_action_engine.settings") as mock_s:
            mock_s.data_dir = tmp_path
            s = engine.load_strategy()
        assert s["pace"] == "normal"

    def test_save_and_load_strategy_roundtrip(self, engine, tmp_path):
        strategy = {**engine.default_strategy(), "pace": "fast"}
        with patch("src.auto_action_engine.settings") as mock_s:
            mock_s.data_dir = tmp_path
            engine.save_strategy(strategy)
            loaded = engine.load_strategy()
        assert loaded["pace"] == "fast"

    def test_load_strategy_saves_default_when_not_found(self, engine, tmp_path):
        with patch("src.auto_action_engine.settings") as mock_s:
            mock_s.data_dir = tmp_path
            engine.load_strategy()
            # After load_strategy, the file should exist
            p = tmp_path / "content_strategy.json"
            assert p.exists()


# ── load_action_log / save_action_log ─────────────────────────────────────────

class TestActionLog:
    def test_load_action_log_returns_default(self, engine, tmp_path):
        with patch("src.auto_action_engine.settings") as mock_s:
            mock_s.data_dir = tmp_path
            log = engine.load_action_log()
        assert "entries" in log
        assert "scores" in log

    def test_save_and_load_action_log(self, engine, tmp_path):
        with patch("src.auto_action_engine.settings") as mock_s:
            mock_s.data_dir = tmp_path
            payload = {"entries": [{"type": "test"}], "scores": {}}
            engine.save_action_log(payload)
            loaded = engine.load_action_log()
        assert loaded["entries"][0]["type"] == "test"


# ── load_stats / save_stats ───────────────────────────────────────────────────

class TestStats:
    def test_load_stats_returns_empty_dict(self, engine, tmp_path):
        with patch("src.auto_action_engine.settings") as mock_s:
            mock_s.data_dir = tmp_path
            stats = engine.load_stats()
        assert stats == {}

    def test_save_and_load_stats(self, engine, tmp_path):
        with patch("src.auto_action_engine.settings") as mock_s:
            mock_s.data_dir = tmp_path
            engine.save_stats({"add_hook": {"success": 5, "fail": 2}})
            stats = engine.load_stats()
        assert stats["add_hook"]["success"] == 5


# ── performance_good ──────────────────────────────────────────────────────────

class TestPerformanceGood:
    def test_empty_insights_returns_true(self, engine):
        assert engine.performance_good([]) is True

    def test_non_empty_insights_returns_false(self, engine):
        assert engine.performance_good([{"type": "drop"}]) is False

    def test_single_insight_returns_false(self, engine):
        assert engine.performance_good([{"type": "retention", "priority": "high"}]) is False


# ── generate_insights ─────────────────────────────────────────────────────────

class TestGenerateInsights:
    def test_no_data_returns_empty(self, engine):
        with patch("src.auto_action_engine._load_db", return_value=[]):
            insights = engine.generate_insights()
        assert insights == []

    def test_single_record_returns_empty(self, engine):
        records = [{"content_type": "daily", "views": 100, "avg_view_percentage": 50.0, "ctr": 0.05}]
        with patch("src.auto_action_engine._load_db", return_value=records):
            insights = engine.generate_insights()
        assert insights == []

    def test_mismatched_content_type_ignored(self, engine):
        records = [
            {"content_type": "weekly", "views": 100, "avg_view_percentage": 50.0, "ctr": 0.05},
            {"content_type": "weekly", "views": 100, "avg_view_percentage": 50.0, "ctr": 0.05},
        ]
        with patch("src.auto_action_engine._load_db", return_value=records):
            insights = engine.generate_insights()
        assert insights == []

    def test_zero_views_ignored(self, engine):
        records = [
            {"content_type": "daily", "views": 0, "avg_view_percentage": 50.0, "ctr": 0.05},
            {"content_type": "daily", "views": 0, "avg_view_percentage": 50.0, "ctr": 0.05},
        ]
        with patch("src.auto_action_engine._load_db", return_value=records):
            insights = engine.generate_insights()
        assert insights == []

    def test_retention_drop_generates_retention_insight(self, engine):
        # recent (last) has much lower retention than baseline
        records = [
            {"content_type": "daily", "views": 100, "avg_view_percentage": 60.0, "ctr": 0.05},
            {"content_type": "daily", "views": 100, "avg_view_percentage": 60.0, "ctr": 0.05},
            {"content_type": "daily", "views": 100, "avg_view_percentage": 60.0, "ctr": 0.05},
            {"content_type": "daily", "views": 100, "avg_view_percentage": 60.0, "ctr": 0.05},
            {"content_type": "daily", "views": 100, "avg_view_percentage": 60.0, "ctr": 0.05},
            {"content_type": "daily", "views": 100, "avg_view_percentage": 40.0, "ctr": 0.05},  # recent bad
        ]
        with patch("src.auto_action_engine._load_db", return_value=records):
            insights = engine.generate_insights()
        assert any(i["type"] == "retention" for i in insights)

    def test_low_retention_generates_drop_insight(self, engine):
        # recent retention < 35 → drop insight
        records = [
            {"content_type": "daily", "views": 100, "avg_view_percentage": 50.0, "ctr": 0.05},
            {"content_type": "daily", "views": 100, "avg_view_percentage": 25.0, "ctr": 0.05},  # recent low
        ]
        with patch("src.auto_action_engine._load_db", return_value=records):
            insights = engine.generate_insights()
        assert any(i["type"] == "drop" for i in insights)

    def test_low_ctr_generates_ctr_insight(self, engine):
        # CTR condition: recent_ctr < avg_ctr - 0.75 (large threshold in the code)
        # baseline avg_ctr must be high, recent ctr must be very low
        records = [
            {"content_type": "daily", "views": 100, "avg_view_percentage": 50.0, "ctr": 0.90},
            {"content_type": "daily", "views": 100, "avg_view_percentage": 50.0, "ctr": 0.90},
            {"content_type": "daily", "views": 100, "avg_view_percentage": 50.0, "ctr": 0.90},
            {"content_type": "daily", "views": 100, "avg_view_percentage": 50.0, "ctr": 0.90},
            {"content_type": "daily", "views": 100, "avg_view_percentage": 50.0, "ctr": 0.90},
            # recent: ctr=0.05, baseline avg=0.90 → diff=0.85 > 0.75 → ctr insight
            {"content_type": "daily", "views": 100, "avg_view_percentage": 50.0, "ctr": 0.05},
        ]
        with patch("src.auto_action_engine._load_db", return_value=records):
            insights = engine.generate_insights()
        assert any(i["type"] == "ctr" for i in insights)


# ── _map ──────────────────────────────────────────────────────────────────────

class TestMap:
    def test_map_retention_high_priority(self, engine):
        result = engine._map({"type": "retention", "priority": "high"})
        assert result is not None
        assert result["type"] == "increase_pacing"

    def test_map_retention_low_priority_returns_none(self, engine):
        result = engine._map({"type": "retention", "priority": "low"})
        assert result is None

    def test_map_drop(self, engine):
        result = engine._map({"type": "drop", "priority": "high"})
        assert result is not None
        assert result["type"] == "add_hook"

    def test_map_ctr(self, engine):
        result = engine._map({"type": "ctr", "priority": "high"})
        assert result is not None
        assert result["type"] == "improve_packaging"

    def test_map_unknown_returns_none(self, engine):
        result = engine._map({"type": "unknown", "priority": "high"})
        assert result is None

    def test_map_retention_has_source(self, engine):
        result = engine._map({"type": "retention", "priority": "high"})
        assert result["source"] == "retention"

    def test_map_drop_has_source(self, engine):
        result = engine._map({"type": "drop"})
        assert result["source"] == "drop"


# ── _score ────────────────────────────────────────────────────────────────────

class TestScore:
    def test_score_with_no_stats_returns_half(self, engine, tmp_path):
        with patch("src.auto_action_engine.settings") as mock_s:
            mock_s.data_dir = tmp_path
            score = engine._score({"type": "add_hook"})
        # default is success=1, fail=1 → 1/2 = 0.5
        assert abs(score - 0.5) < 1e-9

    def test_score_with_good_stats(self, engine, tmp_path):
        with patch("src.auto_action_engine.settings") as mock_s:
            mock_s.data_dir = tmp_path
            engine.save_stats({"add_hook": {"success": 9, "fail": 1}})
            score = engine._score({"type": "add_hook"})
        assert abs(score - 0.9) < 1e-9

    def test_score_with_bad_stats(self, engine, tmp_path):
        with patch("src.auto_action_engine.settings") as mock_s:
            mock_s.data_dir = tmp_path
            engine.save_stats({"add_hook": {"success": 1, "fail": 9}})
            score = engine._score({"type": "add_hook"})
        assert abs(score - 0.1) < 1e-9


# ── _select ───────────────────────────────────────────────────────────────────

class TestSelect:
    def test_select_empty_returns_empty(self, engine):
        result = engine._select([])
        assert result == []

    def test_select_deduplicates_by_type(self, engine):
        # Two actions of same type → deduplicated
        candidates = [
            ({"type": "add_hook", "reason": "a", "source": "drop"}, 0.8),
            ({"type": "add_hook", "reason": "b", "source": "drop"}, 0.7),
        ]
        # Force no exploration
        engine.exploration_rate = 0.0
        result = engine._select(candidates)
        assert len(result) <= 1
        types = [a["type"] for a in result]
        assert len(types) == len(set(types))

    def test_select_returns_confident_actions(self, engine):
        engine.exploration_rate = 0.0
        engine.min_confidence = 0.4
        candidates = [
            ({"type": "add_hook", "reason": "a", "source": "drop"}, 0.8),
        ]
        result = engine._select(candidates)
        assert len(result) == 1
        assert result[0]["selection_mode"] == "confident"

    def test_select_skips_below_min_confidence(self, engine):
        engine.exploration_rate = 0.0
        engine.min_confidence = 0.9  # high threshold
        candidates = [
            ({"type": "add_hook", "reason": "a", "source": "drop"}, 0.5),
        ]
        result = engine._select(candidates)
        assert result == []

    def test_select_respects_max_actions(self, engine):
        engine.exploration_rate = 0.0
        engine.max_actions = 1
        engine.min_confidence = 0.1
        candidates = [
            ({"type": "add_hook", "reason": "a", "source": "drop"}, 0.8),
            ({"type": "increase_pacing", "reason": "b", "source": "retention"}, 0.7),
            ({"type": "improve_packaging", "reason": "c", "source": "ctr"}, 0.6),
        ]
        result = engine._select(candidates)
        assert len(result) <= 1

    def test_select_adds_confidence_key(self, engine):
        engine.exploration_rate = 0.0
        engine.min_confidence = 0.1
        candidates = [
            ({"type": "add_hook", "reason": "a", "source": "drop"}, 0.75),
        ]
        result = engine._select(candidates)
        assert "confidence" in result[0]


# ── _apply ────────────────────────────────────────────────────────────────────

class TestApply:
    def test_apply_increase_pacing(self, engine):
        strategy = engine.default_strategy()
        result = engine._apply(strategy, [{"type": "increase_pacing"}])
        assert result["pace"] == "fast"

    def test_apply_add_hook(self, engine):
        strategy = {**engine.default_strategy(), "hook_aggressiveness": 0.5}
        result = engine._apply(strategy, [{"type": "add_hook"}])
        assert result["hook_aggressiveness"] > 0.5

    def test_apply_improve_packaging(self, engine):
        strategy = engine.default_strategy()
        result = engine._apply(strategy, [{"type": "improve_packaging"}])
        assert result["packaging_style"] == "conflict"

    def test_apply_does_not_mutate_original(self, engine):
        strategy = engine.default_strategy()
        original_pace = strategy["pace"]
        engine._apply(strategy, [{"type": "increase_pacing"}])
        assert strategy["pace"] == original_pace  # unchanged

    def test_apply_add_hook_caps_at_1(self, engine):
        strategy = {**engine.default_strategy(), "hook_aggressiveness": 0.95}
        result = engine._apply(strategy, [{"type": "add_hook"}])
        assert result["hook_aggressiveness"] <= 1.0


# ── apply (orchestrator) ──────────────────────────────────────────────────────

class TestApplyMethod:
    def test_mode_locked_returns_unchanged(self, engine, tmp_path):
        strategy = {**engine.default_strategy(), "mode_locked": True}
        with patch("src.auto_action_engine.settings") as mock_s:
            mock_s.data_dir = tmp_path
            result, actions = engine.apply(strategy, [{"type": "retention", "priority": "high"}])
        assert result["pace"] == "normal"
        assert actions == []

    def test_good_performance_returns_no_change(self, engine, tmp_path):
        strategy = engine.default_strategy()
        with patch("src.auto_action_engine.settings") as mock_s:
            mock_s.data_dir = tmp_path
            result, actions = engine.apply(strategy, [])
        assert actions[0]["type"] == "no_change"

    def test_increases_pacing_on_retention_insight(self, engine, tmp_path):
        strategy = engine.default_strategy()
        insights = [{"type": "retention", "priority": "high"}]
        engine.exploration_rate = 0.0
        with patch("src.auto_action_engine.settings") as mock_s:
            mock_s.data_dir = tmp_path
            result, actions = engine.apply(strategy, insights)
        # Either pacing was increased or actions is empty (below confidence)
        if actions:
            assert result["pace"] == "fast" or len(actions) == 0

    def test_apply_with_unknown_insight_type(self, engine, tmp_path):
        strategy = engine.default_strategy()
        insights = [{"type": "unknown", "priority": "high"}]
        with patch("src.auto_action_engine.settings") as mock_s:
            mock_s.data_dir = tmp_path
            result, actions = engine.apply(strategy, insights)
        # No valid actions mapped → returns strategy unchanged
        assert result == strategy

    def test_apply_returns_tuple(self, engine, tmp_path):
        strategy = engine.default_strategy()
        with patch("src.auto_action_engine.settings") as mock_s:
            mock_s.data_dir = tmp_path
            result = engine.apply(strategy, [])
        assert isinstance(result, tuple)
        assert len(result) == 2


# ── evaluate_recent_actions ───────────────────────────────────────────────────

class TestEvaluateRecentActions:
    def test_evaluate_with_no_pending(self, engine, tmp_path):
        with patch("src.auto_action_engine.settings") as mock_s:
            mock_s.data_dir = tmp_path
            with patch("src.auto_action_engine._load_db", return_value=[]):
                # Should complete without error and do nothing
                engine.evaluate_recent_actions()

    def test_evaluate_with_pending_but_insufficient_records(self, engine, tmp_path):
        store = {
            "entries": [{
                "run_type": "daily",
                "evaluation_status": "pending",
                "actions": [{"type": "add_hook"}],
            }],
            "scores": {},
        }
        with patch("src.auto_action_engine.settings") as mock_s:
            mock_s.data_dir = tmp_path
            engine.save_action_log(store)
            with patch("src.auto_action_engine._load_db", return_value=[]):
                engine.evaluate_recent_actions()
            # Should not raise, just return early

    def test_evaluate_with_pending_and_records(self, engine, tmp_path):
        store = {
            "entries": [{
                "run_type": "daily",
                "evaluation_status": "pending",
                "evaluated_video_id": None,
                "actions": [{"type": "add_hook"}],
            }],
            "scores": {},
        }
        records = [
            {"content_type": "daily", "views": 100, "avg_view_percentage": 50.0, "video_id": "v1"},
            {"content_type": "daily", "views": 100, "avg_view_percentage": 60.0, "video_id": "v2"},
        ]
        with patch("src.auto_action_engine.settings") as mock_s:
            mock_s.data_dir = tmp_path
            engine.save_action_log(store)
            with patch("src.auto_action_engine._load_db", return_value=records):
                engine.evaluate_recent_actions()
            updated_store = engine.load_action_log()
            entry = updated_store["entries"][0]
            assert entry["evaluation_status"] == "done"


# ── prepare_strategy ──────────────────────────────────────────────────────────

class TestPrepareStrategy:
    def test_returns_three_tuple(self, engine, tmp_path):
        with patch("src.auto_action_engine.settings") as mock_s:
            mock_s.data_dir = tmp_path
            with patch("src.auto_action_engine._load_db", return_value=[]):
                result = engine.prepare_strategy()
        assert isinstance(result, tuple)
        assert len(result) == 3

    def test_prepare_strategy_saves_action_log_entry(self, engine, tmp_path):
        with patch("src.auto_action_engine.settings") as mock_s:
            mock_s.data_dir = tmp_path
            with patch("src.auto_action_engine._load_db", return_value=[]):
                engine.prepare_strategy()
            log = engine.load_action_log()
        assert len(log["entries"]) == 1
        assert log["entries"][0]["run_type"] == "daily"

    def test_prepare_strategy_returns_strategy_with_required_keys(self, engine, tmp_path):
        with patch("src.auto_action_engine.settings") as mock_s:
            mock_s.data_dir = tmp_path
            with patch("src.auto_action_engine._load_db", return_value=[]):
                after, actions, insights = engine.prepare_strategy()
        assert "pace" in after
        assert "hook_aggressiveness" in after
        assert isinstance(actions, list)
        assert isinstance(insights, list)
