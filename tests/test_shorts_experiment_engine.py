"""Tests for ShortsExperimentEngine.

Covers:
- Experiment generation (variant count, hook types, template filling)
- ShortsExperimentStore persistence (save / load / update / has_result)
- record_result / get_best_hooks / get_strong_signal
- analyze() integration with CrossLearningEngine and AutoActionEngine (mocked)
- collect_analytics() with mocked YouTube API
"""
from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from src.shorts_experiment_engine import (
    ShortsExperiment,
    ShortsExperimentEngine,
    ShortsExperimentResult,
    ShortsExperimentStore,
    _HOOK_AGG_BOOST,
    _MIN_SIGNAL_COUNT,
    _WINNER_THRESHOLD,
)


# ── helpers ───────────────────────────────────────────────────────────────────

def _story(title: str = "OpenAI releases GPT-5") -> dict[str, Any]:
    return {"title": title, "source": "OpenAI"}


# ── ShortsExperimentStore ─────────────────────────────────────────────────────

class TestShortsExperimentStore:
    def test_save_and_load_experiment(self, tmp_path):
        store = ShortsExperimentStore(data_dir=tmp_path)
        exp = ShortsExperiment(
            experiment_id="exp-1",
            story_title="Test",
            hook_type="conflict",
            hook_text="This changes everything",
            core_text="Core text",
            payoff_text="Follow us",
        )
        store.save_experiment(exp)
        loaded = store.load_experiments()
        assert len(loaded) == 1
        assert loaded[0].experiment_id == "exp-1"
        assert loaded[0].hook_type == "conflict"

    def test_update_experiment(self, tmp_path):
        store = ShortsExperimentStore(data_dir=tmp_path)
        exp = ShortsExperiment(
            experiment_id="exp-2",
            story_title="Test",
            hook_type="curiosity",
            hook_text="Nobody talks about this",
            core_text="Core",
            payoff_text="Follow",
        )
        store.save_experiment(exp)
        store.update_experiment("exp-2", {"video_id": "yt-abc123"})
        updated = store.load_experiments()
        assert updated[0].video_id == "yt-abc123"

    def test_has_result_false_then_true(self, tmp_path):
        store = ShortsExperimentStore(data_dir=tmp_path)
        assert not store.has_result("exp-3")
        result = ShortsExperimentResult(
            experiment_id="exp-3",
            video_id="vid-1",
            retention_3s=0.8,
        )
        store.save_result(result)
        assert store.has_result("exp-3")

    def test_load_results_empty(self, tmp_path):
        store = ShortsExperimentStore(data_dir=tmp_path)
        assert store.load_results() == []

    def test_round_trip_result(self, tmp_path):
        store = ShortsExperimentStore(data_dir=tmp_path)
        r = ShortsExperimentResult(
            experiment_id="exp-4",
            video_id="vid-2",
            retention_3s=0.75,
            avg_watch_pct=0.6,
            completion_rate=0.4,
            views=100,
            is_winner=True,
        )
        store.save_result(r)
        loaded = store.load_results()
        assert loaded[0].retention_3s == 0.75
        assert loaded[0].is_winner is True


# ── ShortsExperimentEngine: generation ───────────────────────────────────────

class TestGenerate:
    def test_returns_up_to_five_variants(self, tmp_path):
        engine = ShortsExperimentEngine(data_dir=tmp_path)
        exps = engine.generate(_story())
        assert 1 <= len(exps) <= 5

    def test_each_variant_has_unique_id(self, tmp_path):
        engine = ShortsExperimentEngine(data_dir=tmp_path)
        exps = engine.generate(_story())
        ids = [e.experiment_id for e in exps]
        assert len(ids) == len(set(ids))

    def test_hook_text_contains_topic(self, tmp_path):
        engine = ShortsExperimentEngine(data_dir=tmp_path)
        exps = engine.generate(_story("OpenAI releases GPT-5"))
        # At least one hook should contain part of the title
        assert any("OpenAI" in e.hook_text for e in exps)

    def test_experiments_persisted(self, tmp_path):
        engine = ShortsExperimentEngine(data_dir=tmp_path)
        exps = engine.generate(_story())
        store = ShortsExperimentStore(data_dir=tmp_path)
        loaded = store.load_experiments()
        assert len(loaded) == len(exps)

    def test_different_hook_types(self, tmp_path):
        engine = ShortsExperimentEngine(data_dir=tmp_path)
        exps = engine.generate(_story())
        hook_types = {e.hook_type for e in exps}
        assert len(hook_types) >= 2


# ── ShortsExperimentEngine: record_result / get_best_hooks / signal ──────────

class TestSignalExtraction:
    def test_get_best_hooks_empty_when_no_results(self, tmp_path):
        engine = ShortsExperimentEngine(data_dir=tmp_path)
        engine.generate(_story())
        assert engine.get_best_hooks() == []

    def test_get_best_hooks_sorted_by_retention(self, tmp_path):
        engine = ShortsExperimentEngine(data_dir=tmp_path)
        exps = engine.generate(_story())
        # Record results in reverse order so the last is worst
        for i, exp in enumerate(exps):
            engine.record_result(
                exp.experiment_id,
                retention_3s=round(0.9 - i * 0.1, 2),
            )
        hooks = engine.get_best_hooks(n=3)
        assert len(hooks) <= 3
        # First hook should correspond to highest retention
        assert hooks[0] == exps[0].hook_text

    def test_strong_signal_none_when_few_winners(self, tmp_path):
        engine = ShortsExperimentEngine(data_dir=tmp_path)
        exps = engine.generate(_story())
        # Record just 1 winner
        engine.record_result(exps[0].experiment_id, retention_3s=0.85)
        assert engine.get_strong_signal() is None

    def test_strong_signal_value_when_enough_winners(self, tmp_path):
        engine = ShortsExperimentEngine(data_dir=tmp_path)
        exps = engine.generate(_story())
        for exp in exps[:_MIN_SIGNAL_COUNT]:
            engine.record_result(exp.experiment_id, retention_3s=_WINNER_THRESHOLD + 0.05)
        signal = engine.get_strong_signal()
        assert signal == _HOOK_AGG_BOOST

    def test_record_result_marks_winner_correctly(self, tmp_path):
        engine = ShortsExperimentEngine(data_dir=tmp_path)
        exps = engine.generate(_story())
        r = engine.record_result(exps[0].experiment_id, retention_3s=_WINNER_THRESHOLD + 0.01)
        assert r.is_winner is True

    def test_record_result_marks_loser_correctly(self, tmp_path):
        engine = ShortsExperimentEngine(data_dir=tmp_path)
        exps = engine.generate(_story())
        r = engine.record_result(exps[0].experiment_id, retention_3s=_WINNER_THRESHOLD - 0.01)
        assert r.is_winner is False


# ── ShortsExperimentEngine: mark_uploaded ────────────────────────────────────

class TestMarkUploaded:
    def test_mark_uploaded_persists_video_id(self, tmp_path):
        engine = ShortsExperimentEngine(data_dir=tmp_path)
        exps = engine.generate(_story())
        engine.mark_uploaded(exps[0].experiment_id, video_id="yt-xyz999")
        store = ShortsExperimentStore(data_dir=tmp_path)
        loaded = store.load_experiments()
        exp = next(e for e in loaded if e.experiment_id == exps[0].experiment_id)
        assert exp.video_id == "yt-xyz999"


# ── ShortsExperimentEngine: analyze ──────────────────────────────────────────

class TestAnalyze:
    def test_analyze_no_results_returns_zero_winners(self, tmp_path):
        engine = ShortsExperimentEngine(data_dir=tmp_path)
        engine.generate(_story())
        summary = engine.analyze()
        assert summary["winners"] == 0
        assert summary["cross_learning"] is False

    def test_analyze_with_winner_triggers_cross_learning(self, tmp_path):
        engine = ShortsExperimentEngine(data_dir=tmp_path)
        exps = engine.generate(_story())
        engine.record_result(exps[0].experiment_id, retention_3s=0.85)

        mock_cl = MagicMock()
        with patch("src.shorts_experiment_engine.CrossLearningEngine", return_value=mock_cl):
            with patch("src.shorts_experiment_engine.AutoActionEngine") as mock_aa_cls:
                mock_aa = MagicMock()
                mock_aa.load_strategy.return_value = {"hook_aggressiveness": 0.5}
                mock_aa_cls.return_value = mock_aa
                summary = engine.analyze()

        assert summary["winners"] == 1
        assert summary["cross_learning"] is True
        mock_cl.learn.assert_called_once()

    def test_analyze_auto_action_boosts_hook_aggressiveness(self, tmp_path):
        engine = ShortsExperimentEngine(data_dir=tmp_path)
        exps = engine.generate(_story())
        engine.record_result(exps[0].experiment_id, retention_3s=0.90)

        with patch("src.shorts_experiment_engine.CrossLearningEngine") as mock_cl_cls:
            mock_cl_cls.return_value = MagicMock()
            with patch("src.shorts_experiment_engine.AutoActionEngine") as mock_aa_cls:
                mock_aa = MagicMock()
                mock_aa.load_strategy.return_value = {"hook_aggressiveness": 0.4}
                mock_aa_cls.return_value = mock_aa
                summary = engine.analyze()

        assert summary["auto_action"] is True
        saved_strategy = mock_aa.save_strategy.call_args[0][0]
        assert saved_strategy["hook_aggressiveness"] > 0.4


# ── collect_analytics ─────────────────────────────────────────────────────────

class TestCollectAnalytics:
    def test_skips_experiments_without_video_id(self, tmp_path):
        engine = ShortsExperimentEngine(data_dir=tmp_path)
        exps = engine.generate(_story())
        # No video_id set
        with patch("src.shorts_experiment_engine.get_video_metrics") as mock_yt:
            collected = engine.collect_analytics(min_age_hours=0.0)
        assert collected == 0
        mock_yt.assert_not_called()

    def test_collects_result_for_uploaded_experiment(self, tmp_path):
        engine = ShortsExperimentEngine(data_dir=tmp_path)
        exps = engine.generate(_story())
        engine.mark_uploaded(exps[0].experiment_id, video_id="yt-abc")
        # Patch the created_at to be old enough
        store = ShortsExperimentStore(data_dir=tmp_path)
        store.update_experiment(exps[0].experiment_id, {"created_at": "2020-01-01T00:00:00+00:00"})

        mock_metrics = {"views": 500, "avg_view_percentage": 40.0}
        with patch("src.shorts_experiment_engine.get_video_metrics", return_value=mock_metrics):
            collected = engine.collect_analytics(min_age_hours=0.0)

        assert collected == 1
        results = store.load_results()
        assert len(results) == 1
        assert results[0].views == 500

    def test_does_not_double_collect(self, tmp_path):
        engine = ShortsExperimentEngine(data_dir=tmp_path)
        exps = engine.generate(_story())
        engine.mark_uploaded(exps[0].experiment_id, video_id="yt-abc")
        store = ShortsExperimentStore(data_dir=tmp_path)
        store.update_experiment(exps[0].experiment_id, {"created_at": "2020-01-01T00:00:00+00:00"})

        mock_metrics = {"views": 100, "avg_view_percentage": 30.0}
        with patch("src.shorts_experiment_engine.get_video_metrics", return_value=mock_metrics):
            first  = engine.collect_analytics(min_age_hours=0.0)
            second = engine.collect_analytics(min_age_hours=0.0)

        assert first == 1
        assert second == 0
