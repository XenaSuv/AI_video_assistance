"""Tests for src/deferred_feedback.py — _DeferredFeedbackCollector.

Targets >= 80% line coverage of src/deferred_feedback.py.
Heavy external deps are patched at the src.deferred_feedback namespace.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch, call

import pytest

# Ensure heavy google-SDK stubs are in place before any project import
for _m in (
    "google.oauth2.credentials",
    "google.auth.exceptions",
    "googleapiclient.http",
    "elevenlabs",
    "elevenlabs.client",
):
    sys.modules.setdefault(_m, MagicMock())

from src.deferred_feedback import _DeferredFeedbackCollector


# ── helpers ───────────────────────────────────────────────────────────────────

def _make_checkpoint(
    tmp_run: Path,
    video_id: str = "vid_abc",
    v3_mode: str = "stable",
    cross_combo: dict | None = None,
) -> None:
    """Write a minimal checkpoint.json into tmp_run."""
    data: dict = {
        "steps": {
            "upload_en": {"video_id": video_id},
            "script": {
                "v3_mode": v3_mode,
                "cross_combo": cross_combo,
            },
        }
    }
    (tmp_run / "checkpoint.json").write_text(json.dumps(data))


def _make_thompson_state_json(tmp_run: Path, n_arms: int = 1) -> None:
    """Write a minimal thompson_state.json so collect() finds this run_dir."""
    state = {
        "arms": [
            {
                "variant_id": f"arm_{i}",
                "type": "question",
                "alpha": 1.0,
                "beta": 1.0,
                "impressions": 100,
                "clicks": 10,
            }
            for i in range(n_arms)
        ]
    }
    (tmp_run / "thompson_state.json").write_text(json.dumps(state))


def _make_script_json(tmp_run: Path, n_scenes: int = 2) -> None:
    data = {
        "scenes": [
            {"scene_intent": "explain", "idx": i}
            for i in range(n_scenes)
        ]
    }
    (tmp_run / "script.json").write_text(json.dumps(data))


# ── collect() — no run dirs ───────────────────────────────────────────────────

class TestCollectNoRunDirs:
    def test_returns_none_when_output_dir_empty(self, tmp_path):
        """collect() returns None when there are no thompson_state.json files."""
        collector = _DeferredFeedbackCollector(current_run_dir=tmp_path / "current")
        with patch("src.deferred_feedback.settings") as mock_settings, \
             patch("src.deferred_feedback.ShortsExperimentEngine") as mock_shorts:
            mock_settings.output_dir = tmp_path
            mock_settings.data_dir = tmp_path
            mock_shorts.return_value.collect_analytics.return_value = 0
            result = collector.collect()
        assert result is None

    def test_collect_shorts_always_called(self, tmp_path):
        """_collect_shorts_analytics is always called, even with no run dirs."""
        collector = _DeferredFeedbackCollector(current_run_dir=tmp_path / "current")
        with patch("src.deferred_feedback.settings") as mock_settings, \
             patch("src.deferred_feedback.ShortsExperimentEngine") as mock_shorts:
            mock_settings.output_dir = tmp_path
            mock_settings.data_dir = tmp_path
            mock_shorts.return_value.collect_analytics.return_value = 0
            collector.collect()
        mock_shorts.return_value.collect_analytics.assert_called_once()


# ── collect() — current run dir is skipped ───────────────────────────────────

class TestCollectSkipsCurrentDir:
    def test_current_run_dir_is_not_processed(self, tmp_path):
        """If the only run dir is the current one, it should be skipped → None."""
        run_dir = tmp_path / "2026-05-18"
        run_dir.mkdir()
        _make_thompson_state_json(run_dir)
        _make_checkpoint(run_dir)

        collector = _DeferredFeedbackCollector(current_run_dir=run_dir)
        with patch("src.deferred_feedback.settings") as mock_settings, \
             patch("src.deferred_feedback.ShortsExperimentEngine") as mock_shorts, \
             patch("src.deferred_feedback.ThompsonBandit") as mock_tb:
            mock_settings.output_dir = tmp_path
            mock_settings.data_dir = tmp_path
            mock_shorts.return_value.collect_analytics.return_value = 0
            result = collector.collect()
        # ThompsonBandit should NOT be instantiated because current dir was skipped
        mock_tb.assert_not_called()
        assert result is None


# ── collect() — checkpoint missing ───────────────────────────────────────────

class TestCollectCheckpointMissing:
    def test_returns_none_when_no_checkpoint(self, tmp_path):
        run_dir = tmp_path / "2026-05-17"
        run_dir.mkdir()
        _make_thompson_state_json(run_dir)
        # No checkpoint.json written

        current = tmp_path / "2026-05-18"
        collector = _DeferredFeedbackCollector(current_run_dir=current)
        with patch("src.deferred_feedback.settings") as mock_settings, \
             patch("src.deferred_feedback.ShortsExperimentEngine") as mock_shorts:
            mock_settings.output_dir = tmp_path
            mock_settings.data_dir = tmp_path
            mock_shorts.return_value.collect_analytics.return_value = 0
            result = collector.collect()
        assert result is None


# ── collect() — no video_id in checkpoint ────────────────────────────────────

class TestCollectNoVideoId:
    def test_returns_none_when_no_video_id(self, tmp_path):
        run_dir = tmp_path / "2026-05-17"
        run_dir.mkdir()
        _make_thompson_state_json(run_dir)
        # Checkpoint without video_id
        (run_dir / "checkpoint.json").write_text(json.dumps({"steps": {}}))

        current = tmp_path / "2026-05-18"
        collector = _DeferredFeedbackCollector(current_run_dir=current)
        with patch("src.deferred_feedback.settings") as mock_settings, \
             patch("src.deferred_feedback.ShortsExperimentEngine") as mock_shorts:
            mock_settings.output_dir = tmp_path
            mock_settings.data_dir = tmp_path
            mock_shorts.return_value.collect_analytics.return_value = 0
            result = collector.collect()
        assert result is None


# ── collect() — bandit has no arms ───────────────────────────────────────────

class TestCollectBanditNoArms:
    def test_returns_none_when_bandit_has_no_arms(self, tmp_path):
        run_dir = tmp_path / "2026-05-17"
        run_dir.mkdir()
        _make_thompson_state_json(run_dir)
        _make_checkpoint(run_dir)

        current = tmp_path / "2026-05-18"
        collector = _DeferredFeedbackCollector(current_run_dir=current)

        mock_bandit = MagicMock()
        mock_bandit.get_state.return_value.arms = []

        with patch("src.deferred_feedback.settings") as mock_settings, \
             patch("src.deferred_feedback.ShortsExperimentEngine") as mock_shorts, \
             patch("src.deferred_feedback.ThompsonBandit", return_value=mock_bandit):
            mock_settings.output_dir = tmp_path
            mock_settings.data_dir = tmp_path
            mock_shorts.return_value.collect_analytics.return_value = 0
            result = collector.collect()
        assert result is None


# ── collect() — youtube metrics missing ──────────────────────────────────────

class TestCollectYouTubeMetricsMissing:
    def test_returns_none_when_yt_metrics_empty(self, tmp_path):
        run_dir = tmp_path / "2026-05-17"
        run_dir.mkdir()
        _make_thompson_state_json(run_dir)
        _make_checkpoint(run_dir)

        current = tmp_path / "2026-05-18"
        collector = _DeferredFeedbackCollector(current_run_dir=current)

        mock_arm = MagicMock()
        mock_arm.variant_id = "arm_0"
        mock_arm.impressions = 100
        mock_arm.clicks = 10
        mock_bandit = MagicMock()
        mock_bandit.get_state.return_value.arms = [mock_arm]

        with patch("src.deferred_feedback.settings") as mock_settings, \
             patch("src.deferred_feedback.ShortsExperimentEngine") as mock_shorts, \
             patch("src.deferred_feedback.ThompsonBandit", return_value=mock_bandit), \
             patch("src.deferred_feedback.get_video_metrics", return_value=None):
            mock_settings.output_dir = tmp_path
            mock_settings.data_dir = tmp_path
            mock_shorts.return_value.collect_analytics.return_value = 0
            result = collector.collect()
        assert result is None


# ── collect() — full happy path ───────────────────────────────────────────────

class TestCollectFullHappyPath:
    def _setup_run_dir(self, tmp_path: Path, name: str = "2026-05-17") -> Path:
        run_dir = tmp_path / name
        run_dir.mkdir()
        _make_thompson_state_json(run_dir)
        _make_checkpoint(run_dir, video_id="vid_123", cross_combo={
            "pace": "fast",
            "persona": "energetic",
            "hook_type": "question",
            "scene_type": "image",
            "topic": "AI news",
        })
        _make_script_json(run_dir)
        return run_dir

    def _make_arm(self, vid: str = "arm_0", impressions: int = 100, clicks: int = 10):
        arm = MagicMock()
        arm.variant_id = vid
        arm.impressions = impressions
        arm.clicks = clicks
        return arm

    def test_returns_preferred_type_string(self, tmp_path):
        run_dir = self._setup_run_dir(tmp_path)
        current = tmp_path / "2026-05-18"

        mock_arm = self._make_arm(impressions=100, clicks=10)
        mock_bandit = MagicMock()
        mock_bandit.get_state.return_value.arms = [mock_arm]
        mock_bandit.suggest_strategy_adjustments.return_value = {
            "preferred_variant_type": "question"
        }

        with patch("src.deferred_feedback.settings") as mock_settings, \
             patch("src.deferred_feedback.ShortsExperimentEngine") as mock_shorts, \
             patch("src.deferred_feedback.ThompsonBandit", return_value=mock_bandit), \
             patch("src.deferred_feedback.get_video_metrics", return_value={
                 "views": 5000,
                 "avg_view_percentage": 55.0,
             }), \
             patch("src.deferred_feedback.get_retention_curve", return_value=[0.9, 0.8, 0.7]), \
             patch("src.deferred_feedback.PerformanceStore") as mock_ps, \
             patch("src.deferred_feedback.SequenceLearningEngine") as mock_seq, \
             patch("src.deferred_feedback.CrossLearningEngine") as mock_cross, \
             patch("src.deferred_feedback.RetentionCorrectionEngine") as mock_rce, \
             patch("src.deferred_feedback._build_scene_map", return_value={0: {"scene_idx": 0, "intent": "explain"}}), \
             patch("src.deferred_feedback._save_retention_state"):
            mock_settings.output_dir = tmp_path
            mock_settings.data_dir = tmp_path
            mock_shorts.return_value.collect_analytics.return_value = 2
            mock_shorts.return_value.analyze.return_value = {
                "winners": 1, "best_hook_type": "question"
            }
            mock_rce.return_value.analyze.return_value = []
            mock_rce.return_value.suggest_strategy_adjustments.return_value = {}
            result = _DeferredFeedbackCollector(current_run_dir=current).collect()

        assert result == "question"

    def test_bandit_update_called_with_new_impressions(self, tmp_path):
        run_dir = self._setup_run_dir(tmp_path)
        current = tmp_path / "2026-05-18"

        mock_arm = self._make_arm(impressions=50, clicks=5)
        mock_bandit = MagicMock()
        mock_bandit.get_state.return_value.arms = [mock_arm]
        mock_bandit.suggest_strategy_adjustments.return_value = {
            "preferred_variant_type": "conflict"
        }

        with patch("src.deferred_feedback.settings") as mock_settings, \
             patch("src.deferred_feedback.ShortsExperimentEngine") as mock_shorts, \
             patch("src.deferred_feedback.ThompsonBandit", return_value=mock_bandit), \
             patch("src.deferred_feedback.get_video_metrics", return_value={
                 "views": 200,
                 "avg_view_percentage": 60.0,
             }), \
             patch("src.deferred_feedback.get_retention_curve", return_value=[]), \
             patch("src.deferred_feedback.PerformanceStore") as mock_ps, \
             patch("src.deferred_feedback.SequenceLearningEngine") as mock_seq, \
             patch("src.deferred_feedback.CrossLearningEngine") as mock_cross, \
             patch("src.deferred_feedback.RetentionCorrectionEngine"):
            mock_settings.output_dir = tmp_path
            mock_settings.data_dir = tmp_path
            mock_shorts.return_value.collect_analytics.return_value = 0
            _DeferredFeedbackCollector(current_run_dir=current).collect()

        # bandit.update should have been called (new_imps = 200 - 50 = 150 > 0)
        mock_bandit.update.assert_called_once()

    def test_bandit_update_skipped_when_no_new_impressions(self, tmp_path):
        run_dir = self._setup_run_dir(tmp_path)
        current = tmp_path / "2026-05-18"

        mock_arm = self._make_arm(impressions=5000, clicks=100)
        mock_bandit = MagicMock()
        mock_bandit.get_state.return_value.arms = [mock_arm]
        mock_bandit.suggest_strategy_adjustments.return_value = {"preferred_variant_type": None}

        with patch("src.deferred_feedback.settings") as mock_settings, \
             patch("src.deferred_feedback.ShortsExperimentEngine") as mock_shorts, \
             patch("src.deferred_feedback.ThompsonBandit", return_value=mock_bandit), \
             patch("src.deferred_feedback.get_video_metrics", return_value={
                 "views": 100,  # less than arm.impressions → new_imps = 0
                 "avg_view_percentage": 50.0,
             }), \
             patch("src.deferred_feedback.get_retention_curve", return_value=[]), \
             patch("src.deferred_feedback.PerformanceStore"), \
             patch("src.deferred_feedback.SequenceLearningEngine"), \
             patch("src.deferred_feedback.CrossLearningEngine"), \
             patch("src.deferred_feedback.RetentionCorrectionEngine"):
            mock_settings.output_dir = tmp_path
            mock_settings.data_dir = tmp_path
            mock_shorts.return_value.collect_analytics.return_value = 0
            _DeferredFeedbackCollector(current_run_dir=current).collect()

        mock_bandit.update.assert_not_called()

    def test_most_common_preferred_type_returned(self, tmp_path):
        """With two run dirs, the most frequent preferred type wins."""
        current = tmp_path / "2026-05-18"
        for date_str in ("2026-05-17", "2026-05-16"):
            self._setup_run_dir(tmp_path, name=date_str)

        mock_arm = self._make_arm()
        mock_bandit = MagicMock()
        mock_bandit.get_state.return_value.arms = [mock_arm]
        mock_bandit.suggest_strategy_adjustments.return_value = {
            "preferred_variant_type": "question"
        }

        with patch("src.deferred_feedback.settings") as mock_settings, \
             patch("src.deferred_feedback.ShortsExperimentEngine") as mock_shorts, \
             patch("src.deferred_feedback.ThompsonBandit", return_value=mock_bandit), \
             patch("src.deferred_feedback.get_video_metrics", return_value={
                 "views": 1000, "avg_view_percentage": 60.0,
             }), \
             patch("src.deferred_feedback.get_retention_curve", return_value=[]), \
             patch("src.deferred_feedback.PerformanceStore"), \
             patch("src.deferred_feedback.SequenceLearningEngine"), \
             patch("src.deferred_feedback.CrossLearningEngine"), \
             patch("src.deferred_feedback.RetentionCorrectionEngine"):
            mock_settings.output_dir = tmp_path
            mock_settings.data_dir = tmp_path
            mock_shorts.return_value.collect_analytics.return_value = 0
            result = _DeferredFeedbackCollector(current_run_dir=current).collect()

        assert result == "question"

    def test_exception_in_process_run_dir_does_not_propagate(self, tmp_path):
        run_dir = self._setup_run_dir(tmp_path)
        current = tmp_path / "2026-05-18"

        with patch("src.deferred_feedback.settings") as mock_settings, \
             patch("src.deferred_feedback.ShortsExperimentEngine") as mock_shorts, \
             patch("src.deferred_feedback.ThompsonBandit", side_effect=RuntimeError("boom")):
            mock_settings.output_dir = tmp_path
            mock_settings.data_dir = tmp_path
            mock_shorts.return_value.collect_analytics.return_value = 0
            # Must not raise — exceptions are swallowed and logged
            result = _DeferredFeedbackCollector(current_run_dir=current).collect()

        assert result is None


# ── _update_perf_store ────────────────────────────────────────────────────────

class TestUpdatePerfStore:
    def test_save_called_with_correct_video_id(self, tmp_path):
        collector = _DeferredFeedbackCollector(current_run_dir=tmp_path)
        cp_data = {"steps": {"script": {"v3_mode": "growth"}}}

        with patch("src.deferred_feedback.PerformanceStore") as mock_ps, \
             patch("src.deferred_feedback.settings") as mock_settings:
            mock_settings.data_dir = tmp_path
            collector._update_perf_store(cp_data, "vid_xyz", 0.65)

        mock_ps.return_value.save.assert_called_once()
        call_kwargs = mock_ps.return_value.save.call_args
        assert call_kwargs.kwargs.get("video_id") == "vid_xyz" or \
               call_kwargs[1].get("video_id") == "vid_xyz" or \
               call_kwargs[0][0] == "vid_xyz"

    def test_strategy_mode_extracted_from_cp_data(self, tmp_path):
        collector = _DeferredFeedbackCollector(current_run_dir=tmp_path)
        cp_data = {"steps": {"script": {"v3_mode": "retention_fix"}}}

        with patch("src.deferred_feedback.PerformanceStore") as mock_ps, \
             patch("src.deferred_feedback.settings") as mock_settings:
            mock_settings.data_dir = tmp_path
            collector._update_perf_store(cp_data, "vid_xyz", 0.5)

        save_kwargs = mock_ps.return_value.save.call_args
        # strategy_mode should be "retention_fix"
        assert "retention_fix" in str(save_kwargs)

    def test_defaults_to_stable_when_v3_mode_missing(self, tmp_path):
        collector = _DeferredFeedbackCollector(current_run_dir=tmp_path)
        cp_data = {"steps": {}}  # no script key

        with patch("src.deferred_feedback.PerformanceStore") as mock_ps, \
             patch("src.deferred_feedback.settings") as mock_settings:
            mock_settings.data_dir = tmp_path
            collector._update_perf_store(cp_data, "vid_xyz", 0.5)

        save_kwargs = mock_ps.return_value.save.call_args
        assert "stable" in str(save_kwargs)


# ── _store_sequence ────────────────────────────────────────────────────────────

class TestStoreSequence:
    def test_skips_when_script_json_missing(self, tmp_path):
        collector = _DeferredFeedbackCollector(current_run_dir=tmp_path)

        with patch("src.deferred_feedback.SequenceLearningEngine") as mock_seq, \
             patch("src.deferred_feedback.settings") as mock_settings:
            mock_settings.data_dir = tmp_path
            collector._store_sequence(tmp_path, retention=0.5)  # no script.json

        mock_seq.return_value.store_sequence.assert_not_called()

    def test_calls_store_sequence_when_script_exists(self, tmp_path):
        _make_script_json(tmp_path)
        collector = _DeferredFeedbackCollector(current_run_dir=tmp_path)

        with patch("src.deferred_feedback.SequenceLearningEngine") as mock_seq, \
             patch("src.deferred_feedback.settings") as mock_settings:
            mock_settings.data_dir = tmp_path
            collector._store_sequence(tmp_path, retention=0.7)

        mock_seq.return_value.store_sequence.assert_called_once()

    def test_uses_scene_intent_from_script(self, tmp_path):
        script_data = {
            "scenes": [
                {"scene_intent": "shock", "idx": 0},
                {"intent": "data", "idx": 1},
            ]
        }
        (tmp_path / "script.json").write_text(json.dumps(script_data))
        collector = _DeferredFeedbackCollector(current_run_dir=tmp_path)

        with patch("src.deferred_feedback.SequenceLearningEngine") as mock_seq, \
             patch("src.deferred_feedback.settings") as mock_settings:
            mock_settings.data_dir = tmp_path
            collector._store_sequence(tmp_path, retention=0.6)

        call_arg = mock_seq.return_value.store_sequence.call_args[0][0]
        intents = [s["intent"] for s in call_arg["scenes"]]
        assert intents == ["shock", "data"]

    def test_exception_swallowed_gracefully(self, tmp_path):
        _make_script_json(tmp_path)
        collector = _DeferredFeedbackCollector(current_run_dir=tmp_path)

        with patch("src.deferred_feedback.SequenceLearningEngine") as mock_seq, \
             patch("src.deferred_feedback.settings") as mock_settings:
            mock_settings.data_dir = tmp_path
            mock_seq.return_value.store_sequence.side_effect = RuntimeError("db err")
            # Must not raise
            collector._store_sequence(tmp_path, retention=0.5)


# ── _learn_cross ───────────────────────────────────────────────────────────────

class TestLearnCross:
    def test_skips_when_no_cross_combo(self, tmp_path):
        collector = _DeferredFeedbackCollector(current_run_dir=tmp_path)
        cp_data = {"steps": {"script": {}}}  # no cross_combo

        with patch("src.deferred_feedback.CrossLearningEngine") as mock_cross, \
             patch("src.deferred_feedback.settings") as mock_settings:
            mock_settings.data_dir = tmp_path
            collector._learn_cross(tmp_path, cp_data, retention=0.5)

        mock_cross.return_value.learn.assert_not_called()

    def test_calls_learn_when_cross_combo_present(self, tmp_path):
        collector = _DeferredFeedbackCollector(current_run_dir=tmp_path)
        cp_data = {
            "steps": {
                "script": {
                    "cross_combo": {
                        "pace": "fast",
                        "persona": "energetic",
                        "hook_type": "question",
                        "scene_type": "image",
                        "topic": "AI news",
                    }
                }
            }
        }

        with patch("src.deferred_feedback.CrossLearningEngine") as mock_cross, \
             patch("src.deferred_feedback.settings") as mock_settings:
            mock_settings.data_dir = tmp_path
            collector._learn_cross(tmp_path, cp_data, retention=0.75)

        mock_cross.return_value.learn.assert_called_once()

    def test_exception_in_learn_is_swallowed(self, tmp_path):
        collector = _DeferredFeedbackCollector(current_run_dir=tmp_path)
        cp_data = {
            "steps": {"script": {"cross_combo": {"pace": "normal"}}}
        }

        with patch("src.deferred_feedback.CrossLearningEngine") as mock_cross, \
             patch("src.deferred_feedback.settings") as mock_settings:
            mock_settings.data_dir = tmp_path
            mock_cross.return_value.learn.side_effect = ValueError("oops")
            # Must not raise
            collector._learn_cross(tmp_path, cp_data, retention=0.5)


# ── _analyze_retention_curve ───────────────────────────────────────────────────

class TestAnalyzeRetentionCurve:
    def test_skips_rce_when_curve_empty(self, tmp_path):
        _make_script_json(tmp_path)
        collector = _DeferredFeedbackCollector(current_run_dir=tmp_path)

        with patch("src.deferred_feedback.RetentionCorrectionEngine") as mock_rce, \
             patch("src.deferred_feedback.get_retention_curve", return_value=[]), \
             patch("src.deferred_feedback.settings") as mock_settings:
            mock_settings.data_dir = tmp_path
            collector._analyze_retention_curve(tmp_path, "vid_abc")

        mock_rce.return_value.analyze.assert_not_called()

    def test_calls_analyze_when_curve_and_script_exist(self, tmp_path):
        _make_script_json(tmp_path)
        collector = _DeferredFeedbackCollector(current_run_dir=tmp_path)

        with patch("src.deferred_feedback.RetentionCorrectionEngine") as mock_rce, \
             patch("src.deferred_feedback.get_retention_curve", return_value=[0.9, 0.8, 0.7]), \
             patch("src.deferred_feedback._build_scene_map", return_value={0: {"scene_idx": 0}}), \
             patch("src.deferred_feedback._save_retention_state") as mock_save, \
             patch("src.deferred_feedback.settings") as mock_settings:
            mock_settings.data_dir = tmp_path
            mock_rce.return_value.analyze.return_value = []
            mock_rce.return_value.suggest_strategy_adjustments.return_value = {}
            collector._analyze_retention_curve(tmp_path, "vid_abc")

        mock_rce.return_value.analyze.assert_called_once()
        mock_save.assert_called_once()

    def test_exception_in_rce_is_swallowed(self, tmp_path):
        _make_script_json(tmp_path)
        collector = _DeferredFeedbackCollector(current_run_dir=tmp_path)

        with patch("src.deferred_feedback.get_retention_curve",
                   side_effect=RuntimeError("network error")), \
             patch("src.deferred_feedback.settings") as mock_settings:
            mock_settings.data_dir = tmp_path
            # Must not raise
            collector._analyze_retention_curve(tmp_path, "vid_abc")

    def test_skips_rce_when_script_missing_but_curve_exists(self, tmp_path):
        # No script.json in tmp_path
        collector = _DeferredFeedbackCollector(current_run_dir=tmp_path)

        with patch("src.deferred_feedback.RetentionCorrectionEngine") as mock_rce, \
             patch("src.deferred_feedback.get_retention_curve", return_value=[0.9, 0.7]), \
             patch("src.deferred_feedback.settings") as mock_settings:
            mock_settings.data_dir = tmp_path
            collector._analyze_retention_curve(tmp_path, "vid_abc")

        # analyze should not be called because script.json is missing
        mock_rce.return_value.analyze.assert_not_called()


# ── _collect_shorts_analytics ──────────────────────────────────────────────────

class TestCollectShortsAnalytics:
    def test_calls_collect_analytics(self, tmp_path):
        collector = _DeferredFeedbackCollector(current_run_dir=tmp_path)

        with patch("src.deferred_feedback.ShortsExperimentEngine") as mock_shorts, \
             patch("src.deferred_feedback.settings") as mock_settings:
            mock_settings.data_dir = tmp_path
            mock_shorts.return_value.collect_analytics.return_value = 3
            mock_shorts.return_value.analyze.return_value = {
                "winners": 2, "best_hook_type": "curiosity"
            }
            collector._collect_shorts_analytics()

        mock_shorts.return_value.collect_analytics.assert_called_once_with(min_age_hours=4.0)

    def test_analyze_called_when_new_results(self, tmp_path):
        collector = _DeferredFeedbackCollector(current_run_dir=tmp_path)

        with patch("src.deferred_feedback.ShortsExperimentEngine") as mock_shorts, \
             patch("src.deferred_feedback.settings") as mock_settings:
            mock_settings.data_dir = tmp_path
            mock_shorts.return_value.collect_analytics.return_value = 5
            mock_shorts.return_value.analyze.return_value = {
                "winners": 1, "best_hook_type": "conflict"
            }
            collector._collect_shorts_analytics()

        mock_shorts.return_value.analyze.assert_called_once()

    def test_analyze_not_called_when_zero_results(self, tmp_path):
        collector = _DeferredFeedbackCollector(current_run_dir=tmp_path)

        with patch("src.deferred_feedback.ShortsExperimentEngine") as mock_shorts, \
             patch("src.deferred_feedback.settings") as mock_settings:
            mock_settings.data_dir = tmp_path
            mock_shorts.return_value.collect_analytics.return_value = 0
            collector._collect_shorts_analytics()

        mock_shorts.return_value.analyze.assert_not_called()

    def test_exception_in_shorts_engine_is_swallowed(self, tmp_path):
        collector = _DeferredFeedbackCollector(current_run_dir=tmp_path)

        with patch("src.deferred_feedback.ShortsExperimentEngine",
                   side_effect=RuntimeError("crash")), \
             patch("src.deferred_feedback.settings") as mock_settings:
            mock_settings.data_dir = tmp_path
            # Must not raise
            collector._collect_shorts_analytics()
