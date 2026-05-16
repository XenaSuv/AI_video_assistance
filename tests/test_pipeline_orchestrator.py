"""Tests for pipeline_orchestrator — stateless helper functions.

The PipelineOrchestrator class itself is an integration boundary (20+ deps).
These tests cover the pure/file-I/O helpers that are fully unit-testable.
conftest.py already stubs numpy and Google SDK modules.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Stub additional deps the orchestrator imports at module level
for _m in (
    "google.oauth2.credentials",
    "google.auth.exceptions",
    "googleapiclient.http",
    "elevenlabs",
    "elevenlabs.client",
):
    sys.modules.setdefault(_m, MagicMock())

from src.pipeline_orchestrator import (
    _build_scene_map,
    _build_v3_context,
    _classify_hook_type,
    _load_cached_script,
    _load_retention_state,
    _save_retention_state,
    _unified_strategy_to_content_strategy,
)
from src.decision_engine_v3 import UnifiedStrategy
from src.script_generator import Scene, VideoScript


# ── _classify_hook_type ───────────────────────────────────────────────────────

class TestClassifyHookType:
    def test_conflict_on_problem_keyword(self):
        assert _classify_hook_type("There is a real problem with this model") == "conflict"

    def test_conflict_on_fail_keyword(self):
        assert _classify_hook_type("Why GPT-4 failed on this benchmark") == "conflict"

    def test_conflict_on_vs_keyword(self):
        assert _classify_hook_type("GPT-4 vs Claude: who wins?") == "conflict"

    def test_conflict_on_threat_keyword(self):
        assert _classify_hook_type("AI is a threat to jobs") == "conflict"

    def test_curiosity_on_nobody_keyword(self):
        assert _classify_hook_type("Nobody is talking about this") == "curiosity"

    def test_curiosity_on_secret_keyword(self):
        assert _classify_hook_type("The hidden secret of neural scaling") == "curiosity"

    def test_curiosity_on_question_mark(self):
        assert _classify_hook_type("Is this the end of GPT?") == "curiosity"

    def test_curiosity_on_wait_keyword(self):
        assert _classify_hook_type("Wait — this changes everything") == "curiosity"

    def test_simple_on_plain_statement(self):
        assert _classify_hook_type("OpenAI released GPT-5 today") == "simple"

    def test_simple_on_empty_string(self):
        assert _classify_hook_type("") == "simple"

    def test_conflict_takes_priority_over_curiosity(self):
        # "problem" is a conflict word; "secret" is curiosity — conflict wins
        hook = "The secret problem with AI scaling"
        assert _classify_hook_type(hook) == "conflict"

    def test_case_insensitive(self):
        assert _classify_hook_type("THE PROBLEM IS REAL") == "conflict"
        assert _classify_hook_type("NOBODY KNOWS THIS") == "curiosity"


# ── _build_scene_map ──────────────────────────────────────────────────────────

def _scene_dict(idx: int, dur: int = 60, intent: str = "explain") -> dict:
    return {"idx": idx, "duration_sec": dur, "scene_intent": intent, "scene_type": "image"}


class TestBuildSceneMap:
    def test_empty_scenes_returns_empty(self):
        assert _build_scene_map([], curve_len=10) == {}

    def test_zero_curve_len_returns_empty(self):
        assert _build_scene_map([_scene_dict(0)], curve_len=0) == {}

    def test_every_bucket_mapped_for_single_scene(self):
        result = _build_scene_map([_scene_dict(0)], curve_len=5)
        assert set(result.keys()) == {0, 1, 2, 3, 4}

    def test_bucket_scene_idx_values_are_valid(self):
        scenes = [_scene_dict(i, dur=30) for i in range(3)]
        result = _build_scene_map(scenes, curve_len=9)
        scene_indices = {v["scene_idx"] for v in result.values()}
        assert scene_indices <= {0, 1, 2}

    def test_intent_preserved(self):
        scenes = [_scene_dict(0, intent="shock")]
        result = _build_scene_map(scenes, curve_len=3)
        assert all(v["intent"] == "shock" for v in result.values())

    def test_longer_scene_gets_more_buckets(self):
        scenes = [_scene_dict(0, dur=90), _scene_dict(1, dur=30)]
        result = _build_scene_map(scenes, curve_len=12)
        buckets_0 = sum(1 for v in result.values() if v["scene_idx"] == 0)
        buckets_1 = sum(1 for v in result.values() if v["scene_idx"] == 1)
        assert buckets_0 > buckets_1

    def test_defaults_when_duration_missing(self):
        scene = {"scene_intent": "explain", "scene_type": "image"}  # no duration_sec
        result = _build_scene_map([scene], curve_len=4)
        assert len(result) > 0


# ── _unified_strategy_to_content_strategy ─────────────────────────────────────

def _strategy(mode: str = "stable", hook_aggressiveness: float = 0.5) -> UnifiedStrategy:
    return UnifiedStrategy(mode=mode, hook_aggressiveness=hook_aggressiveness)


class TestUnifiedStrategyToContentStrategy:
    def test_retention_fix_maps_to_safe(self):
        cs = _unified_strategy_to_content_strategy(_strategy("retention_fix"))
        assert cs.mode == "safe"

    def test_stable_maps_to_balanced(self):
        cs = _unified_strategy_to_content_strategy(_strategy("stable"))
        assert cs.mode == "balanced"

    def test_growth_maps_to_growth(self):
        cs = _unified_strategy_to_content_strategy(_strategy("growth"))
        assert cs.mode == "growth"

    def test_packaging_focus_maps_to_growth(self):
        cs = _unified_strategy_to_content_strategy(_strategy("packaging_focus"))
        assert cs.mode == "growth"

    def test_unknown_mode_defaults_to_balanced(self):
        cs = _unified_strategy_to_content_strategy(_strategy("unknown_mode"))
        assert cs.mode == "balanced"

    def test_exploration_rate_decreases_with_aggressiveness(self):
        low  = _unified_strategy_to_content_strategy(_strategy(hook_aggressiveness=0.0))
        high = _unified_strategy_to_content_strategy(_strategy(hook_aggressiveness=1.0))
        assert low.exploration_rate > high.exploration_rate

    def test_exploration_rate_clamped_between_0_1_and_0_4(self):
        for agr in (0.0, 0.5, 1.0):
            cs = _unified_strategy_to_content_strategy(_strategy(hook_aggressiveness=agr))
            assert 0.1 <= cs.exploration_rate <= 0.4

    def test_retention_fix_has_high_confidence(self):
        cs = _unified_strategy_to_content_strategy(_strategy("retention_fix"))
        assert cs.confidence >= 0.7

    def test_angle_and_format_weights_empty(self):
        cs = _unified_strategy_to_content_strategy(_strategy())
        assert cs.angle_weights == {}
        assert cs.format_weights == {}


# ── _load_retention_state / _save_retention_state ─────────────────────────────

class TestRetentionState:
    def test_returns_empty_dict_when_file_absent(self, tmp_path):
        result = _load_retention_state(tmp_path)
        assert result == {}

    def test_loads_saved_state(self, tmp_path):
        (tmp_path / "retention_state.json").write_text(
            json.dumps({"corrections": [], "adjustments": {"foo": 1}})
        )
        result = _load_retention_state(tmp_path)
        assert result["adjustments"] == {"foo": 1}

    def test_returns_empty_on_corrupt_json(self, tmp_path):
        (tmp_path / "retention_state.json").write_text("not-json{")
        result = _load_retention_state(tmp_path)
        assert result == {}

    def test_save_and_reload_roundtrip(self, tmp_path):
        corrections = [
            MagicMock(to_dict=lambda: {"scene_idx": 0, "drop": {"delta": 0.2}})
        ]
        _save_retention_state(tmp_path, corrections, {"mood": "flat"})
        state = _load_retention_state(tmp_path)
        assert state["adjustments"] == {"mood": "flat"}
        assert len(state["corrections"]) == 1

    def test_save_creates_data_dir(self, tmp_path):
        subdir = tmp_path / "nested" / "data"
        _save_retention_state(subdir, [], {})
        assert (subdir / "retention_state.json").exists()


# ── _load_cached_script ───────────────────────────────────────────────────────

class TestLoadCachedScript:
    def _write_script_json(self, path: Path, n_scenes: int = 2) -> None:
        data = {
            "title": "Test",
            "description": "desc",
            "tags": ["ai"],
            "hook": "hook text",
            "hook_variants": [],
            "scenes": [
                {
                    "heading": f"Scene {i}",
                    "narration": "text",
                    "visual_prompt": "prompt",
                    "duration_sec": 30,
                }
                for i in range(n_scenes)
            ],
        }
        path.write_text(json.dumps(data))

    def test_returns_none_when_file_absent(self, tmp_path):
        assert _load_cached_script(tmp_path / "missing.json") is None

    def test_returns_video_script(self, tmp_path):
        p = tmp_path / "script.json"
        self._write_script_json(p)
        result = _load_cached_script(p)
        assert isinstance(result, VideoScript)

    def test_scene_count_correct(self, tmp_path):
        p = tmp_path / "script.json"
        self._write_script_json(p, n_scenes=3)
        result = _load_cached_script(p)
        assert len(result.scenes) == 3

    def test_scene_idx_assigned_sequentially(self, tmp_path):
        p = tmp_path / "script.json"
        self._write_script_json(p, n_scenes=3)
        result = _load_cached_script(p)
        assert [s.idx for s in result.scenes] == [0, 1, 2]

    def test_title_preserved(self, tmp_path):
        p = tmp_path / "script.json"
        self._write_script_json(p)
        result = _load_cached_script(p)
        assert result.title == "Test"


# ── _build_v3_context ─────────────────────────────────────────────────────────

class TestBuildV3Context:
    def test_returns_dict_with_required_keys(self):
        perf_store = MagicMock()
        perf_store.get_channel_metrics.return_value = {"views": 1000}
        v3_engine = MagicMock()
        v3_engine.load_history.return_value = []

        result = _build_v3_context(
            perf_store, v3_engine,
            thompson_preferred_type="conflict",
            predicted_risks=[{"scene_idx": 1, "probability": 0.3}],
        )

        assert "metrics" in result
        assert "bandit" in result
        assert "prediction" in result
        assert "history" in result

    def test_thompson_type_in_packaging(self):
        perf_store = MagicMock()
        perf_store.get_channel_metrics.return_value = {}
        v3_engine = MagicMock()
        v3_engine.load_history.return_value = []

        result = _build_v3_context(perf_store, v3_engine, "curiosity", [])
        assert result["bandit"]["packaging"]["preferred_variant_type"] == "curiosity"

    def test_none_thompson_type_becomes_empty_string(self):
        perf_store = MagicMock()
        perf_store.get_channel_metrics.return_value = {}
        v3_engine = MagicMock()
        v3_engine.load_history.return_value = []

        result = _build_v3_context(perf_store, v3_engine, None, [])
        assert result["bandit"]["packaging"]["preferred_variant_type"] == ""

    def test_predicted_risks_forwarded(self):
        perf_store = MagicMock()
        perf_store.get_channel_metrics.return_value = {}
        v3_engine = MagicMock()
        v3_engine.load_history.return_value = []
        risks = [{"scene_idx": 2, "probability": 0.7}]

        result = _build_v3_context(perf_store, v3_engine, None, risks)
        assert result["prediction"]["risks"] == risks

    def test_channel_metrics_from_perf_store(self):
        perf_store = MagicMock()
        perf_store.get_channel_metrics.return_value = {"ctr": 0.05}
        v3_engine = MagicMock()
        v3_engine.load_history.return_value = []

        result = _build_v3_context(perf_store, v3_engine, None, [])
        assert result["metrics"] == {"ctr": 0.05}
