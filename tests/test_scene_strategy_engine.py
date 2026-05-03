"""Tests for src/scene_strategy_engine.py"""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.scene_strategy_engine import (
    INTENT_TO_SCENE,
    SceneStrategy,
    SceneStrategyEngine,
    SceneStrategyItem,
    _DEFAULT_SCORES,
)
from src.scene_variety_engine import SceneVarietyEngineV2
from src.script_generator import Scene


# ── helpers ───────────────────────────────────────────────────────────────────

def _scene(idx: int, heading: str = "Heading", narration: str = "Normal text.", **kwargs) -> Scene:
    return Scene(idx=idx, heading=heading, narration=narration, visual_prompt="test", **kwargs)


def _plan(fmt: str = "quick_hit", angle: str = "technical_breakthrough") -> MagicMock:
    p = MagicMock()
    p.editorial_plan = [{"format": fmt, "angle": angle}]
    return p


@pytest.fixture
def engine(tmp_path) -> SceneStrategyEngine:
    return SceneStrategyEngine(data_dir=tmp_path)


# ── INTENT_TO_SCENE ───────────────────────────────────────────────────────────

class TestIntentToScene:
    def test_all_intents_have_candidates(self):
        for intent in ("hook", "explain", "shock", "data", "reaction", "transition"):
            assert intent in INTENT_TO_SCENE
            assert len(INTENT_TO_SCENE[intent]) >= 1

    def test_all_candidate_types_are_valid(self):
        valid = {"image", "text_overlay", "infographic", "cutaway", "diagram"}
        for candidates in INTENT_TO_SCENE.values():
            for t in candidates:
                assert t in valid, f"Unknown scene type: {t!r}"


# ── SceneStrategy ─────────────────────────────────────────────────────────────

class TestSceneStrategy:
    def test_get_returns_item_by_index(self):
        s = SceneStrategy(items=[SceneStrategyItem(0, "hook"), SceneStrategyItem(1, "explain")])
        item = s.get(1)
        assert item is not None
        assert item.intent == "explain"

    def test_get_returns_none_for_missing_index(self):
        s = SceneStrategy(items=[SceneStrategyItem(0, "hook")])
        assert s.get(99) is None

    def test_empty_strategy(self):
        s = SceneStrategy()
        assert s.get(0) is None


# ── build_strategy ────────────────────────────────────────────────────────────

class TestBuildStrategy:
    def test_returns_SceneStrategy(self, engine):
        scenes = [_scene(i) for i in range(3)]
        result = engine.build_strategy(scenes)
        assert isinstance(result, SceneStrategy)

    def test_item_count_matches_scene_count(self, engine):
        scenes = [_scene(i) for i in range(5)]
        result = engine.build_strategy(scenes)
        assert len(result.items) == 5

    def test_first_scene_always_hook(self, engine):
        scenes = [_scene(0, narration="Revenue grew 500%")]
        result = engine.build_strategy(scenes)
        assert result.items[0].intent == "hook"

    def test_index_matches_scene_idx(self, engine):
        scenes = [_scene(i) for i in range(4)]
        strategy = engine.build_strategy(scenes)
        for item, scene in zip(strategy.items, scenes):
            assert item.index == scene.idx

    def test_accepts_none_performance_data(self, engine):
        scenes = [_scene(i) for i in range(3)]
        result = engine.build_strategy(scenes, performance_data=None)
        assert len(result.items) == 3

    def test_accepts_none_editorial_plan(self, engine):
        scenes = [_scene(i) for i in range(3)]
        result = engine.build_strategy(scenes, editorial_plan=None)
        assert len(result.items) == 3

    def test_empty_scenes(self, engine):
        result = engine.build_strategy([])
        assert result.items == []


# ── _detect_intent ────────────────────────────────────────────────────────────

class TestDetectIntent:
    def test_index_0_always_hook(self, engine):
        scene = _scene(0, narration="Why this changes everything forever")
        assert engine._detect_intent(scene, 0, "") == "hook"

    def test_infographic_data_gives_data(self, engine):
        scene = _scene(1, infographic_data={"type": "bar_chart"})
        assert engine._detect_intent(scene, 1, "") == "data"

    def test_percent_gives_data(self, engine):
        scene = _scene(1, narration="Accuracy improved by 15%")
        assert engine._detect_intent(scene, 1, "") == "data"

    def test_million_gives_data(self, engine):
        scene = _scene(1, narration="Company raised 500 million in funding")
        assert engine._detect_intent(scene, 1, "") == "data"

    def test_billion_gives_data(self, engine):
        scene = _scene(2, narration="Valuation hit 2.3 billion dollars")
        assert engine._detect_intent(scene, 2, "") == "data"

    def test_why_gives_explain(self, engine):
        scene = _scene(1, narration="Here is why this matters")
        assert engine._detect_intent(scene, 1, "") == "explain"

    def test_how_gives_explain(self, engine):
        scene = _scene(1, narration="Here is how the mechanism works step by step")
        assert engine._detect_intent(scene, 1, "") == "explain"

    def test_process_gives_explain(self, engine):
        scene = _scene(1, narration="The training process has three phases")
        assert engine._detect_intent(scene, 1, "") == "explain"

    def test_but_gives_shock(self, engine):
        scene = _scene(1, narration="But this changes everything about what we thought")
        assert engine._detect_intent(scene, 1, "") == "shock"

    def test_problem_gives_shock(self, engine):
        scene = _scene(1, narration="The problem is that nobody noticed the flaw")
        assert engine._detect_intent(scene, 1, "") == "shock"

    def test_nobody_gives_shock(self, engine):
        scene = _scene(1, narration="Nobody saw this coming at all")
        assert engine._detect_intent(scene, 1, "") == "shock"

    def test_crazy_gives_reaction(self, engine):
        scene = _scene(1, narration="The result was honestly crazy")
        assert engine._detect_intent(scene, 1, "") == "reaction"

    def test_insane_gives_reaction(self, engine):
        scene = _scene(1, narration="This is insane and changes everything we knew")
        assert engine._detect_intent(scene, 1, "") == "reaction"

    def test_index_3_no_signal_gives_transition(self, engine):
        scene = _scene(3, narration="Some plain text with no strong signal in here")
        assert engine._detect_intent(scene, 3, "") == "transition"

    def test_index_6_no_signal_gives_transition(self, engine):
        scene = _scene(6, narration="More plain unremarkable content with no signal")
        assert engine._detect_intent(scene, 6, "") == "transition"

    def test_angle_hint_applied_on_no_signal(self, engine):
        scene = _scene(1, narration="Plain neutral text about some topic")
        # industry_impact angle → data intent
        assert engine._detect_intent(scene, 1, "industry_impact") == "data"

    def test_angle_hint_applied_threat_to_jobs(self, engine):
        scene = _scene(1, narration="Plain neutral text about some topic")
        assert engine._detect_intent(scene, 1, "threat_to_jobs") == "shock"

    def test_content_signal_beats_angle_hint(self, engine):
        # Data keyword in text beats the angle's "shock" hint
        scene = _scene(1, narration="Revenue grew by 200%")
        assert engine._detect_intent(scene, 1, "threat_to_jobs") == "data"

    def test_data_beats_explain_in_ordering(self, engine):
        # Both "how" and "%" present — data regex checked first
        scene = _scene(1, narration="Here is how the 50% improvement was achieved")
        assert engine._detect_intent(scene, 1, "") == "data"


# ── _optimize_with_feedback ───────────────────────────────────────────────────

class TestOptimizeWithFeedback:
    def test_low_retention_position_changes_intent_to_shock(self, engine):
        scenes = [_scene(i) for i in range(4)]
        strategy = engine.build_strategy(
            scenes, performance_data={"low_retention_positions": [2]}
        )
        assert strategy.get(2).intent == "shock"

    def test_multiple_low_retention_positions(self, engine):
        scenes = [_scene(i) for i in range(6)]
        strategy = engine.build_strategy(
            scenes, performance_data={"low_retention_positions": [1, 3, 5]}
        )
        assert strategy.get(1).intent == "shock"
        assert strategy.get(3).intent == "shock"
        assert strategy.get(5).intent == "shock"

    def test_non_drop_scenes_unchanged(self, engine):
        scenes = [_scene(i) for i in range(4)]
        # Only scene 2 is a drop point
        strategy = engine.build_strategy(
            scenes, performance_data={"low_retention_positions": [2]}
        )
        # Scene 0 must still be hook
        assert strategy.get(0).intent == "hook"
        # Scene 1 and 3 must not be changed to shock
        assert strategy.get(1).intent != "shock"  # no data/explain/etc signal

    def test_low_avg_retention_boosts_weight_for_explain_scenes(self, engine):
        scenes = [_scene(0), _scene(1, narration="Here is how the process works")]
        strategy = engine.build_strategy(
            scenes, performance_data={"avg_retention": 0.3}
        )
        # explain scene gets weight_boost > 0
        assert strategy.get(1).weight_boost > 0

    def test_high_avg_retention_no_weight_boost(self, engine):
        scenes = [_scene(0), _scene(1, narration="Here is how the process works")]
        strategy = engine.build_strategy(
            scenes, performance_data={"avg_retention": 0.8}
        )
        assert strategy.get(1).weight_boost == 0.0

    def test_high_retention_types_merged_into_scores(self, engine):
        scenes = [_scene(i) for i in range(2)]
        engine.build_strategy(
            scenes, performance_data={"high_retention_types": {"infographic": 0.95}}
        )
        assert engine._scores["infographic"] >= 0.95


# ── pick_best ─────────────────────────────────────────────────────────────────

class TestPickBest:
    def test_returns_valid_scene_type(self, engine):
        for intent in INTENT_TO_SCENE:
            t = engine.pick_best(intent)
            assert t in INTENT_TO_SCENE[intent]

    def test_returns_highest_scored_option(self, engine):
        # Manually set infographic higher than image so "data" → "infographic"
        engine._scores["infographic"] = 0.99
        engine._scores["image"] = 0.01
        assert engine.pick_best("data") == "infographic"

    def test_weight_boost_shifts_selection(self, engine):
        # Make image and text_overlay scores equal, then use weight_boost to decide
        engine._scores["image"] = 0.60
        engine._scores["text_overlay"] = 0.55
        # Without boost: image wins for "hook"
        assert engine.pick_best("hook", weight_boost=0.0) == "image"
        # With large boost: text_overlay effectively scores 0.55 + 0.1 > 0.60 → wins
        assert engine.pick_best("hook", weight_boost=0.1) == "text_overlay"

    def test_unknown_intent_falls_back_to_image(self, engine):
        result = engine.pick_best("nonexistent_intent_xyz")
        assert result == "image"


# ── update_scores ─────────────────────────────────────────────────────────────

class TestUpdateScores:
    def test_persists_json_file(self, engine, tmp_path):
        engine.update_scores({"infographic": 0.8})
        path = tmp_path / "scene_performance.json"
        assert path.exists()
        data = json.loads(path.read_text())
        assert "infographic" in data

    def test_ema_update_formula(self, engine):
        engine._scores["image"] = 0.55
        engine.update_scores({"image": 0.75})
        expected = round(0.7 * 0.55 + 0.3 * 0.75, 4)
        assert abs(engine._scores["image"] - expected) < 1e-4

    def test_new_type_added(self, engine):
        engine.update_scores({"future_type": 0.90})
        assert "future_type" in engine._scores

    def test_multiple_types_updated(self, engine, tmp_path):
        engine.update_scores({"image": 0.7, "diagram": 0.6})
        path = tmp_path / "scene_performance.json"
        data = json.loads(path.read_text())
        assert "image" in data
        assert "diagram" in data


# ── _load_scores ──────────────────────────────────────────────────────────────

class TestLoadScores:
    def test_missing_file_uses_defaults(self, tmp_path):
        engine = SceneStrategyEngine(data_dir=tmp_path)
        assert engine._scores == _DEFAULT_SCORES

    def test_existing_file_merged_with_defaults(self, tmp_path):
        path = tmp_path / "scene_performance.json"
        path.write_text(json.dumps({"infographic": 0.90}))
        engine = SceneStrategyEngine(data_dir=tmp_path)
        assert engine._scores["infographic"] == 0.90
        assert "image" in engine._scores  # default still present

    def test_corrupt_file_falls_back_to_defaults(self, tmp_path):
        path = tmp_path / "scene_performance.json"
        path.write_text("not valid json {{}")
        engine = SceneStrategyEngine(data_dir=tmp_path)
        assert engine._scores == _DEFAULT_SCORES

    def test_get_scores_returns_copy(self, engine):
        scores = engine.get_scores()
        scores["image"] = 9999.0
        assert engine._scores["image"] != 9999.0


# ── SceneVarietyEngineV2 (integration) ────────────────────────────────────────

class TestSceneVarietyEngineV2:
    def test_assign_sets_scene_type_from_strategy(self, engine):
        scenes = [_scene(0), _scene(1)]
        strategy = engine.build_strategy(scenes)
        SceneVarietyEngineV2(engine).assign(scenes, strategy)
        for s in scenes:
            from src.scene_variety_engine import SCENE_TYPES
            assert s.scene_type in SCENE_TYPES

    def test_assign_sets_scene_intent_from_strategy(self, engine):
        scenes = [_scene(0), _scene(1)]
        strategy = engine.build_strategy(scenes)
        SceneVarietyEngineV2(engine).assign(scenes, strategy)
        from src.scene_variety_engine import SCENE_INTENTS
        for s in scenes:
            assert s.scene_intent in SCENE_INTENTS

    def test_returns_same_list(self, engine):
        scenes = [_scene(i) for i in range(3)]
        strategy = engine.build_strategy(scenes)
        result = SceneVarietyEngineV2(engine).assign(scenes, strategy)
        assert result is scenes

    def test_no_consecutive_same_type(self, engine):
        scenes = [_scene(i) for i in range(8)]
        strategy = engine.build_strategy(scenes)
        SceneVarietyEngineV2(engine).assign(scenes, strategy)
        for i in range(1, len(scenes)):
            prev, curr = scenes[i-1].scene_type, scenes[i].scene_type
            if curr != "cutaway":
                assert curr != prev, f"Consecutive {prev!r} at scenes {i-1},{i}"

    def test_infographic_data_honoured_over_strategy(self, engine):
        scenes = [
            _scene(0),
            _scene(1, narration="Plain text", infographic_data={"type": "stat_card"}),
        ]
        strategy = engine.build_strategy(scenes)
        # Force a non-infographic intent for scene 1
        item = strategy.get(1)
        if item:
            item.intent = "shock"
        SceneVarietyEngineV2(engine).assign(scenes, strategy)
        assert scenes[1].scene_type == "infographic"

    def test_missing_strategy_item_falls_back(self, engine):
        # Empty strategy: both scenes fall back to "image".
        # enforce_variety then turns scene 1 into "cutaway" (consecutive image→image).
        scenes = [_scene(0), _scene(1)]
        empty_strategy = SceneStrategy()  # no items
        SceneVarietyEngineV2(engine).assign(scenes, empty_strategy)
        assert scenes[0].scene_type == "image"
        assert scenes[1].scene_type == "cutaway"

    def test_default_engine_created_if_none_passed(self):
        # Should not raise even when no engine is injected
        v2 = SceneVarietyEngineV2()
        scenes = [_scene(0), _scene(1)]
        from src.scene_strategy_engine import SceneStrategyEngine
        strat = SceneStrategyEngine().build_strategy(scenes)
        v2.assign(scenes, strat)
        from src.scene_variety_engine import SCENE_TYPES
        for s in scenes:
            assert s.scene_type in SCENE_TYPES

    def test_full_pipeline_with_performance_data(self, engine):
        scenes = [_scene(i) for i in range(6)]
        perf = {
            "low_retention_positions": [2, 4],
            "high_retention_types": {"text_overlay": 0.80},
            "avg_retention": 0.45,
        }
        strategy = engine.build_strategy(scenes, performance_data=perf)
        SceneVarietyEngineV2(engine).assign(scenes, strategy)
        # Drop-point scenes should be "shock" intent → text_overlay or cutaway
        for i in [2, 4]:
            assert scenes[i].scene_intent == "shock"
            from src.scene_variety_engine import SCENE_TYPES
            assert scenes[i].scene_type in SCENE_TYPES
