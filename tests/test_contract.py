"""Tests for the unified contract between DecisionEngineV2 and SceneVarietyEngineV2.

Covers:
- ScenePolicy normalisation and serialisation
- StrategyConfig.scene_policy population from strategy builders
- SceneVarietyEngineV2.assign_from_config() full pipeline
- Avatar frequency management
- Pattern interrupt enforcement
- Priority intent bypass
- update_scene_mix_from_feedback()
- End-to-end: DecisionEngineV2.decide() → assign_from_config()
"""
from __future__ import annotations

import pytest
from dataclasses import dataclass, field
from typing import Any
from unittest.mock import patch

from src.decision_engine_v2 import DecisionEngineV2, ScenePolicy, StrategyConfig, PerformanceStore
from src.scene_variety_engine import SceneVarietyEngineV2
from src.scene_strategy_engine import SceneStrategyEngine


# ── Helpers ────────────────────────────────────────────────────────────────────

def _scene(idx: int, heading: str = "Heading", narration: str = "Narration text"):
    from src.script_generator import Scene
    return Scene(idx=idx, heading=heading, narration=narration, visual_prompt="")


def _scenes(n: int) -> list:
    return [_scene(i) for i in range(n)]


# ── ScenePolicy ────────────────────────────────────────────────────────────────

class TestScenePolicy:
    def test_empty_mix_no_normalisation(self):
        p = ScenePolicy()
        assert p.mix == {}

    def test_already_normalised_mix_unchanged(self):
        mix = {"image": 0.5, "diagram": 0.5}
        p = ScenePolicy(mix=mix)
        assert abs(sum(p.mix.values()) - 1.0) < 0.01

    def test_unnormalised_mix_is_normalised(self):
        p = ScenePolicy(mix={"image": 2.0, "diagram": 2.0, "cutaway": 1.0})
        assert abs(sum(p.mix.values()) - 1.0) < 1e-6

    def test_defaults(self):
        p = ScenePolicy()
        assert p.pattern_interrupt_frequency == 3
        assert p.priority_intents == ["hook"]
        assert p.avoid_repetition is True

    def test_to_dict_round_trip(self):
        p = ScenePolicy(
            mix={"image": 0.6, "cutaway": 0.4},
            pattern_interrupt_frequency=5,
            priority_intents=["hook", "shock"],
            avoid_repetition=False,
        )
        d = p.to_dict()
        p2 = ScenePolicy.from_dict(d)
        assert p2.pattern_interrupt_frequency == 5
        assert p2.priority_intents == ["hook", "shock"]
        assert p2.avoid_repetition is False
        assert abs(p2.mix.get("image", 0) - 0.6) < 1e-6

    def test_from_dict_defaults_on_empty(self):
        p = ScenePolicy.from_dict({})
        assert p.mix == {}
        assert p.pattern_interrupt_frequency == 3


# ── StrategyConfig + ScenePolicy sync ─────────────────────────────────────────

class TestStrategyConfigScenePolicySync:
    def test_scene_mix_copied_to_policy_mix(self):
        mix = {"image": 0.5, "diagram": 0.5}
        cfg = StrategyConfig(scene_mix=mix)
        assert cfg.scene_policy.mix == cfg.scene_mix

    def test_policy_mix_copied_to_scene_mix(self):
        mix = {"text_overlay": 0.7, "cutaway": 0.3}
        policy = ScenePolicy(mix=mix)
        cfg = StrategyConfig(scene_policy=policy)
        assert cfg.scene_mix == mix

    def test_both_set_policy_mix_preserved(self):
        mix_a = {"image": 1.0}
        mix_b = {"cutaway": 1.0}
        cfg = StrategyConfig(
            scene_mix=mix_a,
            scene_policy=ScenePolicy(mix=mix_b),
        )
        # scene_mix normalised; policy.mix kept as given
        assert "image" in cfg.scene_mix
        assert "cutaway" in cfg.scene_policy.mix

    def test_strategy_config_round_trip(self):
        cfg = StrategyConfig(
            mode="safe",
            scene_mix={"image": 0.5, "diagram": 0.5},
            scene_policy=ScenePolicy(pattern_interrupt_frequency=5),
        )
        d   = cfg.to_dict()
        cfg2 = StrategyConfig.from_dict(d)
        assert cfg2.mode == "safe"
        assert cfg2.scene_policy.pattern_interrupt_frequency == 5


# ── Strategy builders populate scene_policy ───────────────────────────────────

class TestStrategyBuildersPolicies:
    def setup_method(self):
        self.engine = DecisionEngineV2(performance_store=PerformanceStore(data_dir=_tmp_dir()))

    def test_growth_policy_fields(self):
        cfg = self.engine._growth_strategy({})
        assert cfg.scene_policy.pattern_interrupt_frequency == 2
        assert "hook" in cfg.scene_policy.priority_intents
        assert "shock" in cfg.scene_policy.priority_intents
        assert cfg.scene_policy.avoid_repetition is True

    def test_safe_policy_fields(self):
        cfg = self.engine._safe_strategy({})
        assert cfg.scene_policy.pattern_interrupt_frequency == 4
        assert cfg.scene_policy.priority_intents == ["hook"]

    def test_experimental_policy_has_mix(self):
        cfg = self.engine._experimental_strategy({})
        assert isinstance(cfg.scene_policy.mix, dict)
        assert len(cfg.scene_policy.mix) > 0

    def test_growth_policy_mix_matches_scene_mix(self):
        cfg = self.engine._growth_strategy({})
        assert cfg.scene_policy.mix == cfg.scene_mix

    def test_safe_policy_mix_matches_scene_mix(self):
        cfg = self.engine._safe_strategy({})
        assert cfg.scene_policy.mix == cfg.scene_mix


# ── assign_from_config: intent assignment ─────────────────────────────────────

class TestAssignFromConfigIntents:
    def setup_method(self):
        self.engine = SceneVarietyEngineV2()
        self.cfg = StrategyConfig(
            mode="growth",
            scene_mix={"text_overlay": 0.5, "image": 0.3, "cutaway": 0.2},
            scene_policy=ScenePolicy(
                mix={"text_overlay": 0.5, "image": 0.3, "cutaway": 0.2},
                pattern_interrupt_frequency=0,   # disable to isolate intent tests
                avoid_repetition=False,
                priority_intents=[],
            ),
            avatar_frequency=0.0,
        )

    def test_scene_0_always_hook(self):
        scenes = _scenes(4)
        self.engine.assign_from_config(scenes, self.cfg)
        assert scenes[0].scene_intent == "hook"

    def test_data_narration_detected(self):
        s = _scenes(2)
        s[1].narration = "Revenue grew 45% last quarter"
        self.engine.assign_from_config(s, self.cfg)
        assert s[1].scene_intent == "data"

    def test_explain_narration_detected(self):
        s = _scenes(2)
        s[1].narration = "Here is how the process works step by step"
        self.engine.assign_from_config(s, self.cfg)
        assert s[1].scene_intent == "explain"

    def test_shock_narration_detected(self):
        s = _scenes(2)
        s[1].narration = "The company secretly banned all internal tools"
        self.engine.assign_from_config(s, self.cfg)
        assert s[1].scene_intent == "shock"

    def test_infographic_data_forces_data_intent(self):
        s = _scenes(2)
        s[1].infographic_data = {"type": "stat_card"}
        self.engine.assign_from_config(s, self.cfg)
        assert s[1].scene_intent == "data"
        assert s[1].scene_type == "infographic"


# ── assign_from_config: priority intents bypass random draw ───────────────────

class TestPriorityIntents:
    def test_hook_scene_uses_pick_best(self):
        """Scene 0 (hook) must get the best type for 'hook', not a random draw."""
        engine = SceneVarietyEngineV2()
        cfg = StrategyConfig(
            mode="growth",
            scene_mix={"image": 0.5, "text_overlay": 0.5},
            scene_policy=ScenePolicy(
                mix={"image": 0.5, "text_overlay": 0.5},
                pattern_interrupt_frequency=0,
                priority_intents=["hook"],
                avoid_repetition=False,
            ),
            avatar_frequency=0.0,
        )
        scenes = _scenes(3)
        # Run many times; scene 0 intent should always be "hook"
        for _ in range(10):
            engine.assign_from_config(scenes, cfg)
            assert scenes[0].scene_intent == "hook"


# ── Pattern interrupt enforcement ─────────────────────────────────────────────

class TestPatternInterrupts:
    def test_cutaway_inserted_at_interval(self):
        engine = SceneVarietyEngineV2()
        cfg = StrategyConfig(
            mode="safe",
            scene_mix={"image": 1.0},
            scene_policy=ScenePolicy(
                mix={"image": 1.0},
                pattern_interrupt_frequency=3,
                priority_intents=[],
                avoid_repetition=False,
            ),
            avatar_frequency=0.0,
        )
        scenes = _scenes(7)
        engine.assign_from_config(scenes, cfg)
        # Positions 3 and 6 (multiples of 3, skipping 0) → cutaway
        assert scenes[3].scene_type == "cutaway"
        assert scenes[6].scene_type == "cutaway"

    def test_zero_frequency_disables_pattern_interrupt(self):
        engine = SceneVarietyEngineV2()
        cfg = StrategyConfig(
            mode="safe",
            scene_mix={"image": 1.0},
            scene_policy=ScenePolicy(
                mix={"image": 1.0},
                pattern_interrupt_frequency=0,
                priority_intents=[],
                avoid_repetition=False,
            ),
            avatar_frequency=0.0,
        )
        scenes = _scenes(6)
        engine.assign_from_config(scenes, cfg)
        # No forced cutaways (scene 0 = hook, others = image unless variety kicks in)
        cutaway_count = sum(1 for s in scenes[1:] if s.scene_type == "cutaway")
        assert cutaway_count == 0


# ── Avatar frequency management ───────────────────────────────────────────────

class TestAvatarFrequency:
    def test_avatar_scenes_created(self):
        engine = SceneVarietyEngineV2()
        cfg = StrategyConfig(
            mode="growth",
            avatar_frequency=0.3,
            scene_mix={"image": 1.0},
            scene_policy=ScenePolicy(
                mix={"image": 1.0},
                pattern_interrupt_frequency=0,
                avoid_repetition=False,
                priority_intents=[],
            ),
        )
        scenes = _scenes(10)
        engine.assign_from_config(scenes, cfg)
        avatar_count = sum(1 for s in scenes if s.scene_type == "avatar")
        assert avatar_count >= 1

    def test_hook_never_replaced_by_avatar(self):
        engine = SceneVarietyEngineV2()
        cfg = StrategyConfig(
            mode="growth",
            avatar_frequency=1.0,   # max frequency
            scene_mix={"image": 1.0},
            scene_policy=ScenePolicy(
                mix={"image": 1.0},
                pattern_interrupt_frequency=0,
                avoid_repetition=False,
                priority_intents=[],
            ),
        )
        scenes = _scenes(5)
        engine.assign_from_config(scenes, cfg)
        # scene 0 (hook) must never become avatar
        assert scenes[0].scene_type != "avatar"

    def test_zero_avatar_frequency_produces_no_avatars(self):
        engine = SceneVarietyEngineV2()
        cfg = StrategyConfig(
            mode="safe",
            avatar_frequency=0.0,
            scene_mix={"image": 1.0},
            scene_policy=ScenePolicy(
                mix={"image": 1.0},
                pattern_interrupt_frequency=0,
                avoid_repetition=False,
                priority_intents=[],
            ),
        )
        scenes = _scenes(8)
        engine.assign_from_config(scenes, cfg)
        assert all(s.scene_type != "avatar" for s in scenes)


# ── Variety enforcement ───────────────────────────────────────────────────────

class TestVarietyEnforcement:
    def test_avoid_repetition_breaks_consecutive_types(self):
        engine = SceneVarietyEngineV2()
        cfg = StrategyConfig(
            mode="safe",
            scene_mix={"image": 1.0},
            scene_policy=ScenePolicy(
                mix={"image": 1.0},
                pattern_interrupt_frequency=0,
                avoid_repetition=True,
                priority_intents=[],
            ),
            avatar_frequency=0.0,
        )
        scenes = _scenes(4)
        engine.assign_from_config(scenes, cfg)
        for i in range(1, len(scenes)):
            if scenes[i - 1].scene_type == scenes[i].scene_type:
                assert scenes[i].scene_type == "cutaway", (
                    f"consecutive types not broken at scene {scenes[i].idx}"
                )

    def test_avoid_repetition_false_allows_duplicates(self):
        engine = SceneVarietyEngineV2()
        cfg = StrategyConfig(
            mode="safe",
            scene_mix={"image": 1.0},
            scene_policy=ScenePolicy(
                mix={"image": 1.0},
                pattern_interrupt_frequency=0,
                avoid_repetition=False,
                priority_intents=[],
            ),
            avatar_frequency=0.0,
        )
        scenes = _scenes(4)
        engine.assign_from_config(scenes, cfg)
        # All non-hook scenes should be "image" (no variety enforcement)
        assert all(s.scene_type == "image" for s in scenes[1:])


# ── infographic_data is always honoured ───────────────────────────────────────

class TestInfographicDataHonoured:
    def test_infographic_data_always_wins(self):
        engine = SceneVarietyEngineV2()
        cfg = StrategyConfig(
            mode="growth",
            scene_mix={"text_overlay": 1.0},
            scene_policy=ScenePolicy(
                mix={"text_overlay": 1.0},
                pattern_interrupt_frequency=0,
                avoid_repetition=False,
                priority_intents=[],
            ),
            avatar_frequency=0.0,
        )
        scenes = _scenes(3)
        scenes[2].infographic_data = {"type": "bar_chart"}
        engine.assign_from_config(scenes, cfg)
        assert scenes[2].scene_type == "infographic"


# ── update_scene_mix_from_feedback ────────────────────────────────────────────

class TestUpdateSceneMixFromFeedback:
    def test_returns_normalised_dict(self):
        de  = DecisionEngineV2(performance_store=PerformanceStore(data_dir=_tmp_dir()))
        sse = SceneStrategyEngine(data_dir=_tmp_dir())
        mix = de.update_scene_mix_from_feedback(sse)
        assert isinstance(mix, dict)
        assert len(mix) > 0
        assert abs(sum(mix.values()) - 1.0) < 1e-4

    def test_mix_keys_match_score_types(self):
        de  = DecisionEngineV2(performance_store=PerformanceStore(data_dir=_tmp_dir()))
        sse = SceneStrategyEngine(data_dir=_tmp_dir())
        mix = de.update_scene_mix_from_feedback(sse)
        expected_keys = set(sse.get_scores().keys())
        assert set(mix.keys()) == expected_keys

    def test_returns_empty_if_no_scores(self):
        de = DecisionEngineV2(performance_store=PerformanceStore(data_dir=_tmp_dir()))

        class _FakeEngine:
            def get_scores(self):
                return {}

        mix = de.update_scene_mix_from_feedback(_FakeEngine())
        assert mix == {}


# ── Full end-to-end contract pipeline ─────────────────────────────────────────

class TestFullContractPipeline:
    def test_decide_then_assign_from_config(self):
        de     = DecisionEngineV2(performance_store=PerformanceStore(data_dir=_tmp_dir()))
        cfg    = de.decide({"avg_watch_pct": 0.60, "recent_trend": "stable"})

        # safe mode expected
        assert cfg.mode == "safe"
        assert isinstance(cfg.scene_policy, ScenePolicy)
        assert cfg.scene_policy.mix

        engine = SceneVarietyEngineV2()
        scenes = _scenes(8)
        engine.assign_from_config(scenes, cfg)

        # Every scene must have a valid type and intent
        from src.scene_variety_engine import SCENE_TYPES, SCENE_INTENTS
        for s in scenes:
            assert s.scene_type in (*SCENE_TYPES, "avatar"), s.scene_type
            assert s.scene_intent in SCENE_INTENTS, s.scene_intent

    def test_growth_mode_pipeline(self):
        de  = DecisionEngineV2(performance_store=PerformanceStore(data_dir=_tmp_dir()))
        cfg = de.decide({"avg_watch_pct": 0.30, "recent_trend": "stable"})
        assert cfg.mode == "growth"

        engine = SceneVarietyEngineV2()
        scenes = _scenes(6)
        engine.assign_from_config(scenes, cfg)
        assert scenes[0].scene_intent == "hook"

    def test_experimental_mode_pipeline(self):
        de  = DecisionEngineV2(performance_store=PerformanceStore(data_dir=_tmp_dir()))
        cfg = de.decide({"avg_watch_pct": 0.50, "recent_trend": "down"})
        assert cfg.mode == "experimental"

        engine = SceneVarietyEngineV2()
        scenes = _scenes(5)
        engine.assign_from_config(scenes, cfg)
        # No crash; all scene types must be valid strings
        from src.scene_variety_engine import SCENE_TYPES
        for s in scenes:
            assert isinstance(s.scene_type, str)

    def test_policy_serialises_and_deserialises_for_next_run(self):
        """StrategyConfig can be stored and recreated for the next pipeline run."""
        de  = DecisionEngineV2(performance_store=PerformanceStore(data_dir=_tmp_dir()))
        cfg = de.decide({"avg_watch_pct": 0.60, "recent_trend": "stable"})

        d    = cfg.to_dict()
        cfg2 = StrategyConfig.from_dict(d)
        assert cfg2.scene_policy.pattern_interrupt_frequency == cfg.scene_policy.pattern_interrupt_frequency
        assert cfg2.scene_policy.priority_intents == cfg.scene_policy.priority_intents

    def test_record_and_decide_uses_stored_metrics(self, tmp_path):
        ps = PerformanceStore(data_dir=tmp_path)
        de = DecisionEngineV2(performance_store=ps)
        cfg = de.decide({"avg_watch_pct": 0.35, "recent_trend": "stable"})
        de.record("vid-001", {"avg_watch_pct": 0.35, "hook_retention": 0.45, "ctr": 0.03}, cfg)

        cfg2 = de.decide()   # uses stored metrics
        assert cfg2.mode in ("growth", "safe", "experimental")


# ── Helpers ────────────────────────────────────────────────────────────────────

def _tmp_dir(tmp_path=None):
    import tempfile, pathlib
    return pathlib.Path(tempfile.mkdtemp())
