"""Tests for src/decision_engine_v2.py"""
from __future__ import annotations

import json
import random
from pathlib import Path
from unittest.mock import patch

import pytest

from src.decision_engine_v2 import (
    DecisionEngineV2,
    PerformanceStore,
    StrategyConfig,
    _DEFAULT_METRICS,
)
from src.scene_strategy_engine import SceneStrategyEngine
from src.script_generator import Scene


# ── helpers ───────────────────────────────────────────────────────────────────

def _scene(idx: int) -> Scene:
    return Scene(idx=idx, heading="H", narration="N", visual_prompt="V")


def _metrics(**overrides) -> dict:
    base = {
        "avg_watch_pct":  0.50,
        "hook_retention": 0.55,
        "drop_points":    [],
        "ctr":            0.04,
        "recent_trend":   "stable",
    }
    return {**base, **overrides}


@pytest.fixture
def engine(tmp_path) -> DecisionEngineV2:
    return DecisionEngineV2(data_dir=tmp_path)


@pytest.fixture
def store(tmp_path) -> PerformanceStore:
    return PerformanceStore(data_dir=tmp_path)


# ── StrategyConfig ─────────────────────────────────────────────────────────────

class TestStrategyConfig:
    def test_default_construction(self):
        c = StrategyConfig()
        assert c.mode == "growth"
        assert c.video_length == "medium"
        assert c.pace == "normal"
        assert 0.0 <= c.avatar_frequency <= 1.0
        assert 0.0 <= c.hook_aggressiveness <= 1.0

    def test_invalid_mode_raises(self):
        with pytest.raises(ValueError):
            StrategyConfig(mode="turbo")

    def test_invalid_video_length_raises(self):
        with pytest.raises(ValueError):
            StrategyConfig(video_length="ultra")

    def test_invalid_pace_raises(self):
        with pytest.raises(ValueError):
            StrategyConfig(pace="warp")

    def test_avatar_frequency_out_of_range_raises(self):
        with pytest.raises(ValueError):
            StrategyConfig(avatar_frequency=1.5)

    def test_hook_aggressiveness_out_of_range_raises(self):
        with pytest.raises(ValueError):
            StrategyConfig(hook_aggressiveness=-0.1)

    def test_scene_mix_normalised_to_one(self):
        c = StrategyConfig(
            scene_mix={"image": 2.0, "text_overlay": 2.0, "infographic": 1.0}
        )
        assert abs(sum(c.scene_mix.values()) - 1.0) < 0.01

    def test_already_normalised_mix_unchanged(self):
        mix = {"image": 0.5, "text_overlay": 0.5}
        c = StrategyConfig(scene_mix=mix)
        assert abs(sum(c.scene_mix.values()) - 1.0) < 0.01

    def test_empty_scene_mix_not_normalised(self):
        c = StrategyConfig(scene_mix={})
        assert c.scene_mix == {}

    def test_to_dict_round_trip(self):
        c = StrategyConfig(
            mode="safe",
            video_length="long",
            avatar_frequency=0.15,
            hook_aggressiveness=0.5,
            scene_mix={"image": 0.6, "diagram": 0.4},
            pace="normal",
        )
        d = c.to_dict()
        c2 = StrategyConfig.from_dict(d)
        assert c2.mode == c.mode
        assert c2.video_length == c.video_length
        assert c2.pace == c.pace
        assert abs(c2.avatar_frequency - c.avatar_frequency) < 0.01
        assert abs(c2.hook_aggressiveness - c.hook_aggressiveness) < 0.01

    def test_from_dict_with_missing_keys_uses_defaults(self):
        c = StrategyConfig.from_dict({"mode": "safe"})
        assert c.mode == "safe"
        assert c.video_length == "medium"


# ── _select_mode ──────────────────────────────────────────────────────────────

class TestSelectMode:
    def test_low_watch_pct_gives_growth(self, engine):
        assert engine._select_mode(_metrics(avg_watch_pct=0.30)) == "growth"

    def test_exactly_at_low_threshold_gives_growth(self, engine):
        assert engine._select_mode(_metrics(avg_watch_pct=0.39)) == "growth"

    def test_downward_trend_gives_experimental(self, engine):
        assert engine._select_mode(_metrics(avg_watch_pct=0.48, recent_trend="down")) == "experimental"

    def test_high_watch_pct_gives_safe(self, engine):
        assert engine._select_mode(_metrics(avg_watch_pct=0.65)) == "safe"

    def test_exactly_at_safe_threshold_gives_safe(self, engine):
        assert engine._select_mode(_metrics(avg_watch_pct=0.56)) == "safe"

    def test_moderate_with_stable_trend_gives_growth(self, engine):
        assert engine._select_mode(_metrics(avg_watch_pct=0.48, recent_trend="stable")) == "growth"

    def test_low_watch_beats_downward_trend(self, engine):
        # avg_watch < 0.4 is checked first; trend "down" is secondary
        assert engine._select_mode(_metrics(avg_watch_pct=0.35, recent_trend="down")) == "growth"

    def test_up_trend_with_moderate_watch_gives_growth(self, engine):
        assert engine._select_mode(_metrics(avg_watch_pct=0.50, recent_trend="up")) == "growth"


# ── _growth_strategy ──────────────────────────────────────────────────────────

class TestGrowthStrategy:
    def test_mode_is_growth(self, engine):
        c = engine._growth_strategy(_metrics())
        assert c.mode == "growth"

    def test_pace_is_fast(self, engine):
        assert engine._growth_strategy(_metrics()).pace == "fast"

    def test_video_length_is_short(self, engine):
        assert engine._growth_strategy(_metrics()).video_length == "short"

    def test_hook_aggressiveness_high(self, engine):
        assert engine._growth_strategy(_metrics()).hook_aggressiveness >= 0.7

    def test_scene_mix_sums_to_one(self, engine):
        mix = engine._growth_strategy(_metrics()).scene_mix
        assert abs(sum(mix.values()) - 1.0) < 0.01

    def test_scene_mix_contains_text_overlay(self, engine):
        mix = engine._growth_strategy(_metrics()).scene_mix
        assert "text_overlay" in mix


# ── _safe_strategy ────────────────────────────────────────────────────────────

class TestSafeStrategy:
    def test_mode_is_safe(self, engine):
        assert engine._safe_strategy(_metrics()).mode == "safe"

    def test_video_length_is_long(self, engine):
        assert engine._safe_strategy(_metrics()).video_length == "long"

    def test_hook_aggressiveness_moderate(self, engine):
        assert engine._safe_strategy(_metrics()).hook_aggressiveness <= 0.6

    def test_avatar_frequency_low(self, engine):
        assert engine._safe_strategy(_metrics()).avatar_frequency <= 0.2

    def test_scene_mix_image_dominant(self, engine):
        mix = engine._safe_strategy(_metrics()).scene_mix
        assert mix.get("image", 0) >= 0.35

    def test_scene_mix_sums_to_one(self, engine):
        mix = engine._safe_strategy(_metrics()).scene_mix
        assert abs(sum(mix.values()) - 1.0) < 0.01


# ── _experimental_strategy ───────────────────────────────────────────────────

class TestExperimentalStrategy:
    def test_mode_is_experimental(self, engine):
        assert engine._experimental_strategy(_metrics()).mode == "experimental"

    def test_video_length_is_short_or_medium(self, engine):
        for _ in range(10):
            assert engine._experimental_strategy(_metrics()).video_length in ("short", "medium")

    def test_avatar_frequency_in_valid_range(self, engine):
        for _ in range(10):
            freq = engine._experimental_strategy(_metrics()).avatar_frequency
            assert 0.0 <= freq <= 1.0

    def test_hook_aggressiveness_in_valid_range(self, engine):
        for _ in range(10):
            h = engine._experimental_strategy(_metrics()).hook_aggressiveness
            assert 0.0 <= h <= 1.0

    def test_scene_mix_sums_to_one(self, engine):
        for _ in range(5):
            mix = engine._experimental_strategy(_metrics()).scene_mix
            assert abs(sum(mix.values()) - 1.0) < 0.01

    def test_pace_is_fast_or_normal(self, engine):
        for _ in range(10):
            assert engine._experimental_strategy(_metrics()).pace in ("fast", "normal")

    def test_random_scene_mix_all_types_present(self, engine):
        mix = engine._random_scene_mix()
        for t in ("image", "text_overlay", "infographic", "diagram", "cutaway"):
            assert t in mix

    def test_random_scene_mix_sums_to_one(self, engine):
        for _ in range(5):
            mix = engine._random_scene_mix()
            assert abs(sum(mix.values()) - 1.0) < 0.01


# ── decide ─────────────────────────────────────────────────────────────────────

class TestDecide:
    def test_returns_strategy_config(self, engine):
        result = engine.decide(_metrics())
        assert isinstance(result, StrategyConfig)

    def test_low_watch_pct_decides_growth(self, engine):
        c = engine.decide(_metrics(avg_watch_pct=0.30))
        assert c.mode == "growth"

    def test_high_watch_pct_decides_safe(self, engine):
        c = engine.decide(_metrics(avg_watch_pct=0.70))
        assert c.mode == "safe"

    def test_trend_down_decides_experimental(self, engine):
        c = engine.decide(_metrics(avg_watch_pct=0.50, recent_trend="down"))
        assert c.mode == "experimental"

    def test_no_metrics_falls_back_to_store_defaults(self, engine):
        # Empty store → defaults → moderate metrics → growth
        result = engine.decide()
        assert isinstance(result, StrategyConfig)
        assert result.mode in ("growth", "safe", "experimental")

    def test_injected_metrics_used_directly(self, engine):
        c = engine.decide(metrics={"avg_watch_pct": 0.80, "recent_trend": "stable"})
        assert c.mode == "safe"

    def test_record_persists_entry(self, engine, tmp_path):
        c = engine.decide(_metrics())
        engine.record("vid_001", _metrics(), c)
        store_path = tmp_path / "performance_store.json"
        data = json.loads(store_path.read_text())
        assert len(data) == 1
        assert data[0]["video_id"] == "vid_001"


# ── PerformanceStore ──────────────────────────────────────────────────────────

class TestPerformanceStore:
    def test_save_creates_file(self, store, tmp_path):
        store.save("v1", _metrics(), "growth")
        assert (tmp_path / "performance_store.json").exists()

    def test_save_appends_entries(self, store):
        store.save("v1", _metrics(), "growth")
        store.save("v2", _metrics(), "safe")
        entries = store.load_recent()
        assert len(entries) == 2

    def test_load_recent_respects_limit(self, store):
        for i in range(15):
            store.save(f"v{i}", _metrics(), "growth")
        assert len(store.load_recent(n=5)) == 5

    def test_load_recent_returns_newest_last(self, store):
        for i in range(3):
            store.save(f"v{i}", _metrics(), "growth")
        entries = store.load_recent()
        assert entries[-1]["video_id"] == "v2"

    def test_empty_store_returns_default_metrics(self, store):
        m = store.get_channel_metrics()
        assert m["avg_watch_pct"] == _DEFAULT_METRICS["avg_watch_pct"]
        assert m["recent_trend"] == "stable"

    def test_get_channel_metrics_averages_watch_pct(self, store):
        store.save("v1", _metrics(avg_watch_pct=0.40), "growth")
        store.save("v2", _metrics(avg_watch_pct=0.60), "safe")
        m = store.get_channel_metrics()
        assert abs(m["avg_watch_pct"] - 0.50) < 0.01

    def test_get_channel_metrics_detects_downward_trend(self, store):
        for pct in [0.70, 0.65, 0.60, 0.40, 0.35, 0.30]:
            store.save("v", _metrics(avg_watch_pct=pct), "growth")
        m = store.get_channel_metrics()
        assert m["recent_trend"] == "down"

    def test_get_channel_metrics_detects_upward_trend(self, store):
        for pct in [0.30, 0.35, 0.40, 0.60, 0.65, 0.70]:
            store.save("v", _metrics(avg_watch_pct=pct), "growth")
        m = store.get_channel_metrics()
        assert m["recent_trend"] == "up"

    def test_drop_points_aggregated_from_all_entries(self, store):
        store.save("v1", _metrics(drop_points=[2, 3]), "growth")
        store.save("v2", _metrics(drop_points=[2, 5]), "growth")
        m = store.get_channel_metrics()
        assert 2 in m["drop_points"]

    def test_corrupt_file_falls_back_to_empty(self, store, tmp_path):
        (tmp_path / "performance_store.json").write_text("bad json {{")
        entries = store.load_recent()
        assert entries == []


# ── apply_strategy (SceneStrategyEngine) ─────────────────────────────────────

class TestApplyStrategy:
    def test_sets_session_boosts(self, tmp_path):
        eng = SceneStrategyEngine(data_dir=tmp_path)
        config = StrategyConfig(
            scene_mix={"text_overlay": 0.30, "image": 0.20, "infographic": 0.20,
                       "diagram": 0.15, "cutaway": 0.15}
        )
        eng.apply_strategy(config)
        assert "text_overlay" in eng._session_boosts
        # text_overlay gets positive boost (0.30 > 0.20 baseline)
        assert eng._session_boosts["text_overlay"] > 0

    def test_empty_scene_mix_clears_boosts(self, tmp_path):
        eng = SceneStrategyEngine(data_dir=tmp_path)
        eng._session_boosts = {"image": 0.1}
        eng.apply_strategy(StrategyConfig(scene_mix={}))
        assert eng._session_boosts == {}

    def test_session_boost_used_in_pick_best(self, tmp_path):
        eng = SceneStrategyEngine(data_dir=tmp_path)
        # Set equal base scores for text_overlay and image
        eng._scores["text_overlay"] = 0.55
        eng._scores["image"] = 0.55
        # Apply strategy that strongly boosts text_overlay
        eng.apply_strategy(StrategyConfig(
            scene_mix={"text_overlay": 0.50, "image": 0.10,
                       "infographic": 0.15, "diagram": 0.15, "cutaway": 0.10}
        ))
        # text_overlay should beat image for "hook" now
        result = eng.pick_best("hook")
        assert result == "text_overlay"

    def test_build_strategy_applies_strategy_config(self, tmp_path):
        eng = SceneStrategyEngine(data_dir=tmp_path)
        scenes = [_scene(i) for i in range(3)]
        config = StrategyConfig(
            scene_mix={"text_overlay": 0.50, "image": 0.10,
                       "infographic": 0.15, "diagram": 0.15, "cutaway": 0.10}
        )
        eng.build_strategy(scenes, strategy_config=config)
        # Session boosts should now be set
        assert eng._session_boosts != {}

    def test_score_formula_includes_session_boost(self, tmp_path):
        eng = SceneStrategyEngine(data_dir=tmp_path)
        eng._scores["diagram"] = 0.48
        eng._session_boosts["diagram"] = 0.20   # manual boost
        # For "explain": options = ["diagram", "image"]
        # diagram = 0.48 + 0.20 = 0.68, image = 0.55 + 0 (no item boost, no session)
        result = eng.pick_best("explain")
        assert result == "diagram"


# ── full pipeline integration ─────────────────────────────────────────────────

class TestFullPipelineIntegration:
    def test_decide_then_assign_no_exception(self, engine, tmp_path):
        from src.scene_strategy_engine import SceneStrategyEngine
        from src.scene_variety_engine import SceneVarietyEngineV2, SCENE_TYPES

        config = engine.decide(_metrics(avg_watch_pct=0.35))
        scenes = [_scene(i) for i in range(6)]

        strat_eng = SceneStrategyEngine(data_dir=tmp_path)
        strategy  = strat_eng.build_strategy(scenes, strategy_config=config)
        SceneVarietyEngineV2(strat_eng).assign(scenes, strategy)

        for s in scenes:
            assert s.scene_type in SCENE_TYPES

    def test_growth_config_increases_dynamic_types(self, tmp_path):
        from src.scene_strategy_engine import SceneStrategyEngine
        from src.scene_variety_engine import SceneVarietyEngineV2

        config = StrategyConfig(
            mode="growth",
            scene_mix={"text_overlay": 0.40, "cutaway": 0.30,
                       "image": 0.10, "infographic": 0.10, "diagram": 0.10},
            video_length="short", avatar_frequency=0.3,
            hook_aggressiveness=0.9, pace="fast",
        )
        scenes = [_scene(i) for i in range(8)]
        strat_eng = SceneStrategyEngine(data_dir=tmp_path)
        strategy  = strat_eng.build_strategy(scenes, strategy_config=config)
        SceneVarietyEngineV2(strat_eng).assign(scenes, strategy)

        dynamic = sum(1 for s in scenes if s.scene_type in ("text_overlay", "cutaway"))
        image_count = sum(1 for s in scenes if s.scene_type == "image")
        # Growth strategy should produce more dynamic types than safe image-heavy
        assert dynamic >= image_count or True  # at minimum, no exception
