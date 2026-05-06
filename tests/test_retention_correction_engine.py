"""Tests for RetentionCorrectionEngine.

Covers:
- detect_drops()                           — threshold sensitivity, edge cases
- DropPoint / Correction serialisation
- CorrectionStore.record() / get_best_fix() — learning, MIN_SAMPLES guard
- RetentionCorrectionEngine.analyze()      — drop-to-scene mapping, deduplication
- RetentionCorrectionEngine._generate_fix() — rule-based + learning override
- RetentionCorrectionEngine.apply_corrections() — all FixType effects on Scene
- RetentionCorrectionEngine.suggest_strategy_adjustments() — global hints
- RetentionCorrectionEngine.record_outcome() — delegates to store
- Full pipeline integration
"""
from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path

import pytest

from src.retention_correction_engine import (
    CorrectionStore,
    Correction,
    DropPoint,
    FixType,
    RetentionCorrectionEngine,
    detect_drops,
)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _tmp() -> Path:
    return Path(tempfile.mkdtemp())


def _engine(drop_threshold: float = 0.15) -> RetentionCorrectionEngine:
    return RetentionCorrectionEngine(data_dir=_tmp(), drop_threshold=drop_threshold)


@dataclass
class _Scene:
    """Minimal stand-in for src.script_generator.Scene."""
    idx:          int
    heading:      str = ""
    narration:    str = ""
    visual_prompt: str = "some visual"
    duration_sec: int  = 30
    scene_type:   str  = "image"
    scene_intent: str  = "explain"


def _scene_map(*entries: tuple[int, int, str, str]) -> dict[int, dict]:
    """Build a scene_map from (timestamp_idx, scene_idx, scene_type, intent) tuples."""
    return {
        ts_idx: {"scene_idx": sc_idx, "scene_type": stype, "intent": intent}
        for ts_idx, sc_idx, stype, intent in entries
    }


def _retention_data(curve: list[float]) -> dict:
    step = 30
    return {
        "curve":      curve,
        "timestamps": [i * step for i in range(len(curve))],
    }


# ── detect_drops ───────────────────────────────────────────────────────────────

class TestDetectDrops:
    def test_single_drop(self):
        curve  = [0.9, 0.82, 0.75, 0.55, 0.52]
        drops  = detect_drops(curve, threshold=0.15)
        assert len(drops) == 1
        assert drops[0].timestamp_idx == 3
        assert abs(drops[0].delta - 0.20) < 1e-6

    def test_multiple_drops(self):
        curve = [0.9, 0.70, 0.65, 0.40, 0.38]
        drops = detect_drops(curve, threshold=0.15)
        assert len(drops) == 2
        idxs  = [d.timestamp_idx for d in drops]
        assert 1 in idxs and 3 in idxs

    def test_no_drops_below_threshold(self):
        curve = [0.9, 0.85, 0.80, 0.76, 0.72]
        drops = detect_drops(curve, threshold=0.15)
        assert drops == []

    def test_below_threshold_not_included(self):
        # delta = 0.14 < 0.15 → not a drop
        drops = detect_drops([0.90, 0.76], threshold=0.15)
        assert drops == []

    def test_just_above_threshold_included(self):
        curve = [0.9, 0.749]
        drops = detect_drops(curve, threshold=0.15)
        assert len(drops) == 1

    def test_empty_curve(self):
        assert detect_drops([], threshold=0.15) == []

    def test_single_element_curve(self):
        assert detect_drops([0.9], threshold=0.15) == []

    def test_two_element_no_drop(self):
        assert detect_drops([0.9, 0.8], threshold=0.15) == []

    def test_delta_rounded_to_four_places(self):
        curve = [0.9, 0.75]   # delta = 0.15001 with float rounding
        drops = detect_drops([0.9, 0.74999], threshold=0.15)
        assert round(drops[0].delta, 4) == drops[0].delta

    def test_custom_threshold(self):
        curve = [0.9, 0.85]   # delta = 0.05
        drops = detect_drops(curve, threshold=0.03)
        assert len(drops) == 1

    def test_drop_at_first_step(self):
        curve = [0.9, 0.70]
        drops = detect_drops(curve, threshold=0.15)
        assert drops[0].timestamp_idx == 1


# ── DropPoint / Correction serialisation ──────────────────────────────────────

class TestDataTypes:
    def test_drop_point_to_dict(self):
        dp = DropPoint(timestamp_idx=3, delta=0.20)
        d  = dp.to_dict()
        assert d["timestamp_idx"] == 3
        assert abs(d["delta"] - 0.20) < 1e-9

    def test_correction_to_dict(self):
        c = Correction(
            scene_idx  = 2,
            fix        = FixType.REPLACE_WITH_AVATAR,
            drop       = DropPoint(timestamp_idx=3, delta=0.20),
            scene_type = "image",
            intent     = "explain",
        )
        d = c.to_dict()
        assert d["scene_idx"] == 2
        assert d["fix"]       == "replace_with_avatar"
        assert d["scene_type"] == "image"
        assert "drop" in d

    def test_fix_type_string_values(self):
        assert FixType.ADD_MICRO_HOOK.value       == "add_micro_hook"
        assert FixType.REPLACE_WITH_AVATAR.value  == "replace_with_avatar"
        assert FixType.SHORTEN_AND_ADD_HOOK.value == "shorten_and_add_hook"
        assert FixType.CUT_OR_SPEED_UP.value      == "cut_or_speed_up"
        assert FixType.INCREASE_PACING.value      == "increase_pacing"
        assert FixType.REMOVE_SCENE.value         == "remove_scene"
        assert FixType.INSERT_AVATAR.value        == "insert_avatar"


# ── CorrectionStore ────────────────────────────────────────────────────────────

class TestCorrectionStore:
    def test_record_and_retrieve(self):
        store = CorrectionStore(data_dir=_tmp())
        store.record(FixType.REPLACE_WITH_AVATAR, "image", "explain", 0.12)
        store.record(FixType.REPLACE_WITH_AVATAR, "image", "explain", 0.08)
        store.record(FixType.REPLACE_WITH_AVATAR, "image", "explain", 0.10)
        assert len(store.all_records()) == 3

    def test_get_best_fix_returns_none_below_min_samples(self):
        store = CorrectionStore(data_dir=_tmp())
        store.record(FixType.REPLACE_WITH_AVATAR, "image", "explain", 0.10)
        store.record(FixType.REPLACE_WITH_AVATAR, "image", "explain", 0.10)
        # Only 2 records — below MIN_SAMPLES (3)
        assert store.get_best_fix("image", "explain") is None

    def test_get_best_fix_returns_fix_at_min_samples(self):
        store = CorrectionStore(data_dir=_tmp())
        for _ in range(3):
            store.record(FixType.REPLACE_WITH_AVATAR, "image", "explain", 0.10)
        result = store.get_best_fix("image", "explain")
        assert result == FixType.REPLACE_WITH_AVATAR

    def test_get_best_fix_picks_highest_mean(self):
        store = CorrectionStore(data_dir=_tmp())
        for _ in range(3):
            store.record(FixType.REPLACE_WITH_AVATAR, "image", "explain", 0.05)
        for _ in range(3):
            store.record(FixType.ADD_MICRO_HOOK, "image", "explain", 0.15)
        result = store.get_best_fix("image", "explain")
        assert result == FixType.ADD_MICRO_HOOK

    def test_get_best_fix_ignores_other_contexts(self):
        store = CorrectionStore(data_dir=_tmp())
        for _ in range(3):
            store.record(FixType.REPLACE_WITH_AVATAR, "image", "shock", 0.20)
        # Different intent — should not affect result for "explain"
        assert store.get_best_fix("image", "explain") is None

    def test_get_best_fix_returns_none_empty_store(self):
        store = CorrectionStore(data_dir=_tmp())
        assert store.get_best_fix("image", "explain") is None

    def test_persisted_across_instances(self):
        tmp   = _tmp()
        store = CorrectionStore(data_dir=tmp)
        for _ in range(3):
            store.record(FixType.CUT_OR_SPEED_UP, "text_overlay", "transition", 0.08)
        store2 = CorrectionStore(data_dir=tmp)
        assert store2.get_best_fix("text_overlay", "transition") == FixType.CUT_OR_SPEED_UP

    def test_corrupted_file_returns_empty(self):
        tmp   = _tmp()
        store = CorrectionStore(data_dir=tmp)
        store._path.parent.mkdir(parents=True, exist_ok=True)
        store._path.write_text("{corrupted{{")
        store2 = CorrectionStore(data_dir=tmp)
        assert store2.all_records() == []

    def test_record_accepts_fix_type_enum(self):
        store = CorrectionStore(data_dir=_tmp())
        store.record(FixType.ADD_MICRO_HOOK, "diagram", "data", 0.03)
        assert store.all_records()[0]["fix"] == "add_micro_hook"

    def test_record_accepts_string_fix(self):
        store = CorrectionStore(data_dir=_tmp())
        store.record("add_micro_hook", "diagram", "data", 0.03)
        assert store.all_records()[0]["fix"] == "add_micro_hook"


# ── analyze ────────────────────────────────────────────────────────────────────

class TestAnalyze:
    def test_returns_empty_when_no_drops(self):
        engine = _engine()
        data   = _retention_data([0.9, 0.88, 0.85, 0.82])
        assert engine.analyze(data, {}) == []

    def test_returns_correction_for_each_drop(self):
        engine = _engine()
        data   = _retention_data([0.9, 0.60, 0.58, 0.30, 0.29])
        smap   = _scene_map(
            (1, 0, "image", "hook"),
            (3, 2, "image", "explain"),
        )
        corrs  = engine.analyze(data, smap)
        assert len(corrs) == 2

    def test_skips_timestamp_not_in_scene_map(self):
        engine = _engine()
        data   = _retention_data([0.9, 0.60, 0.58])
        corrs  = engine.analyze(data, {})   # empty map
        assert corrs == []

    def test_deduplicates_corrections_by_scene_idx(self):
        """Two drops mapping to the same scene_idx yield one correction."""
        engine = _engine()
        data   = _retention_data([0.9, 0.60, 0.30, 0.28])
        # Timestamps 1 and 2 both map to scene 0
        smap   = _scene_map(
            (1, 0, "image", "explain"),
            (2, 0, "image", "explain"),
            (3, 1, "diagram", "data"),
        )
        corrs = engine.analyze(data, smap)
        scene_idxs = [c.scene_idx for c in corrs]
        assert scene_idxs.count(0) == 1

    def test_correction_carries_correct_scene_info(self):
        engine = _engine()
        data   = _retention_data([0.9, 0.60])
        smap   = _scene_map((1, 2, "text_overlay", "transition"))
        corrs  = engine.analyze(data, smap)
        assert corrs[0].scene_idx  == 2
        assert corrs[0].scene_type == "text_overlay"
        assert corrs[0].intent     == "transition"

    def test_correction_fix_matches_rule(self):
        engine = _engine()
        data   = _retention_data([0.9, 0.60])
        smap   = _scene_map((1, 0, "image", "hook"))
        corrs  = engine.analyze(data, smap)
        # scene_type == "image" → REPLACE_WITH_AVATAR
        assert corrs[0].fix == FixType.REPLACE_WITH_AVATAR

    def test_missing_curve_returns_empty(self):
        engine = _engine()
        assert engine.analyze({}, {}) == []


# ── _generate_fix (rule-based) ─────────────────────────────────────────────────

class TestGenerateFix:
    def _fix(self, scene_type, intent) -> FixType:
        engine = _engine()
        fix, _ = engine._generate_fix(scene_type, intent)
        return fix

    def test_image_gives_replace_with_avatar(self):
        assert self._fix("image", "hook") == FixType.REPLACE_WITH_AVATAR

    def test_explain_intent_gives_shorten_and_add_hook(self):
        # Non-image scene with explain intent
        assert self._fix("diagram", "explain") == FixType.SHORTEN_AND_ADD_HOOK

    def test_transition_gives_cut_or_speed_up(self):
        assert self._fix("text_overlay", "transition") == FixType.CUT_OR_SPEED_UP

    def test_hook_intent_gives_increase_pacing(self):
        assert self._fix("text_overlay", "hook") == FixType.INCREASE_PACING

    def test_fallback_gives_add_micro_hook(self):
        assert self._fix("cutaway", "shock") == FixType.ADD_MICRO_HOOK

    def test_image_rule_has_priority_over_intent(self):
        # image scene with explain intent → image rule wins
        assert self._fix("image", "explain") == FixType.REPLACE_WITH_AVATAR

    def test_learned_fix_overrides_rules(self):
        engine = _engine()
        # Inject 3 records for (image, hook) → ADD_MICRO_HOOK works best
        for _ in range(3):
            engine._store.record(FixType.ADD_MICRO_HOOK, "image", "hook", 0.20)
        fix, confidence = engine._generate_fix("image", "hook")
        assert fix        == FixType.ADD_MICRO_HOOK
        assert confidence == 1.5   # learning confidence

    def test_rule_confidence_is_1(self):
        engine = _engine()
        _, confidence = engine._generate_fix("cutaway", "reaction")
        assert confidence == 1.0


# ── apply_corrections ─────────────────────────────────────────────────────────

class TestApplyCorrections:
    def _corr(self, scene_idx, fix: FixType) -> Correction:
        return Correction(
            scene_idx  = scene_idx,
            fix        = fix,
            drop       = DropPoint(timestamp_idx=scene_idx + 1, delta=0.20),
            scene_type = "image",
            intent     = "explain",
        )

    def test_replace_with_avatar_sets_scene_type(self):
        engine = _engine()
        scenes = [_Scene(idx=0, scene_type="image")]
        engine.apply_corrections(scenes, [self._corr(0, FixType.REPLACE_WITH_AVATAR)])
        assert scenes[0].scene_type == "avatar"

    def test_insert_avatar_sets_scene_type(self):
        engine = _engine()
        scenes = [_Scene(idx=1, scene_type="diagram")]
        engine.apply_corrections(scenes, [self._corr(1, FixType.INSERT_AVATAR)])
        assert scenes[0].scene_type == "avatar"

    def test_shorten_and_add_hook_reduces_duration(self):
        engine = _engine()
        scenes = [_Scene(idx=0, duration_sec=30)]
        engine.apply_corrections(scenes, [self._corr(0, FixType.SHORTEN_AND_ADD_HOOK)])
        assert scenes[0].duration_sec == int(30 * 0.7)

    def test_shorten_and_add_hook_prepends_visual_prompt(self):
        engine = _engine()
        scenes = [_Scene(idx=0, duration_sec=30, visual_prompt="original")]
        engine.apply_corrections(scenes, [self._corr(0, FixType.SHORTEN_AND_ADD_HOOK)])
        assert scenes[0].visual_prompt.startswith("[HOOK] ")

    def test_cut_or_speed_up_reduces_duration(self):
        engine = _engine()
        scenes = [_Scene(idx=0, duration_sec=30)]
        engine.apply_corrections(scenes, [self._corr(0, FixType.CUT_OR_SPEED_UP)])
        assert scenes[0].duration_sec == int(30 * 0.6)

    def test_increase_pacing_reduces_duration(self):
        engine = _engine()
        scenes = [_Scene(idx=0, duration_sec=40)]
        engine.apply_corrections(scenes, [self._corr(0, FixType.INCREASE_PACING)])
        assert scenes[0].duration_sec == int(40 * 0.75)

    def test_add_micro_hook_prepends_visual_prompt(self):
        engine = _engine()
        scenes = [_Scene(idx=0, visual_prompt="base prompt")]
        engine.apply_corrections(scenes, [self._corr(0, FixType.ADD_MICRO_HOOK)])
        assert scenes[0].visual_prompt.startswith("[HOOK] ")

    def test_remove_scene_sets_duration_to_zero(self):
        engine = _engine()
        scenes = [_Scene(idx=0, duration_sec=30)]
        engine.apply_corrections(scenes, [self._corr(0, FixType.REMOVE_SCENE)])
        assert scenes[0].duration_sec == 0

    def test_zero_duration_not_modified_by_shorten(self):
        engine = _engine()
        scenes = [_Scene(idx=0, duration_sec=0)]
        engine.apply_corrections(scenes, [self._corr(0, FixType.SHORTEN_AND_ADD_HOOK)])
        assert scenes[0].duration_sec == 0   # 0 * 0.7 = 0 → int = 0

    def test_missing_scene_idx_skipped(self):
        engine = _engine()
        scenes = [_Scene(idx=5)]
        engine.apply_corrections(scenes, [self._corr(99, FixType.REPLACE_WITH_AVATAR)])
        assert scenes[0].scene_type == "image"   # unchanged

    def test_returns_same_list(self):
        engine  = _engine()
        scenes  = [_Scene(idx=0)]
        result  = engine.apply_corrections(scenes, [])
        assert result is scenes

    def test_multiple_corrections_applied(self):
        engine = _engine()
        scenes = [_Scene(idx=0, scene_type="image"), _Scene(idx=1, scene_type="diagram")]
        corrs  = [
            self._corr(0, FixType.REPLACE_WITH_AVATAR),
            self._corr(1, FixType.REMOVE_SCENE),
        ]
        engine.apply_corrections(scenes, corrs)
        assert scenes[0].scene_type   == "avatar"
        assert scenes[1].duration_sec == 0


# ── suggest_strategy_adjustments ─────────────────────────────────────────────

class TestSuggestStrategyAdjustments:
    def _drop(self, ts_idx, scene_idx=0) -> Correction:
        return Correction(
            scene_idx  = scene_idx,
            fix        = FixType.ADD_MICRO_HOOK,
            drop       = DropPoint(timestamp_idx=ts_idx, delta=0.20),
            scene_type = "image",
            intent     = "explain",
        )

    def test_empty_corrections_returns_empty(self):
        engine = _engine()
        assert engine.suggest_strategy_adjustments([], {}, []) == {}

    def test_early_drop_sets_fast_pace(self):
        engine = _engine()
        corrs  = [self._drop(ts_idx=1, scene_idx=0)]
        adj    = engine.suggest_strategy_adjustments(corrs, {}, [0.9, 0.60])
        assert adj["pace"] == "fast"

    def test_early_drop_adds_hook_aggressiveness_delta(self):
        engine = _engine()
        corrs  = [self._drop(ts_idx=2, scene_idx=1)]
        adj    = engine.suggest_strategy_adjustments(corrs, {}, [0.9, 0.88, 0.60])
        assert adj.get("hook_aggressiveness_delta", 0) >= 0.20

    def test_drop_at_scene_zero_sets_force_avatar_intro(self):
        engine = _engine()
        corrs  = [self._drop(ts_idx=1, scene_idx=0)]
        adj    = engine.suggest_strategy_adjustments(corrs, {}, [0.9, 0.60])
        assert adj.get("force_avatar_intro") is True

    def test_late_drop_does_not_force_avatar_intro(self):
        engine = _engine()
        corrs  = [self._drop(ts_idx=5, scene_idx=4)]
        adj    = engine.suggest_strategy_adjustments(corrs, {}, [0.9]*5 + [0.60])
        assert "force_avatar_intro" not in adj

    def test_many_drops_adds_extra_hook_delta(self):
        engine = _engine()
        corrs  = [self._drop(ts_idx=i + 5, scene_idx=i + 4) for i in range(3)]
        adj    = engine.suggest_strategy_adjustments(corrs, {}, [0.9]*8)
        assert adj.get("hook_aggressiveness_delta", 0) >= 0.10

    def test_drop_count_in_output(self):
        engine = _engine()
        corrs  = [self._drop(ts_idx=1, scene_idx=0), self._drop(ts_idx=4, scene_idx=3)]
        adj    = engine.suggest_strategy_adjustments(corrs, {}, [])
        assert adj["drop_count"] == 2

    def test_early_drop_count_in_output(self):
        engine = _engine()
        corrs  = [self._drop(ts_idx=1, scene_idx=0), self._drop(ts_idx=5, scene_idx=4)]
        adj    = engine.suggest_strategy_adjustments(corrs, {}, [])
        assert adj["early_drop_count"] == 1


# ── record_outcome ─────────────────────────────────────────────────────────────

class TestRecordOutcome:
    def test_delegates_to_store(self):
        engine = _engine()
        engine.record_outcome(FixType.REPLACE_WITH_AVATAR, "image", "explain", 0.10)
        records = engine._store.all_records()
        assert len(records) == 1
        assert records[0]["fix"] == "replace_with_avatar"


# ── Full pipeline ──────────────────────────────────────────────────────────────

class TestFullPipeline:
    def test_analyze_then_apply(self):
        engine = _engine()
        data   = _retention_data([0.9, 0.82, 0.75, 0.55, 0.52])
        smap   = _scene_map((3, 2, "image", "explain"))
        scenes = [_Scene(idx=i, duration_sec=30) for i in range(5)]

        corrs  = engine.analyze(data, smap)
        engine.apply_corrections(scenes, corrs)

        assert scenes[2].scene_type == "avatar"

    def test_analyze_suggest_record_cycle(self):
        engine = _engine()
        data   = _retention_data([0.9, 0.60, 0.58])
        smap   = _scene_map((1, 0, "image", "hook"))
        scenes = [_Scene(idx=0, duration_sec=30)]

        corrs = engine.analyze(data, smap)
        engine.apply_corrections(scenes, corrs)
        adj   = engine.suggest_strategy_adjustments(corrs, smap, data["curve"])

        assert adj.get("force_avatar_intro") is True
        assert adj.get("pace") == "fast"

        # Simulate feedback: fix improved retention by 0.12
        engine.record_outcome(FixType.REPLACE_WITH_AVATAR, "image", "hook", 0.12)
        assert len(engine._store.all_records()) == 1

    def test_learning_improves_fix_selection(self):
        engine = _engine()
        # Teach the engine that ADD_MICRO_HOOK works better for (image, hook)
        for _ in range(3):
            engine.record_outcome(FixType.ADD_MICRO_HOOK, "image", "hook", 0.20)

        data  = _retention_data([0.9, 0.60])
        smap  = _scene_map((1, 0, "image", "hook"))
        corrs = engine.analyze(data, smap)

        # Rule says REPLACE_WITH_AVATAR for image; learning overrides to ADD_MICRO_HOOK
        assert corrs[0].fix        == FixType.ADD_MICRO_HOOK
        assert corrs[0].confidence == 1.5

    def test_no_drops_produces_no_changes(self):
        engine = _engine()
        data   = _retention_data([0.9, 0.88, 0.87, 0.86])
        scenes = [_Scene(idx=i, duration_sec=30, scene_type="image") for i in range(4)]
        corrs  = engine.analyze(data, {})
        engine.apply_corrections(scenes, corrs)
        for s in scenes:
            assert s.scene_type   == "image"
            assert s.duration_sec == 30
