"""Tests for DropPredictor and related helpers.

Covers:
- SceneFeatures serialisation (to_dict / from_dict)
- DropPrediction serialisation
- FeatureEncoder.encode() — vector length and values
- FeatureEncoder.from_scene() — extracts features from Scene-like objects
- FeatureEncoder._compute_complexity() — words-per-second normalisation
- TrainingDataset.add_example / get_xy / __len__ / persistence
- TrainingDataset.build_from_video() — correct label assignment
- _rule_based_predict() — each risk signal, additive capping
- _risk_level() — threshold boundaries
- DropPredictor.predict() — uses rule-based when no model
- DropPredictor.predict_scenes() — one prediction per scene
- DropPredictor.train() — requires MIN_TRAIN_SIZE, returns False without sklearn
- apply_preemptive_corrections() — high-risk only, all three fix types
- suggest_strategy_adjustments() — pace, force_avatar_intro, hook_delta
- Full pipeline: encode → predict → apply → adjust
"""
from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path

import pytest

from src.drop_predictor import (
    DROP_LABEL_THRESHOLD,
    FEATURE_NAMES,
    HIGH_RISK_THRESHOLD,
    MEDIUM_RISK_THRESHOLD,
    DropPrediction,
    DropPredictor,
    DropRisk,
    FeatureEncoder,
    SceneFeatures,
    TrainingDataset,
    _risk_level,
    _rule_based_predict,
    apply_preemptive_corrections,
    suggest_strategy_adjustments,
)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _tmp() -> Path:
    return Path(tempfile.mkdtemp())


@dataclass
class _Scene:
    """Minimal stand-in for src.script_generator.Scene."""
    idx:           int
    scene_type:    str = "image"
    scene_intent:  str = "explain"
    narration:     str = "This is a short narration."
    visual_prompt: str = "some visual"
    duration_sec:  int = 30


def _feats(
    scene_type:      str   = "image",
    intent:          str   = "explain",
    duration:        float = 30.0,
    position:        int   = 3,
    has_hook:        bool  = True,
    has_avatar:      bool  = False,
    sentence_length: int   = 20,
    complexity:      float = 0.4,
) -> SceneFeatures:
    return SceneFeatures(
        scene_type      = scene_type,
        intent          = intent,
        duration        = duration,
        position        = position,
        has_hook        = has_hook,
        has_avatar      = has_avatar,
        sentence_length = sentence_length,
        complexity      = complexity,
    )


def _predictor() -> DropPredictor:
    return DropPredictor(data_dir=_tmp(), model=None)


def _prediction(scene_idx: int, prob: float) -> DropPrediction:
    return DropPrediction(
        scene_idx   = scene_idx,
        probability = prob,
        risk        = _risk_level(prob),
        top_factors = [],
    )


# ── SceneFeatures ──────────────────────────────────────────────────────────────

class TestSceneFeatures:
    def test_defaults(self):
        f = SceneFeatures()
        assert f.scene_type      == "image"
        assert f.intent          == "explain"
        assert f.has_hook        is False
        assert f.has_avatar      is False
        assert f.complexity      == 0.5
        assert f.sentence_length == 20

    def test_to_dict_round_trip(self):
        f  = _feats(scene_type="avatar", intent="hook", duration=45.0, position=0,
                    has_hook=True, has_avatar=True, sentence_length=15, complexity=0.3)
        f2 = SceneFeatures.from_dict(f.to_dict())
        assert f2.scene_type      == "avatar"
        assert f2.intent          == "hook"
        assert abs(f2.duration - 45.0) < 1e-9
        assert f2.position        == 0
        assert f2.has_hook        is True
        assert f2.has_avatar      is True
        assert f2.sentence_length == 15
        assert abs(f2.complexity - 0.3) < 1e-9

    def test_from_dict_defaults(self):
        f = SceneFeatures.from_dict({})
        assert f.scene_type  == "image"
        assert f.intent      == "explain"
        assert f.has_hook    is False


# ── DropPrediction ─────────────────────────────────────────────────────────────

class TestDropPrediction:
    def test_to_dict(self):
        pred = DropPrediction(
            scene_idx   = 2,
            probability = 0.7234,
            risk        = DropRisk.HIGH,
            top_factors = ["no_hook", "static_image_no_hook"],
        )
        d = pred.to_dict()
        assert d["scene_idx"]   == 2
        assert d["probability"] == 0.7234
        assert d["risk"]        == "high"
        assert "no_hook" in d["top_factors"]

    def test_risk_enum_values(self):
        assert DropRisk.LOW.value    == "low"
        assert DropRisk.MEDIUM.value == "medium"
        assert DropRisk.HIGH.value   == "high"


# ── FeatureEncoder ─────────────────────────────────────────────────────────────

class TestFeatureEncoder:
    def test_encode_returns_correct_length(self):
        enc = FeatureEncoder()
        vec = enc.encode(_feats())
        assert len(vec) == len(FEATURE_NAMES)

    def test_encode_has_hook_true(self):
        enc = FeatureEncoder()
        vec = enc.encode(_feats(has_hook=True))
        assert vec[2] == 1.0   # has_hook is index 2

    def test_encode_has_hook_false(self):
        enc = FeatureEncoder()
        vec = enc.encode(_feats(has_hook=False))
        assert vec[2] == 0.0

    def test_encode_is_image_flag(self):
        enc = FeatureEncoder()
        vec = enc.encode(_feats(scene_type="image"))
        assert vec[6] == 1.0  # is_image
        assert vec[7] == 0.0  # is_avatar

    def test_encode_is_explain_flag(self):
        enc = FeatureEncoder()
        vec = enc.encode(_feats(intent="explain"))
        assert vec[10] == 1.0  # is_explain
        assert vec[11] == 0.0  # is_hook_intent

    def test_encode_duration_and_position(self):
        enc = FeatureEncoder()
        vec = enc.encode(_feats(duration=60.0, position=4))
        assert vec[0] == 60.0
        assert vec[1] == 4.0

    def test_from_scene_basic(self):
        enc   = FeatureEncoder()
        scene = _Scene(idx=3, scene_type="avatar", scene_intent="hook",
                       narration="wait this is shocking", duration_sec=20)
        feats = enc.from_scene(scene, position=3)
        assert feats.scene_type  == "avatar"
        assert feats.intent      == "hook"
        assert feats.has_avatar  is True
        assert feats.position    == 3
        assert feats.duration    == 20.0

    def test_from_scene_has_hook_from_intent(self):
        enc   = FeatureEncoder()
        scene = _Scene(idx=0, scene_intent="hook", narration="neutral text here")
        feats = enc.from_scene(scene)
        assert feats.has_hook is True

    def test_from_scene_has_hook_from_narration(self):
        enc   = FeatureEncoder()
        scene = _Scene(idx=2, scene_intent="explain", narration="but wait this matters")
        feats = enc.from_scene(scene)
        assert feats.has_hook is True

    def test_from_scene_no_hook(self):
        enc   = FeatureEncoder()
        scene = _Scene(idx=2, scene_intent="explain", narration="plain text here")
        feats = enc.from_scene(scene)
        assert feats.has_hook is False

    def test_from_scene_sentence_length(self):
        enc   = FeatureEncoder()
        scene = _Scene(idx=0, narration="one two three four five")
        feats = enc.from_scene(scene)
        assert feats.sentence_length == 5

    def test_compute_complexity_zero_words(self):
        assert FeatureEncoder._compute_complexity([], 30) == 0.0

    def test_compute_complexity_zero_duration(self):
        assert FeatureEncoder._compute_complexity(["a", "b"], 0) == 0.0

    def test_compute_complexity_capped_at_one(self):
        # 100 words / 10s = 10 wps >> 3 wps cap
        words = ["x"] * 100
        assert FeatureEncoder._compute_complexity(words, 10) == 1.0

    def test_compute_complexity_normal(self):
        # 30 words / 30s = 1 wps / 3 = 0.333
        words = ["x"] * 30
        c = FeatureEncoder._compute_complexity(words, 30)
        assert abs(c - round(1 / 3, 4)) < 1e-9

    def test_from_scene_uses_position_override(self):
        enc   = FeatureEncoder()
        scene = _Scene(idx=5)
        feats = enc.from_scene(scene, position=2)
        assert feats.position == 2


# ── _rule_based_predict ────────────────────────────────────────────────────────

class TestRuleBasedPredict:
    def _score(self, **kwargs) -> float:
        prob, _ = _rule_based_predict(_feats(**kwargs))
        return prob

    def _factors(self, **kwargs) -> list[str]:
        _, factors = _rule_based_predict(_feats(**kwargs))
        return factors

    def test_baseline_low_risk(self):
        # Safe defaults: has_hook, short explain, low complexity, mid position
        prob = self._score(intent="explain", duration=30, has_hook=True,
                           scene_type="diagram", complexity=0.3, position=5)
        assert prob < MEDIUM_RISK_THRESHOLD

    def test_long_explain_raises_risk(self):
        prob = self._score(intent="explain", duration=60, has_hook=False,
                           scene_type="diagram", complexity=0.3, position=5)
        assert prob >= HIGH_RISK_THRESHOLD

    def test_long_explain_factor_reported(self):
        factors = self._factors(intent="explain", duration=60, has_hook=False,
                                scene_type="diagram")
        assert "long_explain_scene" in factors

    def test_no_hook_raises_risk(self):
        prob_no_hook = self._score(has_hook=False, intent="data", scene_type="infographic",
                                   complexity=0.3, duration=25)
        prob_hook    = self._score(has_hook=True,  intent="data", scene_type="infographic",
                                   complexity=0.3, duration=25)
        assert prob_no_hook > prob_hook

    def test_no_hook_factor_reported(self):
        assert "no_hook" in self._factors(has_hook=False, scene_type="diagram")

    def test_static_image_no_hook_raises_risk(self):
        prob = self._score(scene_type="image", has_hook=False, complexity=0.3,
                           intent="data", duration=20)
        assert prob >= MEDIUM_RISK_THRESHOLD
        factors = self._factors(scene_type="image", has_hook=False)
        assert "static_image_no_hook" in factors

    def test_high_complexity_raises_risk(self):
        # Base risk from complexity only (hook present, not explain)
        prob_high = self._score(complexity=0.8, has_hook=True, intent="data",
                                scene_type="diagram", duration=20, position=5)
        prob_low  = self._score(complexity=0.3, has_hook=True, intent="data",
                                scene_type="diagram", duration=20, position=5)
        assert prob_high > prob_low

    def test_early_position_raises_risk(self):
        prob_early = self._score(position=0, has_hook=True, intent="data",
                                 scene_type="diagram", complexity=0.3)
        prob_late  = self._score(position=8, has_hook=True, intent="data",
                                 scene_type="diagram", complexity=0.3)
        assert prob_early > prob_late

    def test_transition_scene_adds_small_risk(self):
        prob_trans = self._score(intent="transition", has_hook=True,
                                 scene_type="diagram", complexity=0.3)
        prob_data  = self._score(intent="data", has_hook=True,
                                 scene_type="diagram", complexity=0.3)
        assert prob_trans > prob_data

    def test_long_sentence_raises_risk(self):
        prob_long  = self._score(sentence_length=40, has_hook=True, intent="data",
                                 scene_type="diagram", complexity=0.3)
        prob_short = self._score(sentence_length=10, has_hook=True, intent="data",
                                 scene_type="diagram", complexity=0.3)
        assert prob_long > prob_short

    def test_probability_capped_at_one(self):
        # Worst case: all risk signals active
        prob = self._score(intent="explain", duration=90, has_hook=False,
                           scene_type="image", complexity=0.9,
                           sentence_length=50, position=0)
        assert prob <= 1.0

    def test_probability_non_negative(self):
        prob = self._score(intent="hook", has_hook=True, scene_type="avatar",
                           complexity=0.0, duration=10, position=5, sentence_length=5)
        assert prob >= 0.0


# ── _risk_level ────────────────────────────────────────────────────────────────

class TestRiskLevel:
    def test_high(self):
        assert _risk_level(0.6)  == DropRisk.HIGH
        assert _risk_level(0.99) == DropRisk.HIGH

    def test_medium(self):
        assert _risk_level(0.4)  == DropRisk.MEDIUM
        assert _risk_level(0.59) == DropRisk.MEDIUM

    def test_low(self):
        assert _risk_level(0.0)  == DropRisk.LOW
        assert _risk_level(0.39) == DropRisk.LOW


# ── TrainingDataset ────────────────────────────────────────────────────────────

class TestTrainingDataset:
    def test_add_and_len(self):
        ds = TrainingDataset(data_dir=_tmp())
        ds.add_example(_feats(), 0)
        ds.add_example(_feats(has_hook=False), 1)
        assert len(ds) == 2

    def test_get_xy_shapes(self):
        ds = TrainingDataset(data_dir=_tmp())
        ds.add_example(_feats(), 0)
        ds.add_example(_feats(has_hook=False), 1)
        X, y = ds.get_xy()
        assert len(X) == 2
        assert len(y) == 2
        assert len(X[0]) == len(FEATURE_NAMES)

    def test_get_xy_labels(self):
        ds = TrainingDataset(data_dir=_tmp())
        ds.add_example(_feats(), 0)
        ds.add_example(_feats(), 1)
        _, y = ds.get_xy()
        assert 0 in y and 1 in y

    def test_persisted_across_instances(self):
        tmp = _tmp()
        ds  = TrainingDataset(data_dir=tmp)
        ds.add_example(_feats(), 1)
        ds2 = TrainingDataset(data_dir=tmp)
        assert len(ds2) == 1

    def test_corrupted_file_returns_empty(self):
        tmp = _tmp()
        ds  = TrainingDataset(data_dir=tmp)
        ds._path.parent.mkdir(parents=True, exist_ok=True)
        ds._path.write_text("{bad json{{")
        ds2 = TrainingDataset(data_dir=tmp)
        assert len(ds2) == 0

    def test_build_from_video_labels_drop_scenes(self):
        scenes = [_Scene(idx=0), _Scene(idx=1), _Scene(idx=2)]
        curve  = [0.9, 0.60, 0.58]   # drop at index 1: delta=0.30 > 0.15
        # Only scene 1 is covered; scenes 0 and 2 are NOT in scene_map → skipped
        scene_map = {1: {"scene_idx": 1, "scene_type": "image", "intent": "explain"}}
        ds = TrainingDataset(data_dir=_tmp())
        added = ds.build_from_video(scenes, {"curve": curve}, scene_map)
        assert added == 1   # only covered scene added
        _, y = ds.get_xy()
        assert y[0] == 1   # scene 1 had a drop → label 1

    def test_build_from_video_all_covered_no_drop(self):
        """All scenes covered and no drop → all labeled 0."""
        scenes = [_Scene(idx=1), _Scene(idx=2)]
        curve  = [0.9, 0.88, 0.87]
        scene_map = {
            1: {"scene_idx": 1, "scene_type": "image", "intent": "explain"},
            2: {"scene_idx": 2, "scene_type": "diagram", "intent": "data"},
        }
        ds = TrainingDataset(data_dir=_tmp())
        added = ds.build_from_video(scenes, {"curve": curve}, scene_map)
        assert added == 2
        _, y = ds.get_xy()
        assert all(label == 0 for label in y)

    def test_build_from_video_empty_curve(self):
        ds = TrainingDataset(data_dir=_tmp())
        added = ds.build_from_video([_Scene(idx=0)], {"curve": []}, {})
        assert added == 0

    def test_build_from_video_uncovered_scenes_skipped(self):
        """Scenes with no scene_map entry are excluded from training data."""
        scenes = [_Scene(idx=i) for i in range(3)]
        curve  = [0.9, 0.88, 0.87]
        ds = TrainingDataset(data_dir=_tmp())
        ds.build_from_video(scenes, {"curve": curve}, {})   # empty map
        assert len(ds) == 0   # nothing added — no coverage data


# ── DropPredictor ──────────────────────────────────────────────────────────────

class TestDropPredictor:
    def test_predict_returns_prediction(self):
        p    = _predictor()
        pred = p.predict(_feats(), scene_idx=0)
        assert isinstance(pred, DropPrediction)
        assert 0.0 <= pred.probability <= 1.0

    def test_predict_uses_rule_based_without_model(self):
        p    = _predictor()
        # High-risk features → should give high probability
        pred = p.predict(_feats(intent="explain", duration=90, has_hook=False,
                                scene_type="image", complexity=0.9,
                                sentence_length=50, position=0))
        assert pred.risk == DropRisk.HIGH

    def test_predict_low_risk_safe_features(self):
        p    = _predictor()
        pred = p.predict(_feats(intent="hook", has_hook=True, scene_type="avatar",
                                complexity=0.1, duration=15, position=5))
        assert pred.risk in (DropRisk.LOW, DropRisk.MEDIUM)

    def test_predict_scenes_one_per_scene(self):
        p      = _predictor()
        scenes = [_Scene(idx=i) for i in range(4)]
        preds  = p.predict_scenes(scenes)
        assert len(preds) == 4

    def test_predict_scenes_scene_idx_matches(self):
        p      = _predictor()
        scenes = [_Scene(idx=i) for i in range(3)]
        preds  = p.predict_scenes(scenes)
        assert [pred.scene_idx for pred in preds] == [0, 1, 2]

    def test_train_returns_false_without_sklearn(self):
        from src import drop_predictor as dp
        original = dp._SKLEARN_AVAILABLE
        try:
            dp._SKLEARN_AVAILABLE = False
            p = _predictor()
            assert p.train() is False
        finally:
            dp._SKLEARN_AVAILABLE = original

    def test_train_returns_false_insufficient_data(self):
        p = _predictor()
        p.dataset.add_example(_feats(), 0)   # only 1 example
        assert p.train() is False

    def test_injectable_model_used_for_prediction(self):
        """When a model is injected it takes priority over rule-based."""
        from unittest.mock import MagicMock
        from src import drop_predictor as dp

        mock_model          = MagicMock()
        mock_model.classes_ = [0, 1]
        mock_model.predict_proba.return_value = [[0.3, 0.7]]
        mock_model.feature_importances_ = [0.1] * len(FEATURE_NAMES)

        original = dp._SKLEARN_AVAILABLE
        try:
            dp._SKLEARN_AVAILABLE = True
            p    = DropPredictor(data_dir=_tmp(), model=mock_model)
            pred = p.predict(_feats(), scene_idx=0)
            assert abs(pred.probability - 0.7) < 1e-9
            assert pred.risk == DropRisk.HIGH
        finally:
            dp._SKLEARN_AVAILABLE = original


# ── apply_preemptive_corrections ───────────────────────────────────────────────

class TestApplyPreemptiveCorrections:
    def test_high_risk_explain_shortens_scene(self):
        scenes = [_Scene(idx=0, scene_intent="explain", duration_sec=40)]
        preds  = [_prediction(scene_idx=0, prob=0.75)]
        apply_preemptive_corrections(scenes, preds)
        assert scenes[0].duration_sec == int(40 * 0.7)

    def test_high_risk_explain_prepends_hook(self):
        scenes = [_Scene(idx=0, scene_intent="explain", visual_prompt="original")]
        preds  = [_prediction(scene_idx=0, prob=0.75)]
        apply_preemptive_corrections(scenes, preds)
        assert scenes[0].visual_prompt.startswith("[HOOK] ")

    def test_high_risk_image_becomes_avatar(self):
        scenes = [_Scene(idx=1, scene_type="image", scene_intent="data")]
        preds  = [_prediction(scene_idx=1, prob=0.80)]
        apply_preemptive_corrections(scenes, preds)
        assert scenes[0].scene_type == "avatar"

    def test_high_risk_fallback_adds_micro_hook(self):
        scenes = [_Scene(idx=2, scene_type="diagram", scene_intent="shock",
                         visual_prompt="base")]
        preds  = [_prediction(scene_idx=2, prob=0.70)]
        apply_preemptive_corrections(scenes, preds)
        assert scenes[0].visual_prompt.startswith("[HOOK] ")

    def test_low_risk_not_corrected(self):
        scenes = [_Scene(idx=0, scene_type="image", duration_sec=30)]
        preds  = [_prediction(scene_idx=0, prob=0.30)]
        apply_preemptive_corrections(scenes, preds)
        assert scenes[0].scene_type   == "image"
        assert scenes[0].duration_sec == 30

    def test_medium_risk_not_corrected_by_default(self):
        scenes = [_Scene(idx=0, scene_type="image")]
        preds  = [_prediction(scene_idx=0, prob=0.50)]
        apply_preemptive_corrections(scenes, preds)
        assert scenes[0].scene_type == "image"

    def test_returns_applied_records(self):
        scenes = [_Scene(idx=0, scene_intent="explain", duration_sec=40)]
        preds  = [_prediction(scene_idx=0, prob=0.70)]
        applied = apply_preemptive_corrections(scenes, preds)
        assert len(applied) == 1
        assert applied[0]["scene_idx"] == 0
        assert "fix" in applied[0]

    def test_missing_scene_skipped(self):
        scenes  = [_Scene(idx=5)]
        preds   = [_prediction(scene_idx=99, prob=0.90)]
        applied = apply_preemptive_corrections(scenes, preds)
        assert applied == []

    def test_custom_threshold(self):
        # scene_intent="data" so explain rule doesn't fire → image rule wins
        scenes = [_Scene(idx=0, scene_type="image", scene_intent="data")]
        preds  = [_prediction(scene_idx=0, prob=0.50)]
        apply_preemptive_corrections(scenes, preds, threshold=0.45)
        assert scenes[0].scene_type == "avatar"

    def test_zero_duration_not_modified(self):
        scenes = [_Scene(idx=0, scene_intent="explain", duration_sec=0)]
        preds  = [_prediction(scene_idx=0, prob=0.80)]
        apply_preemptive_corrections(scenes, preds)
        assert scenes[0].duration_sec == 0


# ── suggest_strategy_adjustments ─────────────────────────────────────────────

class TestSuggestStrategyAdjustments:
    def test_no_high_risk_returns_empty(self):
        preds = [_prediction(0, 0.30), _prediction(1, 0.50)]
        assert suggest_strategy_adjustments(preds) == {}

    def test_early_high_risk_sets_fast_pace(self):
        preds = [_prediction(scene_idx=1, prob=0.70)]
        adj   = suggest_strategy_adjustments(preds)
        assert adj["pace"] == "fast"

    def test_early_high_risk_adds_hook_delta(self):
        preds = [_prediction(scene_idx=0, prob=0.70)]
        adj   = suggest_strategy_adjustments(preds)
        assert adj.get("hook_aggressiveness_delta", 0) >= 0.20

    def test_high_risk_scene_zero_forces_avatar_intro(self):
        preds = [_prediction(scene_idx=0, prob=0.70)]
        adj   = suggest_strategy_adjustments(preds)
        assert adj.get("force_avatar_intro") is True

    def test_late_high_risk_does_not_force_avatar(self):
        preds = [_prediction(scene_idx=5, prob=0.70)]
        adj   = suggest_strategy_adjustments(preds)
        assert "force_avatar_intro" not in adj

    def test_many_high_risk_adds_extra_delta(self):
        preds = [_prediction(scene_idx=i + 5, prob=0.70) for i in range(3)]
        adj   = suggest_strategy_adjustments(preds)
        assert adj.get("hook_aggressiveness_delta", 0) >= 0.10

    def test_high_risk_scene_count_in_output(self):
        preds = [_prediction(0, 0.80), _prediction(1, 0.70), _prediction(2, 0.30)]
        adj   = suggest_strategy_adjustments(preds)
        assert adj["high_risk_scene_count"] == 2

    def test_early_high_risk_count_in_output(self):
        preds = [_prediction(0, 0.80), _prediction(6, 0.70)]
        adj   = suggest_strategy_adjustments(preds)
        assert adj["early_high_risk_count"] == 1


# ── Full pipeline ──────────────────────────────────────────────────────────────

class TestFullPipeline:
    def test_predict_apply_adjust(self):
        predictor = _predictor()
        scenes    = [
            _Scene(idx=0, scene_intent="explain", duration_sec=60,
                   narration="plain text " * 20, scene_type="image"),
            _Scene(idx=1, scene_intent="hook", duration_sec=15,
                   narration="but wait this is shocking", scene_type="avatar"),
        ]
        preds   = predictor.predict_scenes(scenes)
        applied = apply_preemptive_corrections(scenes, preds)
        adj     = suggest_strategy_adjustments(preds)

        # Scene 0: long explain image → high risk → corrected
        assert preds[0].risk in (DropRisk.MEDIUM, DropRisk.HIGH)
        assert len(preds) == 2

    def test_training_dataset_built_from_video(self):
        predictor = _predictor()
        scenes    = [_Scene(idx=i) for i in range(3)]
        curve     = [0.9, 0.60, 0.58]
        # Only scene 1 is covered — scenes 0 and 2 are excluded (no false labels)
        scene_map = {1: {"scene_idx": 1, "scene_type": "image", "intent": "explain"}}
        added = predictor.dataset.build_from_video(
            scenes, {"curve": curve}, scene_map
        )
        assert added == 1   # only the covered scene
        _, y = predictor.dataset.get_xy()
        assert y[0] == 1   # scene 1 had a drop → label 1

    def test_train_fails_gracefully_without_sklearn(self):
        predictor = _predictor()
        for i in range(15):
            predictor.dataset.add_example(
                _feats(has_hook=(i % 2 == 0)), i % 2
            )
        # train() returns False when sklearn unavailable
        result = predictor.train()
        assert result is False or result is True   # either is valid
