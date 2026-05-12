"""Tests for EmotionEngine — affective arc layer."""
from __future__ import annotations

import dataclasses

import pytest

from src.emotion_engine import (
    AVATAR_STYLE_MAP,
    EMOTION_MAP,
    EMOTIONS,
    SHORT_EMOTION_FLOW,
    VISUAL_MAP,
    VOICE_MAP,
    EmotionEngine,
    EmotionStats,
    _FLOW_DAMPING,
    _HIGH_TENSION_THRESHOLD,
    _SHORT_INTENSITY_MAP,
    _TENSION_SCORE,
    get_emotion_engine,
)
from src.script_generator import Scene, VideoScript


# ── helpers ───────────────────────────────────────────────────────────────────

def _scene(idx: int = 0, intent: str = "explain") -> Scene:
    return Scene(idx=idx, heading="h", narration="n", visual_prompt="v",
                 scene_intent=intent)


def _script(intents: list[str] = None) -> VideoScript:
    intents = intents or ["shock", "explain", "twist"]
    scenes = [_scene(i, intent) for i, intent in enumerate(intents)]
    return VideoScript(title="Test", description="d", tags=[], hook="h", scenes=scenes)


# ── EMOTIONS / MAPS ───────────────────────────────────────────────────────────

class TestConstants:
    def test_emotions_list_has_all_expected(self):
        for e in ("shock", "curiosity", "doubt", "surprise", "neutral"):
            assert e in EMOTIONS

    def test_emotion_map_covers_main_intents(self):
        for intent in ("shock", "conflict", "twist", "explain", "data", "reaction",
                       "curiosity", "warning", "mistake", "insider"):
            assert intent in EMOTION_MAP, intent

    def test_emotion_map_values_are_tuples(self):
        for intent, val in EMOTION_MAP.items():
            assert isinstance(val, tuple) and len(val) == 2
            emotion, intensity = val
            assert emotion in EMOTIONS, f"{intent} maps to unknown emotion '{emotion}'"
            assert 0.0 <= intensity <= 1.0, f"{intent} intensity {intensity} out of range"

    def test_voice_map_values_are_ssml_tokens_or_empty(self):
        allowed = {"[EMPHASIS]", "[PAUSE_SHORT]", "[PAUSE_LONG]", ""}
        for emo, tag in VOICE_MAP.items():
            assert tag in allowed, f"{emo} → unknown tag '{tag}'"

    def test_visual_map_values_are_known_directives(self):
        allowed = {"zoom_in", "cut_fast", "slow_zoom", "subtitle_focus"}
        for emo, vis in VISUAL_MAP.items():
            assert vis in allowed, f"{emo} → unknown visual '{vis}'"

    def test_avatar_style_map_values_are_known(self):
        allowed = {"normal", "energetic", "thinking", "serious"}
        for emo, style in AVATAR_STYLE_MAP.items():
            assert style in allowed, f"{emo} → unknown style '{style}'"

    def test_short_emotion_flow_has_four(self):
        assert len(SHORT_EMOTION_FLOW) == 4

    def test_short_emotion_flow_all_known(self):
        for e in SHORT_EMOTION_FLOW:
            assert e in EMOTIONS, e

    def test_short_intensity_map_covers_flow(self):
        for e in SHORT_EMOTION_FLOW:
            assert e in _SHORT_INTENSITY_MAP, e
            assert 0.0 <= _SHORT_INTENSITY_MAP[e] <= 1.0

    def test_tension_score_all_between_0_1(self):
        for intent, score in _TENSION_SCORE.items():
            assert 0.0 <= score <= 1.0, intent

    def test_thresholds_sane(self):
        assert 0.0 < _HIGH_TENSION_THRESHOLD < 1.0
        assert 0.0 < _FLOW_DAMPING < 1.0


# ── Scene dataclass new fields ────────────────────────────────────────────────

class TestSceneFields:
    def test_scene_has_emotion_field(self):
        s = _scene()
        assert hasattr(s, "emotion")

    def test_scene_emotion_default_neutral(self):
        s = _scene()
        assert s.emotion == "neutral"

    def test_scene_has_emotion_intensity_field(self):
        s = _scene()
        assert hasattr(s, "emotion_intensity")

    def test_scene_emotion_intensity_default_zero(self):
        s = _scene()
        assert s.emotion_intensity == 0.0

    def test_scene_has_avatar_style_field(self):
        s = _scene()
        assert hasattr(s, "avatar_style")

    def test_scene_avatar_style_default_normal(self):
        s = _scene()
        assert s.avatar_style == "normal"


# ── EmotionEngine._assign_emotion ────────────────────────────────────────────

class TestAssignEmotion:
    def test_shock_intent_gives_shock_emotion(self, tmp_path):
        engine = EmotionEngine(tmp_path)
        result = engine._assign_emotion(_scene(intent="shock"))
        assert result.emotion == "shock"

    def test_twist_intent_gives_surprise_emotion(self, tmp_path):
        engine = EmotionEngine(tmp_path)
        result = engine._assign_emotion(_scene(intent="twist"))
        assert result.emotion == "surprise"

    def test_explain_intent_gives_neutral(self, tmp_path):
        engine = EmotionEngine(tmp_path)
        result = engine._assign_emotion(_scene(intent="explain"))
        assert result.emotion == "neutral"

    def test_unknown_intent_gives_neutral(self, tmp_path):
        engine = EmotionEngine(tmp_path)
        result = engine._assign_emotion(_scene(intent="unknown_xyz"))
        assert result.emotion == "neutral"

    def test_intensity_set_from_map(self, tmp_path):
        engine = EmotionEngine(tmp_path)
        result = engine._assign_emotion(_scene(intent="shock"))
        _, expected = EMOTION_MAP["shock"]
        assert result.emotion_intensity == expected

    def test_avatar_style_set(self, tmp_path):
        engine = EmotionEngine(tmp_path)
        result = engine._assign_emotion(_scene(intent="shock"))
        assert result.avatar_style == "energetic"

    def test_does_not_mutate_input(self, tmp_path):
        engine = EmotionEngine(tmp_path)
        original = _scene(intent="shock")
        engine._assign_emotion(original)
        assert original.emotion == "neutral"


# ── EmotionEngine._adjust_flow ────────────────────────────────────────────────

class TestAdjustFlow:
    def test_consecutive_same_emotion_damped(self, tmp_path):
        engine = EmotionEngine(tmp_path)
        s1 = dataclasses.replace(_scene(0, "shock"), emotion="shock", emotion_intensity=0.9)
        s2 = dataclasses.replace(_scene(1, "shock"), emotion="shock", emotion_intensity=0.9)
        result = engine._adjust_flow([s1, s2])
        assert result[1].emotion_intensity < result[0].emotion_intensity

    def test_damping_factor_applied(self, tmp_path):
        engine = EmotionEngine(tmp_path)
        s1 = dataclasses.replace(_scene(0), emotion="shock", emotion_intensity=0.9)
        s2 = dataclasses.replace(_scene(1), emotion="shock", emotion_intensity=0.9)
        result = engine._adjust_flow([s1, s2])
        assert abs(result[1].emotion_intensity - 0.9 * _FLOW_DAMPING) < 0.01

    def test_different_emotions_not_damped(self, tmp_path):
        engine = EmotionEngine(tmp_path)
        s1 = dataclasses.replace(_scene(0), emotion="shock",    emotion_intensity=0.9)
        s2 = dataclasses.replace(_scene(1), emotion="curiosity",emotion_intensity=0.7)
        result = engine._adjust_flow([s1, s2])
        assert result[1].emotion_intensity == 0.7

    def test_single_scene_unchanged(self, tmp_path):
        engine = EmotionEngine(tmp_path)
        s = dataclasses.replace(_scene(0), emotion="shock", emotion_intensity=0.9)
        result = engine._adjust_flow([s])
        assert result[0].emotion_intensity == 0.9


# ── EmotionEngine._sync_with_tension ─────────────────────────────────────────

class TestSyncWithTension:
    def test_high_tension_scene_boosted_to_threshold(self, tmp_path):
        engine = EmotionEngine(tmp_path)
        s = dataclasses.replace(_scene(intent="shock"), emotion="shock", emotion_intensity=0.3)
        result = engine._sync_with_tension([s])
        assert result[0].emotion_intensity >= _HIGH_TENSION_THRESHOLD

    def test_low_tension_scene_not_boosted(self, tmp_path):
        engine = EmotionEngine(tmp_path)
        s = dataclasses.replace(_scene(intent="data"), emotion="calm", emotion_intensity=0.25)
        result = engine._sync_with_tension([s])
        assert result[0].emotion_intensity == 0.25

    def test_high_conflict_intensity_boosts_all_high_tension_scenes(self, tmp_path):
        engine = EmotionEngine(tmp_path)
        s = dataclasses.replace(_scene(intent="shock"), emotion="shock", emotion_intensity=0.5)
        result = engine._sync_with_tension([s], conflict_intensity=0.9)
        assert result[0].emotion_intensity >= _HIGH_TENSION_THRESHOLD

    def test_already_high_intensity_unchanged(self, tmp_path):
        engine = EmotionEngine(tmp_path)
        s = dataclasses.replace(_scene(intent="shock"), emotion="shock", emotion_intensity=0.95)
        result = engine._sync_with_tension([s])
        assert result[0].emotion_intensity == 0.95


# ── EmotionEngine._enforce_short_flow ────────────────────────────────────────

class TestEnforceShortFlow:
    def test_first_scene_gets_shock(self, tmp_path):
        engine = EmotionEngine(tmp_path)
        scenes = [_scene(i, "explain") for i in range(4)]
        result = engine._enforce_short_flow(scenes)
        assert result[0].emotion == "shock"

    def test_second_scene_gets_curiosity(self, tmp_path):
        engine = EmotionEngine(tmp_path)
        scenes = [_scene(i) for i in range(4)]
        result = engine._enforce_short_flow(scenes)
        assert result[1].emotion == "curiosity"

    def test_fourth_scene_gets_surprise(self, tmp_path):
        engine = EmotionEngine(tmp_path)
        scenes = [_scene(i) for i in range(4)]
        result = engine._enforce_short_flow(scenes)
        assert result[3].emotion == "surprise"

    def test_flow_wraps_after_four(self, tmp_path):
        engine = EmotionEngine(tmp_path)
        scenes = [_scene(i) for i in range(5)]
        result = engine._enforce_short_flow(scenes)
        assert result[4].emotion == SHORT_EMOTION_FLOW[0]

    def test_intensities_from_short_intensity_map(self, tmp_path):
        engine = EmotionEngine(tmp_path)
        scenes = [_scene(i) for i in range(4)]
        result = engine._enforce_short_flow(scenes)
        for r in result:
            assert r.emotion_intensity == _SHORT_INTENSITY_MAP[r.emotion]


# ── EmotionEngine._apply_visual ───────────────────────────────────────────────

class TestApplyVisual:
    def test_empty_visual_direction_filled(self, tmp_path):
        engine = EmotionEngine(tmp_path)
        s = dataclasses.replace(_scene(intent="shock"), emotion="shock", visual_direction="")
        result = engine._apply_visual([s])
        assert result[0].visual_direction == VISUAL_MAP["shock"]

    def test_existing_visual_direction_preserved(self, tmp_path):
        engine = EmotionEngine(tmp_path)
        s = dataclasses.replace(_scene(intent="shock"), emotion="shock", visual_direction="slow_zoom")
        result = engine._apply_visual([s])
        assert result[0].visual_direction == "slow_zoom"


# ── EmotionEngine.apply ───────────────────────────────────────────────────────

class TestApply:
    def test_returns_video_script(self, tmp_path):
        engine = EmotionEngine(tmp_path)
        result = engine.apply(_script())
        assert isinstance(result, VideoScript)

    def test_all_scenes_have_emotion(self, tmp_path):
        engine = EmotionEngine(tmp_path)
        result = engine.apply(_script(["shock", "explain", "twist", "data"]))
        for scene in result.scenes:
            assert scene.emotion in EMOTIONS

    def test_all_scenes_have_intensity(self, tmp_path):
        engine = EmotionEngine(tmp_path)
        result = engine.apply(_script())
        for scene in result.scenes:
            assert 0.0 <= scene.emotion_intensity <= 1.0

    def test_all_scenes_have_avatar_style(self, tmp_path):
        engine = EmotionEngine(tmp_path)
        result = engine.apply(_script())
        for scene in result.scenes:
            assert scene.avatar_style in ("normal", "energetic", "thinking", "serious")

    def test_script_title_preserved(self, tmp_path):
        engine = EmotionEngine(tmp_path)
        result = engine.apply(_script())
        assert result.title == "Test"

    def test_is_shorts_enforces_flow(self, tmp_path):
        engine = EmotionEngine(tmp_path)
        result = engine.apply(_script(["explain"] * 4), is_shorts=True)
        assert result.scenes[0].emotion == "shock"
        assert result.scenes[1].emotion == "curiosity"

    def test_original_scenes_not_mutated(self, tmp_path):
        engine = EmotionEngine(tmp_path)
        original = _script()
        original_emotions = [s.emotion for s in original.scenes]
        engine.apply(original)
        assert [s.emotion for s in original.scenes] == original_emotions


# ── EmotionEngine.short_emotion_flow ─────────────────────────────────────────

class TestShortEmotionFlow:
    def test_returns_list_of_tuples(self):
        flow = EmotionEngine.short_emotion_flow(4)
        assert len(flow) == 4
        for emo, intensity in flow:
            assert emo in EMOTIONS
            assert 0.0 <= intensity <= 1.0

    def test_first_is_shock(self):
        flow = EmotionEngine.short_emotion_flow(4)
        assert flow[0][0] == "shock"

    def test_wraps_correctly(self):
        flow = EmotionEngine.short_emotion_flow(5)
        assert flow[4][0] == SHORT_EMOTION_FLOW[0]

    def test_zero_segments_returns_empty(self):
        assert EmotionEngine.short_emotion_flow(0) == []


# ── EmotionEngine.emotion_curve ───────────────────────────────────────────────

class TestEmotionCurve:
    def test_returns_list_of_dicts(self, tmp_path):
        engine = EmotionEngine(tmp_path)
        script = engine.apply(_script(["shock", "explain", "twist"]))
        curve = engine.emotion_curve(script)
        assert len(curve) == 3
        for item in curve:
            assert "emotion" in item
            assert "intensity" in item

    def test_intensity_values_in_range(self, tmp_path):
        engine = EmotionEngine(tmp_path)
        script = engine.apply(_script())
        for item in engine.emotion_curve(script):
            assert 0.0 <= item["intensity"] <= 1.0


# ── EmotionEngine.log_stats / load_stats ─────────────────────────────────────

class TestLogStats:
    def test_log_creates_file(self, tmp_path):
        engine = EmotionEngine(tmp_path)
        script = engine.apply(_script())
        engine.log_stats(script, video_id="vid123")
        assert (tmp_path / "emotion_stats.jsonl").exists()

    def test_log_returns_emotion_stats(self, tmp_path):
        engine = EmotionEngine(tmp_path)
        script = engine.apply(_script())
        stats = engine.log_stats(script)
        assert isinstance(stats, EmotionStats)

    def test_load_stats_empty_on_new_store(self, tmp_path):
        engine = EmotionEngine(tmp_path)
        assert engine.load_stats() == []

    def test_load_stats_returns_records(self, tmp_path):
        engine = EmotionEngine(tmp_path)
        script = engine.apply(_script())
        engine.log_stats(script)
        records = engine.load_stats()
        assert len(records) == 1

    def test_dominant_emotion_set(self, tmp_path):
        engine = EmotionEngine(tmp_path)
        script = engine.apply(_script(["shock", "shock", "shock", "explain"]))
        stats = engine.log_stats(script)
        assert stats.dominant_emotion == "shock"

    def test_avg_intensity_positive(self, tmp_path):
        engine = EmotionEngine(tmp_path)
        script = engine.apply(_script())
        stats = engine.log_stats(script)
        assert stats.avg_intensity > 0.0

    def test_scene_count_matches(self, tmp_path):
        engine = EmotionEngine(tmp_path)
        script = engine.apply(_script(["shock", "explain", "twist"]))
        stats = engine.log_stats(script)
        assert stats.scene_count == 3

    def test_load_respects_n(self, tmp_path):
        engine = EmotionEngine(tmp_path)
        script = engine.apply(_script())
        for _ in range(5):
            engine.log_stats(script)
        assert len(engine.load_stats(n=3)) == 3


# ── Singleton ─────────────────────────────────────────────────────────────────

class TestSingleton:
    def test_returns_engine(self):
        assert isinstance(get_emotion_engine(), EmotionEngine)

    def test_same_instance(self):
        assert get_emotion_engine() is get_emotion_engine()
