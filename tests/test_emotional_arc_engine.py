"""Tests for EmotionalArcEngine — global emotional journey layer."""
from __future__ import annotations

import dataclasses
import json

import pytest

from src.emotional_arc_engine import (
    EMOTIONAL_ARCS,
    SHORT_ARC,
    ArcStats,
    EmotionalArcEngine,
    _EMOTION_ALIAS,
    _EMOTION_INTENT_SYNC,
    _INTENT_SYNC_THRESHOLD,
    _resolve,
    get_arc_engine,
)
from src.emotion_engine import AVATAR_STYLE_MAP, EMOTIONS
from src.script_generator import Scene, VideoScript


# ── helpers ───────────────────────────────────────────────────────────────────

def _scene(idx: int = 0, intent: str = "explain") -> Scene:
    return Scene(idx=idx, heading="h", narration="n", visual_prompt="v",
                 scene_intent=intent)


def _script(n: int = 4, intent: str = "explain") -> VideoScript:
    scenes = [_scene(i, intent) for i in range(n)]
    return VideoScript(title="Test", description="d", tags=[], hook="h", scenes=scenes)


def _arc_engine(tmp_path) -> EmotionalArcEngine:
    return EmotionalArcEngine(data_dir=tmp_path)


# ── _resolve / alias ──────────────────────────────────────────────────────────

class TestResolve:
    def test_known_alias_maps_correctly(self):
        assert _resolve("tension") == "shock"
        assert _resolve("relief") == "calm"
        assert _resolve("realism") == "neutral"
        assert _resolve("unresolved") == "doubt"

    def test_core_emotion_passes_through(self):
        for emo in EMOTIONS:
            assert _resolve(emo) == emo

    def test_unknown_passes_through(self):
        assert _resolve("xyz_unknown") == "xyz_unknown"

    def test_all_alias_values_are_core_emotions(self):
        for alias, target in _EMOTION_ALIAS.items():
            assert target in EMOTIONS, f"alias {alias!r} → {target!r} not in EMOTIONS"


# ── EMOTIONAL_ARCS constants ──────────────────────────────────────────────────

class TestArcConstants:
    def test_all_four_arcs_present(self):
        for arc in ("suspicion_reveal", "hype_breakdown", "confusion_clarity", "calm_to_fear"):
            assert arc in EMOTIONAL_ARCS

    def test_each_arc_has_four_steps(self):
        for arc_name, steps in EMOTIONAL_ARCS.items():
            assert len(steps) == 4, f"{arc_name} has {len(steps)} steps"

    def test_arc_intensities_in_range(self):
        for arc_name, steps in EMOTIONAL_ARCS.items():
            for emo, intensity in steps:
                assert 0.0 <= intensity <= 1.0, f"{arc_name} step {emo!r} intensity {intensity}"

    def test_arc_emotions_resolve_to_core(self):
        for arc_name, steps in EMOTIONAL_ARCS.items():
            for emo, _ in steps:
                resolved = _resolve(emo)
                assert resolved in EMOTIONS, (
                    f"{arc_name} step {emo!r} resolves to {resolved!r} — not in EMOTIONS"
                )

    def test_short_arc_has_four_steps(self):
        assert len(SHORT_ARC) == 4

    def test_short_arc_intensities_in_range(self):
        for emo, intensity in SHORT_ARC:
            assert 0.0 <= intensity <= 1.0

    def test_intent_sync_values_are_valid_intents(self):
        valid_intents = {
            "explain", "shock", "conflict", "twist", "reaction",
            "curiosity", "warning", "mistake", "insider", "data", "hook",
        }
        for emo, intent in _EMOTION_INTENT_SYNC.items():
            assert intent in valid_intents, f"{emo} → {intent!r} unknown intent"

    def test_intent_sync_threshold_sane(self):
        assert 0.0 < _INTENT_SYNC_THRESHOLD < 1.0


# ── _get_arc ──────────────────────────────────────────────────────────────────

class TestGetArc:
    def test_known_arc_returned(self, tmp_path):
        eng = _arc_engine(tmp_path)
        arc = eng._get_arc("hype_breakdown")
        assert arc == EMOTIONAL_ARCS["hype_breakdown"]

    def test_unknown_arc_falls_back_to_suspicion_reveal(self, tmp_path):
        eng = _arc_engine(tmp_path)
        arc = eng._get_arc("nonexistent_arc")
        assert arc == EMOTIONAL_ARCS["suspicion_reveal"]


# ── _map_arc_to_scenes ────────────────────────────────────────────────────────

class TestMapArcToScenes:
    def test_empty_scenes_returns_empty(self, tmp_path):
        eng = _arc_engine(tmp_path)
        result = eng._map_arc_to_scenes([], EMOTIONAL_ARCS["suspicion_reveal"])
        assert result == []

    def test_first_scene_gets_first_arc_emotion(self, tmp_path):
        eng = _arc_engine(tmp_path)
        scenes = [_scene(i) for i in range(4)]
        arc = EMOTIONAL_ARCS["suspicion_reveal"]
        result = eng._map_arc_to_scenes(scenes, arc)
        expected_first = _resolve(arc[0][0])
        assert result[0].emotion == expected_first

    def test_last_scene_gets_last_arc_emotion(self, tmp_path):
        eng = _arc_engine(tmp_path)
        scenes = [_scene(i) for i in range(4)]
        arc = EMOTIONAL_ARCS["suspicion_reveal"]
        result = eng._map_arc_to_scenes(scenes, arc)
        expected_last = _resolve(arc[-1][0])
        assert result[-1].emotion == expected_last

    def test_intensity_set_correctly(self, tmp_path):
        eng = _arc_engine(tmp_path)
        scenes = [_scene(i) for i in range(4)]
        arc = EMOTIONAL_ARCS["suspicion_reveal"]
        result = eng._map_arc_to_scenes(scenes, arc)
        # Last scene gets last step's intensity
        assert result[-1].emotion_intensity == arc[-1][1]

    def test_avatar_style_set_from_map(self, tmp_path):
        eng = _arc_engine(tmp_path)
        scenes = [_scene(0)]
        arc = [("shock", 0.9)]
        result = eng._map_arc_to_scenes(scenes, arc)
        assert result[0].avatar_style == AVATAR_STYLE_MAP.get("shock", "normal")

    def test_single_scene_gets_first_step(self, tmp_path):
        eng = _arc_engine(tmp_path)
        arc = EMOTIONAL_ARCS["hype_breakdown"]
        result = eng._map_arc_to_scenes([_scene(0)], arc)
        assert result[0].emotion == _resolve(arc[0][0])

    def test_eight_scenes_distributed_across_four_steps(self, tmp_path):
        eng = _arc_engine(tmp_path)
        scenes = [_scene(i) for i in range(8)]
        arc = EMOTIONAL_ARCS["suspicion_reveal"]
        result = eng._map_arc_to_scenes(scenes, arc)
        # 8 scenes, 4 steps — each step covers 2 scenes
        # scenes 0,1 → step 0; scenes 6,7 → step 3
        assert result[0].emotion == _resolve(arc[0][0])
        assert result[7].emotion == _resolve(arc[3][0])

    def test_does_not_mutate_input(self, tmp_path):
        eng = _arc_engine(tmp_path)
        scenes = [_scene(i) for i in range(4)]
        original_emotions = [s.emotion for s in scenes]
        eng._map_arc_to_scenes(scenes, EMOTIONAL_ARCS["suspicion_reveal"])
        assert [s.emotion for s in scenes] == original_emotions

    def test_all_result_emotions_are_core(self, tmp_path):
        eng = _arc_engine(tmp_path)
        for arc_name, arc in EMOTIONAL_ARCS.items():
            scenes = [_scene(i) for i in range(6)]
            result = eng._map_arc_to_scenes(scenes, arc)
            for r in result:
                assert r.emotion in EMOTIONS, f"{arc_name}: {r.emotion!r} not in EMOTIONS"

    def test_intensity_rounded_to_three_decimals(self, tmp_path):
        eng = _arc_engine(tmp_path)
        scenes = [_scene(0)]
        arc = [("curiosity", 0.123456789)]
        result = eng._map_arc_to_scenes(scenes, arc)
        assert result[0].emotion_intensity == round(0.123456789, 3)


# ── _sync_intents ─────────────────────────────────────────────────────────────

class TestSyncIntents:
    def test_high_intensity_sync_mapped_emotion_updates_intent(self, tmp_path):
        eng = _arc_engine(tmp_path)
        s = dataclasses.replace(_scene(0, "explain"),
                                emotion="surprise", emotion_intensity=0.90)
        result = eng._sync_intents([s])
        assert result[0].scene_intent == "twist"

    def test_low_intensity_does_not_update_intent(self, tmp_path):
        eng = _arc_engine(tmp_path)
        s = dataclasses.replace(_scene(0, "explain"),
                                emotion="surprise", emotion_intensity=0.50)
        result = eng._sync_intents([s])
        assert result[0].scene_intent == "explain"

    def test_at_threshold_updates_intent(self, tmp_path):
        eng = _arc_engine(tmp_path)
        s = dataclasses.replace(_scene(0, "explain"),
                                emotion="surprise",
                                emotion_intensity=_INTENT_SYNC_THRESHOLD)
        result = eng._sync_intents([s])
        assert result[0].scene_intent == "twist"

    def test_emotion_not_in_sync_map_leaves_intent(self, tmp_path):
        eng = _arc_engine(tmp_path)
        s = dataclasses.replace(_scene(0, "explain"),
                                emotion="calm", emotion_intensity=0.90)
        result = eng._sync_intents([s])
        assert result[0].scene_intent == "explain"

    def test_does_not_mutate_input(self, tmp_path):
        eng = _arc_engine(tmp_path)
        s = dataclasses.replace(_scene(0, "explain"),
                                emotion="surprise", emotion_intensity=0.90)
        eng._sync_intents([s])
        assert s.scene_intent == "explain"


# ── apply ─────────────────────────────────────────────────────────────────────

class TestApply:
    def test_returns_video_script(self, tmp_path):
        eng = _arc_engine(tmp_path)
        result = eng.apply(_script(4), "suspicion_reveal")
        assert isinstance(result, VideoScript)

    def test_all_scenes_have_core_emotion(self, tmp_path):
        eng = _arc_engine(tmp_path)
        result = eng.apply(_script(4), "hype_breakdown")
        for s in result.scenes:
            assert s.emotion in EMOTIONS

    def test_all_intensities_in_range(self, tmp_path):
        eng = _arc_engine(tmp_path)
        result = eng.apply(_script(4), "confusion_clarity")
        for s in result.scenes:
            assert 0.0 <= s.emotion_intensity <= 1.0

    def test_title_preserved(self, tmp_path):
        eng = _arc_engine(tmp_path)
        result = eng.apply(_script(4), "calm_to_fear")
        assert result.title == "Test"

    def test_scene_count_preserved(self, tmp_path):
        eng = _arc_engine(tmp_path)
        result = eng.apply(_script(8), "suspicion_reveal")
        assert len(result.scenes) == 8

    def test_original_script_not_mutated(self, tmp_path):
        eng = _arc_engine(tmp_path)
        original = _script(4)
        original_emotions = [s.emotion for s in original.scenes]
        eng.apply(original, "suspicion_reveal")
        assert [s.emotion for s in original.scenes] == original_emotions

    def test_unknown_arc_type_falls_back(self, tmp_path):
        eng = _arc_engine(tmp_path)
        result = eng.apply(_script(4), "nonexistent_arc")
        assert isinstance(result, VideoScript)
        for s in result.scenes:
            assert s.emotion in EMOTIONS

    def test_first_scene_emotion_matches_first_arc_step(self, tmp_path):
        eng = _arc_engine(tmp_path)
        result = eng.apply(_script(4), "suspicion_reveal")
        expected = _resolve(EMOTIONAL_ARCS["suspicion_reveal"][0][0])
        assert result.scenes[0].emotion == expected

    def test_last_scene_emotion_matches_last_arc_step(self, tmp_path):
        eng = _arc_engine(tmp_path)
        result = eng.apply(_script(4), "suspicion_reveal")
        expected = _resolve(EMOTIONAL_ARCS["suspicion_reveal"][-1][0])
        assert result.scenes[-1].emotion == expected

    def test_high_intensity_arc_syncs_intent(self, tmp_path):
        eng = _arc_engine(tmp_path)
        result = eng.apply(_script(4), "suspicion_reveal")
        # Last step is surprise at 1.0 → should sync intent to "twist"
        assert result.scenes[-1].scene_intent == "twist"


# ── select_arc ────────────────────────────────────────────────────────────────

class TestSelectArc:
    def test_hype_signal_returns_hype_breakdown(self, tmp_path):
        eng = _arc_engine(tmp_path)
        assert eng.select_arc("This is overhyped AI news") == "hype_breakdown"

    def test_tutorial_signal_returns_confusion_clarity(self, tmp_path):
        eng = _arc_engine(tmp_path)
        assert eng.select_arc("How to understand this AI explained") == "confusion_clarity"

    def test_fear_signal_returns_calm_to_fear(self, tmp_path):
        eng = _arc_engine(tmp_path)
        assert eng.select_arc("AI job replacement warning") == "calm_to_fear"

    def test_no_signal_returns_suspicion_reveal(self, tmp_path):
        eng = _arc_engine(tmp_path)
        assert eng.select_arc("New AI model released today") == "suspicion_reveal"

    def test_accepts_video_script(self, tmp_path):
        eng = _arc_engine(tmp_path)
        script = VideoScript(
            title="The biggest AI breakthrough ever",
            description="d", tags=[], hook="h", scenes=[],
        )
        assert eng.select_arc(script) == "hype_breakdown"

    def test_case_insensitive(self, tmp_path):
        eng = _arc_engine(tmp_path)
        assert eng.select_arc("HOW TO USE AI") == "confusion_clarity"


# ── select_arc_from_editorial ─────────────────────────────────────────────────

class TestSelectArcFromEditorial:
    def test_hype_vs_reality_theme(self, tmp_path):
        eng = _arc_engine(tmp_path)
        result = eng.select_arc_from_editorial({"theme": "hype_vs_reality"})
        assert result == "hype_breakdown"

    def test_ai_hype_belief(self, tmp_path):
        eng = _arc_engine(tmp_path)
        result = eng.select_arc_from_editorial({"belief_applied": "ai_hype"})
        assert result == "hype_breakdown"

    def test_breakdown_format(self, tmp_path):
        eng = _arc_engine(tmp_path)
        result = eng.select_arc_from_editorial({"format": "breakdown"})
        assert result == "confusion_clarity"

    def test_tools_misuse_theme(self, tmp_path):
        eng = _arc_engine(tmp_path)
        result = eng.select_arc_from_editorial({"theme": "tools_misuse"})
        assert result == "confusion_clarity"

    def test_temporal_conflict(self, tmp_path):
        eng = _arc_engine(tmp_path)
        result = eng.select_arc_from_editorial({"conflict_type": "temporal"})
        assert result == "calm_to_fear"

    def test_future_uncertainty_theme(self, tmp_path):
        eng = _arc_engine(tmp_path)
        result = eng.select_arc_from_editorial({"theme": "future_uncertainty"})
        assert result == "calm_to_fear"

    def test_empty_editorial_falls_to_keyword(self, tmp_path):
        eng = _arc_engine(tmp_path)
        result = eng.select_arc_from_editorial({})
        assert result in EMOTIONAL_ARCS

    def test_returns_valid_arc_type(self, tmp_path):
        eng = _arc_engine(tmp_path)
        for editorial in [
            {"theme": "hype_vs_reality"},
            {"format": "breakdown"},
            {"conflict_type": "temporal"},
            {},
        ]:
            assert eng.select_arc_from_editorial(editorial) in EMOTIONAL_ARCS


# ── short_arc_emotions ────────────────────────────────────────────────────────

class TestShortArcEmotions:
    def test_returns_four_tuples(self):
        result = EmotionalArcEngine.short_arc_emotions()
        assert len(result) == 4

    def test_all_emotions_are_core(self):
        for emo, _ in EmotionalArcEngine.short_arc_emotions():
            assert emo in EMOTIONS

    def test_intensities_in_range(self):
        for _, intensity in EmotionalArcEngine.short_arc_emotions():
            assert 0.0 <= intensity <= 1.0

    def test_first_is_shock(self):
        result = EmotionalArcEngine.short_arc_emotions()
        assert result[0][0] == "shock"

    def test_third_is_surprise(self):
        result = EmotionalArcEngine.short_arc_emotions()
        assert result[2][0] == "surprise"


# ── arc_curve ─────────────────────────────────────────────────────────────────

class TestArcCurve:
    def test_returns_list_of_dicts(self, tmp_path):
        eng = _arc_engine(tmp_path)
        script = eng.apply(_script(4), "suspicion_reveal")
        curve = eng.arc_curve(script)
        assert len(curve) == 4
        for item in curve:
            assert "idx" in item
            assert "emotion" in item
            assert "intensity" in item

    def test_intensities_in_range(self, tmp_path):
        eng = _arc_engine(tmp_path)
        script = eng.apply(_script(4), "suspicion_reveal")
        for item in eng.arc_curve(script):
            assert 0.0 <= item["intensity"] <= 1.0

    def test_intent_and_avatar_present(self, tmp_path):
        eng = _arc_engine(tmp_path)
        script = eng.apply(_script(4), "hype_breakdown")
        for item in eng.arc_curve(script):
            assert "intent" in item
            assert "avatar" in item


# ── log_stats / load_stats ────────────────────────────────────────────────────

class TestLogStats:
    def test_creates_file(self, tmp_path):
        eng = _arc_engine(tmp_path)
        script = eng.apply(_script(4), "suspicion_reveal")
        eng.log_stats(script, "suspicion_reveal", "vid001")
        assert (tmp_path / "arc_stats.jsonl").exists()

    def test_returns_arc_stats(self, tmp_path):
        eng = _arc_engine(tmp_path)
        script = eng.apply(_script(4), "hype_breakdown")
        stats = eng.log_stats(script, "hype_breakdown")
        assert isinstance(stats, ArcStats)

    def test_arc_type_recorded(self, tmp_path):
        eng = _arc_engine(tmp_path)
        script = eng.apply(_script(4), "calm_to_fear")
        stats = eng.log_stats(script, "calm_to_fear", "v1")
        assert stats.arc_type == "calm_to_fear"

    def test_scene_count_correct(self, tmp_path):
        eng = _arc_engine(tmp_path)
        script = eng.apply(_script(6), "suspicion_reveal")
        stats = eng.log_stats(script, "suspicion_reveal")
        assert stats.scene_count == 6

    def test_avg_intensity_positive(self, tmp_path):
        eng = _arc_engine(tmp_path)
        script = eng.apply(_script(4), "suspicion_reveal")
        stats = eng.log_stats(script, "suspicion_reveal")
        assert stats.avg_intensity > 0.0

    def test_emotion_sequence_length_matches_scene_count(self, tmp_path):
        eng = _arc_engine(tmp_path)
        script = eng.apply(_script(5), "hype_breakdown")
        stats = eng.log_stats(script, "hype_breakdown")
        assert len(stats.emotion_sequence) == 5

    def test_load_stats_empty_on_new_store(self, tmp_path):
        eng = _arc_engine(tmp_path)
        assert eng.load_stats() == []

    def test_load_stats_returns_records(self, tmp_path):
        eng = _arc_engine(tmp_path)
        script = eng.apply(_script(4), "suspicion_reveal")
        eng.log_stats(script, "suspicion_reveal")
        assert len(eng.load_stats()) == 1

    def test_load_respects_n(self, tmp_path):
        eng = _arc_engine(tmp_path)
        script = eng.apply(_script(4), "suspicion_reveal")
        for _ in range(7):
            eng.log_stats(script, "suspicion_reveal")
        assert len(eng.load_stats(n=3)) == 3

    def test_arc_stats_to_dict_roundtrip(self, tmp_path):
        eng = _arc_engine(tmp_path)
        script = eng.apply(_script(4), "hype_breakdown")
        stats = eng.log_stats(script, "hype_breakdown", "vid42")
        d = stats.to_dict()
        restored = ArcStats.from_dict(d)
        assert restored.arc_type == stats.arc_type
        assert restored.video_id == stats.video_id
        assert restored.scene_count == stats.scene_count


# ── arc_type_distribution ─────────────────────────────────────────────────────

class TestArcTypeDistribution:
    def test_returns_dict_with_all_arc_types(self, tmp_path):
        eng = _arc_engine(tmp_path)
        dist = eng.arc_type_distribution()
        for arc in EMOTIONAL_ARCS:
            assert arc in dist

    def test_empty_store_returns_zeros(self, tmp_path):
        eng = _arc_engine(tmp_path)
        dist = eng.arc_type_distribution()
        assert all(v == 0 for v in dist.values())

    def test_counts_correct(self, tmp_path):
        eng = _arc_engine(tmp_path)
        script = eng.apply(_script(4), "suspicion_reveal")
        eng.log_stats(script, "suspicion_reveal")
        eng.log_stats(script, "suspicion_reveal")
        dist = eng.arc_type_distribution()
        assert dist["suspicion_reveal"] == 2
        assert dist["hype_breakdown"] == 0


# ── singleton ─────────────────────────────────────────────────────────────────

class TestSingleton:
    def test_returns_engine(self):
        assert isinstance(get_arc_engine(), EmotionalArcEngine)

    def test_same_instance(self):
        assert get_arc_engine() is get_arc_engine()
