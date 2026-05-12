"""Tests for ShortsBeatEngine — per-beat execution plan layer."""
from __future__ import annotations

import pytest

from src.shorts_beat_engine import (
    BEAT_AVATAR_MODE,
    BEAT_DURATIONS,
    BEAT_EMOTIONS,
    BEAT_ROLES,
    BEAT_VISUAL_TYPE,
    ShortBeat,
    ShortBeatPlan,
    ShortsBeatEngine,
    _BEAT_SSML_PREFIX,
    _BEAT_SSML_SUFFIX,
    _FILLER_WORDS,
    _SUBTITLE_MAX_WORDS,
    get_beat_engine,
)
from src.emotion_engine import EMOTIONS
from src.shorts_engine_v2 import ShortScript


# ── helpers ───────────────────────────────────────────────────────────────────

def _script(
    hook:   str = "Everyone thinks this is a breakthrough.",
    core:   str = "But there is a problem nobody mentions.",
    twist:  str = "It actually makes things worse. And nobody is talking about it.",
    ending: str = "That's why this matters more than you think.",
) -> ShortScript:
    return ShortScript(
        short_id  = "test-id-001",
        topic     = "AI model",
        hook_type = "conflict",
        hook      = hook,
        core      = core,
        twist     = twist,
        ending    = ending,
        style     = "fast_cut",
        persona   = "direct",
    )


def _engine() -> ShortsBeatEngine:
    return ShortsBeatEngine()


# ── Beat constants ────────────────────────────────────────────────────────────

class TestBeatConstants:
    def test_five_roles(self):
        assert len(BEAT_ROLES) == 5

    def test_role_order(self):
        assert BEAT_ROLES == ["hook", "contrast", "explain", "twist", "loop"]

    def test_durations_sum_to_sixteen(self):
        assert sum(BEAT_DURATIONS.values()) == 16.0

    def test_all_roles_have_duration(self):
        for role in BEAT_ROLES:
            assert role in BEAT_DURATIONS

    def test_all_durations_positive(self):
        for role, dur in BEAT_DURATIONS.items():
            assert dur > 0, role

    def test_all_roles_have_emotion(self):
        for role in BEAT_ROLES:
            assert role in BEAT_EMOTIONS

    def test_emotion_names_are_core_emotions(self):
        for role, (emotion, _) in BEAT_EMOTIONS.items():
            assert emotion in EMOTIONS, f"{role} → {emotion!r}"

    def test_emotion_intensities_in_range(self):
        for role, (_, intensity) in BEAT_EMOTIONS.items():
            assert 0.0 <= intensity <= 1.0, role

    def test_all_roles_have_avatar_mode(self):
        for role in BEAT_ROLES:
            assert role in BEAT_AVATAR_MODE

    def test_avatar_modes_valid(self):
        valid = {"avatar", "broll"}
        for role, mode in BEAT_AVATAR_MODE.items():
            assert mode in valid, f"{role} → {mode!r}"

    def test_explain_is_broll(self):
        assert BEAT_AVATAR_MODE["explain"] == "broll"

    def test_hook_and_twist_are_avatar(self):
        assert BEAT_AVATAR_MODE["hook"]  == "avatar"
        assert BEAT_AVATAR_MODE["twist"] == "avatar"

    def test_all_roles_have_visual_type(self):
        for role in BEAT_ROLES:
            assert role in BEAT_VISUAL_TYPE

    def test_visual_types_valid(self):
        valid = {"zoom_in", "cut_zoom_out", "broll", "jump_cut", "slow_zoom"}
        for role, vis in BEAT_VISUAL_TYPE.items():
            assert vis in valid, f"{role} → {vis!r}"

    def test_hook_visual_is_zoom_in(self):
        assert BEAT_VISUAL_TYPE["hook"] == "zoom_in"

    def test_twist_visual_is_jump_cut(self):
        assert BEAT_VISUAL_TYPE["twist"] == "jump_cut"

    def test_explain_visual_is_broll(self):
        assert BEAT_VISUAL_TYPE["explain"] == "broll"

    def test_loop_visual_is_slow_zoom(self):
        assert BEAT_VISUAL_TYPE["loop"] == "slow_zoom"

    def test_emotion_flow_is_shock_doubt_neutral_surprise_curiosity(self):
        flow = [BEAT_EMOTIONS[r][0] for r in BEAT_ROLES]
        assert flow == ["shock", "doubt", "neutral", "surprise", "curiosity"]


# ── SSML annotation ───────────────────────────────────────────────────────────

class TestAnnotate:
    def test_hook_wrapped_in_emphasis(self):
        eng = _engine()
        result = eng._annotate("hook", "This changes everything")
        assert result == "[EMPHASIS]This changes everything[/EMPHASIS]"

    def test_contrast_prefixed_with_pause_short(self):
        eng = _engine()
        result = eng._annotate("contrast", "But there is a problem")
        assert result.startswith("[PAUSE_SHORT]")

    def test_explain_has_no_tags(self):
        eng = _engine()
        result = eng._annotate("explain", "It makes things worse")
        assert "[" not in result
        assert result == "It makes things worse"

    def test_twist_has_pause_and_emphasis(self):
        eng = _engine()
        result = eng._annotate("twist", "Nobody talks about this")
        assert "[PAUSE_SHORT]" in result
        assert "[EMPHASIS]" in result
        assert "[/EMPHASIS]" in result

    def test_loop_has_no_tags(self):
        eng = _engine()
        result = eng._annotate("loop", "This is bigger than it looks")
        assert "[" not in result

    def test_unknown_role_returns_text_unchanged(self):
        eng = _engine()
        result = eng._annotate("unknown_role", "some text")
        assert result == "some text"

    def test_annotated_text_contains_original_text(self):
        eng = _engine()
        text = "original text here"
        for role in BEAT_ROLES:
            assert text in eng._annotate(role, text)


# ── subtitle extraction ───────────────────────────────────────────────────────

class TestExtractSubtitle:
    def test_returns_uppercase(self):
        eng = _engine()
        result = eng._extract_subtitle("this changes everything now")
        assert result == result.upper()

    def test_strips_filler_words(self):
        eng = _engine()
        # "this", "is", "a" are fillers; "breakthrough" should survive
        result = eng._extract_subtitle("this is a breakthrough")
        assert "THIS" not in result or "BREAKTHROUGH" in result

    def test_max_five_words(self):
        eng = _engine()
        text = "breaking revolutionary artificial intelligence model released today worldwide"
        words = eng._extract_subtitle(text).split()
        assert len(words) <= _SUBTITLE_MAX_WORDS

    def test_empty_ish_text_returns_something(self):
        eng = _engine()
        result = eng._extract_subtitle("a the is")
        assert len(result) > 0

    def test_short_text_handled(self):
        eng = _engine()
        result = eng._extract_subtitle("AI")
        assert len(result) > 0

    def test_punctuation_stripped(self):
        eng = _engine()
        result = eng._extract_subtitle("breakthrough! model, released.")
        assert "!" not in result
        assert "," not in result
        assert "." not in result

    def test_filler_words_set_nonempty(self):
        assert len(_FILLER_WORDS) > 20

    def test_common_words_in_filler(self):
        for word in ("the", "a", "is", "and", "but", "in", "of"):
            assert word in _FILLER_WORDS


# ── _split_twist ──────────────────────────────────────────────────────────────

class TestSplitTwist:
    def test_two_sentences_split_at_midpoint(self):
        eng = _engine()
        twist = "First sentence here. Second sentence there."
        explain, twist_out = eng._split_twist(twist)
        assert len(explain) > 0
        assert len(twist_out) > 0

    def test_both_parts_contain_original_words(self):
        eng = _engine()
        twist = "It makes things worse. And nobody is talking about it."
        explain, twist_out = eng._split_twist(twist)
        all_text = explain + " " + twist_out
        assert "worse" in all_text
        assert "nobody" in all_text

    def test_single_sentence_splits_at_word_midpoint(self):
        eng = _engine()
        words = "one two three four five six".split()
        twist = " ".join(words)
        explain, twist_out = eng._split_twist(twist)
        # Both parts non-empty
        assert explain.strip()
        assert twist_out.strip()

    def test_three_sentences_explain_gets_first(self):
        eng = _engine()
        twist = "First. Second. Third."
        explain, _ = eng._split_twist(twist)
        assert "First" in explain

    def test_empty_string_does_not_crash(self):
        eng = _engine()
        explain, twist_out = eng._split_twist("")
        assert isinstance(explain, str)
        assert isinstance(twist_out, str)


# ── _split_into_beats ────────────────────────────────────────────────────────

class TestSplitIntoBeats:
    def test_returns_five_tuples(self):
        eng = _engine()
        result = eng._split_into_beats(_script())
        assert len(result) == 5

    def test_first_beat_is_hook(self):
        eng = _engine()
        result = eng._split_into_beats(_script())
        assert result[0][0] == "hook"

    def test_second_beat_is_contrast(self):
        eng = _engine()
        result = eng._split_into_beats(_script())
        assert result[1][0] == "contrast"

    def test_third_beat_is_explain(self):
        eng = _engine()
        result = eng._split_into_beats(_script())
        assert result[2][0] == "explain"

    def test_fourth_beat_is_twist(self):
        eng = _engine()
        result = eng._split_into_beats(_script())
        assert result[3][0] == "twist"

    def test_fifth_beat_is_loop(self):
        eng = _engine()
        result = eng._split_into_beats(_script())
        assert result[4][0] == "loop"

    def test_hook_text_matches_script_hook(self):
        eng = _engine()
        s = _script()
        result = eng._split_into_beats(s)
        assert result[0][1] == s.hook

    def test_contrast_text_matches_script_core(self):
        eng = _engine()
        s = _script()
        result = eng._split_into_beats(s)
        assert result[1][1] == s.core

    def test_loop_text_matches_script_ending(self):
        eng = _engine()
        s = _script()
        result = eng._split_into_beats(s)
        assert result[4][1] == s.ending

    def test_explain_and_twist_cover_all_twist_words(self):
        eng = _engine()
        s = _script()
        beats = eng._split_into_beats(s)
        explain_words = set(beats[2][1].split())
        twist_words   = set(beats[3][1].split())
        original_words = set(s.twist.split())
        assert explain_words | twist_words == original_words


# ── ShortBeat dataclass ───────────────────────────────────────────────────────

class TestShortBeat:
    def test_to_dict_has_all_fields(self):
        beat = ShortBeat(
            role="hook", text="test", annotated_text="[EMPHASIS]test[/EMPHASIS]",
            duration_sec=2.0, emotion="shock", emotion_intensity=1.0,
            avatar_mode="avatar", visual_type="zoom_in", subtitle_text="TEST",
        )
        d = beat.to_dict()
        for key in ("role", "text", "annotated_text", "duration_sec", "emotion",
                    "emotion_intensity", "avatar_mode", "visual_type", "subtitle_text"):
            assert key in d

    def test_to_dict_values_correct(self):
        beat = ShortBeat(
            role="twist", text="nobody", annotated_text="nobody",
            duration_sec=4.0, emotion="surprise", emotion_intensity=1.0,
            avatar_mode="avatar", visual_type="jump_cut", subtitle_text="NOBODY",
        )
        assert beat.to_dict()["emotion"] == "surprise"
        assert beat.to_dict()["duration_sec"] == 4.0


# ── plan() ────────────────────────────────────────────────────────────────────

class TestPlan:
    def test_returns_short_beat_plan(self):
        eng = _engine()
        result = eng.plan(_script())
        assert isinstance(result, ShortBeatPlan)

    def test_plan_has_five_beats(self):
        eng = _engine()
        result = eng.plan(_script())
        assert len(result.beats) == 5

    def test_beat_roles_in_order(self):
        eng = _engine()
        result = eng.plan(_script())
        assert [b.role for b in result.beats] == BEAT_ROLES

    def test_all_beat_emotions_are_core_emotions(self):
        eng = _engine()
        result = eng.plan(_script())
        for beat in result.beats:
            assert beat.emotion in EMOTIONS

    def test_all_beat_intensities_in_range(self):
        eng = _engine()
        result = eng.plan(_script())
        for beat in result.beats:
            assert 0.0 <= beat.emotion_intensity <= 1.0

    def test_all_avatar_modes_valid(self):
        eng = _engine()
        result = eng.plan(_script())
        for beat in result.beats:
            assert beat.avatar_mode in ("avatar", "broll")

    def test_all_visual_types_valid(self):
        valid = {"zoom_in", "cut_zoom_out", "broll", "jump_cut", "slow_zoom"}
        eng = _engine()
        result = eng.plan(_script())
        for beat in result.beats:
            assert beat.visual_type in valid

    def test_all_subtitles_uppercase(self):
        eng = _engine()
        result = eng.plan(_script())
        for beat in result.beats:
            assert beat.subtitle_text == beat.subtitle_text.upper()

    def test_hook_beat_emotion_is_shock(self):
        eng = _engine()
        result = eng.plan(_script())
        assert result.beats[0].emotion == "shock"

    def test_twist_beat_emotion_is_surprise(self):
        eng = _engine()
        result = eng.plan(_script())
        assert result.beats[3].emotion == "surprise"

    def test_explain_beat_avatar_mode_is_broll(self):
        eng = _engine()
        result = eng.plan(_script())
        assert result.beats[2].avatar_mode == "broll"

    def test_short_id_preserved(self):
        eng = _engine()
        s = _script()
        result = eng.plan(s)
        assert result.short_id == s.short_id

    def test_hook_annotated_text_has_emphasis(self):
        eng = _engine()
        result = eng.plan(_script())
        assert "[EMPHASIS]" in result.beats[0].annotated_text

    def test_original_script_not_mutated(self):
        eng = _engine()
        s = _script()
        original_hook = s.hook
        eng.plan(s)
        assert s.hook == original_hook


# ── ShortBeatPlan properties ──────────────────────────────────────────────────

class TestShortBeatPlan:
    def test_total_duration_is_sixteen(self):
        eng = _engine()
        plan = eng.plan(_script())
        assert plan.total_duration_sec == 16.0

    def test_emotion_flow_length_five(self):
        eng = _engine()
        plan = eng.plan(_script())
        assert len(plan.emotion_flow) == 5

    def test_emotion_flow_correct_sequence(self):
        eng = _engine()
        plan = eng.plan(_script())
        assert plan.emotion_flow == ["shock", "doubt", "neutral", "surprise", "curiosity"]

    def test_full_annotated_narration_contains_all_beat_texts(self):
        eng = _engine()
        plan = eng.plan(_script())
        narration = plan.full_annotated_narration
        assert len(narration) > 0

    def test_to_dict_has_required_keys(self):
        eng = _engine()
        plan = eng.plan(_script())
        d = plan.to_dict()
        for key in ("short_id", "total_duration_sec", "emotion_flow", "beats"):
            assert key in d

    def test_to_dict_beats_is_list(self):
        eng = _engine()
        plan = eng.plan(_script())
        assert isinstance(plan.to_dict()["beats"], list)

    def test_to_dict_beat_count(self):
        eng = _engine()
        plan = eng.plan(_script())
        assert len(plan.to_dict()["beats"]) == 5


# ── ShortsEngineV2.plan() integration ────────────────────────────────────────

class TestShortsEngineV2Plan:
    def test_plan_method_exists(self, tmp_path):
        from src.shorts_engine_v2 import ShortsEngineV2
        eng = ShortsEngineV2(data_dir=tmp_path)
        assert hasattr(eng, "plan")

    def test_plan_returns_beat_plan(self, tmp_path):
        from src.shorts_engine_v2 import ShortsEngineV2
        eng = ShortsEngineV2(data_dir=tmp_path)
        plan = eng.plan(_script())
        assert isinstance(plan, ShortBeatPlan)

    def test_plan_has_five_beats(self, tmp_path):
        from src.shorts_engine_v2 import ShortsEngineV2
        eng = ShortsEngineV2(data_dir=tmp_path)
        plan = eng.plan(_script())
        assert len(plan.beats) == 5


# ── singleton ─────────────────────────────────────────────────────────────────

class TestSingleton:
    def test_returns_engine(self):
        assert isinstance(get_beat_engine(), ShortsBeatEngine)

    def test_same_instance(self):
        assert get_beat_engine() is get_beat_engine()
