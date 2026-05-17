"""Tests for PresentationEngine — direction layer between Humanizer and Voice/Video."""
from __future__ import annotations

import pytest

from src.presentation_engine import (
    INTENT_MAP,
    PresentationEngine,
    _emphasise_first_sentence,
    _first_sentence_end,
    _punchify,
)
from src.script_generator import Scene, VideoScript


# ── helpers ───────────────────────────────────────────────────────────────────

def _scene(
    idx: int = 0,
    intent: str = "explain",
    narration: str = "This is test narration.",
) -> Scene:
    return Scene(
        idx=idx,
        heading=f"Scene {idx}",
        narration=narration,
        visual_prompt="test prompt",
        scene_intent=intent,
    )


def _script(*scenes: Scene) -> VideoScript:
    return VideoScript(
        title="Test",
        description="desc",
        tags=[],
        hook="hook",
        scenes=list(scenes),
    )


def _strategy(fmt: str = "long") -> dict:
    return {"format": fmt}


# ── INTENT_MAP ────────────────────────────────────────────────────────────────

class TestIntentMap:
    def test_all_intents_have_four_keys(self):
        for intent, mapping in INTENT_MAP.items():
            assert {"voice", "visual", "pause", "subtitle"} == set(mapping.keys()), intent

    def test_shock_maps_to_zoom_in(self):
        assert INTENT_MAP["shock"]["visual"] == "zoom_in"

    def test_curiosity_maps_to_slow_zoom(self):
        assert INTENT_MAP["curiosity"]["visual"] == "slow_zoom"

    def test_twist_maps_to_cut_fast(self):
        assert INTENT_MAP["twist"]["visual"] == "cut_fast"

    def test_explain_maps_to_subtitle_focus(self):
        assert INTENT_MAP["explain"]["visual"] == "subtitle_focus"

    def test_shock_has_bold_highlight_subtitle(self):
        assert INTENT_MAP["shock"]["subtitle"] == "bold_highlight"

    def test_explain_has_normal_subtitle(self):
        assert INTENT_MAP["explain"]["subtitle"] == "normal"


# ── PresentationEngine.apply — intent mapping ─────────────────────────────────

class TestIntentMapping:
    def test_shock_sets_zoom_in(self):
        engine = PresentationEngine()
        script = _script(_scene(intent="shock"))
        result = engine.apply(script, _strategy())
        assert result.scenes[0].visual_direction == "zoom_in"

    def test_curiosity_sets_slow_zoom(self):
        engine = PresentationEngine()
        script = _script(_scene(intent="curiosity"))
        result = engine.apply(script, _strategy())
        assert result.scenes[0].visual_direction == "slow_zoom"

    def test_twist_sets_cut_fast(self):
        engine = PresentationEngine()
        script = _script(_scene(intent="twist"))
        result = engine.apply(script, _strategy())
        assert result.scenes[0].visual_direction == "cut_fast"

    def test_explain_sets_subtitle_focus(self):
        engine = PresentationEngine()
        script = _script(_scene(intent="explain"))
        result = engine.apply(script, _strategy())
        assert result.scenes[0].visual_direction == "subtitle_focus"

    def test_unknown_intent_falls_back_to_explain(self):
        engine = PresentationEngine()
        sc = _scene(intent="unknown_intent")
        script = _script(sc)
        result = engine.apply(script, _strategy())
        assert result.scenes[0].visual_direction == "subtitle_focus"


# ── Voice direction ───────────────────────────────────────────────────────────

class TestVoiceDirection:
    def test_shock_injects_emphasis(self):
        engine = PresentationEngine()
        script = _script(_scene(intent="shock", narration="This changes everything. More text."))
        result = engine.apply(script, _strategy())
        narration = result.scenes[0].narration
        assert "[EMPHASIS]" in narration
        assert "[/EMPHASIS]" in narration

    def test_curiosity_injects_pause_short(self):
        engine = PresentationEngine()
        script = _script(_scene(intent="curiosity", narration="Nobody talks about this."))
        result = engine.apply(script, _strategy())
        assert result.scenes[0].narration.startswith("[PAUSE_SHORT]")

    def test_twist_injects_pause_long(self):
        engine = PresentationEngine()
        script = _script(_scene(intent="twist", narration="Here is the twist."))
        result = engine.apply(script, _strategy())
        assert result.scenes[0].narration.startswith("[PAUSE_LONG]")

    def test_explain_no_annotation(self):
        engine = PresentationEngine()
        narration = "This is a neutral explanation."
        script = _script(_scene(intent="explain", narration=narration))
        result = engine.apply(script, _strategy())
        # No PAUSE or EMPHASIS added for neutral voice
        assert "[PAUSE" not in result.scenes[0].narration
        assert "[EMPHASIS]" not in result.scenes[0].narration

    def test_no_duplicate_pause_annotation(self):
        engine = PresentationEngine()
        # Narration already has a pause token
        narration = "[PAUSE_SHORT]Some text."
        script = _script(_scene(intent="curiosity", narration=narration))
        result = engine.apply(script, _strategy())
        assert result.scenes[0].narration.count("[PAUSE_SHORT]") == 1

    def test_reaction_intent_injects_pause_short(self):
        engine = PresentationEngine()
        # "reaction" maps to voice="curious" → should prepend [PAUSE_SHORT]
        script = _script(_scene(intent="reaction", narration="What a surprise."))
        result = engine.apply(script, _strategy())
        assert result.scenes[0].narration.startswith("[PAUSE_SHORT]")

    def test_voice_direction_field_set(self):
        engine = PresentationEngine()
        script = _script(_scene(intent="shock"))
        result = engine.apply(script, _strategy())
        assert result.scenes[0].voice_direction == "strong"


# ── Visual direction (Shorts mode) ────────────────────────────────────────────

class TestShortsMode:
    def test_shorts_forces_visual_cycle(self):
        engine = PresentationEngine()
        scenes = [_scene(idx=i, intent="explain") for i in range(4)]
        script = _script(*scenes)
        result = engine.apply(script, _strategy("short"))
        visuals = [s.visual_direction for s in result.scenes]
        assert "zoom_in" in visuals
        assert "cut_fast" in visuals

    def test_shorts_forces_bold_highlight_subtitles(self):
        engine = PresentationEngine()
        scenes = [_scene(idx=i, intent="explain") for i in range(3)]
        script = _script(*scenes)
        result = engine.apply(script, _strategy("short"))
        for scene in result.scenes:
            assert scene.subtitle_style == "bold_highlight"

    def test_shorts_forces_short_pacing(self):
        engine = PresentationEngine()
        scenes = [_scene(idx=i, intent="explain") for i in range(3)]
        script = _script(*scenes)
        result = engine.apply(script, _strategy("short"))
        for scene in result.scenes:
            assert scene.pause_after == "short"

    def test_shorts_also_triggered_by_shorts_format_string(self):
        engine = PresentationEngine()
        script = _script(_scene(intent="explain"))
        result = engine.apply(script, _strategy("shorts"))
        assert result.scenes[0].subtitle_style == "bold_highlight"


# ── Pacing ────────────────────────────────────────────────────────────────────

class TestPacing:
    def test_shock_gets_short_pause(self):
        engine = PresentationEngine()
        script = _script(_scene(intent="shock"))
        result = engine.apply(script, _strategy())
        assert result.scenes[0].pause_after == "short"

    def test_pause_after_field_set_for_all_scenes(self):
        engine = PresentationEngine()
        scenes = [_scene(idx=i) for i in range(5)]
        script = _script(*scenes)
        result = engine.apply(script, _strategy())
        for scene in result.scenes:
            assert scene.pause_after in ("short", "long", "none")


# ── Persona layer ─────────────────────────────────────────────────────────────

class TestPersonaLayer:
    def test_direct_persona_punchifies_short_lines(self):
        engine = PresentationEngine()
        narration = "This is a short line."
        script = _script(_scene(intent="explain", narration=narration))
        result = engine.apply(script, _strategy(), persona="direct")
        assert result.scenes[0].narration.rstrip().endswith("!")

    def test_direct_persona_leaves_long_lines_unchanged(self):
        engine = PresentationEngine()
        # >100 chars — should not be punchified
        narration = ("This is a very long narration sentence that exceeds one hundred characters "
                     "and should not get an exclamation mark at the end.")
        script = _script(_scene(intent="explain", narration=narration))
        result = engine.apply(script, _strategy(), persona="direct")
        # Should not end with "!" (was not punchified due to length)
        assert not result.scenes[0].narration.rstrip().endswith("!")

    def test_serious_persona_adds_pause_on_shock(self):
        engine = PresentationEngine()
        # Start with no pause annotation
        narration = "Something shocking happened."
        script = _script(_scene(intent="shock", narration=narration))
        result = engine.apply(script, _strategy(), persona="serious")
        # Shock+serious should have at least one pause annotation
        assert "[PAUSE" in result.scenes[0].narration

    def test_default_persona_is_direct(self):
        engine = PresentationEngine()
        narration = "Short line."
        script = _script(_scene(intent="explain", narration=narration))
        result = engine.apply(script, _strategy())
        assert result.scenes[0].narration.rstrip().endswith("!")


# ── Subtitle style ────────────────────────────────────────────────────────────

class TestSubtitleStyle:
    def test_shock_subtitle_bold_highlight(self):
        engine = PresentationEngine()
        script = _script(_scene(intent="shock"))
        result = engine.apply(script, _strategy())
        assert result.scenes[0].subtitle_style == "bold_highlight"

    def test_explain_subtitle_normal(self):
        engine = PresentationEngine()
        script = _script(_scene(intent="explain"))
        result = engine.apply(script, _strategy())
        assert result.scenes[0].subtitle_style == "normal"


# ── Script is returned (not mutated in place) ─────────────────────────────────

class TestApplyReturnValue:
    def test_returns_same_script_object(self):
        engine = PresentationEngine()
        script = _script(_scene())
        result = engine.apply(script, _strategy())
        assert result is script

    def test_all_scenes_processed(self):
        engine = PresentationEngine()
        scenes = [_scene(idx=i) for i in range(6)]
        script = _script(*scenes)
        result = engine.apply(script, _strategy())
        assert len(result.scenes) == 6
        for scene in result.scenes:
            assert scene.voice_direction != ""
            assert scene.visual_direction != ""


# ── Helper functions ──────────────────────────────────────────────────────────

class TestHelpers:
    def test_first_sentence_end_at_period(self):
        assert _first_sentence_end("Hello world. More text.") == 12

    def test_first_sentence_end_at_exclamation(self):
        assert _first_sentence_end("Wow! More text.") == 4

    def test_first_sentence_end_at_question(self):
        assert _first_sentence_end("Really? Yes.") == 7

    def test_first_sentence_end_no_punctuation(self):
        text = "no punctuation here"
        assert _first_sentence_end(text) == len(text)

    def test_emphasise_first_sentence_wraps_correctly(self):
        text = "This is key. Rest follows."
        result = _emphasise_first_sentence(text)
        assert result.startswith("[EMPHASIS]This is key.[/EMPHASIS]")
        assert "Rest follows." in result

    def test_emphasise_idempotent(self):
        text = "[EMPHASIS]Already wrapped.[/EMPHASIS] More."
        assert _emphasise_first_sentence(text) == text

    def test_punchify_short_period_line(self):
        assert _punchify("Short line.") == "Short line!"

    def test_punchify_leaves_question_marks(self):
        assert _punchify("Is this right?") == "Is this right?"

    def test_punchify_leaves_existing_exclamation(self):
        assert _punchify("Already exciting!") == "Already exciting!"

    def test_punchify_does_not_touch_ellipsis(self):
        assert _punchify("Trailing ellipsis...") == "Trailing ellipsis..."

    def test_punchify_multiline(self):
        text = "Line one.\nLine two."
        result = _punchify(text)
        assert "Line one!" in result
        assert "Line two!" in result
