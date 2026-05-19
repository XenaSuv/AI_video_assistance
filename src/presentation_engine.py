"""Presentation Engine — direction layer between Humanizer and Voice/Video.

This is not a text generator. It is a director.

  Editorial → Script → Humanizer → **Presentation** → Voice/Video

The engine reads each scene's ``scene_intent`` and translates it into three
concrete output signals:

  voice_direction  — SSML annotation strategy (injected into narration)
  visual_direction — clip motion directive (read by VideoGenerator)
  pause_after      — breath between scenes (read by VideoGenerator)

plus a ``subtitle_style`` hint for the Subtitle/Caption layer.

Intent map (the authoritative source of truth):

  explain   → neutral voice   | subtitle_focus | no pause
  shock     → strong voice    | zoom_in        | pause short
  curiosity → pause voice     | slow_zoom      | no pause
  twist     → drop voice      | cut_fast       | pause short
  data      → neutral voice   | subtitle_focus | no pause
  reaction  → curious voice   | zoom_in        | pause short

Persona layer (applied after intent mapping):

  direct   — terminal "." on punchy lines → "!" for punch energy
  serious  — prepend [PAUSE_SHORT] on shock/twist lines
  curious  — keep existing pause; no override

Shorts mode (activated when strategy["format"] == "short"):
  - All pacing forced to "short"
  - All subtitles forced to "bold_highlight"
  - Visual rotation every scene: zoom_in → cut_fast → slow_zoom → …

SSML annotation tokens recognised by voice_generator.annotated_to_ssml():
  [PAUSE_SHORT]   → <break time="300ms"/>
  [PAUSE_LONG]    → <break time="700ms"/>
  [EMPHASIS]      → <emphasis level="strong">
  [/EMPHASIS]     → </emphasis>
"""
from __future__ import annotations

import re
from typing import Any

from loguru import logger

from src.script_generator import Scene, VideoScript

# ── Intent → presentation mapping ────────────────────────────────────────────

INTENT_MAP: dict[str, dict[str, str]] = {
    "explain":   {"voice": "neutral",  "visual": "subtitle_focus", "pause": "none",  "subtitle": "normal"},
    "shock":     {"voice": "strong",   "visual": "zoom_in",        "pause": "short", "subtitle": "bold_highlight"},
    "curiosity": {"voice": "pause",    "visual": "slow_zoom",      "pause": "none",  "subtitle": "bold_highlight"},
    "twist":     {"voice": "drop",     "visual": "cut_fast",       "pause": "short", "subtitle": "bold_highlight"},
    "data":      {"voice": "neutral",  "visual": "subtitle_focus", "pause": "none",  "subtitle": "normal"},
    "reaction":  {"voice": "curious",  "visual": "zoom_in",        "pause": "short", "subtitle": "bold_highlight"},
    "conflict":  {"voice": "strong",   "visual": "cut_fast",       "pause": "short", "subtitle": "bold_highlight"},
    "mistake":   {"voice": "drop",     "visual": "cut_fast",       "pause": "short", "subtitle": "bold_highlight"},
    "insider":   {"voice": "pause",    "visual": "slow_zoom",      "pause": "none",  "subtitle": "bold_highlight"},
}

_FALLBACK_INTENT = INTENT_MAP["explain"]

# Rotating visual sequence for Shorts fast-pacing mode
_SHORTS_VISUAL_CYCLE = ["zoom_in", "cut_fast", "slow_zoom", "cut_fast"]


# ── Engine ────────────────────────────────────────────────────────────────────

class PresentationEngine:
    """Apply voice, visual, pacing, and persona direction to a VideoScript.

    Usage::

        engine = PresentationEngine()
        script = engine.apply(script, strategy, persona="direct")
        # script.scenes now have voice_direction, visual_direction, etc.
        # scene.narration has SSML annotation tokens injected
    """

    def apply(
        self,
        script: VideoScript,
        strategy: dict[str, Any],
        persona: str = "direct",
    ) -> VideoScript:
        """Apply all direction layers in order and return the modified script."""
        is_shorts = strategy.get("format", "") in ("short", "shorts")

        scenes = list(script.scenes)
        scenes = self._map_intent(scenes)
        scenes = self._inject_voice_direction(scenes)
        scenes = self._inject_visual_direction(scenes, is_shorts=is_shorts)
        scenes = self._inject_pacing(scenes, is_shorts=is_shorts)
        scenes = self._apply_persona(scenes, persona)
        scenes = self._inject_subtitle_style(scenes, is_shorts=is_shorts)

        script.scenes = scenes
        logger.info(
            f"PresentationEngine: applied to {len(scenes)} scenes "
            f"(persona={persona!r}, shorts={is_shorts})"
        )
        return script

    # ── mapping ───────────────────────────────────────────────────────────────

    def _map_intent(self, scenes: list[Scene]) -> list[Scene]:
        """Resolve each scene's intent directives from INTENT_MAP."""
        for scene in scenes:
            mapping = INTENT_MAP.get(scene.scene_intent, _FALLBACK_INTENT)
            scene.voice_direction   = mapping["voice"]
            scene.visual_direction  = mapping["visual"]
            scene.pause_after       = mapping["pause"]
            scene.subtitle_style    = mapping["subtitle"]
        return scenes

    # ── voice ─────────────────────────────────────────────────────────────────

    def _inject_voice_direction(self, scenes: list[Scene]) -> list[Scene]:
        """Inject SSML annotation tokens into scene.narration based on voice_direction.

        The voice_generator.annotated_to_ssml() function converts these tokens
        to real SSML before sending to ElevenLabs.
        """
        for scene in scenes:
            direction = scene.voice_direction
            text = scene.narration

            if direction == "strong":
                # Wrap first sentence in [EMPHASIS] for punch
                text = _emphasise_first_sentence(text)
            elif direction == "pause":
                # Brief dramatic pause before the narration (curiosity build)
                if not text.startswith("[PAUSE"):
                    text = "[PAUSE_SHORT]" + text
            elif direction == "drop":
                # Longer pause for twist/reveal moments
                if not text.startswith("[PAUSE"):
                    text = "[PAUSE_LONG]" + text
            elif direction == "curious":
                # Light pause to invite the viewer in
                if not text.startswith("[PAUSE"):
                    text = "[PAUSE_SHORT]" + text
            # "neutral" → no annotation

            scene.narration = text
        return scenes

    # ── visual ────────────────────────────────────────────────────────────────

    def _inject_visual_direction(
        self,
        scenes: list[Scene],
        is_shorts: bool = False,
    ) -> list[Scene]:
        """Set visual_direction. In Shorts mode, forces a rotating fast cycle."""
        if is_shorts:
            for i, scene in enumerate(scenes):
                scene.visual_direction = _SHORTS_VISUAL_CYCLE[i % len(_SHORTS_VISUAL_CYCLE)]
        # Long-form: visual_direction already set by _map_intent — nothing to override
        return scenes

    # ── pacing ────────────────────────────────────────────────────────────────

    def _inject_pacing(
        self,
        scenes: list[Scene],
        is_shorts: bool = False,
    ) -> list[Scene]:
        """Assign pause_after for scene transitions.

        In Shorts mode every scene gets a "short" pause to maintain tempo.
        In long-form, alternate short/none creates natural breathing room.
        """
        if is_shorts:
            for scene in scenes:
                scene.pause_after = "short"
        else:
            for i, scene in enumerate(scenes):
                # Intent-map value already set; only override "none" scenes
                # every other position to add minimal rhythm.
                if scene.pause_after == "none" and i % 2 == 0:
                    scene.pause_after = "short"
        return scenes

    # ── persona ───────────────────────────────────────────────────────────────

    def _apply_persona(self, scenes: list[Scene], persona: str) -> list[Scene]:
        """Light persona touch applied on top of intent-driven direction."""
        if persona == "direct":
            for scene in scenes:
                scene.narration = _punchify(scene.narration)
        elif persona == "serious":
            for scene in scenes:
                if scene.scene_intent in ("shock", "twist"):
                    # Extra gravitas: prepend a pause if not already annotated
                    if "[PAUSE" not in scene.narration:
                        scene.narration = "[PAUSE_SHORT]" + scene.narration
        # "curious" persona — intent-driven pauses already sufficient
        return scenes

    # ── subtitles ─────────────────────────────────────────────────────────────

    def _inject_subtitle_style(
        self,
        scenes: list[Scene],
        is_shorts: bool = False,
    ) -> list[Scene]:
        """Force bold_highlight subtitles in Shorts mode."""
        if is_shorts:
            for scene in scenes:
                scene.subtitle_style = "bold_highlight"
        return scenes


# ── Helpers ───────────────────────────────────────────────────────────────────

def _first_sentence_end(text: str) -> int:
    """Return the index of the first sentence terminator (. ! ?) + 1."""
    match = re.search(r"[.!?]", text)
    return match.end() if match else len(text)


def _emphasise_first_sentence(text: str) -> str:
    """Wrap the first sentence in [EMPHASIS]...[/EMPHASIS] tokens."""
    if "[EMPHASIS]" in text:
        return text
    end = _first_sentence_end(text)
    first, rest = text[:end], text[end:]
    return f"[EMPHASIS]{first}[/EMPHASIS]{rest}"


def _punchify(text: str) -> str:
    """For 'direct' persona: replace trailing period with '!' on short lines."""
    lines = text.split("\n")
    result = []
    for line in lines:
        stripped = line.rstrip()
        # Only punch up short declarative lines (≤100 chars) that end with "."
        if stripped.endswith(".") and len(stripped) <= 100 and not stripped.endswith(".."):
            stripped = stripped[:-1] + "!"
        result.append(stripped)
    return "\n".join(result)
