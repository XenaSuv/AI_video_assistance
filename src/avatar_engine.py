"""Avatar Engine — intent-aware orchestration of HeyGen avatar clips.

The avatar is not a visual effect. It is a source of trust, retention,
and recognisability.  Done wrong it lands in the uncanny valley.
Done right it reads as "this is like a real creator".

This engine sits between PresentationEngine and the render layer:

    Script → Humanizer → Presentation → **Avatar** → Voice → Video

Responsibilities:
    1. Decide when to use avatar vs B-roll (not every scene needs a face)
    2. Map persona → avatar_id (multi-avatar A/B strategy)
    3. Map intent → avatar_style (closeup for shock/twist, normal for explain)
    4. Add subtitle overlay on avatar clips (text = retention)
    5. Apply Shorts zoom effect for hook/short-mode clips

Intent → avatar_style mapping
──────────────────────────────
    shock     → closeup   face fills frame — creates impact
    twist     → closeup   reveal moment — dramatic pull
    hook      → closeup   stops the scroll — must be face-to-camera
    conflict  → closeup   direct challenge — confrontational energy
    warning   → normal    authority without aggression
    explain   → normal    calm delivery — trust signal
    curiosity → normal    inviting, open
    data      → normal    neutral — let the numbers speak
    reaction  → normal    relatable, not intense

Avatar intents (use face-to-camera):
    shock, twist, hook, conflict, warning

B-roll intents (use stock video instead):
    explain, data, curiosity, reaction

Persona → avatar_id (multi-avatar A/B)
───────────────────────────────────────
    direct       → HEYGEN_AVATAR_ID_DIRECT       (or default)
    serious      → HEYGEN_AVATAR_ID_SERIOUS       (or default)
    curious      → HEYGEN_AVATAR_ID_CURIOUS       (or default)
    professional → HEYGEN_AVATAR_ID_PROFESSIONAL  (or default)
    casual       → HEYGEN_AVATAR_ID_CASUAL        (or default)

If the persona-specific env var is empty, falls back to HEYGEN_AVATAR_ID.

Shorts mode
───────────
    - zoom_clip() applied after generation (face pushed 15% closer)
    - subtitle_text always burned in (bold, high-contrast)
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path
from typing import Any

from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import settings
import src.ffmpeg_utils as ffmpeg_utils
from src.heygen_presenter import generate_heygen_clip


# ── Intent mappings ───────────────────────────────────────────────────────────

# Intents where a face-to-camera avatar dominates (stops scroll, creates trust)
AVATAR_INTENTS: frozenset[str] = frozenset({
    "shock", "twist", "hook", "conflict", "warning",
})

# HeyGen avatar_style per intent
_INTENT_STYLE: dict[str, str] = {
    "shock":     "closeup",
    "twist":     "closeup",
    "hook":      "closeup",
    "conflict":  "closeup",
    "warning":   "normal",
    "explain":   "normal",
    "curiosity": "normal",
    "data":      "normal",
    "reaction":  "normal",
    "mistake":   "normal",
    "insider":   "normal",
}

# Persona → settings attribute name for persona-specific avatar_id overrides
_PERSONA_AVATAR_ATTR: dict[str, str] = {
    "direct":       "heygen_avatar_id_direct",
    "serious":      "heygen_avatar_id_serious",
    "curious":      "heygen_avatar_id_curious",
    "professional": "heygen_avatar_id_professional",
    "casual":       "heygen_avatar_id_casual",
}


# ── Engine ────────────────────────────────────────────────────────────────────

class AvatarEngine:
    """Intent-aware HeyGen clip generator with subtitle and zoom layers.

    Public interface
    ────────────────
    should_use_avatar(intent)
        Returns True for intents that benefit from face-to-camera delivery.

    render(audio_path, out_path, *, intent, persona, is_shorts, subtitle_text, aspect)
        Generate a HeyGen clip, add subtitles, apply Shorts zoom.
        Returns out_path on success, None on failure (always non-fatal).

    avatar_style(intent)
        The HeyGen avatar_style string for a given intent.

    avatar_id(persona)
        The avatar_id to use for a given persona (falls back to default).
    """

    def __init__(self) -> None:
        self._api_key    = settings.heygen_api_key
        self._default_id = settings.heygen_avatar_id
        self._voice_id   = settings.heygen_voice_id
        self._enabled    = settings.heygen_enabled

    # ── decision ──────────────────────────────────────────────────────────────

    def should_use_avatar(self, intent: str) -> bool:
        """True for high-impact intents — face-to-camera beats B-roll here."""
        return intent in AVATAR_INTENTS

    # ── lookups ───────────────────────────────────────────────────────────────

    def avatar_style(self, intent: str) -> str:
        """Return HeyGen avatar_style for *intent*."""
        return _INTENT_STYLE.get(intent, "normal")

    def avatar_id(self, persona: str) -> str:
        """Resolve the avatar_id for *persona*.

        Checks the persona-specific settings field first; falls back to the
        default HEYGEN_AVATAR_ID so a single avatar always works.
        """
        attr = _PERSONA_AVATAR_ATTR.get(persona, "")
        if attr:
            override = getattr(settings, attr, "").strip()
            if override:
                logger.debug(f"AvatarEngine: persona={persona!r} → avatar_id from {attr}")
                return override
        return self._default_id

    # ── render ────────────────────────────────────────────────────────────────

    def render(
        self,
        audio_path: Path,
        out_path: Path,
        *,
        intent: str = "explain",
        persona: str = "direct",
        is_shorts: bool = False,
        subtitle_text: str = "",
        aspect: str = "9:16",
    ) -> Path | None:
        """Generate a HeyGen avatar clip, add subtitles, apply Shorts zoom.

        All steps are non-fatal — a failure at any layer returns None so the
        caller can fall back to its normal B-roll/text pipeline.

        Parameters
        ──────────
        audio_path    — ElevenLabs MP3 to lip-sync against
        out_path      — final output MP4 path
        intent        — scene_intent string (shock | explain | …)
        persona       — delivery persona (direct | serious | curious | …)
        is_shorts     — activates 15% zoom-in for tighter face framing
        subtitle_text — text to burn as animated captions (empty = skip)
        aspect        — HeyGen video aspect ("9:16" for Shorts, "16:9" default)
        """
        if not self._enabled:
            logger.debug("AvatarEngine: HeyGen disabled — skipping")
            return None

        aid   = self.avatar_id(persona)
        style = self.avatar_style(intent)
        raw   = out_path.with_stem(out_path.stem + "_av_raw")

        logger.info(
            f"AvatarEngine: intent={intent!r} persona={persona!r} "
            f"style={style!r} avatar_id={aid!r} shorts={is_shorts}"
        )

        # ── Step 1: HeyGen clip ───────────────────────────────────────────────
        clip = generate_heygen_clip(
            audio_path,
            raw,
            api_key=self._api_key,
            avatar_id=aid,
            aspect=aspect,
            avatar_style=style,
            fallback_text=subtitle_text,
            voice_id=self._voice_id,
        )
        if clip is None:
            return None

        # ── Step 2: Shorts zoom ───────────────────────────────────────────────
        if is_shorts:
            zoomed = out_path.with_stem(out_path.stem + "_av_zoom")
            clip = ffmpeg_utils.zoom_clip(clip, zoomed, scale=1.15)

        # ── Step 3: Subtitle overlay ──────────────────────────────────────────
        if subtitle_text:
            subbed = out_path.with_stem(out_path.stem + "_av_sub")
            clip = ffmpeg_utils.burn_captions(
                clip,
                subtitle_text,
                subbed,
                font_size=60 if is_shorts else 52,
                y_pct=0.80,
                color="white",
            )

        # ── Finalise ──────────────────────────────────────────────────────────
        try:
            shutil.copy2(str(clip), str(out_path))
        except Exception as exc:
            logger.warning(f"AvatarEngine: copy to out_path failed: {exc}")
            return None
        finally:
            # Clean up intermediate files
            for tmp in (
                raw,
                out_path.with_stem(out_path.stem + "_av_zoom"),
                out_path.with_stem(out_path.stem + "_av_sub"),
            ):
                if tmp.exists() and tmp != out_path:
                    tmp.unlink(missing_ok=True)

        logger.info(f"AvatarEngine: clip ready — {out_path.name}")
        return out_path

    def render_for_scene(
        self,
        scene: Any,
        audio_path: Path,
        assembled_dir: Path,
        *,
        persona: str = "direct",
        aspect: str = "16:9",
    ) -> Path | None:
        """Convenience wrapper for long-form pipeline scene rendering.

        Reads intent and narration directly from the Scene object so the
        pipeline_orchestrator / video_generator don't need to unpack them.
        """
        intent   = getattr(scene, "scene_intent", "explain")
        narration = getattr(scene, "narration", "")
        idx       = getattr(scene, "idx", 0)
        out_path  = assembled_dir / f"heygen_scene_{idx:02d}.mp4"

        return self.render(
            audio_path,
            out_path,
            intent=intent,
            persona=persona,
            is_shorts=False,
            subtitle_text=narration,
            aspect=aspect,
        )
