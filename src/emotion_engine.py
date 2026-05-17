"""Emotion Engine — the affective layer that gives content a feeling.

The difference between tension and emotion:
    tension   = structural pressure (high/low)
    emotion   = what that pressure feels like (shock, doubt, fear…)

Emotion is the glue between all other layers:
    hooks + sequence + tension + conflict + identity → felt through emotion.

What this engine adds to every scene:
    emotion           — the named feeling the viewer should experience
    emotion_intensity — 0.0-1.0 scalar (damped on consecutive same emotions)
    avatar_style      — avatar delivery mode derived from emotion

Emotional arc principle:
    shock → shock → shock → shock  (❌ numbing)
    curiosity → doubt → surprise → relief → tension  (✅ alive)

Integration:
    PipelineOrchestrator._step_script()
        → PresentationEngine.apply()
        → EmotionEngine.apply()         ← here

    ShortsEngineV2: use EmotionEngine.short_emotion_flow(n) for Shorts arc

    Dashboard: emotion_curve() → per-scene intensity chart
"""
from __future__ import annotations

import dataclasses
import json
import sys
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import settings
from src.script_generator import Scene, VideoScript


# ── Emotion vocabulary ────────────────────────────────────────────────────────

EMOTIONS: list[str] = [
    "shock", "curiosity", "doubt", "surprise",
    "excitement", "fear", "calm", "neutral",
]


# ── Intent → (emotion, base intensity) ───────────────────────────────────────

EMOTION_MAP: dict[str, tuple[str, float]] = {
    "hook":      ("shock",      0.90),
    "shock":     ("shock",      0.90),
    "conflict":  ("doubt",      0.80),
    "twist":     ("surprise",   1.00),
    "reaction":  ("excitement", 0.80),
    "curiosity": ("curiosity",  0.70),
    "warning":   ("fear",       0.75),
    "mistake":   ("doubt",      0.75),
    "insider":   ("curiosity",  0.65),
    "data":      ("calm",       0.25),
    "explain":   ("neutral",    0.30),
}

_DEFAULT_EMOTION: tuple[str, float] = ("neutral", 0.30)


# ── Emotion → SSML voice tag prefix ──────────────────────────────────────────
# Tokens understood by voice_generator.annotated_to_ssml()

VOICE_MAP: dict[str, str] = {
    "shock":      "[EMPHASIS]",
    "surprise":   "[EMPHASIS]",
    "excitement": "[EMPHASIS]",
    "curiosity":  "[PAUSE_SHORT]",
    "doubt":      "[PAUSE_SHORT]",
    "fear":       "[PAUSE_LONG]",
    "calm":       "",
    "neutral":    "",
}


# ── Emotion → visual direction ────────────────────────────────────────────────
# Complements PresentationEngine's INTENT_MAP; only applied when no direction set

VISUAL_MAP: dict[str, str] = {
    "shock":      "zoom_in",
    "surprise":   "cut_fast",
    "excitement": "zoom_in",
    "curiosity":  "slow_zoom",
    "doubt":      "subtitle_focus",
    "fear":       "slow_zoom",
    "calm":       "subtitle_focus",
    "neutral":    "subtitle_focus",
}


# ── Emotion → avatar delivery style ──────────────────────────────────────────

AVATAR_STYLE_MAP: dict[str, str] = {
    "shock":      "energetic",
    "surprise":   "energetic",
    "excitement": "energetic",
    "curiosity":  "thinking",
    "doubt":      "thinking",
    "fear":       "serious",
    "calm":       "normal",
    "neutral":    "normal",
}


# ── Shorts: enforced emotion arc ──────────────────────────────────────────────

SHORT_EMOTION_FLOW: list[str] = ["shock", "curiosity", "doubt", "surprise"]

_SHORT_INTENSITY_MAP: dict[str, float] = {
    "shock":     0.90,
    "curiosity": 0.75,
    "doubt":     0.70,
    "surprise":  1.00,
}


# ── Tension scoring per scene intent ─────────────────────────────────────────

_TENSION_SCORE: dict[str, float] = {
    "shock":     0.90,
    "conflict":  0.85,
    "twist":     0.90,
    "warning":   0.75,
    "mistake":   0.70,
    "reaction":  0.60,
    "curiosity": 0.50,
    "insider":   0.45,
    "explain":   0.30,
    "data":      0.20,
}
_HIGH_TENSION_THRESHOLD: float = 0.80
_FLOW_DAMPING:           float = 0.70   # intensity multiplier for same-emotion repeat


# ── Emotion stats dataclass (for analytics logging) ───────────────────────────

@dataclass
class EmotionStats:
    """Per-video emotion summary, written to data/emotion_stats.jsonl."""
    video_id:         str
    title:            str
    dominant_emotion: str
    avg_intensity:    float
    emotion_sequence: list[str]
    scene_count:      int
    recorded_at:      str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ── Engine ────────────────────────────────────────────────────────────────────

class EmotionEngine:
    """Assigns and shapes the emotional arc of a video script.

    Public interface
    ─────────────────
    apply(script, *, is_shorts, conflict_intensity)
        → VideoScript: every Scene has emotion / emotion_intensity / avatar_style

    short_emotion_flow(n_segments)
        → list[(emotion, intensity)]: enforced Shorts arc

    emotion_curve(script)
        → list[dict]: per-scene data for dashboard line chart

    log_stats(script, video_id)
        → EmotionStats: persists to data/emotion_stats.jsonl

    load_stats(n)
        → list[EmotionStats]: last n runs for dashboard
    """

    def __init__(self, data_dir: Path | None = None) -> None:
        self._dir  = data_dir or settings.data_dir
        self._path = self._dir / "emotion_stats.jsonl"
        self._dir.mkdir(parents=True, exist_ok=True)

    # ── main entry point ──────────────────────────────────────────────────────

    def apply(
        self,
        script: VideoScript,
        *,
        is_shorts: bool = False,
        conflict_intensity: float = 0.0,
    ) -> VideoScript:
        """Enrich every Scene with emotion / emotion_intensity / avatar_style.

        Steps:
          1. _assign_emotions:  intent → emotion from EMOTION_MAP
          2. _adjust_flow:      dampen repeating emotions (×0.70)
          3. _sync_with_tension: high-tension intents boost intensity
          4. _enforce_short_flow: Shorts override (shock→curiosity→doubt→surprise)
          5. _apply_visual:     fill empty visual_direction from VISUAL_MAP
        """
        scenes = [self._assign_emotion(s) for s in script.scenes]
        scenes = self._adjust_flow(scenes)
        scenes = self._sync_with_tension(scenes, conflict_intensity)
        if is_shorts:
            scenes = self._enforce_short_flow(scenes)
        scenes = self._apply_visual(scenes)
        return dataclasses.replace(script, scenes=scenes)

    # ── assign emotion per scene ──────────────────────────────────────────────

    def _assign_emotion(self, scene: Scene) -> Scene:
        """Map scene_intent → emotion + intensity + avatar_style."""
        emotion, intensity = EMOTION_MAP.get(scene.scene_intent, _DEFAULT_EMOTION)
        return dataclasses.replace(
            scene,
            emotion          = emotion,
            emotion_intensity= round(intensity, 3),
            avatar_style     = AVATAR_STYLE_MAP.get(emotion, "normal"),
        )

    # ── dampen consecutive same emotions ──────────────────────────────────────

    def _adjust_flow(self, scenes: list[Scene]) -> list[Scene]:
        """Reduce intensity when the same emotion repeats back-to-back.

        Prevents the viewer from going numb to a single sustained tone.
        Damping is multiplicative: same × 0.70 each repetition.
        """
        result = list(scenes)
        for i in range(1, len(result)):
            if result[i].emotion == result[i - 1].emotion:
                new_intensity = round(result[i].emotion_intensity * _FLOW_DAMPING, 3)
                result[i] = dataclasses.replace(result[i], emotion_intensity=new_intensity)
        return result

    # ── sync emotion intensity with scene tension ─────────────────────────────

    def _sync_with_tension(
        self,
        scenes: list[Scene],
        conflict_intensity: float = 0.0,
    ) -> list[Scene]:
        """Boost emotion_intensity for high-tension scenes.

        Uses scene_intent-derived tension + the editorial conflict_intensity
        (from NarrativeConflictEngine) as a global floor for the whole script.
        """
        result = list(scenes)
        for i, scene in enumerate(result):
            scene_tension   = _TENSION_SCORE.get(scene.scene_intent, 0.3)
            effective       = max(scene_tension, conflict_intensity)
            if effective > _HIGH_TENSION_THRESHOLD:
                boosted = max(scene.emotion_intensity, _HIGH_TENSION_THRESHOLD)
                if boosted != scene.emotion_intensity:
                    result[i] = dataclasses.replace(
                        scene, emotion_intensity=round(boosted, 3)
                    )
        return result

    # ── enforce Shorts emotion arc ────────────────────────────────────────────

    def _enforce_short_flow(self, scenes: list[Scene]) -> list[Scene]:
        """Override emotions with the Shorts rhythm: shock→curiosity→doubt→surprise.

        Shorts viewers need a predictable emotional roller-coaster.
        Each position in SHORT_EMOTION_FLOW is tuned for maximum 3-second retention.
        """
        result = []
        for i, scene in enumerate(scenes):
            emotion   = SHORT_EMOTION_FLOW[i % len(SHORT_EMOTION_FLOW)]
            intensity = _SHORT_INTENSITY_MAP.get(emotion, 0.70)
            result.append(dataclasses.replace(
                scene,
                emotion           = emotion,
                emotion_intensity = round(intensity, 3),
                avatar_style      = AVATAR_STYLE_MAP.get(emotion, "normal"),
            ))
        return result

    # ── fill visual direction from emotion ────────────────────────────────────

    def _apply_visual(self, scenes: list[Scene]) -> list[Scene]:
        """Fill empty visual_direction using VISUAL_MAP.

        Does NOT override values already set by PresentationEngine.
        """
        result = []
        for scene in scenes:
            if scene.visual_direction:
                result.append(scene)
            else:
                visual = VISUAL_MAP.get(scene.emotion, "subtitle_focus")
                result.append(dataclasses.replace(scene, visual_direction=visual))
        return result

    # ── Shorts static helper ──────────────────────────────────────────────────

    @staticmethod
    def short_emotion_flow(n_segments: int) -> list[tuple[str, float]]:
        """Return (emotion, intensity) for each segment in a Short.

        Example for n_segments=4:
            [("shock", 0.9), ("curiosity", 0.75), ("doubt", 0.7), ("surprise", 1.0)]
        """
        return [
            (
                SHORT_EMOTION_FLOW[i % len(SHORT_EMOTION_FLOW)],
                _SHORT_INTENSITY_MAP.get(SHORT_EMOTION_FLOW[i % len(SHORT_EMOTION_FLOW)], 0.70),
            )
            for i in range(n_segments)
        ]

    # ── emotion curve for dashboard ───────────────────────────────────────────

    def emotion_curve(self, script: VideoScript) -> list[dict[str, Any]]:
        """Return per-scene emotion data for dashboard visualisation."""
        return [
            {
                "idx":       s.idx,
                "intent":    s.scene_intent,
                "emotion":   s.emotion,
                "intensity": s.emotion_intensity,
                "avatar":    s.avatar_style,
            }
            for s in script.scenes
        ]

    # ── analytics persistence ─────────────────────────────────────────────────

    def log_stats(self, script: VideoScript, video_id: str = "") -> EmotionStats:
        """Persist emotion curve summary to data/emotion_stats.jsonl."""
        curve = self.emotion_curve(script)
        emotions = [p["emotion"] for p in curve]
        intensities = [p["intensity"] for p in curve]

        from collections import Counter
        dominant = Counter(emotions).most_common(1)[0][0] if emotions else "neutral"
        avg_int  = round(sum(intensities) / len(intensities), 3) if intensities else 0.0

        stats = EmotionStats(
            video_id         = video_id,
            title            = script.title,
            dominant_emotion = dominant,
            avg_intensity    = avg_int,
            emotion_sequence = emotions,
            scene_count      = len(curve),
        )
        try:
            with open(self._path, "a") as f:
                f.write(json.dumps(stats.to_dict()) + "\n")
        except Exception as exc:
            logger.warning(f"EmotionEngine: stats write failed: {exc}")
        logger.info(
            f"EmotionEngine: {len(curve)} scenes, dominant={dominant}, "
            f"avg_intensity={avg_int:.2f}"
        )
        return stats

    def load_stats(self, n: int = 20) -> list[EmotionStats]:
        """Load the last *n* emotion stat records."""
        if not self._path.exists():
            return []
        records: list[EmotionStats] = []
        for line in self._path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
                records.append(EmotionStats(**{
                    k: v for k, v in d.items()
                    if k in EmotionStats.__dataclass_fields__
                }))
            except Exception as exc:
                logger.debug(f"Skipped malformed record: {exc}")
        return records[-n:]


# ── Module-level singleton ────────────────────────────────────────────────────

_engine: EmotionEngine | None = None


def get_emotion_engine() -> EmotionEngine:
    """Return the module-level EmotionEngine singleton."""
    global _engine
    if _engine is None:
        _engine = EmotionEngine()
    return _engine
