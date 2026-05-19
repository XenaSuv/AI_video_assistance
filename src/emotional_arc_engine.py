"""Emotional Arc Engine — global emotional journey of a video.

EmotionEngine assigns emotions locally (per scene_intent).
EmotionalArcEngine shapes emotions globally (across the whole video).

The difference:
    EmotionEngine  → each scene feels correct in isolation
    ArcEngine      → the viewer travels somewhere by the end

Four arcs, matched to four content patterns:

    suspicion_reveal  — curiosity → doubt → tension → surprise
    (best for AI news)  "something's off" → "wait, really?" → "reveal"

    hype_breakdown    — excitement → doubt → doubt(strong) → realism
    (AI announcements) "this looks amazing" → "but here's the problem"

    confusion_clarity — curiosity → curiosity(deep) → calm → calm(settled)
    (tutorials)        "what is this?" → "I get it now" → "relief"

    calm_to_fear      — neutral → curiosity → shock → doubt(unresolved)
    (consequences)     "seems fine" → "wait" → "this is serious" → "..."

Arc application:
    n scenes are divided into len(arc) steps (linear interpolation).
    Each step's emotion + intensity overrides what EmotionEngine set.
    avatar_style is updated from AVATAR_STYLE_MAP.

Ordering in pipeline:
    PresentationEngine → EmotionEngine → EmotionalArcEngine → save

Integration:
    PipelineOrchestrator._step_script()
        → EmotionalArcEngine().apply(script, arc_type=select_arc(...))

    Shorts: EmotionalArcEngine.short_arc_emotions() → [(emotion, intensity)×4]
"""
from __future__ import annotations

import dataclasses
import json
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import settings
from src.emotion_engine import AVATAR_STYLE_MAP
from src.script_generator import Scene, VideoScript

# ── Emotion name aliases (expressive arc names → EMOTIONS vocabulary) ─────────
# Arcs use descriptive names ("tension", "realism") for readability.
# These are normalised to the 8 core emotions before being applied.

_EMOTION_ALIAS: dict[str, str] = {
    "tension":     "shock",
    "conflict":    "doubt",
    "confusion":   "curiosity",
    "exploration": "curiosity",
    "insight":     "calm",
    "relief":      "calm",
    "concern":     "fear",
    "unresolved":  "doubt",
    "realism":     "neutral",
}


def _resolve(emotion: str) -> str:
    """Normalise alias → core emotion. Unknown names pass through unchanged."""
    return _EMOTION_ALIAS.get(emotion, emotion)


# ── Arc step type (named for readability in arc definitions) ──────────────────

ArcStep = tuple[str, float]   # (emotion_or_alias, intensity 0.0-1.0)


# ── Emotional arcs ────────────────────────────────────────────────────────────

EMOTIONAL_ARCS: dict[str, list[ArcStep]] = {
    "suspicion_reveal": [
        ("curiosity",  0.60),   # "something's off here"
        ("doubt",      0.75),   # "wait, is this right?"
        ("tension",    0.90),   # "the pressure builds"
        ("surprise",   1.00),   # "the reveal"
    ],
    "hype_breakdown": [
        ("excitement", 0.80),   # "this looks amazing"
        ("doubt",      0.60),   # "but wait…"
        ("conflict",   0.90),   # "here's the real problem"
        ("realism",    0.45),   # "the sober landing"
    ],
    "confusion_clarity": [
        ("confusion",  0.65),   # "what is this exactly?"
        ("exploration",0.70),   # "let me work through it…"
        ("insight",    0.60),   # "the click moment"
        ("relief",     0.40),   # "settled, calm, clear"
    ],
    "calm_to_fear": [
        ("neutral",    0.30),   # "seems fine at first"
        ("concern",    0.60),   # "wait, what?"
        ("tension",    0.90),   # "this is serious"
        ("unresolved", 0.70),   # "left hanging — watch next time"
    ],
}

# Shorts: compressed 4-beat arc tuned for 3-second retention
SHORT_ARC: list[ArcStep] = [
    ("shock",     1.00),
    ("doubt",     0.70),
    ("surprise",  1.00),
    ("curiosity", 0.60),
]

# When arc emotion is strong, sync scene_intent for PresentationEngine/AvatarEngine
_EMOTION_INTENT_SYNC: dict[str, str] = {
    "surprise":   "twist",
    "doubt":      "conflict",
    "shock":      "shock",
    "tension":    "shock",
    "fear":       "warning",
    "concern":    "warning",
    "excitement": "reaction",
    "curiosity":  "curiosity",
}
_INTENT_SYNC_THRESHOLD: float = 0.75


# ── Arc-selection signals ─────────────────────────────────────────────────────

_HYPE_SIGNALS       = {"hype", "overhyped", "revolutionary", "breakthrough", "amazing",
                        "game-changer", "changes everything", "biggest"}
_TUTORIAL_SIGNALS   = {"how", "why", "explained", "guide", "understand", "breakdown",
                        "learn", "deep dive", "tutorial"}
_FEAR_SIGNALS       = {"warning", "danger", "risk", "threat", "concern", "worse",
                        "problem", "replace", "layoff", "safety"}


# ── Arc stats dataclass ───────────────────────────────────────────────────────

@dataclass
class ArcStats:
    """One arc record: which arc was used and what the emotion sequence looked like."""
    arc_type:        str
    video_id:        str
    scene_count:     int
    emotion_sequence: list[str]
    avg_intensity:   float
    recorded_at:     str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ArcStats":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# ── Engine ────────────────────────────────────────────────────────────────────

class EmotionalArcEngine:
    """Shapes the global emotional journey of a VideoScript.

    Public interface
    ─────────────────
    apply(script, arc_type)
        → VideoScript: scenes re-shaped with arc emotion distribution

    select_arc(script_or_text)
        → str: keyword-based arc selection from title/hook

    select_arc_from_editorial(editorial)
        → str: arc selection using enriched editorial (theme/belief/conflict)

    short_arc_emotions()
        → list[(emotion, intensity)]: 4-beat Shorts arc

    arc_curve(script)
        → list[dict]: per-scene arc data for dashboard line chart

    log_stats(script, arc_type, video_id)
        → ArcStats: persists to data/arc_stats.jsonl

    load_stats(n)
        → list[ArcStats]: last n arc records
    """

    def __init__(self, data_dir: Path | None = None) -> None:
        self._dir  = data_dir or settings.data_dir
        self._path = self._dir / "arc_stats.jsonl"
        self._dir.mkdir(parents=True, exist_ok=True)

    # ── main entry point ──────────────────────────────────────────────────────

    def apply(self, script: VideoScript, arc_type: str) -> VideoScript:
        """Re-shape scene emotions to follow the named arc's global distribution.

        Overrides EmotionEngine's local per-scene assignments with the arc's
        globally-shaped emotion/intensity sequence.  Does not touch
        voice_direction or visual_direction set by PresentationEngine.
        """
        arc = self._get_arc(arc_type)
        scenes = self._map_arc_to_scenes(script.scenes, arc)
        scenes = self._sync_intents(scenes)
        logger.debug(
            f"EmotionalArcEngine: arc={arc_type!r} "
            f"scenes={len(scenes)} steps={len(arc)}"
        )
        return dataclasses.replace(script, scenes=scenes)

    # ── arc retrieval ─────────────────────────────────────────────────────────

    def _get_arc(self, arc_type: str) -> list[ArcStep]:
        """Return the arc steps, falling back to suspicion_reveal."""
        return EMOTIONAL_ARCS.get(arc_type, EMOTIONAL_ARCS["suspicion_reveal"])

    # ── arc → scenes distribution ─────────────────────────────────────────────

    def _map_arc_to_scenes(
        self, scenes: list[Scene], arc: list[ArcStep]
    ) -> list[Scene]:
        """Distribute arc steps linearly across scenes.

        For n=8 scenes and 4 arc steps, each step covers 2 scenes:
            scenes 0-1 → step 0  (curiosity 0.60)
            scenes 2-3 → step 1  (doubt 0.75)
            scenes 4-5 → step 2  (tension 0.90)
            scenes 6-7 → step 3  (surprise 1.00)
        """
        n_scenes = len(scenes)
        n_steps  = len(arc)
        if n_scenes == 0 or n_steps == 0:
            return list(scenes)

        result = []
        for i, scene in enumerate(scenes):
            # Linear interpolation: map scene index → arc step index
            step_idx = int(i / max(n_scenes - 1, 1) * (n_steps - 1)) if n_scenes > 1 else 0
            step_idx = min(step_idx, n_steps - 1)

            raw_emotion, intensity = arc[step_idx]
            emotion = _resolve(raw_emotion)

            result.append(dataclasses.replace(
                scene,
                emotion           = emotion,
                emotion_intensity = round(intensity, 3),
                avatar_style      = AVATAR_STYLE_MAP.get(emotion, "normal"),
            ))
        return result

    # ── intent sync ───────────────────────────────────────────────────────────

    def _sync_intents(self, scenes: list[Scene]) -> list[Scene]:
        """Update scene_intent for scenes where the arc emotion is strong.

        This gives PresentationEngine and AvatarEngine the right cue
        when the arc overrides with an emotion they understand (e.g.,
        'surprise' → 'twist' → cut_fast visual + strong voice).
        Only fires when emotion_intensity ≥ _INTENT_SYNC_THRESHOLD.
        """
        result = []
        for scene in scenes:
            if (scene.emotion_intensity >= _INTENT_SYNC_THRESHOLD
                    and scene.emotion in _EMOTION_INTENT_SYNC):
                new_intent = _EMOTION_INTENT_SYNC[scene.emotion]
                result.append(dataclasses.replace(scene, scene_intent=new_intent))
            else:
                result.append(scene)
        return result

    # ── arc selection ─────────────────────────────────────────────────────────

    def select_arc(self, source: "VideoScript | str") -> str:
        """Select arc type from a VideoScript or raw text string.

        Priority:
          hype signals        → hype_breakdown
          tutorial signals    → confusion_clarity
          fear/warning signals→ calm_to_fear
          default             → suspicion_reveal
        """
        if isinstance(source, VideoScript):
            text = (source.title + " " + source.hook).lower()
        else:
            text = str(source).lower()

        if any(s in text for s in _HYPE_SIGNALS):
            return "hype_breakdown"
        if any(s in text for s in _TUTORIAL_SIGNALS):
            return "confusion_clarity"
        if any(s in text for s in _FEAR_SIGNALS):
            return "calm_to_fear"
        return "suspicion_reveal"

    def select_arc_from_editorial(self, editorial: dict[str, Any]) -> str:
        """Select arc using enriched editorial (theme / belief / conflict_type).

        Uses NarrativeIdentityEngine and NarrativeConflictEngine output if
        present, otherwise falls back to keyword-based select_arc().
        """
        theme         = editorial.get("theme", "")
        belief        = editorial.get("belief_applied", "")
        conflict_type = editorial.get("conflict_type", "")
        fmt           = editorial.get("format", "")
        topic         = editorial.get("topic", "") + " " + editorial.get("hook", "")

        # Hype-belief or hype theme → breakdown narrative
        if theme == "hype_vs_reality" or belief == "ai_hype":
            return "hype_breakdown"

        # Tutorial / explainer format → guide viewer to clarity
        if fmt in ("breakdown",) or theme in ("tools_misuse", "hidden_limits"):
            return "confusion_clarity"

        # Temporal conflict (past position revisited) or fear signals
        if conflict_type == "temporal" or theme == "future_uncertainty":
            return "calm_to_fear"

        # Fall through to keyword matching
        return self.select_arc(topic)

    # ── Shorts static helper ──────────────────────────────────────────────────

    @staticmethod
    def short_arc_emotions() -> list[tuple[str, float]]:
        """Return the 4-beat Shorts arc as (emotion, intensity) tuples.

        Beat 1: shock      (1.0)  — opener punch
        Beat 2: doubt      (0.7)  — "but wait…"
        Beat 3: surprise   (1.0)  — the twist
        Beat 4: curiosity  (0.6)  — the loop hook
        """
        return [(_resolve(emo), intensity) for emo, intensity in SHORT_ARC]

    # ── arc curve for dashboard ───────────────────────────────────────────────

    def arc_curve(self, script: VideoScript) -> list[dict[str, Any]]:
        """Return per-scene arc data for the dashboard line chart."""
        return [
            {
                "idx":       s.idx,
                "emotion":   s.emotion,
                "intensity": s.emotion_intensity,
                "intent":    s.scene_intent,
                "avatar":    s.avatar_style,
            }
            for s in script.scenes
        ]

    # ── analytics persistence ─────────────────────────────────────────────────

    def log_stats(
        self, script: VideoScript, arc_type: str, video_id: str = ""
    ) -> ArcStats:
        """Persist arc usage stats to data/arc_stats.jsonl."""
        curve      = self.arc_curve(script)
        emotions   = [p["emotion"] for p in curve]
        intensities= [p["intensity"] for p in curve]
        avg_int    = round(sum(intensities) / len(intensities), 3) if intensities else 0.0

        stats = ArcStats(
            arc_type        = arc_type,
            video_id        = video_id,
            scene_count     = len(curve),
            emotion_sequence= emotions,
            avg_intensity   = avg_int,
        )
        try:
            with open(self._path, "a") as f:
                f.write(json.dumps(stats.to_dict()) + "\n")
        except Exception as exc:
            logger.warning(f"EmotionalArcEngine: stats write failed: {exc}")
        logger.info(
            f"EmotionalArcEngine: arc={arc_type!r} "
            f"scenes={len(curve)} avg_intensity={avg_int:.2f}"
        )
        return stats

    def load_stats(self, n: int = 20) -> list[ArcStats]:
        """Load the last *n* arc stat records."""
        if not self._path.exists():
            return []
        records: list[ArcStats] = []
        for line in self._path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                records.append(ArcStats.from_dict(json.loads(line)))
            except Exception as exc:
                logger.debug(f"Skipped malformed record: {exc}")
        return records[-n:]

    def arc_type_distribution(self) -> dict[str, int]:
        """Count how often each arc type has been used."""
        records = self.load_stats(n=200)
        counts: dict[str, int] = {k: 0 for k in EMOTIONAL_ARCS}
        for r in records:
            counts[r.arc_type] = counts.get(r.arc_type, 0) + 1
        return counts


# ── Module-level singleton ────────────────────────────────────────────────────

_engine: EmotionalArcEngine | None = None


def get_arc_engine() -> EmotionalArcEngine:
    """Return the module-level EmotionalArcEngine singleton."""
    global _engine
    if _engine is None:
        _engine = EmotionalArcEngine()
    return _engine
