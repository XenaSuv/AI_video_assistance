"""Shorts Beat Engine — per-beat execution plan for 16-second Shorts.

Transforms a ShortScript's four text sections into five timed beats:

    HOOK     (0–2s)   shock      avatar  zoom_in
    CONTRAST (2–5s)   doubt      avatar  cut_zoom_out
    EXPLAIN  (5–9s)   neutral    broll   broll
    TWIST    (9–13s)  surprise   avatar  jump_cut
    LOOP     (13–16s) curiosity  avatar  slow_zoom

Every 2–3 seconds the beat changes: new frame, new emotion, new text.

The golden rule:
    каждые 2–3 секунды → изменение
    (смена кадра / смена эмоции / смена текста / смена темпа)

SSML voice logic per beat:
    hook     → fast + [EMPHASIS]...[/EMPHASIS]
    contrast → [PAUSE_SHORT] + slower feel
    explain  → normal rate, no tags
    twist    → [PAUSE_SHORT] then [EMPHASIS]...[/EMPHASIS]
    loop     → normal, slightly softer

Subtitle rule: NOT the full sentence — 2–5 keywords in CAPS only.

Integration:
    ShortsEngineV2.plan(script) → ShortBeatPlan
    → video assembly: beat.visual_type + beat.avatar_mode + beat.duration_sec
    → TTS:            beat.annotated_text (SSML tags pre-applied)
    → subtitles:      beat.subtitle_text  (2–5 words, UPPERCASE)
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from src.shorts_engine_v2 import ShortScript


# ── Beat role ordering ────────────────────────────────────────────────────────

BEAT_ROLES: list[str] = ["hook", "contrast", "explain", "twist", "loop"]

# ── Durations (seconds) ───────────────────────────────────────────────────────

BEAT_DURATIONS: dict[str, float] = {
    "hook":     2.0,
    "contrast": 3.0,
    "explain":  4.0,
    "twist":    4.0,
    "loop":     3.0,
}

# ── Emotion per beat (name, base intensity) ───────────────────────────────────

BEAT_EMOTIONS: dict[str, tuple[str, float]] = {
    "hook":     ("shock",     1.00),
    "contrast": ("doubt",     0.80),
    "explain":  ("neutral",   0.50),
    "twist":    ("surprise",  1.00),
    "loop":     ("curiosity", 0.60),
}

# ── Avatar / b-roll mode per beat ─────────────────────────────────────────────
# avatar = face + trust signal; broll = context + rest from avatar

BEAT_AVATAR_MODE: dict[str, str] = {
    "hook":     "avatar",
    "contrast": "avatar",
    "explain":  "broll",
    "twist":    "avatar",
    "loop":     "avatar",
}

# ── Visual cut type per beat ──────────────────────────────────────────────────

BEAT_VISUAL_TYPE: dict[str, str] = {
    "hook":     "zoom_in",
    "contrast": "cut_zoom_out",
    "explain":  "broll",
    "twist":    "jump_cut",
    "loop":     "slow_zoom",
}

# ── SSML annotation tags applied around each beat's text ─────────────────────

_BEAT_SSML_PREFIX: dict[str, str] = {
    "hook":     "[EMPHASIS]",
    "contrast": "[PAUSE_SHORT]",
    "explain":  "",
    "twist":    "[PAUSE_SHORT][EMPHASIS]",
    "loop":     "",
}
_BEAT_SSML_SUFFIX: dict[str, str] = {
    "hook":     "[/EMPHASIS]",
    "contrast": "",
    "explain":  "",
    "twist":    "[/EMPHASIS]",
    "loop":     "",
}

# ── Subtitle filler-word filter ───────────────────────────────────────────────

_FILLER_WORDS: frozenset[str] = frozenset({
    "a", "an", "the", "is", "it", "in", "on", "at", "to", "of",
    "and", "but", "for", "or", "so", "this", "that", "with", "by",
    "from", "be", "are", "was", "were", "has", "have", "had",
    "do", "does", "did", "not", "no", "as", "its", "you", "your",
    "we", "our", "they", "their", "he", "she", "his", "her", "i",
    "me", "my", "about", "just", "than", "more", "very", "even",
    "if", "when", "what", "how", "why", "who", "which", "there",
    "here", "can", "will", "would", "could", "should", "now",
})

_SUBTITLE_MAX_WORDS: int = 5


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class ShortBeat:
    """One timed beat in a Short — the atomic unit of the execution layer."""
    role:              str    # hook | contrast | explain | twist | loop
    text:              str    # raw narration text
    annotated_text:    str    # text with SSML tags pre-applied
    duration_sec:      float  # target beat duration in seconds
    emotion:           str    # shock | doubt | neutral | surprise | curiosity
    emotion_intensity: float  # 0.0–1.0
    avatar_mode:       str    # avatar | broll
    visual_type:       str    # zoom_in | cut_zoom_out | broll | jump_cut | slow_zoom
    subtitle_text:     str    # 2–5 words, UPPERCASE

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ShortBeatPlan:
    """Complete 5-beat execution plan derived from a ShortScript."""
    short_id: str
    beats:    list[ShortBeat]

    @property
    def total_duration_sec(self) -> float:
        return sum(b.duration_sec for b in self.beats)

    @property
    def emotion_flow(self) -> list[str]:
        """Returns the emotion sequence: shock → doubt → neutral → surprise → curiosity."""
        return [b.emotion for b in self.beats]

    @property
    def full_annotated_narration(self) -> str:
        """Full narration string with SSML tags — ready for TTS input."""
        return " ".join(b.annotated_text for b in self.beats)

    def to_dict(self) -> dict[str, Any]:
        return {
            "short_id":           self.short_id,
            "total_duration_sec": self.total_duration_sec,
            "emotion_flow":       self.emotion_flow,
            "beats":              [b.to_dict() for b in self.beats],
        }


# ── Engine ────────────────────────────────────────────────────────────────────

class ShortsBeatEngine:
    """Converts a ShortScript into a 5-beat execution plan.

    Public interface
    ─────────────────
    plan(script) → ShortBeatPlan
        Maps ShortScript.{hook, core, twist, ending} to 5 timed beats.
        The "twist" field is split at a sentence boundary to produce
        the EXPLAIN beat and the TWIST beat.
    """

    def plan(self, script: ShortScript) -> ShortBeatPlan:
        """Return the 5-beat execution plan for a ShortScript."""
        raw_beats = self._split_into_beats(script)
        beats     = [self._build_beat(role, text) for role, text in raw_beats]
        return ShortBeatPlan(short_id=script.short_id, beats=beats)

    # ── text splitting ────────────────────────────────────────────────────────

    def _split_into_beats(self, script: ShortScript) -> list[tuple[str, str]]:
        """Map ShortScript's 4 fields to 5 (role, text) pairs.

        ShortScript sections → beats:
            hook    → HOOK
            core    → CONTRAST
            twist   → EXPLAIN (first half) + TWIST (second half)
            ending  → LOOP
        """
        explain_text, twist_text = self._split_twist(script.twist)
        return [
            ("hook",     script.hook),
            ("contrast", script.core),
            ("explain",  explain_text),
            ("twist",    twist_text),
            ("loop",     script.ending),
        ]

    def _split_twist(self, twist: str) -> tuple[str, str]:
        """Split the twist section at the first sentence boundary past the midpoint.

        Falls back to word-midpoint split for single-sentence twists.
        """
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", twist) if s.strip()]
        if len(sentences) >= 2:
            mid = max(1, len(sentences) // 2)
            return " ".join(sentences[:mid]), " ".join(sentences[mid:])
        words = twist.split()
        if not words:
            return twist, twist
        mid = max(1, len(words) // 2)
        return " ".join(words[:mid]), " ".join(words[mid:])

    # ── single beat assembly ──────────────────────────────────────────────────

    def _build_beat(self, role: str, text: str) -> ShortBeat:
        emotion, intensity = BEAT_EMOTIONS[role]
        return ShortBeat(
            role              = role,
            text              = text,
            annotated_text    = self._annotate(role, text),
            duration_sec      = BEAT_DURATIONS[role],
            emotion           = emotion,
            emotion_intensity = intensity,
            avatar_mode       = BEAT_AVATAR_MODE[role],
            visual_type       = BEAT_VISUAL_TYPE[role],
            subtitle_text     = self._extract_subtitle(text),
        )

    # ── SSML annotation ───────────────────────────────────────────────────────

    def _annotate(self, role: str, text: str) -> str:
        """Wrap text with the SSML tags appropriate for this beat's role."""
        prefix = _BEAT_SSML_PREFIX.get(role, "")
        suffix = _BEAT_SSML_SUFFIX.get(role, "")
        if prefix or suffix:
            return f"{prefix}{text}{suffix}"
        return text

    # ── subtitle keyword extraction ───────────────────────────────────────────

    def _extract_subtitle(self, text: str, max_words: int = _SUBTITLE_MAX_WORDS) -> str:
        """Extract 2–5 key words from text, uppercase.

        Strips punctuation, removes filler words.
        Falls back to first words if no keywords found.
        """
        raw_words   = text.split()
        clean       = [re.sub(r"[^\w'-]", "", w) for w in raw_words]
        keywords    = [
            w for w in clean
            if w and w.lower() not in _FILLER_WORDS and len(w) > 2
        ]
        chosen = keywords[:max_words] if keywords else [w for w in clean if w][:max_words]
        return " ".join(w.upper() for w in chosen) if chosen else text[:30].upper()


# ── Module-level singleton ────────────────────────────────────────────────────

_engine: ShortsBeatEngine | None = None


def get_beat_engine() -> ShortsBeatEngine:
    """Return the module-level ShortsBeatEngine singleton."""
    global _engine
    if _engine is None:
        _engine = ShortsBeatEngine()
    return _engine
