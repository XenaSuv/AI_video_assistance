"""Audience Profile — learned audience preferences built from retention feedback.

The profile is updated after every published video.  It records what angles,
formats, personas, hook patterns, and emotions the audience responds to — and
what they don't.

Update rules (each call to update_from_feedback):
    retention > 0.65  → reinforce angle / persona / format / emotion  (+0.05)
    retention < 0.35  → penalise disliked_patterns (+0.05), decay angle score (−0.03)
    all values clamped to [0.0, 1.0]

Storage: data/audience_profile.json  (plain JSON dict, recreated from scratch on first run)
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from loguru import logger


_PROFILE_FILE = "audience_profile.json"

_RETENTION_GOOD  = 0.65   # reinforce
_RETENTION_BAD   = 0.35   # penalise
_REINFORCE_DELTA = 0.05
_PENALISE_DELTA  = 0.03
_MIN_SCORE = 0.0
_MAX_SCORE = 1.0

_BLANK_PROFILE: dict[str, Any] = {
    "preferred_angles":        {},
    "preferred_formats":       {},
    "preferred_hook_patterns": {},
    "disliked_patterns":       {},
    "retention_patterns":      {},
    "emotion_preferences":     {},
    "persona_affinity":        {},
}


class AudienceProfile:
    """Persistent learned audience preferences.

    Public interface
    ────────────────
    update_from_feedback(angle, retention, *, format, persona, hook_pattern, emotions)
        → update all preference dicts, save

    get_angle_score(angle)         → float (0–1)
    get_format_score(format_type)  → float (0–1)
    get_persona_affinity(persona)  → float (0–1)
    get_emotion_score(emotion)     → float (0–1)
    is_disliked(pattern)           → bool
    top_angles(n)                  → list[tuple[str, float]]
    top_personas(n)                → list[tuple[str, float]]
    save()                         → None
    """

    def __init__(self, data_dir: Path | None = None) -> None:
        if data_dir is None:
            from config import settings
            data_dir = settings.data_dir
        data_dir = Path(data_dir)
        data_dir.mkdir(parents=True, exist_ok=True)
        self._path = data_dir / _PROFILE_FILE
        self._p: dict[str, Any] = self._load()

    # ── feedback integration ──────────────────────────────────────────────────

    def update_from_feedback(
        self,
        angle:        str,
        retention:    float,
        *,
        format_type:  str = "",
        persona:      str = "",
        hook_pattern: str = "",
        emotions:     list[str] | None = None,
    ) -> None:
        """Adjust preference scores based on a single video's retention result."""
        if retention >= _RETENTION_GOOD:
            self._reinforce("preferred_angles",        angle)
            if format_type:
                self._reinforce("preferred_formats",   format_type)
            if persona:
                self._reinforce("persona_affinity",    persona)
            if hook_pattern:
                self._reinforce("preferred_hook_patterns", hook_pattern)
            for emo in (emotions or []):
                self._reinforce("emotion_preferences", emo)
        elif retention <= _RETENTION_BAD:
            self._decay("preferred_angles", angle)
            if format_type:
                self._decay("preferred_formats", format_type)
            if hook_pattern:
                self._penalise("disliked_patterns", hook_pattern)

        # Always record raw retention average for the angle
        self._update_retention_pattern(angle, retention)

        self.save()
        logger.debug(
            f"AudienceProfile: updated angle={angle!r} retention={retention:.2f}"
        )

    # ── score accessors ───────────────────────────────────────────────────────

    def get_angle_score(self, angle: str) -> float:
        return float(self._p["preferred_angles"].get(angle, 0.0))

    def get_format_score(self, format_type: str) -> float:
        return float(self._p["preferred_formats"].get(format_type, 0.0))

    def get_persona_affinity(self, persona: str) -> float:
        return float(self._p["persona_affinity"].get(persona, 0.0))

    def get_emotion_score(self, emotion: str) -> float:
        return float(self._p["emotion_preferences"].get(emotion, 0.0))

    def get_angle_retention(self, angle: str) -> float:
        """Return the running average retention for this angle (0–1)."""
        rp = self._p["retention_patterns"].get(angle)
        if not rp:
            return 0.0
        return float(rp.get("avg", 0.0))

    def is_disliked(self, pattern: str) -> bool:
        return pattern in self._p["disliked_patterns"]

    # ── analytics ─────────────────────────────────────────────────────────────

    def top_angles(self, n: int = 5) -> list[tuple[str, float]]:
        items = sorted(
            self._p["preferred_angles"].items(), key=lambda x: x[1], reverse=True
        )
        return items[:n]

    def top_personas(self, n: int = 5) -> list[tuple[str, float]]:
        items = sorted(
            self._p["persona_affinity"].items(), key=lambda x: x[1], reverse=True
        )
        return items[:n]

    def top_emotions(self, n: int = 5) -> list[tuple[str, float]]:
        items = sorted(
            self._p["emotion_preferences"].items(), key=lambda x: x[1], reverse=True
        )
        return items[:n]

    def as_dict(self) -> dict[str, Any]:
        return dict(self._p)

    def save(self) -> None:
        try:
            self._path.write_text(json.dumps(self._p, indent=2))
        except Exception as exc:
            logger.warning(f"AudienceProfile: save failed: {exc}")

    # ── internals ─────────────────────────────────────────────────────────────

    def _reinforce(self, bucket: str, key: str) -> None:
        if not key:
            return
        current = float(self._p[bucket].get(key, 0.0))
        self._p[bucket][key] = min(_MAX_SCORE, current + _REINFORCE_DELTA)

    def _decay(self, bucket: str, key: str) -> None:
        if not key:
            return
        current = float(self._p[bucket].get(key, 0.0))
        self._p[bucket][key] = max(_MIN_SCORE, current - _PENALISE_DELTA)

    def _penalise(self, bucket: str, key: str) -> None:
        if not key:
            return
        current = float(self._p[bucket].get(key, 0.0))
        self._p[bucket][key] = min(_MAX_SCORE, current + _REINFORCE_DELTA)

    def _update_retention_pattern(self, angle: str, retention: float) -> None:
        """Maintain running average retention per angle."""
        rp = self._p["retention_patterns"].setdefault(angle, {"avg": 0.0, "count": 0})
        n = int(rp.get("count", 0))
        avg = float(rp.get("avg", 0.0))
        n += 1
        rp["avg"]   = avg + (retention - avg) / n   # online mean
        rp["count"] = n

    def _load(self) -> dict[str, Any]:
        if not self._path.exists():
            return {k: dict(v) for k, v in _BLANK_PROFILE.items()}
        try:
            data = json.loads(self._path.read_text())
            if isinstance(data, dict):
                # back-fill any keys added in future versions
                for key, default in _BLANK_PROFILE.items():
                    data.setdefault(key, dict(default))
                return data
        except Exception as exc:
            logger.warning(f"AudienceProfile: load failed, starting fresh: {exc}")
        return {k: dict(v) for k, v in _BLANK_PROFILE.items()}


# ── module-level singleton ────────────────────────────────────────────────────

_profile: AudienceProfile | None = None


def get_audience_profile() -> AudienceProfile:
    """Return the module-level AudienceProfile singleton."""
    global _profile
    if _profile is None:
        _profile = AudienceProfile()
    return _profile
