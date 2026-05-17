"""Narrative Conflict Engine — emotional tension layer for character-driven content.

The insight: people don't watch for perfect explanations.
They watch for tension, doubt, and conflict.

Three types of conflict:
    external  — "everyone says X, but they're wrong"
    internal  — "this looks right, but something feels off"
    temporal  — "I said X before. I was wrong."

Types 2 and 3 are what make a character feel alive.

Conflict intensity (0.0 → 1.0):
    > 0.67  → strong assertion, no hedging
    0.33-0.67 → measured doubt, specific uncertainty
    < 0.33  → soft reservation, open question

Integration:
    EditorialBrain._enrich_with_narrative()
        → NarrativeConflictEngine.apply(editorial)
        → adds conflict_type / conflict_line / conflict_intensity

    Shorts pipeline:
        HOOK → CONFLICT (conflict_line) → TWIST → END

    PresentationEngine:
        scene_intent="conflict" → strong voice, cut_fast, bold_highlight
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import settings


# ── Conflict types ────────────────────────────────────────────────────────────

CONFLICT_TYPES: list[str] = ["external", "internal", "temporal"]


# ── Conflict templates by type × intensity band ───────────────────────────────

_CONFLICT_TEMPLATES: dict[str, dict[str, list[str]]] = {
    "external": {
        "high": [
            "Everyone's wrong about this. Here's the actual story.",
            "The consensus on this is broken. And I'll show you why.",
            "What they're saying and what's actually happening are two different things.",
            "This is not what they're telling you.",
        ],
        "medium": [
            "Most people are missing something here.",
            "The headline is not the real story.",
            "Here's what the announcement didn't mention.",
            "The obvious take on this is incomplete.",
        ],
        "low": [
            "I'm not sure everyone's read this correctly.",
            "Worth questioning the obvious interpretation here.",
            "There's another way to look at this.",
            "The framing might be leading us somewhere wrong.",
        ],
    },
    "internal": {
        "high": [
            "Something is seriously off here. I can't ignore it.",
            "I want to believe this. But I genuinely can't.",
            "This doesn't add up. And I've been sitting with that.",
            "My gut is telling me something is wrong with this picture.",
        ],
        "medium": [
            "This looks impressive. But something feels wrong.",
            "My gut says one thing. The data says another.",
            "I'm genuinely uncertain about where this lands.",
            "I've been going back and forth on this one.",
        ],
        "low": [
            "I have some reservations about this.",
            "Still working out what this actually means.",
            "Not sure how I feel about this yet.",
            "This deserves more skepticism than it's getting.",
        ],
    },
    "temporal": {
        "high": [
            "I said this would work. I was wrong. Here's why.",
            "Last time I covered this, I got it wrong. Correcting that now.",
            "I was too confident before. Things are more complicated.",
            "I owe you an update. My earlier take missed something.",
        ],
        "medium": [
            "When I first saw this, I thought one thing. Now I'm not so sure.",
            "I've been watching this develop. My view has shifted.",
            "This is exactly what I was tracking — but the outcome surprised me.",
            "My earlier read was too simple. Here's what changed.",
        ],
        "low": [
            "This connects to something I said before. With a twist.",
            "Worth revisiting my earlier take on this topic.",
            "My first instinct was close. There's more nuance here.",
            "I've updated my thinking on this since we last covered it.",
        ],
    },
}

# Intensity band thresholds
_HIGH_THRESHOLD = 0.67
_LOW_THRESHOLD  = 0.33


# ── Memory dataclass ──────────────────────────────────────────────────────────

@dataclass
class ConflictEntry:
    """One conflict record: which type, topic, intensity, and generated line."""
    conflict_type: str
    topic:         str
    intensity:     float
    line:          str
    video_id:      str = ""
    recorded_at:   str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ConflictEntry":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# ── Memory store ──────────────────────────────────────────────────────────────

class ConflictMemory:
    """Append-only store of conflict history.

    Enables callbacks: "This is exactly what I warned about."
    Enables the dashboard to show the drama arc of the channel.
    """

    def __init__(self, data_dir: Path | None = None) -> None:
        self._dir  = data_dir or settings.data_dir
        self._path = self._dir / "conflict_memory.jsonl"
        self._dir.mkdir(parents=True, exist_ok=True)

    def record(
        self,
        conflict_type: str,
        topic: str,
        intensity: float,
        line: str,
        video_id: str = "",
    ) -> ConflictEntry:
        entry = ConflictEntry(
            conflict_type=conflict_type,
            topic=topic,
            intensity=round(intensity, 3),
            line=line,
            video_id=video_id,
        )
        try:
            with open(self._path, "a") as f:
                f.write(json.dumps(entry.to_dict()) + "\n")
        except Exception as exc:
            logger.warning(f"ConflictMemory: write failed: {exc}")
        return entry

    def load_recent(self, n: int = 20) -> list[ConflictEntry]:
        if not self._path.exists():
            return []
        entries: list[ConflictEntry] = []
        for line in self._path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(ConflictEntry.from_dict(json.loads(line)))
            except Exception as exc:
                logger.debug(f"Skipped malformed record: {exc}")
        return entries[-n:]

    def get_topic_history(self, topic: str, n: int = 5) -> list[ConflictEntry]:
        """Return past conflicts on *topic* for callback generation."""
        recent = self.load_recent(n=100)
        topic_lower = topic.lower()
        matches = [
            e for e in recent
            if topic_lower in e.topic.lower()
            or any(w in e.topic.lower() for w in topic_lower.split()[:3] if len(w) > 3)
        ]
        return matches[-n:]

    def stats(self) -> dict[str, Any]:
        """Aggregate conflict stats for dashboard display."""
        recent = self.load_recent(n=100)
        type_counts: dict[str, int] = {t: 0 for t in CONFLICT_TYPES}
        total_intensity = 0.0
        for e in recent:
            type_counts[e.conflict_type] = type_counts.get(e.conflict_type, 0) + 1
            total_intensity += e.intensity
        avg_intensity = round(total_intensity / len(recent), 3) if recent else 0.0
        dominant = max(type_counts, key=lambda k: type_counts[k]) if recent else "external"
        return {
            "total":        len(recent),
            "type_counts":  type_counts,
            "avg_intensity": avg_intensity,
            "dominant_type": dominant,
        }


# ── Engine ────────────────────────────────────────────────────────────────────

class NarrativeConflictEngine:
    """Injects tension into every editorial plan.

    Public interface
    ─────────────────
    apply(editorial)
        → dict: adds conflict_type / conflict_line / conflict_intensity

    generate_conflict_line(conflict_type, topic, intensity, reference)
        → str: context-aware conflict phrase

    record(topic, conflict_type, intensity, line, video_id)
        → ConflictEntry: saves to conflict_memory.jsonl

    get_conflict_history(topic)
        → list[ConflictEntry]: past conflicts on this topic

    stats()
        → dict: type counts, avg intensity, dominant type
    """

    def __init__(self, data_dir: Path | None = None) -> None:
        self._memory = ConflictMemory(data_dir)

    # ── main entry point ──────────────────────────────────────────────────────

    def apply(self, editorial: dict[str, Any]) -> dict[str, Any]:
        """Enrich *editorial* with conflict layer.

        Adds:
            editorial["conflict_type"]       → "external" | "internal" | "temporal"
            editorial["conflict_intensity"]  → float 0.0–1.0
            editorial["conflict_line"]       → generated tension phrase for script
        """
        editorial = editorial.copy()

        conflict_type, intensity = self._select_conflict(editorial)
        topic     = editorial.get("topic") or editorial.get("hook") or ""
        reference = editorial.get("reference", "")

        line = self.generate_conflict_line(
            conflict_type, topic, intensity, reference=reference
        )

        editorial["conflict_type"]      = conflict_type
        editorial["conflict_intensity"] = round(intensity, 3)
        editorial["conflict_line"]      = line

        logger.debug(
            f"NarrativeConflict: type={conflict_type} intensity={intensity:.2f} "
            f"topic={topic[:40]!r}"
        )
        return editorial

    # ── conflict selection ────────────────────────────────────────────────────

    def _uncertainty_score(self, editorial: dict[str, Any]) -> float:
        """How uncertain should the character be? Higher = more internal conflict."""
        novelty      = editorial.get("novelty_score",      0.5)
        controversy  = editorial.get("controversy_score",  0.5)
        # Novel + low-controversy topics are genuinely uncertain
        return min(1.0, novelty * 0.5 + (1.0 - controversy) * 0.5)

    def _intensity_score(self, editorial: dict[str, Any]) -> float:
        """How strongly should the conflict land?"""
        controversy   = editorial.get("controversy_score", 0.5)
        belief        = editorial.get("belief_applied",    "")
        # Strong worldview activation → stronger conflict
        belief_boost  = 0.2 if belief in ("ai_hype", "ai_risk") else 0.05
        return min(1.0, max(0.1, controversy * 0.7 + belief_boost + 0.05))

    def _select_conflict(
        self, editorial: dict[str, Any]
    ) -> tuple[str, float]:
        """Choose conflict type and intensity based on editorial context."""
        has_past    = bool(editorial.get("reference"))
        uncertainty = self._uncertainty_score(editorial)
        intensity   = self._intensity_score(editorial)

        # Temporal: we have a past take and some reason to revisit it
        if has_past and uncertainty > 0.3:
            return "temporal", intensity

        # Internal: high uncertainty on a novel/complex topic
        if uncertainty > 0.65:
            return "internal", intensity * 0.75   # internal = slightly softer

        # External: default — the character vs. the world
        return "external", intensity

    # ── conflict line generation ──────────────────────────────────────────────

    def generate_conflict_line(
        self,
        conflict_type: str,
        topic: str,
        intensity: float,
        reference: str = "",
    ) -> str:
        """Generate a context-aware conflict phrase.

        For temporal conflicts with a past reference, weaves in the actual
        prior take: "I said: 'X.' I was wrong."
        """
        band = (
            "high"   if intensity > _HIGH_THRESHOLD else
            "medium" if intensity > _LOW_THRESHOLD  else
            "low"
        )
        templates = (
            _CONFLICT_TEMPLATES.get(conflict_type, {}).get(band)
            or _CONFLICT_TEMPLATES["external"]["medium"]
        )

        idx = hash(topic[:30]) % len(templates)
        base_line = templates[idx]

        # Temporal: personalise with the actual past take
        if conflict_type == "temporal" and reference:
            past_content = (
                reference.split(":", 1)[-1].strip()[:60].split(".")[0].strip()
            )
            if past_content:
                if band == "high":
                    return f"I said: '{past_content}.' I was wrong."
                if band == "medium":
                    return f"Last time I said: '{past_content}.' Here's what changed."

        return base_line

    # ── memory persistence ────────────────────────────────────────────────────

    def record(
        self,
        topic: str,
        conflict_type: str,
        intensity: float,
        line: str,
        video_id: str = "",
    ) -> ConflictEntry:
        """Persist one conflict entry after a video is published."""
        entry = self._memory.record(
            conflict_type=conflict_type,
            topic=topic,
            intensity=intensity,
            line=line,
            video_id=video_id,
        )
        logger.info(
            f"ConflictMemory: recorded [{conflict_type}] "
            f"intensity={intensity:.2f} topic={topic!r}"
        )
        return entry

    def get_conflict_history(self, topic: str) -> list[ConflictEntry]:
        """Return past conflict entries for *topic*."""
        return self._memory.get_topic_history(topic)

    def stats(self) -> dict[str, Any]:
        """Aggregate stats for dashboard display."""
        return self._memory.stats()


# ── Module-level singleton ────────────────────────────────────────────────────

_engine: NarrativeConflictEngine | None = None


def get_conflict_engine() -> NarrativeConflictEngine:
    """Return the module-level NarrativeConflictEngine singleton."""
    global _engine
    if _engine is None:
        _engine = NarrativeConflictEngine()
    return _engine
