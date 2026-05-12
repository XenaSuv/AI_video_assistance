"""Editorial Identity — the channel's immutable editorial DNA.

Defines what the channel believes, what it prioritises, and what it avoids.
This replaces GPT randomness with a deterministic voice.

The identity can be evolved via feedback (update_priorities / update_from_feedback),
and is persisted to data/editorial_identity.json so learning survives restarts.
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from config import settings


# ── Default identity ──────────────────────────────────────────────────────────

EDITORIAL_IDENTITY: dict[str, Any] = {
    "voice": "direct",         # how we speak: direct | measured | provocative
    "tone": "skeptical",       # emotional register: skeptical | urgent | calm
    "bias": "anti-hype",       # permanent editorial lean: anti-hype | pro-safety | neutral

    # What we care about most — ordered by priority. Feedback can reorder these.
    "priorities": [
        "hidden_risks",
        "misconceptions",
        "real_world_impact",
    ],

    # Story types we don't touch
    "avoid": [
        "generic_news",
        "press_release_style",
    ],

    # Signature opening structures that define the channel's rhetorical style
    "signature_patterns": [
        "everyone_thinks_x_but",
        "this_looks_good_but",
        "nobody_talks_about",
    ],
}


# ── Identity dataclass ────────────────────────────────────────────────────────

@dataclass
class EditorialIdentity:
    """Live editorial identity — loaded from config, updatable via feedback."""

    voice:              str        = "direct"
    tone:               str        = "skeptical"
    bias:               str        = "anti-hype"
    priorities:         list[str]  = field(default_factory=lambda: [
        "hidden_risks", "misconceptions", "real_world_impact"
    ])
    avoid:              list[str]  = field(default_factory=lambda: [
        "generic_news", "press_release_style"
    ])
    signature_patterns: list[str]  = field(default_factory=lambda: [
        "everyone_thinks_x_but", "this_looks_good_but", "nobody_talks_about"
    ])

    # ── serialisation ─────────────────────────────────────────────────────────

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "EditorialIdentity":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    @classmethod
    def default(cls) -> "EditorialIdentity":
        return cls.from_dict(EDITORIAL_IDENTITY)

    # ── persistence ───────────────────────────────────────────────────────────

    @classmethod
    def load(cls, data_dir: Path | None = None) -> "EditorialIdentity":
        """Load from data/editorial_identity.json, falling back to defaults."""
        path = (data_dir or settings.data_dir) / "editorial_identity.json"
        if path.exists():
            try:
                return cls.from_dict(json.loads(path.read_text()))
            except Exception as exc:
                logger.warning(f"EditorialIdentity: load failed ({exc}), using defaults")
        return cls.default()

    def save(self, data_dir: Path | None = None) -> None:
        """Persist to data/editorial_identity.json."""
        path = (data_dir or settings.data_dir) / "editorial_identity.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2))

    # ── feedback-driven updates ───────────────────────────────────────────────

    def promote_priority(self, priority: str) -> None:
        """Move a priority to the front of the list (feedback reinforcement)."""
        if priority in self.priorities:
            self.priorities.remove(priority)
        self.priorities.insert(0, priority)

    def add_priority(self, priority: str) -> None:
        """Append a new priority if not already present."""
        if priority not in self.priorities:
            self.priorities.append(priority)

    def top_priority(self) -> str:
        """Return the current highest-priority editorial goal."""
        return self.priorities[0] if self.priorities else "real_world_impact"
