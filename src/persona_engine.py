"""Persona Engine — the character layer that turns reporting into commentary.

The difference between a news bot and an AI influencer:
    news bot   → "OpenAI released GPT-5"
    influencer → "This looks impressive. But here's what nobody is saying."

This engine provides the character's voice, opinions, and memory.

Architecture:
    Editorial → Script → Humanizer (+PersonaEngine) → Presentation → Avatar → Video

Three things make a character:
    Voice       — HOW they say things (tone / style / energy)
    Opinion     — WHAT they actually think (bias / archetype)
    Consistency — that it's the SAME character every time (signatures / memory)

Archetype: skeptic_insider
    "I know what's happening behind the scenes — and it's not what they're saying."
    → Best fit for AI commentary: not a fanboy, not a doomer, a realist with receipts.

Content formats (four, not one):
    hot_take   — "Unpopular opinion: this is a step backward."   (highest retention)
    reaction   — "Just saw this. My take:"                       (fastest to produce)
    breakdown  — "Let me explain what's actually going on."      (tutorial energy)
    prediction — "Here's what happens next with AI agents."      (builds following)

Shorts opinion format (replaces generic news):
    Hook → Opinion ("But here's the thing...") → Twist → Signature kicker

Memory:
    Stores opinions, topics, predictions per video.
    Enables callbacks: "Last week I said X. And now..."
    File: data/persona_memory.jsonl
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import settings


# ── Archetype configuration ───────────────────────────────────────────────────

DEFAULT_PERSONA: dict[str, Any] = {
    "archetype":  "skeptic_insider",
    "tone":       "direct",
    "attitude":   "slightly skeptical",
    "energy":     "controlled",
    "style":      "short punchy sentences. One idea per line. Strong verbs.",
    "bias":       "calls_out_hype",
    "rules": [
        "Always find the angle the press release didn't mention",
        "Challenge the optimistic headline — but back it up",
        "Speak like you know more than you're saying",
        "Short sentences. Never more than two clauses.",
        "End with a question, a prediction, or a challenge",
        "Never hype without a 'but'",
    ],
}

# ── Signature phrases (brand → recognition) ───────────────────────────────────

SIGNATURES: dict[str, list[str]] = {
    "opener": [
        "Here's the thing nobody is saying.",
        "Wait — before you react to this.",
        "I've been watching this closely.",
        "This matters more than the headline.",
    ],
    "transition": [
        "But here's where it gets interesting.",
        "That's not even the real story.",
        "This is where it breaks.",
        "Stay with me on this.",
    ],
    "kicker": [
        "That's the real story.",
        "Remember this moment.",
        "Watch what happens next.",
        "Don't say I didn't warn you.",
    ],
}

# ── Opinion injection templates ────────────────────────────────────────────────

_OPINION_TEMPLATES: dict[str, list[str]] = {
    "calls_out_hype": [
        "This looks impressive. But here's what nobody is saying.",
        "Everyone's excited. Here's why they probably shouldn't be.",
        "Sounds like a breakthrough. Let me show you the other side.",
    ],
    "insider_take": [
        "I've been watching this. Here's what's actually happening.",
        "Behind the headline, something else is going on.",
        "What the announcement didn't mention:",
    ],
    "skeptic_push": [
        "This is where it breaks.",
        "The real question nobody is asking:",
        "Here's the problem with this narrative.",
    ],
}

# ── Content format detection ───────────────────────────────────────────────────

CONTENT_FORMATS: dict[str, dict[str, str]] = {
    "hot_take":   {
        "opener":      "Unpopular opinion:",
        "angle_bias":  "overhyped_vs_reality",
        "energy":      "high",
    },
    "reaction":   {
        "opener":      "Just saw this.",
        "angle_bias":  "industry_impact",
        "energy":      "fast",
    },
    "breakdown":  {
        "opener":      "Let me break this down.",
        "angle_bias":  "technical_breakthrough",
        "energy":      "measured",
    },
    "prediction": {
        "opener":      "Here's what happens next.",
        "angle_bias":  "what_this_means_for_you",
        "energy":      "confident",
    },
}

_FORMAT_SIGNALS: dict[str, set[str]] = {
    "hot_take":   {"wrong", "controversial", "unpopular", "actually", "nobody", "hype", "overhyped"},
    "prediction": {"will", "next", "future", "going to", "heading", "predict", "bet"},
    "reaction":   {"just", "breaking", "released", "announced", "launches", "reveals"},
    "breakdown":  {"how", "why", "explained", "works", "understand", "deep dive"},
}

# Angle score boosts for skeptic_insider archetype
_SKEPTIC_ANGLE_BOOST: dict[str, float] = {
    "overhyped_vs_reality":   0.25,   # primary bias — always seek the counter-narrative
    "threat_to_jobs":         0.10,   # insider knows what's really at stake
    "industry_impact":        0.05,
    "technical_breakthrough": 0.00,   # neutral — only if actually a breakthrough
    "what_this_means_for_you": 0.05,
}


# ── Memory dataclass ──────────────────────────────────────────────────────────

@dataclass
class MemoryEntry:
    """One opinion record: this character said X about Y on date D."""
    entry_type:  str           # opinion | prediction | hot_take | reaction
    topic:       str           # short topic phrase
    content:     str           # the actual opinion / prediction text
    video_id:    str = ""      # YouTube video_id (filled after upload)
    recorded_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "MemoryEntry":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# ── Memory store ──────────────────────────────────────────────────────────────

class PersonaMemory:
    """Append-only memory of past opinions and predictions.

    Enables the character to reference its own history:
        "Last week I said this would happen. And it did."
    """

    def __init__(self, data_dir: Path | None = None) -> None:
        self._dir  = data_dir or settings.data_dir
        self._path = self._dir / "persona_memory.jsonl"
        self._dir.mkdir(parents=True, exist_ok=True)

    def record(
        self,
        entry_type: str,
        topic: str,
        content: str,
        video_id: str = "",
    ) -> MemoryEntry:
        entry = MemoryEntry(
            entry_type=entry_type,
            topic=topic,
            content=content,
            video_id=video_id,
        )
        try:
            with open(self._path, "a") as f:
                f.write(json.dumps(entry.to_dict()) + "\n")
        except Exception as exc:
            logger.warning(f"PersonaMemory: write failed: {exc}")
        return entry

    def load_recent(self, n: int = 20) -> list[MemoryEntry]:
        if not self._path.exists():
            return []
        entries: list[MemoryEntry] = []
        for line in self._path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(MemoryEntry.from_dict(json.loads(line)))
            except Exception:
                pass
        return entries[-n:]

    def get_callbacks(self, topic: str, n: int = 2) -> list[str]:
        """Return callback phrases referencing past opinions on *topic*.

        E.g. "I said this before about {topic}: {opinion}"
        """
        recent = self.load_recent(n=50)
        topic_lower = topic.lower()
        matches = [
            e for e in recent
            if topic_lower in e.topic.lower() or
               any(word in e.topic.lower() for word in topic_lower.split()[:3])
        ]
        if not matches:
            return []

        callbacks = []
        for entry in matches[-n:]:
            if entry.entry_type == "prediction":
                callbacks.append(f"I predicted this: {entry.content}")
            else:
                callbacks.append(f"I said this before: {entry.content}")
        return callbacks


# ── Engine ────────────────────────────────────────────────────────────────────

class PersonaEngine:
    """The character layer: voice + opinion + memory.

    Public interface
    ────────────────
    detect_format(story_text)
        → "hot_take" | "reaction" | "breakdown" | "prediction"

    inject_opinion(text, hook_type="")
        → str: rule-based opinion transformation, no LLM

    get_signature(position, seed="")
        → str: rotating brand phrase at opener / transition / kicker positions

    get_callback(topic)
        → str: callback to past opinions ("I said this before...")

    opinionize_core(core_fact, hook_type)
        → str: transforms the Short's 2-6s core segment into an opinionated take

    as_humanizer_persona(story_text="")
        → dict: enriched persona dict for HumanizerAgent (rules + style + bias)

    bias_angle_scores(scores)
        → dict: amplifies skeptic-aligned angles by _SKEPTIC_ANGLE_BOOST

    record(topic, opinion, entry_type, video_id="")
        → MemoryEntry: saves to persona_memory.jsonl

    get_callbacks(topic)
        → list[str]: past opinions on this topic for "I told you" moments
    """

    def __init__(
        self,
        data_dir: Path | None = None,
        config: dict[str, Any] | None = None,
    ) -> None:
        self._config = config or DEFAULT_PERSONA
        self._memory = PersonaMemory(data_dir)

    # ── format detection ──────────────────────────────────────────────────────

    def detect_format(self, story_text: str) -> str:
        """Keyword-based format detection (like extract_features in HookPatternEngine)."""
        text = story_text.lower()
        best: str = "reaction"
        best_count = 0
        for fmt, signals in _FORMAT_SIGNALS.items():
            count = sum(1 for kw in signals if kw in text)
            if count > best_count:
                best_count = count
                best = fmt
        return best

    # ── opinion injection ─────────────────────────────────────────────────────

    def inject_opinion(self, text: str, hook_type: str = "") -> str:
        """Add the persona's opinion layer to *text*.

        Rule-based, no LLM. Fast enough to call inside the generation loop.
        The hook_type (curiosity/conflict/warning/mistake/insider) biases
        which opinion template is selected.
        """
        bias = self._config.get("bias", "calls_out_hype")

        if hook_type in ("conflict", "mistake"):
            templates = _OPINION_TEMPLATES["skeptic_push"]
        elif hook_type in ("insider",):
            templates = _OPINION_TEMPLATES["insider_take"]
        else:
            templates = _OPINION_TEMPLATES.get(bias, _OPINION_TEMPLATES["calls_out_hype"])

        idx = hash(text[:40]) % len(templates)
        opinion = templates[idx]

        return f"{opinion} {text}".strip()

    # ── signature phrases ─────────────────────────────────────────────────────

    def get_signature(self, position: str, seed: str = "") -> str:
        """Return a rotating brand phrase for the given position.

        position: "opener" | "transition" | "kicker"
        seed: used to vary the selection while staying deterministic
        """
        phrases = SIGNATURES.get(position, SIGNATURES["kicker"])
        idx = hash(seed or position) % len(phrases)
        return phrases[idx]

    # ── memory callbacks ──────────────────────────────────────────────────────

    def get_callback(self, topic: str) -> str:
        """Return a single callback string if there's a past opinion on topic."""
        callbacks = self._memory.get_callbacks(topic, n=1)
        return callbacks[0] if callbacks else ""

    # ── Shorts core transform ─────────────────────────────────────────────────

    def opinionize_core(self, core_fact: str, hook_type: str) -> str:
        """Transform the Short's 2-6s core segment into an opinionated take.

        Input:  "OpenAI just released GPT-5 with major reasoning upgrades."
        Output: "This looks impressive. But here's what nobody is saying.
                 OpenAI just released GPT-5 with major reasoning upgrades."
        """
        opinion_prefix = self.inject_opinion("", hook_type).strip()
        if not opinion_prefix:
            return core_fact
        return f"{opinion_prefix} {core_fact}".strip()

    # ── humanizer persona ─────────────────────────────────────────────────────

    def as_humanizer_persona(self, story_text: str = "") -> dict[str, Any]:
        """Return an enriched persona dict for HumanizerAgent.

        Merges the archetype config with story-context format detection
        so the humanizer GPT prompt gets the full character brief.
        """
        fmt = self.detect_format(story_text) if story_text else "reaction"
        fmt_config = CONTENT_FORMATS.get(fmt, CONTENT_FORMATS["reaction"])
        return {
            "style":     self._config["style"],
            "rules":     self._config["rules"],
            "archetype": self._config["archetype"],
            "attitude":  self._config["attitude"],
            "format":    fmt,
            "opener":    fmt_config["opener"],
            "bias":      self._config["bias"],
            "signature": self.get_signature("opener", seed=story_text[:20]),
        }

    # ── angle biasing ─────────────────────────────────────────────────────────

    def bias_angle_scores(
        self,
        scores: dict[str, float],
    ) -> dict[str, float]:
        """Amplify angles that match the skeptic_insider archetype.

        Call this after _angle_score() in EditorialBrain to tilt selection
        toward counter-narrative and insider angles.
        """
        biased: dict[str, float] = {}
        for angle, score in scores.items():
            boost = _SKEPTIC_ANGLE_BOOST.get(angle, 0.0)
            biased[angle] = round(score + boost, 4)
        return biased

    # ── memory ────────────────────────────────────────────────────────────────

    def record(
        self,
        topic: str,
        opinion: str,
        entry_type: str = "opinion",
        video_id: str = "",
    ) -> MemoryEntry:
        """Persist one opinion to persona_memory.jsonl."""
        entry = self._memory.record(
            entry_type=entry_type,
            topic=topic,
            content=opinion,
            video_id=video_id,
        )
        logger.info(f"PersonaMemory: recorded [{entry_type}] for {topic!r}")
        return entry

    def get_callbacks(self, topic: str) -> list[str]:
        """Return past-opinion callback strings for *topic*."""
        return self._memory.get_callbacks(topic)


# ── Module-level singleton ────────────────────────────────────────────────────

_engine: PersonaEngine | None = None


def get_default_engine() -> PersonaEngine:
    """Return the module-level PersonaEngine singleton."""
    global _engine
    if _engine is None:
        _engine = PersonaEngine()
    return _engine
