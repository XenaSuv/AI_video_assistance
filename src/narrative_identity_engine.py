"""Narrative Identity Engine — temporal dimension of the channel's personality.

The difference between a content machine and an influencer:
    content machine → video → video → video
    influencer       → character → development → recurring themes → story

"Narrative identity = who this channel is over time."

Three things build narrative identity:
    Beliefs  — the channel has a fixed worldview it applies to everything
    Themes   — certain topics keep coming back (recognizable recurring motifs)
    Memory   — past takes, predictions, and pattern references ("I said this...")

How it works in the pipeline:
    EditorialBrain._plan_story()
        → NarrativeIdentityEngine.apply(editorial, topic)
        → enriches the editorial dict with:
            theme         — recurring theme this story belongs to
            arc_key       — long-term narrative arc being advanced
            arc_phrase    — "This is another example of what we've been seeing..."
            reference     — past memory callback for the script generator
            callback_phrase — "Here we go again." / "I called this."
            belief_applied — which core belief shaped the angle

Architecture:
    BELIEFS   → static worldview config
    THEMES    → list of recurring topic motifs
    ARCS      → long-term storylines (accumulate episodes over time)
    CALLBACKS → signature callback phrases
    NarrativeMemory → append-only JSONL store (narrative_memory.jsonl)
    NarrativeIdentityEngine → applies all layers to an editorial dict
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


# ── Core Beliefs (the channel's fixed worldview) ──────────────────────────────

BELIEFS: dict[str, str] = {
    "ai_hype":    "overestimated",   # AI announcements are mostly overblown
    "ai_risk":    "underestimated",  # Real dangers go underreported
    "tools":      "overused_wrong",  # People use AI tools incorrectly
    "future":     "uncertain",       # Nobody actually knows what's next
    "incumbents": "threatened",      # Big players are more fragile than they appear
    "open_source": "underrated",     # Open models outperform their press coverage
}

# Belief value → angle key override (when belief fires, suggest this angle)
_BELIEF_ANGLE_MAP: dict[str, str] = {
    "overestimated":   "overhyped_vs_reality",
    "underestimated":  "threat_to_jobs",
    "overused_wrong":  "what_this_means_for_you",
    "uncertain":       "overhyped_vs_reality",
    "threatened":      "industry_impact",
    "underrated":      "technical_breakthrough",
}

# Keyword → belief key (first match wins)
_BELIEF_SIGNALS: list[tuple[str, str]] = [
    ("breakthrough",        "ai_hype"),
    ("revolutionary",       "ai_hype"),
    ("changes everything",  "ai_hype"),
    ("outperforms",         "ai_hype"),
    ("state of the art",    "ai_hype"),
    ("gpt-",                "ai_hype"),
    ("replace",             "ai_risk"),
    ("layoff",              "ai_risk"),
    ("alignment",           "ai_risk"),
    ("safety concern",      "ai_risk"),
    ("hallucination",       "tools"),
    ("misuse",              "tools"),
    ("mistake",             "tools"),
    ("limitation",          "tools"),
    ("actually",            "tools"),
    ("open source",         "open_source"),
    ("open-source",         "open_source"),
    ("llama",               "open_source"),
    ("mistral",             "open_source"),
    ("acquire",             "incumbents"),
    ("merge",               "incumbents"),
    ("shutdown",            "incumbents"),
    ("pivot",               "incumbents"),
]


# ── Recurring Themes ──────────────────────────────────────────────────────────

THEMES: list[str] = [
    "hype_vs_reality",
    "tools_misuse",
    "hidden_limits",
    "future_uncertainty",
    "industry_consolidation",
    "open_vs_closed",
]

_THEME_SIGNALS: dict[str, set[str]] = {
    "hype_vs_reality": {
        "hype", "overhyped", "claimed", "promised", "revolutionary",
        "breakthrough", "game-changer", "changes everything",
    },
    "tools_misuse": {
        "hallucination", "wrong", "misuse", "mistake", "failed",
        "error", "limitation", "actually doesn't", "not ready",
    },
    "hidden_limits": {
        "limit", "ceiling", "plateau", "benchmark", "performance",
        "real-world", "actually", "in practice", "but",
    },
    "future_uncertainty": {
        "future", "predict", "2025", "2026", "next year",
        "will", "what happens", "where this goes",
    },
    "industry_consolidation": {
        "acquire", "merge", "deal", "partnership",
        "microsoft", "google", "amazon", "nvidia",
    },
    "open_vs_closed": {
        "open source", "open-source", "meta", "llama",
        "mistral", "closed", "proprietary", "api-only",
    },
}


# ── Narrative Arcs (long-term storylines) ─────────────────────────────────────

ARCS: dict[str, dict[str, Any]] = {
    "benchmarks_lie": {
        "label":    "Benchmarks Don't Tell the Whole Story",
        "keywords": {"benchmark", "evaluation", "test score", "leaderboard", "sota", "performance"},
        "status":   "developing",
        "summary":  "Every model claims SOTA. Real-world performance tells a different story.",
        "phrase":   "This is another chapter in the benchmarks story.",
        "episodes": [],
    },
    "ai_hype_cycle": {
        "label":    "The AI Hype Cycle",
        "keywords": {"breakthrough", "revolutionary", "changes everything", "release", "new model"},
        "status":   "developing",
        "summary":  "Each announcement promises more than it delivers.",
        "phrase":   "And here's another example of what we've been tracking.",
        "episodes": [],
    },
    "job_displacement": {
        "label":    "What AI Is Actually Doing to Jobs",
        "keywords": {"jobs", "replace", "automation", "workers", "hiring", "layoff", "displacement"},
        "status":   "developing",
        "summary":  "The real employment story is more complicated than headlines suggest.",
        "phrase":   "This connects to the bigger employment story we've been following.",
        "episodes": [],
    },
    "open_closed_war": {
        "label":    "Open Source vs Closed Models",
        "keywords": {"open source", "open-source", "meta", "mistral", "llama", "closed", "proprietary"},
        "status":   "developing",
        "summary":  "The battle between open and closed AI is reshaping the industry.",
        "phrase":   "This is the open vs closed story playing out again.",
        "episodes": [],
    },
    "big_tech_control": {
        "label":    "Who Controls AI",
        "keywords": {"microsoft", "google", "amazon", "nvidia", "acquire", "invest", "control", "monopoly"},
        "status":   "developing",
        "summary":  "Power in AI is concentrating faster than anyone is reporting.",
        "phrase":   "This is part of the consolidation pattern we keep seeing.",
        "episodes": [],
    },
}


# ── Signature Callbacks ───────────────────────────────────────────────────────

CALLBACKS: list[str] = [
    "And this is exactly what I warned about.",
    "Here we go again.",
    "Remember what I said? This is it.",
    "I've been watching this for months.",
    "Nobody wanted to hear this. But here it is.",
    "I called this. And now it's happening.",
    "This is the pattern. Watch it repeat.",
    "We talked about this before. The story continues.",
]


# ── Memory dataclass ──────────────────────────────────────────────────────────

@dataclass
class NarrativeEntry:
    """One narrative record: topic + theme + arc + content + context."""
    entry_type:  str           # opinion | prediction | arc_episode | callback_used
    topic:       str
    content:     str
    theme:       str = ""
    arc_key:     str = ""
    video_id:    str = ""
    recorded_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "NarrativeEntry":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# ── Memory store ──────────────────────────────────────────────────────────────

class NarrativeMemory:
    """Append-only store of the channel's narrative history.

    Tracks which arcs have been advanced, which topics recur, and what
    predictions were made — enabling "I've been covering this for months."
    """

    def __init__(self, data_dir: Path | None = None) -> None:
        self._dir  = data_dir or settings.data_dir
        self._path = self._dir / "narrative_memory.jsonl"
        self._dir.mkdir(parents=True, exist_ok=True)

    def record(
        self,
        entry_type: str,
        topic: str,
        content: str,
        theme: str = "",
        arc_key: str = "",
        video_id: str = "",
    ) -> NarrativeEntry:
        entry = NarrativeEntry(
            entry_type=entry_type,
            topic=topic,
            content=content,
            theme=theme,
            arc_key=arc_key,
            video_id=video_id,
        )
        try:
            with open(self._path, "a") as f:
                f.write(json.dumps(entry.to_dict()) + "\n")
        except Exception as exc:
            logger.warning(f"NarrativeMemory: write failed: {exc}")
        return entry

    def load_recent(self, n: int = 50) -> list[NarrativeEntry]:
        if not self._path.exists():
            return []
        entries: list[NarrativeEntry] = []
        for line in self._path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(NarrativeEntry.from_dict(json.loads(line)))
            except Exception:
                pass
        return entries[-n:]

    def get_relevant(self, topic: str, n: int = 2) -> list[NarrativeEntry]:
        """Return entries relevant to *topic* for memory callbacks."""
        recent = self.load_recent(n=100)
        topic_lower = topic.lower()
        words = topic_lower.split()[:4]
        matches = [
            e for e in recent
            if topic_lower in e.topic.lower()
            or any(w in e.topic.lower() for w in words if len(w) > 3)
        ]
        return matches[-n:]

    def get_arc_episodes(self, arc_key: str, n: int = 3) -> list[NarrativeEntry]:
        """Return the most recent episodes for a given arc."""
        recent = self.load_recent(n=100)
        return [e for e in recent if e.arc_key == arc_key][-n:]

    def topic_count(self, topic: str) -> int:
        """How many times has this topic appeared in memory?"""
        topic_lower = topic.lower()
        return sum(1 for e in self.load_recent(n=100) if topic_lower in e.topic.lower())

    def arc_episode_count(self, arc_key: str) -> int:
        return len(self.get_arc_episodes(arc_key, n=100))


# ── Engine ────────────────────────────────────────────────────────────────────

class NarrativeIdentityEngine:
    """Injects temporal narrative identity into each editorial plan.

    Transforms isolated per-video plans into chapters of an ongoing story:
        beliefs  → the angle is never neutral
        themes   → videos start to rhyme
        arcs     → viewers begin to follow a developing storyline
        memory   → "I said this before" callbacks build trust

    Public interface
    ─────────────────
    apply(editorial, topic)
        → dict: enriched editorial with theme / arc / reference / callback

    record(topic, content, entry_type, theme, arc_key, video_id)
        → NarrativeEntry: persists to narrative_memory.jsonl

    get_signature_callback(topic, has_prediction)
        → str: context-aware callback phrase

    arc_status()
        → dict: current episode counts per arc
    """

    def __init__(self, data_dir: Path | None = None) -> None:
        self._memory = NarrativeMemory(data_dir)

    # ── main entry point ──────────────────────────────────────────────────────

    def apply(self, editorial: dict[str, Any], topic: str) -> dict[str, Any]:
        """Enrich *editorial* with narrative identity layers.

        Modifies (non-destructively):
            editorial["theme"]           → recurring theme key
            editorial["arc_key"]         → narrative arc this advances
            editorial["arc_label"]       → human-readable arc name
            editorial["arc_phrase"]      → "This is another example of..."
            editorial["belief_applied"]  → which core belief fired
            editorial["reference"]       → past-memory callback text
            editorial["callback_phrase"] → signature callback phrase
        """
        editorial = editorial.copy()
        if not editorial.get("topic"):
            editorial["topic"] = topic

        editorial = self._inject_belief(editorial)
        editorial = self._inject_theme(editorial)
        editorial = self._inject_arc(editorial, topic)
        editorial = self._inject_memory(editorial, topic)

        return editorial

    # ── belief injection ──────────────────────────────────────────────────────

    def _inject_belief(self, editorial: dict[str, Any]) -> dict[str, Any]:
        """Apply worldview to angle selection — the channel is never neutral."""
        topic = (editorial.get("topic") or "").lower()
        hook  = (editorial.get("hook")  or "").lower()
        text  = topic + " " + hook

        for signal, belief_key in _BELIEF_SIGNALS:
            if signal in text:
                belief_val = BELIEFS.get(belief_key, "")
                angle_override = _BELIEF_ANGLE_MAP.get(belief_val, "")
                editorial["belief_applied"] = belief_key
                # Only override angle if not already set by EditorialBrain
                if angle_override and not editorial.get("belief_angle"):
                    editorial["belief_angle"] = angle_override
                    logger.debug(
                        f"NarrativeIdentity: belief '{belief_key}' → angle '{angle_override}'"
                    )
                break

        return editorial

    # ── theme injection ───────────────────────────────────────────────────────

    def _inject_theme(self, editorial: dict[str, Any]) -> dict[str, Any]:
        """Tag the editorial with the recurring theme it belongs to."""
        topic = (editorial.get("topic") or "").lower()
        hook  = (editorial.get("hook")  or "").lower()
        text  = topic + " " + hook

        best_theme = ""
        best_count = 0
        for theme, signals in _THEME_SIGNALS.items():
            count = sum(1 for s in signals if s in text)
            if count > best_count:
                best_count = count
                best_theme = theme

        if not best_theme:
            # Fallback: use the theme we've covered least recently (keep variety)
            recent = self._memory.load_recent(n=20)
            recent_themes = [e.theme for e in recent if e.theme]
            for theme in THEMES:
                if theme not in recent_themes:
                    best_theme = theme
                    break
            best_theme = best_theme or THEMES[0]

        editorial["theme"] = best_theme
        return editorial

    # ── arc injection ─────────────────────────────────────────────────────────

    def _inject_arc(self, editorial: dict[str, Any], topic: str) -> dict[str, Any]:
        """Match the story to a long-term narrative arc."""
        text = (topic + " " + (editorial.get("hook") or "")).lower()

        for arc_key, arc in ARCS.items():
            if any(kw in text for kw in arc["keywords"]):
                editorial["arc_key"]   = arc_key
                editorial["arc_label"] = arc["label"]
                editorial["arc_phrase"] = arc["phrase"]
                episode_count = self._memory.arc_episode_count(arc_key)
                if episode_count >= 2:
                    editorial["arc_episode_count"] = episode_count
                logger.debug(
                    f"NarrativeIdentity: arc '{arc_key}' matched (episode #{episode_count + 1})"
                )
                break

        return editorial

    # ── memory injection ──────────────────────────────────────────────────────

    def _inject_memory(self, editorial: dict[str, Any], topic: str) -> dict[str, Any]:
        """Surface past opinions and predictions as reference material."""
        relevant = self._memory.get_relevant(topic, n=2)
        if not relevant:
            return editorial

        most_recent = relevant[-1]

        # Build reference text for script generator
        if most_recent.entry_type == "prediction":
            ref = f"I predicted this: {most_recent.content}"
        else:
            ref = f"We covered this before: {most_recent.content}"
        editorial["reference"] = ref

        # Build signature callback
        has_prediction = any(e.entry_type == "prediction" for e in relevant)
        phrase = self.get_signature_callback(topic, has_prediction=has_prediction)
        if phrase:
            editorial["callback_phrase"] = phrase

        return editorial

    # ── signature callback ────────────────────────────────────────────────────

    def get_signature_callback(self, topic: str, has_prediction: bool = False) -> str:
        """Return a context-aware signature callback phrase.

        Prioritises:
          1. Prediction confirmation → "Remember what I said? This is it."
          2. Recurring topic (≥2 appearances) → "Here we go again."
          3. Single prior appearance → hash-rotated CALLBACKS phrase
          4. No history → empty string
        """
        if has_prediction:
            return "Remember what I said? This is it."

        count = self._memory.topic_count(topic)
        if count >= 2:
            return "Here we go again."
        if count == 1:
            idx = hash(topic[:30]) % len(CALLBACKS)
            return CALLBACKS[idx]
        return ""

    # ── memory persistence ────────────────────────────────────────────────────

    def record(
        self,
        topic: str,
        content: str,
        entry_type: str = "opinion",
        theme: str = "",
        arc_key: str = "",
        video_id: str = "",
    ) -> NarrativeEntry:
        """Persist one narrative entry after a video is published."""
        entry = self._memory.record(
            entry_type=entry_type,
            topic=topic,
            content=content,
            theme=theme,
            arc_key=arc_key,
            video_id=video_id,
        )
        logger.info(
            f"NarrativeMemory: recorded [{entry_type}] topic={topic!r} "
            f"theme={theme!r} arc={arc_key!r}"
        )
        return entry

    # ── arc status ────────────────────────────────────────────────────────────

    def arc_status(self) -> dict[str, dict[str, Any]]:
        """Return current episode count and status for all arcs."""
        result = {}
        for arc_key, arc in ARCS.items():
            episodes = self._memory.get_arc_episodes(arc_key, n=100)
            result[arc_key] = {
                "label":         arc["label"],
                "status":        arc["status"],
                "episode_count": len(episodes),
                "summary":       arc["summary"],
                "last_episode":  episodes[-1].recorded_at if episodes else None,
            }
        return result


# ── Module-level singleton ────────────────────────────────────────────────────

_engine: NarrativeIdentityEngine | None = None


def get_narrative_engine() -> NarrativeIdentityEngine:
    """Return the module-level NarrativeIdentityEngine singleton."""
    global _engine
    if _engine is None:
        _engine = NarrativeIdentityEngine()
    return _engine
