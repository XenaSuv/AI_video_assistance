"""Shorts Engine V2 — Shorts as first-class citizens, not pipeline byproducts.

Core philosophy change:
    OLD: long video → trim → Short
    NEW: Short = atomic unit of attention, built from scratch

Every Short is one idea delivered in 25–30 seconds:
    0–2s   HOOK      — single sentence that stops the scroll
    2–6s   CONTEXT   — why this matters (one sentence)
    6–15s  DEVELOP   — the actual content
    15–25s PAYOFF    — the reveal / answer / twist
    25–30s LOOP      — cliffhanger that makes them rewatch

Visual style: fast text cuts every 1–2s (no Ken Burns, no DALL-E).
Avatar (HeyGen) is the primary visual when available.

Hook taxonomy (what actually retains past 3 seconds):
    wait_what       — unexpected fact / contradiction
    conflict        — tension, failure, battle
    personal_trigger — "you're doing X wrong"
    negative        — counter-intuitive contrarian take
    curiosity       — secret / hidden knowledge

Primary metric: retention_3s  (not views, not likes)
"""
from __future__ import annotations

import json
import sys
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import settings


# ── Hook templates ────────────────────────────────────────────────────────────

_HOOK_PATTERNS: dict[str, list[str]] = {
    "wait_what": [
        "{topic} just did something nobody expected",
        "Wait — {topic} lied and nobody noticed",
        "This {topic} fact changes everything",
        "{topic} made a mistake nobody is talking about",
    ],
    "conflict": [
        "AI is getting worse, not better — here's the proof",
        "{topic} is under attack and here's what's really happening",
        "This is the problem with {topic} nobody admits",
        "{topic} vs reality: someone is lying",
    ],
    "personal_trigger": [
        "You're using {topic} completely wrong",
        "You're misunderstanding {topic} — and it's costing you",
        "Stop using {topic} like this",
        "Everyone is missing this about {topic}",
    ],
    "negative": [
        "{topic} might actually fail — here's why",
        "Why {topic} is more overhyped than you think",
        "The hidden downside of {topic} experts won't tell you",
        "{topic} isn't what they're selling you",
    ],
    "curiosity": [
        "Nobody is talking about this in {topic}",
        "The secret behind {topic} that changes how you see it",
        "This is what {topic} actually looks like behind the scenes",
        "What they don't show you about {topic}",
    ],
}

_LOOP_ENDINGS = [
    "…and that's where it gets scary.",
    "…but here's the part nobody tells you.",
    "…watch this again and you'll see it.",
    "…this is just the beginning.",
    "…and it's already happening to you.",
]


# ── Dataclasses ───────────────────────────────────────────────────────────────

@dataclass
class ShortScript:
    """One standalone Short: 4 segments, single idea, ≤30 seconds."""
    short_id:       str
    topic:          str
    hook_type:      str
    hook:           str           # 0–2s spoken text
    context:        str           # 2–6s
    develop:        str           # 6–15s
    payoff:         str           # 15–25s
    loop:           str           # 25–30s cliffhanger
    style:          str = "fast_cut"
    use_avatar:     bool = True
    created_at:     str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    @property
    def full_narration(self) -> str:
        return " ".join([self.hook, self.context, self.develop, self.payoff, self.loop])

    @property
    def visual_text(self) -> str:
        """Large on-screen text — the hook, shown for the first 3 seconds."""
        return self.hook

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ShortScript":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class ShortResult:
    """Production output: rendered MP4 + upload state."""
    short_id:   str
    video_path: str = ""
    video_id:   str = ""          # YouTube video_id after upload
    uploaded:   bool = False
    error:      str = ""
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ── Script builder ────────────────────────────────────────────────────────────

def _extract_topic(story: dict[str, Any]) -> str:
    title = story.get("title", "AI")
    words = title.split()
    return " ".join(words[:4]) if words else "AI"


def _pick_hook(topic: str, hook_type: str, seed: str) -> str:
    templates = _HOOK_PATTERNS[hook_type]
    idx = hash(seed) % len(templates)
    return templates[idx].format(topic=topic)


def _pick_loop() -> str:
    return _LOOP_ENDINGS[0]   # deterministic; ShortsEngineV2 cycles them


def _build_script_from_story(
    story: dict[str, Any],
    hook_type: str,
    hook: str,
    loop_idx: int = 0,
) -> ShortScript:
    """Build a 4-segment Short script from story metadata.

    Keeps segments short and punchy — no filler, no transitions.
    Content is derived from the story title + summary without GPT
    so the pipeline works offline and costs nothing extra.
    """
    title   = story.get("title", "AI update")
    summary = story.get("summary", story.get("description", title))
    source  = story.get("source", "")

    # Trim summary to a single impactful sentence
    first_sentence = summary.split(".")[0].strip()
    if len(first_sentence) > 140:
        first_sentence = first_sentence[:137] + "…"

    source_tag = f" — via {source}" if source else ""
    loop = _LOOP_ENDINGS[loop_idx % len(_LOOP_ENDINGS)]

    return ShortScript(
        short_id  = str(uuid.uuid4()),
        topic     = _extract_topic(story),
        hook_type = hook_type,
        hook      = hook,
        context   = first_sentence + source_tag + ".",
        develop   = (
            f"Here's what's actually happening: {first_sentence.lower()}. "
            f"And this is bigger than it looks."
        ),
        payoff    = (
            f"The real implication? {title}. "
            f"This shifts how we should think about AI entirely."
        ),
        loop      = loop,
        use_avatar = settings.heygen_enabled,
    )


# ── Engine ────────────────────────────────────────────────────────────────────

class ShortsEngineV2:
    """Build 5–10 standalone Shorts from a single story.

    Each Short tests a different hook type — the whole batch is an experiment.
    The ShortsExperimentEngine tracks which hook types win on retention_3s and
    feeds those patterns back into the daily pipeline's hook selection.
    """

    def __init__(self, data_dir: Path | None = None) -> None:
        self._data_dir = data_dir or settings.data_dir
        self._scripts_path = self._data_dir / "shorts_scripts_v2.jsonl"
        self._data_dir.mkdir(parents=True, exist_ok=True)

    # ── generation ────────────────────────────────────────────────────────────

    def generate(self, story: dict[str, Any], max_shorts: int = 5) -> list[ShortScript]:
        """Return up to *max_shorts* ShortScript objects for *story*.

        Generates one variant per hook type family, so the batch covers the
        full hypothesis space in one pass.
        """
        topic      = _extract_topic(story)
        title      = story.get("title", topic)
        hook_types = list(_HOOK_PATTERNS.keys())[:max_shorts]
        scripts: list[ShortScript] = []

        for i, hook_type in enumerate(hook_types):
            hook   = _pick_hook(topic, hook_type, seed=title + hook_type)
            script = _build_script_from_story(story, hook_type, hook, loop_idx=i)
            scripts.append(script)
            self._persist(script)
            logger.info(
                f"ShortsV2 [{hook_type}] → {hook!r}"
            )

        return scripts

    def generate_hooks(self, topic: str, n: int = 10) -> list[dict[str, str]]:
        """Return raw hook variants for *topic* across all types.

        Used by the daily pipeline to inject proven hook patterns into
        editorial_brain's hook_candidates list.
        """
        hooks: list[dict[str, str]] = []
        for hook_type, templates in _HOOK_PATTERNS.items():
            for tmpl in templates[:max(1, n // len(_HOOK_PATTERNS))]:
                hooks.append({
                    "text":      tmpl.format(topic=topic),
                    "hook_type": hook_type,
                })
        return hooks[:n]

    # ── persistence ───────────────────────────────────────────────────────────

    def _persist(self, script: ShortScript) -> None:
        try:
            with open(self._scripts_path, "a") as f:
                f.write(json.dumps(script.to_dict()) + "\n")
        except Exception as exc:
            logger.warning(f"ShortsEngineV2: persist failed: {exc}")

    def load_scripts(self) -> list[ShortScript]:
        if not self._scripts_path.exists():
            return []
        scripts = []
        for line in self._scripts_path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                scripts.append(ShortScript.from_dict(json.loads(line)))
            except Exception:
                pass
        return scripts
