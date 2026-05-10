"""Shorts Engine V2 — experiment system, not a video generator.

Flow:
    editorial (topic + core_fact + angle)
        ↓
    _generate_hooks()       → 5 typed hook hypotheses
        ↓
    _build_variants()       → ShortScript per hook (style + persona assigned)
        ↓
    publish                 → video_id per variant
        ↓
    learn(short, metrics)   → retention_3s stored by {hook_type, style, persona}
        ↓
    get_top_hooks()         → best combinations → feed editorial_brain

Why this matters:
    Shorts don't just publish content — they run controlled experiments on hook
    effectiveness.  Every Short is one data point.  After 10–20 Shorts the
    system knows which hook types, visual styles, and delivery personas retain
    past 3 seconds.  That knowledge flows back into the long-form pipeline,
    making every daily video's hook better than the last.

Hook taxonomy (5 types, each tests a different psychological trigger):
    conflict   — tension / battle / failure ("This is the problem with X")
    curiosity  — hidden knowledge ("Nobody is talking about X")
    mistake    — personal relevance ("You're using X wrong")
    warning    — urgency / risk ("This might actually fail")
    insider    — exclusive access ("What insiders know about X")

Visual styles (matched to hook psychology):
    fast_cut     — rapid cuts every 1-2s; conflict / mistake hooks
    zoom_text    — text zooms in on key words; curiosity / insider hooks
    avatar_focus — avatar fills frame; warning hooks (authority signal)

Personas (delivery energy matched to message tone):
    direct   — straight to the point; conflict / mistake hooks
    serious  — calm authority; warning hooks
    curious  — inviting, questioning; curiosity / insider hooks

Primary metric: retention_3s   (everything else is noise until this is good)
Learning store: data/shorts_learning_v2.jsonl
"""
from __future__ import annotations

import json
import sys
import uuid
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import settings


# ── Constants ─────────────────────────────────────────────────────────────────

HOOK_TYPES = ["conflict", "curiosity", "mistake", "warning", "insider"]

_HOOK_TEMPLATES: dict[str, list[str]] = {
    "conflict": [
        "This is the problem with {topic} nobody admits",
        "{topic} vs reality: someone is lying",
        "{topic} is under attack — here's what's really happening",
        "AI is getting worse, not better — here's the proof",
    ],
    "curiosity": [
        "Nobody is talking about {topic}",
        "What insiders won't tell you about {topic}",
        "This is what {topic} actually looks like behind the scenes",
        "The secret behind {topic} that changes everything",
    ],
    "mistake": [
        "You're using {topic} completely wrong",
        "Stop using {topic} like this",
        "Everyone is missing this about {topic}",
        "You're misunderstanding {topic} — and it's costing you",
    ],
    "warning": [
        "This might actually fail — here's why",
        "{topic} is heading in a dangerous direction",
        "Warning: {topic} just crossed a line",
        "This {topic} decision could backfire badly",
    ],
    "insider": [
        "What insiders know about {topic}",
        "Here's what they don't show you about {topic}",
        "The {topic} detail experts are whispering about",
        "Behind the scenes: what actually happened with {topic}",
    ],
}

_STYLE_MAP: dict[str, str] = {
    "conflict": "fast_cut",
    "curiosity": "zoom_text",
    "mistake":   "fast_cut",
    "warning":   "avatar_focus",
    "insider":   "zoom_text",
}

_PERSONA_MAP: dict[str, str] = {
    "conflict": "direct",
    "curiosity": "curious",
    "mistake":   "direct",
    "warning":   "serious",
    "insider":   "curious",
}

_LOOP_ENDINGS: list[str] = [
    "…and that's where it gets scary.",
    "…but here's the part nobody tells you.",
    "…watch this again and you'll see it.",
    "…this is just the beginning.",
    "…and it's already happening to you.",
]

# Minimum retention_3s to be considered a winner
_WINNER_THRESHOLD = 0.65


# ── Dataclasses ───────────────────────────────────────────────────────────────

@dataclass
class ShortScript:
    """One standalone Short — atomic unit of attention, 25-30 seconds."""
    short_id:   str
    topic:      str
    hook_type:  str
    hook:       str        # 0–2s: scroll-stopper
    core:       str        # 2–6s: one-sentence context
    twist:      str        # 6–15s: the actual substance
    ending:     str        # 15–30s: payoff + loop cliffhanger
    style:      str        # fast_cut | zoom_text | avatar_focus
    persona:    str        # direct | serious | curious
    use_avatar: bool       = True
    created_at: str        = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    @property
    def full_narration(self) -> str:
        return " ".join([self.hook, self.core, self.twist, self.ending])

    @property
    def visual_text(self) -> str:
        return self.hook

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ShortScript":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class ShortResult:
    """Production output — rendered MP4 + upload state."""
    short_id:   str
    video_path: str  = ""
    video_id:   str  = ""
    uploaded:   bool = False
    error:      str  = ""
    created_at: str  = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class LearningRecord:
    """One data point: which combination produced which retention_3s."""
    short_id:     str
    hook_type:    str
    style:        str
    persona:      str
    topic:        str
    retention_3s: float
    avg_watch:    float = 0.0
    completion:   float = 0.0
    recorded_at:  str   = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "LearningRecord":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# ── Helpers ───────────────────────────────────────────────────────────────────

def _extract_topic(source: dict[str, Any]) -> str:
    """Return a short topic phrase from story or editorial dict."""
    title = source.get("topic") or source.get("title", "AI")
    words = title.split()
    return " ".join(words[:4]) if words else "AI"


def _pick_template(hook_type: str, seed: str) -> str:
    templates = _HOOK_TEMPLATES[hook_type]
    return templates[hash(seed) % len(templates)]


def _build_script_from_editorial(
    editorial: dict[str, Any],
    hook_type: str,
    hook_text: str,
    loop_idx: int = 0,
) -> ShortScript:
    topic     = editorial.get("topic", "AI")
    core_fact = editorial.get("core_fact", topic)
    angle     = editorial.get("angle", "")

    first_sentence = core_fact.split(".")[0].strip()
    if len(first_sentence) > 120:
        first_sentence = first_sentence[:117] + "…"

    loop    = _LOOP_ENDINGS[loop_idx % len(_LOOP_ENDINGS)]
    style   = _STYLE_MAP[hook_type]
    persona = _PERSONA_MAP[hook_type]

    twist = (
        f"Here's what's actually happening: {first_sentence.lower()}."
        if not angle
        else angle.rstrip(".") + "."
    )

    return ShortScript(
        short_id   = str(uuid.uuid4()),
        topic      = topic,
        hook_type  = hook_type,
        hook       = hook_text,
        core       = first_sentence + ".",
        twist      = twist,
        ending     = f"The real implication? {loop}",
        style      = style,
        persona    = persona,
        use_avatar = settings.heygen_enabled,
    )


# ── Engine ────────────────────────────────────────────────────────────────────

class ShortsEngineV2:
    """Shorts as a laboratory: generate hypotheses → measure → learn → feed system.

    Public interface:
        generate(editorial)         → list[ShortScript]
        learn(short, metrics)       → None
        get_top_hooks(n)            → list[LearningRecord]  sorted by retention_3s
        get_best_combination()      → dict | None  best {hook_type, style, persona}
        generate_hooks(topic, n)    → list[dict]   for daily pipeline injection
    """

    def __init__(self, data_dir: Path | None = None) -> None:
        self._data_dir      = data_dir or settings.data_dir
        self._scripts_path  = self._data_dir / "shorts_scripts_v2.jsonl"
        self._learning_path = self._data_dir / "shorts_learning_v2.jsonl"
        self._data_dir.mkdir(parents=True, exist_ok=True)

    # ── core pipeline ─────────────────────────────────────────────────────────

    def generate(
        self,
        editorial: dict[str, Any],
        max_shorts: int = 5,
    ) -> list[ShortScript]:
        """Return one ShortScript per hook type, up to *max_shorts*.

        *editorial* must contain:
            topic     — short topic phrase ("ChatGPT", "OpenAI")
            core_fact — the key fact in one sentence
            angle     — optional editorial slant / twist
        """
        topic  = editorial.get("topic", _extract_topic(editorial))
        hooks  = self._generate_hooks(editorial)[:max_shorts]
        return self._build_variants(editorial, hooks)

    def _generate_hooks(self, editorial: dict[str, Any]) -> list[dict[str, str]]:
        """Return one typed hook dict per HOOK_TYPE."""
        topic = editorial.get("topic", "AI")
        seed  = editorial.get("core_fact", topic)
        return [
            {
                "type": ht,
                "text": _pick_template(ht, seed=seed + ht).format(topic=topic),
            }
            for ht in HOOK_TYPES
        ]

    def _build_variants(
        self,
        editorial: dict[str, Any],
        hooks: list[dict[str, str]],
    ) -> list[ShortScript]:
        scripts: list[ShortScript] = []
        for i, hook in enumerate(hooks):
            script = _build_script_from_editorial(
                editorial, hook["type"], hook["text"], loop_idx=i
            )
            scripts.append(script)
            self._save_script(script)
            logger.info(
                f"ShortsV2 [{hook['type']}|{script.style}|{script.persona}] "
                f"→ {hook['text']!r}"
            )
        return scripts

    def _select_style(self, hook: dict[str, str]) -> str:
        return _STYLE_MAP.get(hook["type"], "fast_cut")

    def _select_persona(self, hook: dict[str, str]) -> str:
        return _PERSONA_MAP.get(hook["type"], "curious")

    # ── learning ──────────────────────────────────────────────────────────────

    def learn(
        self,
        short: ShortScript | dict[str, Any],
        metrics: dict[str, float],
    ) -> LearningRecord:
        """Record what combination produced what retention_3s.

        Call this after collecting YouTube analytics for an uploaded Short.
        *short* can be a ShortScript or a plain dict with the same keys.
        """
        if isinstance(short, ShortScript):
            short_id  = short.short_id
            hook_type = short.hook_type
            style     = short.style
            persona   = short.persona
            topic     = short.topic
        else:
            short_id  = short.get("short_id",  "unknown")
            hook_type = short.get("hook_type", "conflict")
            style     = short.get("style",     "fast_cut")
            persona   = short.get("persona",   "direct")
            topic     = short.get("topic",     "AI")

        record = LearningRecord(
            short_id     = short_id,
            hook_type    = hook_type,
            style        = style,
            persona      = persona,
            topic        = topic,
            retention_3s = round(float(metrics.get("retention_3s", 0.0)), 4),
            avg_watch    = round(float(metrics.get("avg_watch",    0.0)), 4),
            completion   = round(float(metrics.get("completion",   0.0)), 4),
        )
        self._save_learning(record)

        status = "winner" if record.retention_3s >= _WINNER_THRESHOLD else "loser"
        logger.info(
            f"ShortsV2 learn: [{hook_type}|{style}|{persona}] "
            f"retention_3s={record.retention_3s:.2f} → {status}"
        )
        return record

    def get_top_hooks(self, n: int = 5) -> list[LearningRecord]:
        """Return top-n LearningRecords sorted by retention_3s (highest first)."""
        records = self._load_learning()
        return sorted(records, key=lambda r: r.retention_3s, reverse=True)[:n]

    def get_best_combination(self) -> dict[str, str | float] | None:
        """Return the {hook_type, style, persona} combination with the highest
        average retention_3s (requires ≥3 data points per combination).
        """
        records = self._load_learning()
        if not records:
            return None

        bucket: dict[tuple[str, str, str], list[float]] = defaultdict(list)
        for r in records:
            key = (r.hook_type, r.style, r.persona)
            bucket[key].append(r.retention_3s)

        candidates = {
            key: sum(vals) / len(vals)
            for key, vals in bucket.items()
            if len(vals) >= 3
        }
        if not candidates:
            return None

        best = max(candidates, key=lambda k: candidates[k])
        result: dict[str, str | float] = {
            "hook_type":         best[0],
            "style":             best[1],
            "persona":           best[2],
            "avg_retention_3s":  round(candidates[best], 4),
        }
        return result

    # ── daily pipeline feed ───────────────────────────────────────────────────

    def generate_hooks(self, topic: str, n: int = 10) -> list[dict[str, str]]:
        """Return proven + template hooks for *topic* for daily pipeline injection.

        Proven winners (from learning store) come first; templates fill the rest.
        """
        hooks: list[dict[str, str]] = []

        # Proven winners first
        top = self.get_top_hooks(n=n)
        for rec in top:
            template = _pick_template(rec.hook_type, seed=topic + rec.hook_type)
            hooks.append({
                "text":      template.format(topic=topic),
                "hook_type": rec.hook_type,
                "source":    "learned",
                "retention_3s": str(rec.retention_3s),
            })

        # Fill remainder with template hooks across all types
        if len(hooks) < n:
            seen_types = {h["hook_type"] for h in hooks}
            for hook_type, templates in _HOOK_TEMPLATES.items():
                if hook_type in seen_types:
                    continue
                for tmpl in templates[:max(1, (n - len(hooks)) // len(_HOOK_TEMPLATES))]:
                    hooks.append({
                        "text":      tmpl.format(topic=topic),
                        "hook_type": hook_type,
                        "source":    "template",
                    })
                    if len(hooks) >= n:
                        break
                if len(hooks) >= n:
                    break

        return hooks[:n]

    # ── persistence ───────────────────────────────────────────────────────────

    def _save_script(self, script: ShortScript) -> None:
        self._append(self._scripts_path, script.to_dict())

    def _save_learning(self, record: LearningRecord) -> None:
        self._append(self._learning_path, record.to_dict())

    def _append(self, path: Path, record: dict[str, Any]) -> None:
        try:
            with open(path, "a") as f:
                f.write(json.dumps(record) + "\n")
        except Exception as exc:
            logger.warning(f"ShortsEngineV2: write failed ({path.name}): {exc}")

    def load_scripts(self) -> list[ShortScript]:
        return self._load_jsonl(self._scripts_path, ShortScript.from_dict)

    def _load_learning(self) -> list[LearningRecord]:
        return self._load_jsonl(self._learning_path, LearningRecord.from_dict)

    def _load_jsonl(self, path: Path, factory: Any) -> list:
        if not path.exists():
            return []
        records = []
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                records.append(factory(json.loads(line)))
            except Exception:
                pass
        return records
