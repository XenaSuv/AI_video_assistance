"""Hook Pattern Engine — learn WHY hooks work, not just which strings work.

Instead of storing "Nobody is talking about this…" and learning that exact
string, this engine extracts the structural features that made it work:

    text   → features → pattern → score → generate

Features extracted from every hook:
    pattern   — primary pattern family (hidden_info / conflict / mistake /
                 warning / insider)
    emotion   — the psychological trigger (intrigue / fear / challenge /
                 urgency / exclusivity)
    structure — how the sentence is constructed (gap / direct_callout /
                 question / challenge_authority / reveal)
    tone      — delivery energy (insider / direct / urgent / curious)

This gives the system two levels of learning:

    Level 1 — single-feature patterns
        conflict → avg 0.68
        hidden_info → avg 0.71

    Level 2 — multi-feature combos (the powerful upgrade)
        conflict + urgent + direct_callout → avg 0.74
        hidden_info + intrigue + gap      → avg 0.71

The combination level is how top creators operate: they don't just pick
"conflict", they pick "conflict + urgency + direct personal callout".

Data file: data/hook_patterns.jsonl — append-only, one record per hook.

Integration:
    # After each Short's analytics arrive:
    engine.record(hook_text, hook_type, retention_3s, short_id)

    # Before generating next batch:
    top = engine.get_top_patterns()
    hooks = [engine.generate_from_pattern(p["pattern"], topic) for p in top[:3]]

    # Cross-learning combo:
    best = engine.get_best_combo()
    cross_engine.learn({"hook_pattern": best["pattern"], ...})
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import settings

# ── Feature signals ───────────────────────────────────────────────────────────

_PATTERN_SIGNALS: dict[str, set[str]] = {
    "hidden_info": {"nobody", "secret", "hidden", "won't tell", "don't show",
                    "behind the scenes", "what they know", "not talking"},
    "conflict":    {"problem", "attack", "vs ", "battle", "lie", "lying",
                    "collapse", "war", "getting worse", "real problem"},
    "mistake":     {"wrong", "mistake", "misunderstand", "stop ", "missing",
                    "completely wrong", "you're using", "doing this"},
    "warning":     {"fail", "warning", "danger", "risk", "backfire", "scary",
                    "terrifying", "might actually", "heading in", "could"},
    "insider":     {"insider", "insiders", "what they don't", "actually looks",
                    "behind the scenes", "whispering", "experts"},
}

_EMOTION_SIGNALS: dict[str, set[str]] = {
    "fear":        {"scary", "terrifying", "danger", "risk", "collapse", "fail"},
    "intrigue":    {"nobody", "secret", "hidden", "not talking", "won't tell"},
    "challenge":   {"wrong", "stop", "misunderstand", "missing", "completely"},
    "urgency":     {"warning", "backfire", "about to", "heading in", "might actually"},
    "exclusivity": {"insider", "insiders", "what they know", "experts", "whispering"},
}

_STRUCTURE_SIGNALS: dict[str, set[str]] = {
    "direct_callout":      {"you're", "you are", "stop ", "your ", "you've"},
    "question":            {"?"},
    "gap":                 {"nobody", "secret", "hidden", "what they", "won't tell"},
    "challenge_authority": {"problem", "vs ", "battle", "lie", "worse"},
    "reveal":              {"here's", "here is", "this is", "actually", "real"},
}

_TONE_SIGNALS: dict[str, set[str]] = {
    "insider":  {"insider", "insiders", "secret", "behind", "what they"},
    "direct":   {"wrong", "stop ", "misunderstand", "you're", "completely"},
    "urgent":   {"warning", "fail", "danger", "scary", "about to", "might"},
    "curious":  {"nobody", "hidden", "actually", "wait", "secret"},
}

# ── Generation templates (one set per pattern) ────────────────────────────────

_PATTERN_TEMPLATES: dict[str, list[str]] = {
    "hidden_info": [
        "Nobody is talking about {topic}",
        "The secret about {topic} they won't tell you",
        "What insiders know about {topic}",
        "Here's what they don't show you about {topic}",
    ],
    "conflict": [
        "This is the real problem with {topic}",
        "{topic} vs reality — someone is lying",
        "{topic} is getting worse, not better",
        "The war over {topic} nobody is covering",
    ],
    "mistake": [
        "You're using {topic} completely wrong",
        "Stop doing this with {topic}",
        "Everyone is misunderstanding {topic}",
        "You're missing this about {topic}",
    ],
    "warning": [
        "This might actually fail — {topic}",
        "Warning: {topic} is heading in the wrong direction",
        "{topic} could backfire badly",
        "{topic} is about to cross a dangerous line",
    ],
    "insider": [
        "What insiders know about {topic}",
        "Here's what they don't show you about {topic}",
        "The {topic} detail experts are whispering about",
        "Behind the scenes: what really happened with {topic}",
    ],
    "general": [
        "This changes everything about {topic}",
        "{topic} just did something unexpected",
        "The {topic} update everyone missed",
    ],
}

_MIN_COMBO_POINTS = 2   # min data points to trust a multi-feature combo score


# ── Dataclasses ───────────────────────────────────────────────────────────────

@dataclass
class HookPatternRecord:
    """One observation: this hook text → these features → this retention_3s."""
    hook_text:    str
    hook_type:    str
    pattern:      str
    emotion:      str
    structure:    str
    tone:         str
    retention_3s: float
    short_id:     str = ""
    recorded_at:  str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "HookPatternRecord":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class PatternScore:
    """Aggregated score for a single-feature pattern."""
    pattern:      str
    avg_score:    float
    count:        int
    top_emotion:  str = ""
    top_structure: str = ""
    top_tone:     str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ComboScore:
    """Aggregated score for a multi-feature combination."""
    pattern:   str
    tone:      str
    structure: str
    avg_score: float
    count:     int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ── Feature extraction ────────────────────────────────────────────────────────

def extract_features(hook_text: str) -> dict[str, str]:
    """Rule-based feature extraction. Replaceable with an LLM call later.

    Returns a dict with keys: pattern, emotion, structure, tone.
    Falls back to "general" / "neutral" / "reveal" / "curious" when no
    signal matches.
    """
    text = hook_text.lower()

    def _match(signals: dict[str, set[str]]) -> str:
        best: str | None = None
        best_count = 0
        for label, keywords in signals.items():
            count = sum(1 for kw in keywords if kw in text)
            if count > best_count:
                best_count = count
                best = label
        return best or next(iter(signals))  # fallback to first key

    pattern   = _match(_PATTERN_SIGNALS) if any(
        kw in text for kws in _PATTERN_SIGNALS.values() for kw in kws
    ) else "general"
    emotion   = _match(_EMOTION_SIGNALS) if any(
        kw in text for kws in _EMOTION_SIGNALS.values() for kw in kws
    ) else "neutral"
    structure = _match(_STRUCTURE_SIGNALS) if any(
        kw in text for kws in _STRUCTURE_SIGNALS.values() for kw in kws
    ) else "reveal"
    tone      = _match(_TONE_SIGNALS) if any(
        kw in text for kws in _TONE_SIGNALS.values() for kw in kws
    ) else "curious"

    return {
        "pattern":   pattern,
        "emotion":   emotion,
        "structure": structure,
        "tone":      tone,
    }


# ── Engine ────────────────────────────────────────────────────────────────────

class HookPatternEngine:
    """Learn structural patterns from hook performance data.

    Public interface:
        record(hook_text, hook_type, retention_3s, short_id)
            → extracts features and stores one HookPatternRecord

        get_top_patterns(n)
            → list[PatternScore] sorted by avg retention_3s

        get_best_combo(n)
            → list[ComboScore]  sorted by avg retention_3s
               (requires ≥ _MIN_COMBO_POINTS per combo)

        generate_from_pattern(pattern, topic, seed)
            → str  — a hook text using the best template for that pattern

        generate_best_hooks(topic, n)
            → list[dict]  — hooks generated from top patterns, proven first
    """

    def __init__(self, data_dir: Path | None = None) -> None:
        self._data_dir = data_dir or settings.data_dir
        self._path     = self._data_dir / "hook_patterns.jsonl"
        self._data_dir.mkdir(parents=True, exist_ok=True)

    # ── recording ─────────────────────────────────────────────────────────────

    def record(
        self,
        hook_text:    str,
        hook_type:    str,
        retention_3s: float,
        short_id:     str = "",
    ) -> HookPatternRecord:
        """Extract features from *hook_text* and persist one learning record."""
        features = extract_features(hook_text)
        rec = HookPatternRecord(
            hook_text    = hook_text,
            hook_type    = hook_type,
            retention_3s = round(retention_3s, 4),
            short_id     = short_id,
            **features,
        )
        self._append(rec.to_dict())
        logger.info(
            f"HookPattern recorded: [{features['pattern']}|{features['tone']}|"
            f"{features['structure']}] retention_3s={retention_3s:.2f}"
        )
        return rec

    # ── analysis ──────────────────────────────────────────────────────────────

    def get_top_patterns(self, n: int = 5) -> list[PatternScore]:
        """Return patterns ranked by average retention_3s."""
        records = self._load()
        if not records:
            return []

        groups: dict[str, list[HookPatternRecord]] = defaultdict(list)
        for r in records:
            groups[r.pattern].append(r)

        scores: list[PatternScore] = []
        for pattern, recs in groups.items():
            vals = [r.retention_3s for r in recs]
            # Most common emotion/structure/tone for this pattern
            top_emotion   = _most_common(r.emotion   for r in recs)
            top_structure = _most_common(r.structure for r in recs)
            top_tone      = _most_common(r.tone      for r in recs)
            scores.append(PatternScore(
                pattern       = pattern,
                avg_score     = round(sum(vals) / len(vals), 4),
                count         = len(vals),
                top_emotion   = top_emotion,
                top_structure = top_structure,
                top_tone      = top_tone,
            ))

        return sorted(scores, key=lambda s: s.avg_score, reverse=True)[:n]

    def get_best_combo(self, n: int = 5) -> list[ComboScore]:
        """Return multi-feature combos ranked by avg retention_3s.

        Only combos with ≥ _MIN_COMBO_POINTS data points are included to
        avoid over-fitting to single observations.
        """
        records = self._load()
        if not records:
            return []

        bucket: dict[tuple[str, str, str], list[float]] = defaultdict(list)
        for r in records:
            key = (r.pattern, r.tone, r.structure)
            bucket[key].append(r.retention_3s)

        combos: list[ComboScore] = []
        for (pattern, tone, structure), vals in bucket.items():
            if len(vals) < _MIN_COMBO_POINTS:
                continue
            combos.append(ComboScore(
                pattern   = pattern,
                tone      = tone,
                structure = structure,
                avg_score = round(sum(vals) / len(vals), 4),
                count     = len(vals),
            ))

        return sorted(combos, key=lambda c: c.avg_score, reverse=True)[:n]

    # ── generation ────────────────────────────────────────────────────────────

    def generate_from_pattern(
        self,
        pattern: str,
        topic:   str,
        seed:    str = "",
    ) -> str:
        """Return a hook text using the best known template for *pattern*."""
        templates = _PATTERN_TEMPLATES.get(pattern, _PATTERN_TEMPLATES["general"])
        idx = hash(seed or topic) % len(templates)
        return templates[idx].format(topic=topic)

    def generate_best_hooks(self, topic: str, n: int = 5) -> list[dict[str, str]]:
        """Generate hooks from top proven patterns.

        Returns a list of dicts with keys: text, pattern, source.
        Proven patterns come first; falls back to full template set if not
        enough data yet.
        """
        hooks: list[dict[str, str]] = []

        top_patterns = self.get_top_patterns(n=n)
        for ps in top_patterns:
            text = self.generate_from_pattern(ps.pattern, topic, seed=topic + ps.pattern)
            hooks.append({
                "text":    text,
                "pattern": ps.pattern,
                "tone":    ps.top_tone,
                "source":  "pattern_learned",
                "avg_retention_3s": str(ps.avg_score),
            })

        # Fill remainder from templates if not enough proven patterns
        if len(hooks) < n:
            seen = {h["pattern"] for h in hooks}
            for pattern in _PATTERN_TEMPLATES:
                if pattern in seen or pattern == "general":
                    continue
                text = self.generate_from_pattern(pattern, topic)
                hooks.append({
                    "text":    text,
                    "pattern": pattern,
                    "tone":    "",
                    "source":  "template",
                })
                if len(hooks) >= n:
                    break

        return hooks[:n]

    # ── persistence ───────────────────────────────────────────────────────────

    def _append(self, record: dict[str, Any]) -> None:
        try:
            with open(self._path, "a") as f:
                f.write(json.dumps(record) + "\n")
        except Exception as exc:
            logger.warning(f"HookPatternEngine: write failed: {exc}")

    def _load(self) -> list[HookPatternRecord]:
        if not self._path.exists():
            return []
        records: list[HookPatternRecord] = []
        for line in self._path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                records.append(HookPatternRecord.from_dict(json.loads(line)))
            except Exception as exc:
                logger.debug(f"Skipped malformed record: {exc}")
        return records


# ── Helpers ───────────────────────────────────────────────────────────────────

def _most_common(values: Any) -> str:
    counts: dict[str, int] = defaultdict(int)
    for v in values:
        counts[v] += 1
    return max(counts, key=lambda k: counts[k]) if counts else ""
