"""Editorial Judgment — orchestrates the full editorial decision pipeline.

The difference:
    Before:  story → script
    Now:     story → judgment → narrative → delivery

EditorialJudgment produces a structured editorial decision:
    {
        "story":            {...},
        "angle":            "hype_vs_reality",
        "persona":          {"style": "skeptical_insider", "voice": "...", "rules": [...]},
        "format":           {"type": "hot_take", "pacing": "fast", "structure": [...]},
        "editorial_intent": {"goal": "challenge perception", "emotion": "doubt", "priority": "misconceptions"}
    }

This output is consumed by:
    ScriptGenerator  — angle + goal shape the prompt
    HumanizerAgent   — persona.rules drive rewrite heuristics
    PresentationEngine — format.structure guides scene ordering

Feedback integration:
    update_from_feedback({"angle": "hype_vs_reality", "retention": 0.72})
    → promotes that angle's priority to the top of identity.priorities
    → persists the updated identity so it survives restarts
"""
from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from config import settings
from src.editorial.angle_selector import AngleSelector
from src.editorial.editorial_identity import EditorialIdentity
from src.editorial.format_selector import FormatSelector
from src.editorial.persona_mapper import PersonaMapper
from src.editorial.story_ranker import StoryRanker

# ── Feedback threshold ────────────────────────────────────────────────────────

_RETENTION_WINNER_THRESHOLD: float = 0.60   # retention above this reinforces the angle


# ── Feedback record ───────────────────────────────────────────────────────────

@dataclass
class JudgmentFeedback:
    """One feedback data point linking an angle to a measured retention score."""
    angle:       str
    retention:   float
    video_id:    str = ""
    recorded_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "JudgmentFeedback":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# ── Main engine ───────────────────────────────────────────────────────────────

class EditorialJudgment:
    """Produces a deterministic editorial decision for a story or list of stories.

    Public interface
    ─────────────────
    judge(story)            → judgment dict for a single story
    process(stories)        → judgment dict for the top-ranked story in a list
    update_from_feedback(f) → reinforce winning angles in identity.priorities
    load_feedback(n)        → list[JudgmentFeedback] from JSONL store
    """

    def __init__(
        self,
        identity:  EditorialIdentity | None = None,
        data_dir:  Path | None = None,
    ) -> None:
        self._dir      = data_dir or settings.data_dir
        self._dir.mkdir(parents=True, exist_ok=True)
        self._fb_path  = self._dir / "editorial_feedback.jsonl"
        self.identity  = identity or EditorialIdentity.load(self._dir)
        self._ranker   = StoryRanker()
        self._angles   = AngleSelector()
        self._personas = PersonaMapper()
        self._formats  = FormatSelector()

    # ── public interface ──────────────────────────────────────────────────────

    def judge(self, story: dict[str, Any]) -> dict[str, Any]:
        """Return a full editorial judgment for a single story dict.

        story signals consumed:
            hype, risk, tool, controversy, importance, seen_recently,
            misconception, impact
        """
        angle  = self._angles.select(story, self.identity)
        intent = self._angles.intent(angle)
        return {
            "story":            story,
            "angle":            angle,
            "persona":          self._personas.map(angle),
            "format":           self._formats.select(angle),
            "editorial_intent": intent,
        }

    def process(self, stories: list[dict[str, Any]]) -> dict[str, Any]:
        """Rank stories and return a judgment for the top-scoring one.

        Returns an empty dict if stories is empty.
        """
        if not stories:
            return {}
        ranked  = self._ranker.rank(stories, self.identity)
        _, best = ranked[0]
        judgment = self.judge(best)
        logger.debug(
            f"EditorialJudgment: selected angle={judgment['angle']!r} "
            f"persona={judgment['persona']['style']!r} "
            f"format={judgment['format']['type']!r}"
        )
        return judgment

    # ── feedback integration ──────────────────────────────────────────────────

    def update_from_feedback(
        self,
        feedback: dict[str, Any],
        persist: bool = True,
    ) -> None:
        """Reinforce winning angles in identity.priorities.

        feedback must contain:
            angle      str     which angle was used
            retention  float   measured 30s retention (0–1)

        If retention > _RETENTION_WINNER_THRESHOLD:
            → the angle's priority is promoted to position 0
            → identity is persisted so learning survives restarts
        """
        angle     = feedback.get("angle", "")
        retention = float(feedback.get("retention", 0.0))
        video_id  = feedback.get("video_id", "")

        if not angle:
            return

        record = JudgmentFeedback(angle=angle, retention=retention, video_id=video_id)
        self._append_feedback(record)

        if retention > _RETENTION_WINNER_THRESHOLD:
            from src.editorial.angle_selector import ANGLE_INTENT
            priority = ANGLE_INTENT.get(angle, {}).get("priority", "")
            if priority:
                self.identity.promote_priority(priority)
                logger.info(
                    f"EditorialJudgment: retention={retention:.2f} → "
                    f"promoted priority={priority!r} for angle={angle!r}"
                )
                if persist:
                    self.identity.save(self._dir)

    def load_feedback(self, n: int = 50) -> list[JudgmentFeedback]:
        """Load the last n feedback records."""
        if not self._fb_path.exists():
            return []
        records: list[JudgmentFeedback] = []
        for line in self._fb_path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                records.append(JudgmentFeedback.from_dict(json.loads(line)))
            except Exception as exc:
                logger.debug(f"Skipped malformed record: {exc}")
        return records[-n:]

    def winning_angles(self, min_retention: float = _RETENTION_WINNER_THRESHOLD) -> list[str]:
        """Return angles that consistently beat the retention threshold."""
        records = self.load_feedback()
        by_angle: dict[str, list[float]] = {}
        for r in records:
            by_angle.setdefault(r.angle, []).append(r.retention)
        winners = [
            angle for angle, scores in by_angle.items()
            if (sum(scores) / len(scores)) >= min_retention
        ]
        return winners

    # ── persistence helpers ───────────────────────────────────────────────────

    def _append_feedback(self, record: JudgmentFeedback) -> None:
        try:
            with open(self._fb_path, "a") as f:
                f.write(json.dumps(record.to_dict()) + "\n")
        except Exception as exc:
            logger.warning(f"EditorialJudgment: feedback write failed: {exc}")


# ── Module-level singleton ────────────────────────────────────────────────────

_engine: EditorialJudgment | None = None


def get_judgment_engine() -> EditorialJudgment:
    """Return the module-level EditorialJudgment singleton."""
    global _engine
    if _engine is None:
        _engine = EditorialJudgment()
    return _engine
