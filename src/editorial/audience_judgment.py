"""Audience-Aware Editorial Judgment — balances editorial identity with learned audience taste.

Replaces the purely deterministic angle selection with a scored blend of three signals:

    1. Editorial identity priorities  (what WE want to say)
    2. Audience learned preferences   (what the audience responds to)
    3. Historical retention memory    (what actually worked)

The result is still deterministic (highest total score wins) but adapts over
time as the audience profile accumulates feedback.

Integration in the pipeline
────────────────────────────
AudienceAwareJudgment.judge(story) is called from EditorialJudgment.judge() when
an AudienceProfile is available, replacing the plain AngleSelector result.

Scoring weights (configurable via class attributes):
    IDENTITY_WEIGHT   = 2.0   — editorial priorities are the foundation
    AUDIENCE_WEIGHT   = 1.0   — audience preference modulates but doesn't override
    RETENTION_WEIGHT  = 1.5   — proven track-record gets a meaningful boost
"""
from __future__ import annotations

from typing import Any

from loguru import logger

from src.editorial.angle_selector import ANGLES, ANGLE_INTENT
from src.editorial.persona_mapper import ANGLE_PERSONA, PERSONAS
from src.editorial.format_selector import ANGLE_FORMAT, FORMATS
from src.editorial.editorial_identity import EditorialIdentity
from src.editorial.audience_profile import AudienceProfile
from src.editorial.editorial_memory import EditorialMemory


# ── Scoring weights ───────────────────────────────────────────────────────────

IDENTITY_WEIGHT:  float = 2.0
AUDIENCE_WEIGHT:  float = 1.0
RETENTION_WEIGHT: float = 1.5

# Minimum persona affinity — below this threshold fall back to "explainer"
_MIN_PERSONA_AFFINITY: float = 0.30


class AudienceAwareJudgment:
    """Audience-aware editorial decision engine.

    Public interface
    ────────────────
    judge(story)            → full judgment dict (same schema as EditorialJudgment)
    select_angle(story)     → str
    select_persona(angle)   → str
    filter_hooks(hooks)     → list[dict]
    score_angles(story)     → dict[str, float]   (for debugging / dashboard)
    """

    def __init__(
        self,
        identity:  EditorialIdentity | None = None,
        audience:  AudienceProfile   | None = None,
        memory:    EditorialMemory    | None = None,
        data_dir   = None,
    ) -> None:
        from config import settings
        _dir = data_dir or settings.data_dir

        self.identity = identity or EditorialIdentity.load(_dir)
        self.audience = audience or AudienceProfile(data_dir=_dir)
        self.memory   = memory   or EditorialMemory(data_dir=_dir)

    # ── public interface ──────────────────────────────────────────────────────

    def judge(self, story: dict[str, Any]) -> dict[str, Any]:
        """Return a full editorial judgment shaped by both identity and audience taste."""
        angle   = self.select_angle(story)
        persona = self.select_persona(angle)
        fmt_key = ANGLE_FORMAT.get(angle, "explainer")
        intent  = dict(ANGLE_INTENT.get(angle, ANGLE_INTENT["impact"]))

        judgment: dict[str, Any] = {
            "story":            story,
            "angle":            angle,
            "persona":          dict(PERSONAS.get(persona, PERSONAS["explainer"])),
            "format":           dict(FORMATS.get(fmt_key, FORMATS["explainer"])),
            "editorial_intent": intent,
            "audience_scores":  self.score_angles(story),
        }

        # Narrative callback from memory
        topic = str(story.get("title", story.get("topic", "")))
        if topic:
            try:
                callback = self.memory.get_callback_line(topic)
                if callback:
                    judgment["callback"] = callback
                if self.memory.topic_is_recent(topic):
                    judgment["topic_is_recent"] = True
            except Exception as _e:
                logger.debug(f"AudienceAwareJudgment: memory lookup failed: {_e}")

        logger.debug(
            f"AudienceAwareJudgment: angle={angle!r} persona={persona!r} "
            f"scores={judgment['audience_scores']}"
        )
        return judgment

    def select_angle(self, story: dict[str, Any]) -> str:
        """Score all candidate angles and return the highest-scoring one.

        Scoring (additive):
            identity priority match → IDENTITY_WEIGHT
            audience preference     → score × AUDIENCE_WEIGHT
            retention history       → avg_retention × RETENTION_WEIGHT
        """
        scored = self.score_angles(story)
        return max(scored, key=lambda a: scored[a])

    def select_persona(self, angle: str) -> str:
        """Return the canonical persona for this angle, with audience affinity check.

        If the audience shows low affinity for the canonical persona (< threshold),
        fall back to "explainer" which is universally accessible.
        """
        canonical = ANGLE_PERSONA.get(angle, "explainer")
        affinity  = self.audience.get_persona_affinity(canonical)

        # Only apply affinity fallback once the profile has learned something
        # (affinity == 0.0 means "no data yet", which is neutral, not negative)
        if affinity > 0.0 and affinity < _MIN_PERSONA_AFFINITY:
            logger.debug(
                f"AudienceAwareJudgment: low affinity for {canonical!r} "
                f"({affinity:.2f}) → falling back to explainer"
            )
            return "explainer"

        return canonical

    def filter_hooks(self, hooks: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Remove hooks whose pattern the audience consistently dislikes."""
        filtered = [
            h for h in hooks
            if not self.audience.is_disliked(h.get("pattern", ""))
        ]
        removed = len(hooks) - len(filtered)
        if removed:
            logger.debug(f"AudienceAwareJudgment: filtered {removed} disliked hook(s)")
        return filtered

    def score_angles(self, story: dict[str, Any]) -> dict[str, float]:
        """Return a score for every candidate angle — useful for debugging and dashboard."""
        scores: dict[str, float] = {}

        # story-signal → angle map (same as AngleSelector)
        signal_angle = {
            "hype": "hype_vs_reality",
            "risk": "hidden_risks",
            "tool": "misuse",
        }

        for angle in ANGLES:
            score = 0.0

            # 1. Identity priority bonus
            priority = ANGLE_INTENT.get(angle, {}).get("priority", "")
            if priority in self.identity.priorities:
                score += IDENTITY_WEIGHT

            # 2. Direct story-signal match (binary, from AngleSelector logic)
            for sig, sig_angle in signal_angle.items():
                if story.get(sig) and angle == sig_angle:
                    score += IDENTITY_WEIGHT * 0.5   # half-weight signal bonus

            # 3. Audience preference score
            score += self.audience.get_angle_score(angle) * AUDIENCE_WEIGHT

            # 4. Proven retention history
            score += self.audience.get_angle_retention(angle) * RETENTION_WEIGHT

            scores[angle] = round(score, 4)

        return scores


# ── module-level singleton ────────────────────────────────────────────────────

_engine: AudienceAwareJudgment | None = None


def get_audience_judgment() -> AudienceAwareJudgment:
    """Return the module-level AudienceAwareJudgment singleton."""
    global _engine
    if _engine is None:
        _engine = AudienceAwareJudgment()
    return _engine
