"""Angle Selector — deterministic angle from story signals.

Angle = the editorial meaning of the video ("why does this matter to the viewer").
Not picked randomly — derived directly from story signals + identity priority.

Priority order:
    hype signal   → hype_vs_reality  (anti-hype identity's core play)
    risk signal   → hidden_risks     (expose what nobody mentions)
    tool signal   → misuse           (how is it being used wrong?)
    default       → impact           (explain why this matters)

Each angle has a matching editorial_intent:
    goal      — what cognitive shift the viewer should have after watching
    emotion   — primary emotion to target (from EMOTIONS vocabulary)
    priority  — which identity priority this serves
"""
from __future__ import annotations

from typing import Any

from src.editorial.editorial_identity import EditorialIdentity


# ── Angles ────────────────────────────────────────────────────────────────────

ANGLES: list[str] = ["hype_vs_reality", "hidden_risks", "misuse", "impact"]

# Editorial intent carried by each angle
ANGLE_INTENT: dict[str, dict[str, str]] = {
    "hype_vs_reality": {
        "goal":     "challenge perception",
        "emotion":  "doubt",
        "priority": "misconceptions",
    },
    "hidden_risks": {
        "goal":     "expose danger",
        "emotion":  "fear",
        "priority": "hidden_risks",
    },
    "misuse": {
        "goal":     "expose misuse",
        "emotion":  "shock",
        "priority": "real_world_impact",
    },
    "impact": {
        "goal":     "explain significance",
        "emotion":  "curiosity",
        "priority": "real_world_impact",
    },
}


class AngleSelector:
    """Select the editorial angle for a story deterministically."""

    def select(
        self,
        story: dict[str, Any],
        identity: EditorialIdentity | None = None,
    ) -> str:
        """Return the angle that best matches this story's signals.

        Signal priority: hype → hidden_risks → misuse → impact (default).
        The identity bias is already applied upstream by StoryRanker, so
        the highest-scored story naturally surfaces the right signal here.
        """
        # Anti-hype identity elevates hype stories to its flagship angle
        if story.get("hype"):
            return "hype_vs_reality"

        if story.get("risk"):
            return "hidden_risks"

        if story.get("tool"):
            return "misuse"

        return "impact"

    def intent(self, angle: str) -> dict[str, str]:
        """Return the editorial_intent dict for an angle."""
        return dict(ANGLE_INTENT.get(angle, ANGLE_INTENT["impact"]))
