"""Format Selector — deterministic format from angle.

Format shapes the video's structural rhythm, not just its topic.

Each format has:
    type        — the production format label
    pacing      — energy/speed profile
    structure   — ordered list of scene roles
    description — what the viewer experiences
"""
from __future__ import annotations

# ── Format definitions ────────────────────────────────────────────────────────

FORMATS: dict[str, dict[str, object]] = {
    "hot_take": {
        "type":        "hot_take",
        "pacing":      "fast",
        "structure":   ["hook", "flip", "evidence", "punchline"],
        "description": "Provocative opener that challenges a widely held belief",
    },
    "warning": {
        "type":        "warning",
        "pacing":      "measured",
        "structure":   ["calm_open", "reveal_risk", "evidence", "call_to_action"],
        "description": "Starts safe, escalates to a specific danger, ends with agency",
    },
    "breakdown": {
        "type":        "breakdown",
        "pacing":      "deliberate",
        "structure":   ["problem", "breakdown", "real_example", "verdict"],
        "description": "Dissects how something is being used (or misused) in practice",
    },
    "explainer": {
        "type":        "explainer",
        "pacing":      "steady",
        "structure":   ["hook", "context", "explanation", "impact", "takeaway"],
        "description": "Takes the viewer from 'I heard about this' to 'I understand this'",
    },
}

# Angle → format mapping (deterministic)
ANGLE_FORMAT: dict[str, str] = {
    "hype_vs_reality": "hot_take",
    "hidden_risks":    "warning",
    "misuse":          "breakdown",
    "impact":          "explainer",
}


class FormatSelector:
    """Select the production format for a given angle."""

    def select(self, angle: str) -> dict[str, object]:
        """Return the full format dict for this angle."""
        fmt_key = ANGLE_FORMAT.get(angle, "explainer")
        return dict(FORMATS[fmt_key])

    def format_type(self, angle: str) -> str:
        """Return just the format type string."""
        return ANGLE_FORMAT.get(angle, "explainer")
