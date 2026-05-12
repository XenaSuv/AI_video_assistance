"""Persona Mapper — deterministic persona from angle.

Persona ≠ random.  Persona = function of angle.

Each persona carries:
    style       — delivery register (skeptical_insider / analyst / builder / explainer)
    voice       — how the script sounds
    rules       — concrete writing constraints passed to ScriptGenerator
"""
from __future__ import annotations

from typing import Any


# ── Persona definitions ───────────────────────────────────────────────────────

PERSONAS: dict[str, dict[str, Any]] = {
    "skeptical_insider": {
        "style": "skeptical_insider",
        "voice": "direct, confident, always finds the catch",
        "rules": [
            "start with what everyone believes, then flip it",
            "use 'but here's what they're not telling you'",
            "never endorse — always interrogate",
        ],
    },
    "analyst": {
        "style": "analyst",
        "voice": "measured, evidence-driven, calm under pressure",
        "rules": [
            "lead with the data, not the emotion",
            "name the risk explicitly before explaining it",
            "end with a concrete implication for the viewer",
        ],
    },
    "builder": {
        "style": "builder",
        "voice": "practical, tool-first, no patience for abstractions",
        "rules": [
            "ask 'how would you actually use this?'",
            "call out misuse patterns with specifics",
            "prefer concrete examples over theoretical concerns",
        ],
    },
    "explainer": {
        "style": "explainer",
        "voice": "clear, curious, accessible without dumbing down",
        "rules": [
            "assume smart viewer, not informed viewer",
            "use one vivid analogy per video",
            "end with 'why this matters to you'",
        ],
    },
}

# Angle → persona mapping (deterministic)
ANGLE_PERSONA: dict[str, str] = {
    "hype_vs_reality": "skeptical_insider",
    "hidden_risks":    "analyst",
    "misuse":          "builder",
    "impact":          "explainer",
}


class PersonaMapper:
    """Map an angle to its canonical persona."""

    def map(self, angle: str) -> dict[str, Any]:
        """Return the full persona dict for this angle."""
        persona_key = ANGLE_PERSONA.get(angle, "explainer")
        return dict(PERSONAS[persona_key])

    def style(self, angle: str) -> str:
        """Return just the persona style string."""
        return ANGLE_PERSONA.get(angle, "explainer")
