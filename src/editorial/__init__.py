"""Editorial sub-package — deterministic editorial judgment layer.

Pipeline position:
    story → EditorialJudgment → NarrativeIdentityEngine → NarrativeConflictEngine → delivery

Exports:
    EditorialIdentity   — channel voice/bias/priority config
    StoryRanker         — score stories with editorial bias
    AngleSelector       — deterministic angle from story signals
    PersonaMapper       — persona from angle
    FormatSelector      — format from angle
    EditorialJudgment   — orchestrator + feedback integration
    get_judgment_engine — module-level singleton
"""
from src.editorial.angle_selector import ANGLE_INTENT, AngleSelector
from src.editorial.editorial_identity import EDITORIAL_IDENTITY, EditorialIdentity
from src.editorial.editorial_judgment import EditorialJudgment, get_judgment_engine
from src.editorial.format_selector import ANGLE_FORMAT, FormatSelector
from src.editorial.persona_mapper import ANGLE_PERSONA, PersonaMapper
from src.editorial.story_ranker import StoryRanker

__all__ = [
    "EditorialIdentity",
    "EDITORIAL_IDENTITY",
    "StoryRanker",
    "AngleSelector",
    "ANGLE_INTENT",
    "PersonaMapper",
    "ANGLE_PERSONA",
    "FormatSelector",
    "ANGLE_FORMAT",
    "EditorialJudgment",
    "get_judgment_engine",
]
