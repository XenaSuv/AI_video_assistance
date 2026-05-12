"""Editorial sub-package — deterministic editorial judgment layer.

Pipeline position:
    story → EditorialJudgment → NarrativeIdentityEngine → NarrativeConflictEngine → delivery

Exports:
    EditorialIdentity       — channel voice/bias/priority config
    StoryRanker             — score stories with editorial bias
    AngleSelector           — deterministic angle from story signals
    PersonaMapper           — persona from angle
    FormatSelector          — format from angle
    EditorialJudgment       — orchestrator + feedback integration
    get_judgment_engine     — module-level singleton
    EditorialMemory         — persistent cross-video memory (angles, predictions, retention)
    get_editorial_memory    — module-level singleton
    AudienceProfile         — learned audience preferences
    get_audience_profile    — module-level singleton
    AudienceAwareJudgment   — identity × audience × retention scored judgment
    get_audience_judgment   — module-level singleton
"""
from src.editorial.editorial_identity import EditorialIdentity, EDITORIAL_IDENTITY
from src.editorial.story_ranker import StoryRanker
from src.editorial.angle_selector import AngleSelector, ANGLE_INTENT
from src.editorial.persona_mapper import PersonaMapper, ANGLE_PERSONA
from src.editorial.format_selector import FormatSelector, ANGLE_FORMAT
from src.editorial.editorial_judgment import EditorialJudgment, get_judgment_engine
from src.editorial.editorial_memory import EditorialMemory, get_editorial_memory
from src.editorial.audience_profile import AudienceProfile, get_audience_profile
from src.editorial.audience_judgment import AudienceAwareJudgment, get_audience_judgment

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
    "EditorialMemory",
    "get_editorial_memory",
    "AudienceProfile",
    "get_audience_profile",
    "AudienceAwareJudgment",
    "get_audience_judgment",
]
