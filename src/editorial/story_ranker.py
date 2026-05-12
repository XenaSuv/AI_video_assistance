"""Story Ranker — score stories with editorial bias.

Scoring components (additive):
  +importance     — intrinsic importance score from the scraper
  +controversy    — conflictual stories are editorially interesting
  +anti-hype bias — hyped stories get boosted when identity.bias == "anti-hype"
  +priority match — story signals that match identity.priorities get a bonus
  -seen_recently  — dedup penalty: don't repeat topics we just covered
"""
from __future__ import annotations

from typing import Any

from src.editorial.editorial_identity import EditorialIdentity


# ── Scoring constants ─────────────────────────────────────────────────────────

_CONTROVERSY_BONUS:  float = 2.0
_ANTI_HYPE_BONUS:    float = 1.5
_PRIORITY_BONUS:     float = 1.0
_SEEN_PENALTY:       float = 2.0
_MEMORY_REPEAT_PENALTY: float = 1.5   # extra penalty if topic appears in editorial memory

# Map story signal keys → identity priority names they satisfy
_SIGNAL_TO_PRIORITY: dict[str, str] = {
    "risk":          "hidden_risks",
    "misconception": "misconceptions",
    "impact":        "real_world_impact",
}


class StoryRanker:
    """Scores and ranks a list of stories against an editorial identity."""

    def rank(
        self,
        stories: list[dict[str, Any]],
        identity: EditorialIdentity,
    ) -> list[tuple[float, dict[str, Any]]]:
        """Return stories sorted by editorial score (highest first).

        Each story dict may contain:
            importance     float   0–10    scraper signal strength
            controversy    float   0–1     how divisive the topic is
            hype           bool            is this a hyped announcement?
            risk           bool            does it expose a risk?
            seen_recently  bool            covered in the last 30 days?
            misconception  bool            corrects a common false belief?
            impact         bool            has real-world consequences?
        """
        scored = []
        for story in stories:
            score = self._score(story, identity)
            scored.append((score, story))
        return sorted(scored, key=lambda x: x[0], reverse=True)

    def _score(self, story: dict[str, Any], identity: EditorialIdentity) -> float:
        score = 0.0

        # 1. Intrinsic importance
        score += float(story.get("importance", 0))

        # 2. Controversy bonus — editorial loves conflict
        controversy = float(story.get("controversy", 0))
        if controversy > 0:
            score += _CONTROVERSY_BONUS

        # 3. Anti-hype bias — hyped stories are more interesting to debunk
        if identity.bias == "anti-hype" and story.get("hype"):
            score += _ANTI_HYPE_BONUS

        # 4. Priority match — reward stories that hit our editorial priorities
        for signal, priority in _SIGNAL_TO_PRIORITY.items():
            if story.get(signal) and priority in identity.priorities:
                score += _PRIORITY_BONUS

        # 5. Dedup penalty
        if story.get("seen_recently"):
            score -= _SEEN_PENALTY

        # 6. Editorial memory repeat penalty — avoid topics we've already covered
        topic = str(story.get("title", story.get("topic", "")))
        if topic:
            try:
                from src.editorial.editorial_memory import get_editorial_memory
                if get_editorial_memory().topic_is_recent(topic):
                    score -= _MEMORY_REPEAT_PENALTY
            except Exception:
                pass

        return score
