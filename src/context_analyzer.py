"""Context Analyzer — editorial input → structured content context.

Converts the editorial description of a video topic into a machine-readable
context dict that makes the Decision Engine content-aware.

Key insight
-----------
    breaking news → "fast + aggressive + urgent"
    tutorial      → "slow + clear + teacher"
    AI research   → "slower + analytical"

Without this layer the decision engine is blind to what the video is *about*
and can only react to past performance metrics.

Context shape (output of analyze())
-------------------------------------
{
    "content_type": "breaking" | "tutorial" | "news" | "review" | "opinion",
    "topic":        str,
    "novelty":      float,   # 0–1 — how new/fresh is the information
    "complexity":   float,   # 0–1 — how technically dense
    "urgency":      float,   # 0–1 — how time-sensitive
    "topic_signals": {
        "mentions_openai":    bool,
        "mentions_google":    bool,
        "mentions_meta":      bool,
        "is_research":        bool,
        "is_safety":          bool,
        "is_regulation":      bool,
        "is_product_launch":  bool,
    }
}

Strategy overrides per content type
-------------------------------------
    "breaking"  → pace=fast,   hook_aggressiveness=1.0, persona=urgent
    "tutorial"  → pace=slow,   hook_aggressiveness=0.4, persona=teacher
    "review"    → pace=normal, hook_aggressiveness=0.6, persona=balanced
    "opinion"   → pace=fast,   hook_aggressiveness=0.8, persona=curious
    "news"      → (no override — metrics-based decision applies)

Topic-signal refinements applied on top
-----------------------------------------
    is_research     → pace=slow (overrides content-type pace)
    mentions_openai → persona=insider
    urgency ≥ 0.8   → pace=fast, hook_aggressiveness boosted (unless tutorial)

Integration
-----------
    analyzer = ContextAnalyzer()
    ctx      = analyzer.analyze(editorial_input)

    # Apply to strategy (orchestrator pattern):
    for attr, val in ctx.to_strategy_overrides().items():
        if hasattr(strategy, attr):
            setattr(strategy, attr, val)

    # Feed into cross-learning:
    cross_engine.recommend(context={"topic": ctx.topic})

    # Feed into decision engine context dict:
    decision_context["context"] = ctx.to_dict()
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from loguru import logger

# ── Content type detection keywords ───────────────────────────────────────────

_BREAKING_KW   = {"breaking", "just announced", "urgent", "live:", "developing", "exclusive"}
_TUTORIAL_KW   = {"how to", "guide", "tutorial", "step-by-step", "step by step",
                  "learn ", "beginner", "explained", "complete course"}
_REVIEW_KW     = {"review", " vs ", "comparison", "ranked", "best of", "worst"}
_OPINION_KW    = {"opinion", "why i ", "i think", "unpopular", "hot take", "controversial"}

# Topic signal keywords
_OPENAI_KW     = {"openai", "gpt-", "chatgpt", "o1", "o3", "sora"}
_GOOGLE_KW     = {"google", "gemini", "bard", "deepmind", "vertex"}
_META_KW       = {"meta ", "llama", "facebook ai"}
_RESEARCH_KW   = {"research", "paper", "study", "arxiv", "benchmark", "published"}
_SAFETY_KW     = {"safety", "alignment", "risk", "bias", "harmful", "guardrail"}
_REGULATION_KW = {"regulation", "ban", "law", "policy", "congress", " eu ", "gdpr", "act"}
_LAUNCH_KW     = {"launch", "release", "new model", "announced", "introducing", "unveil"}


# ── Output dataclass ──────────────────────────────────────────────────────────

@dataclass
class ContentContext:
    """Structured content context produced by ContextAnalyzer.analyze()."""

    content_type:  str              = "news"
    topic:         str              = ""
    novelty:       float            = 0.5
    complexity:    float            = 0.5
    urgency:       float            = 0.5
    topic_signals: dict[str, bool]  = field(default_factory=dict)

    # ── Serialisation ──────────────────────────────────────────────────────────

    def to_dict(self) -> dict[str, Any]:
        return {
            "content_type":  self.content_type,
            "topic":         self.topic,
            "novelty":       self.novelty,
            "complexity":    self.complexity,
            "urgency":       self.urgency,
            "topic_signals": dict(self.topic_signals),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ContentContext":
        return cls(
            content_type  = d.get("content_type",  "news"),
            topic         = d.get("topic",         ""),
            novelty       = float(d.get("novelty",   0.5)),
            complexity    = float(d.get("complexity", 0.5)),
            urgency       = float(d.get("urgency",    0.5)),
            topic_signals = dict(d.get("topic_signals", {})),
        )

    # ── Decision-engine integration ────────────────────────────────────────────

    def to_mode_hint(self) -> str | None:
        """Advisory mode that the Decision Engine may adopt.

        Returns None when context gives no strong signal — defer to metrics.
        """
        if self.content_type == "breaking" or self.urgency >= 0.8:
            return "growth"
        if self.content_type == "tutorial":
            return "stable"
        if self.novelty >= 0.8:
            return "growth"
        return None

    def to_strategy_overrides(self) -> dict[str, Any]:
        """Partial strategy dict to overlay on the base metrics-driven strategy.

        Content-type overrides are applied first; topic-signal refinements
        on top.  Caller is responsible for applying only fields that exist on
        the strategy object being overridden.
        """
        overrides: dict[str, Any] = {}

        # ── Content-type base ──────────────────────────────────────────────────
        if self.content_type == "breaking":
            overrides.update({"pace": "fast", "hook_aggressiveness": 1.0, "persona": "urgent"})

        elif self.content_type == "tutorial":
            overrides.update({"pace": "slow", "hook_aggressiveness": 0.4, "persona": "teacher"})

        elif self.content_type == "review":
            overrides.update({"pace": "normal", "hook_aggressiveness": 0.6, "persona": "balanced"})

        elif self.content_type == "opinion":
            overrides.update({"pace": "fast", "hook_aggressiveness": 0.8, "persona": "curious"})

        # ── Topic-signal refinements ───────────────────────────────────────────
        if self.topic_signals.get("is_research"):
            overrides["pace"] = "slow"

        if self.topic_signals.get("mentions_openai"):
            overrides["persona"] = "insider"

        if self.urgency >= 0.8 and self.content_type != "tutorial":
            overrides["pace"] = "fast"
            current_agg = overrides.get("hook_aggressiveness", 0.5)
            overrides["hook_aggressiveness"] = max(current_agg, 0.9)

        if self.complexity >= 0.8:
            overrides.setdefault("persona", "clear_explainer")

        return overrides


# ── ContextAnalyzer ───────────────────────────────────────────────────────────

class ContextAnalyzer:
    """Convert editorial input into a structured ContentContext.

    Typical usage::

        analyzer = ContextAnalyzer()
        ctx      = analyzer.analyze(editorial_input)

        # In SystemOrchestrator:
        for attr, val in ctx.to_strategy_overrides().items():
            if hasattr(strategy, attr):
                setattr(strategy, attr, val)
    """

    def analyze(self, editorial: dict[str, Any]) -> ContentContext:
        """Main entry point.  editorial → ContentContext."""
        topic   = str(editorial.get("topic") or editorial.get("angle") or "")
        summary = str(editorial.get("summary") or "")
        text    = f"{topic} {summary}".lower()

        content_type  = self._detect_type(editorial, text)
        topic_signals = self._extract_topic_signals(text)

        ctx = ContentContext(
            content_type  = content_type,
            topic         = topic,
            novelty       = float(editorial.get("novelty_score", editorial.get("novelty", 0.5))),
            complexity    = float(editorial.get("complexity",    0.5)),
            urgency       = float(editorial.get("urgency",       0.5)),
            topic_signals = topic_signals,
        )

        logger.info(
            f"ContextAnalyzer: type={ctx.content_type!r} topic={ctx.topic!r} "
            f"novelty={ctx.novelty:.2f} complexity={ctx.complexity:.2f} "
            f"urgency={ctx.urgency:.2f} signals={ctx.topic_signals}"
        )
        return ctx

    # ── Type detection ─────────────────────────────────────────────────────────

    def _detect_type(self, editorial: dict[str, Any], text_lower: str) -> str:
        """Priority: explicit flags > keyword match > default 'news'."""
        if editorial.get("is_breaking"):
            return "breaking"
        if editorial.get("is_tutorial"):
            return "tutorial"

        if any(kw in text_lower for kw in _BREAKING_KW):
            return "breaking"
        if any(kw in text_lower for kw in _TUTORIAL_KW):
            return "tutorial"
        if any(kw in text_lower for kw in _REVIEW_KW):
            return "review"
        if any(kw in text_lower for kw in _OPINION_KW):
            return "opinion"

        return "news"

    # ── Topic signal extraction ────────────────────────────────────────────────

    def _extract_topic_signals(self, text_lower: str) -> dict[str, bool]:
        """Keyword-based topic signal detection."""
        return {
            "mentions_openai":   any(kw in text_lower for kw in _OPENAI_KW),
            "mentions_google":   any(kw in text_lower for kw in _GOOGLE_KW),
            "mentions_meta":     any(kw in text_lower for kw in _META_KW),
            "is_research":       any(kw in text_lower for kw in _RESEARCH_KW),
            "is_safety":         any(kw in text_lower for kw in _SAFETY_KW),
            "is_regulation":     any(kw in text_lower for kw in _REGULATION_KW),
            "is_product_launch": any(kw in text_lower for kw in _LAUNCH_KW),
        }
