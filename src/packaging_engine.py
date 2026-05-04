"""Packaging Engine — synchronised hook / title / thumbnail concept from editorial plan.

Takes the editorial plan (angle + conflict + story context) and the channel
strategy (StrategyConfig from DecisionEngineV2) and produces three mutually
consistent packaging outputs that drive viewer acquisition:

    hook              — opening 5-10 s spoken line  (CTR + retention start)
    title             — YouTube title, 5–10 words, conflict-first
    thumbnail_concept — {text, style, emotion} spec for ThumbnailGenerator

Flow
----
editorial_plan + strategy_config
    ↓
PackagingEngine.generate()
    ↓
PackagingResult {idea, hook, title, thumbnail}
    ↓
EditorialBrain   → editorial_plan["core_idea"]  (optional write-back)
ScriptGenerator  → script.hook
ThumbnailGenerator → packaging.thumbnail concept
YouTubeUploader  → packaging.title

Hook consistency guarantee
--------------------------
Thumbnail text, video title, and the first 10 seconds of spoken narration
must all surface the same *core idea*.  PackagingEngine enforces this by
deriving hook, title, and thumbnail from one shared *idea* string.

Performance store
-----------------
data/packaging_performance.json — append-only list of
  {video_id, timestamp, title, title_pattern, thumbnail_style, ctr}
Used by PackagingStore.get_best_patterns() to learn which formulas win.
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import settings


# ── Angle → core idea ─────────────────────────────────────────────────────────

# Keyed by editorial_brain angle_key AND simplified aliases from spec
_ANGLE_KEY_TO_IDEA: dict[str, str] = {
    # real keys from editorial_brain.ANGLE_CANDIDATES
    "technical_breakthrough": "Everyone is hyped — but there is a hidden problem",
    "overhyped_vs_reality":   "Everyone is hyped — but there is a hidden problem",
    "industry_impact":        "This changes everything for the industry",
    "threat_to_jobs":         "This could cost people their jobs",
    "what_this_means_for_you": "Here is what this really means for you",
    # simplified aliases (as in spec)
    "hype_vs_reality": "Everyone is hyped — but there is a hidden problem",
    "breakthrough":    "This changes everything",
    "risk":            "This could go wrong",
}

# Keyed by editorial_brain conflict string
_CONFLICT_TO_IDEA: dict[str, str] = {
    "hype vs reality":        "Everyone is hyped — but there is a hidden problem",
    "innovation vs adoption": "This changes everything for the industry",
    "growth vs jobs":         "This could cost people their jobs",
    "promise vs practical use": "Here is what this really means for you",
}

_DEFAULT_IDEA = "Something important is happening"


# ── Title patterns (per idea family) ─────────────────────────────────────────

_TITLE_PATTERNS: dict[str, tuple[str, str]] = {
    # (pattern_key, title_text)
    "hype_problem": ("hype_problem",  "The Hidden Problem No One Talks About"),
    "game_changer": ("game_changer",  "This Changes Everything"),
    "jobs_risk":    ("jobs_risk",     "The Jobs That Are Disappearing Now"),
    "practical":    ("practical",     "What You Actually Need to Know"),
    "default":      ("default",       "What You Need to Know About This"),
}


# ── Thumbnail concepts (per idea family) ──────────────────────────────────────

_THUMBNAIL_CONCEPTS: dict[str, dict[str, str]] = {
    "hype_problem": {"text": "THE PROBLEM",    "style": "high_contrast", "emotion": "shock"},
    "game_changer": {"text": "GAME CHANGER",   "style": "clean",         "emotion": "curiosity"},
    "jobs_risk":    {"text": "JOBS AT RISK",   "style": "high_contrast", "emotion": "shock"},
    "practical":    {"text": "WHAT THIS MEANS","style": "clean",         "emotion": "curiosity"},
    "default":      {"text": "WHAT IS THIS?",  "style": "neutral",       "emotion": "curiosity"},
}


# ── Result type ───────────────────────────────────────────────────────────────

@dataclass
class PackagingResult:
    """All packaging outputs derived from a single core idea.

    All three outputs (hook, title, thumbnail) are guaranteed to express the
    same *idea* so that thumbnail → title → opening hook form a coherent funnel.
    """
    idea: str
    hook: str
    title: str
    thumbnail: dict[str, str]
    title_pattern: str = field(default="default", repr=False)

    def to_dict(self) -> dict[str, Any]:
        return {
            "idea":          self.idea,
            "hook":          self.hook,
            "title":         self.title,
            "thumbnail":     self.thumbnail,
            "title_pattern": self.title_pattern,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "PackagingResult":
        return cls(
            idea          = d.get("idea",          ""),
            hook          = d.get("hook",          ""),
            title         = d.get("title",         ""),
            thumbnail     = d.get("thumbnail",     {}),
            title_pattern = d.get("title_pattern", "default"),
        )


# ── PackagingEngine ───────────────────────────────────────────────────────────

class PackagingEngine:
    """Produce synchronised hook / title / thumbnail from editorial plan + strategy.

    All three outputs are derived from one *core idea* extracted from the
    editorial plan, so they remain consistent across the full pipeline.
    """

    # ── Public API ─────────────────────────────────────────────────────────────

    def generate(
        self,
        editorial_plan: Any,
        strategy_config: Any,
    ) -> PackagingResult:
        """Main entry point.

        *editorial_plan* may be:
        - an ``EditorialPlan`` object  (``editorial_plan.editorial_plan[0]``)
        - a list of plan dicts        (first entry used)
        - a single plan dict          (used directly)

        *strategy_config* may be a ``StrategyConfig`` object or a plain dict
        with a ``"mode"`` key.
        """
        plan_entry = self._extract_plan_entry(editorial_plan)
        idea       = self._extract_core_idea(plan_entry)
        hook       = self._generate_hook(idea, strategy_config)
        title      = self._generate_title(idea, hook)
        thumbnail  = self._generate_thumbnail(idea, hook)
        pattern    = self._idea_to_family(idea)

        result = PackagingResult(
            idea          = idea,
            hook          = hook,
            title         = title,
            thumbnail     = thumbnail,
            title_pattern = pattern,
        )

        self._log_result(result)
        return result

    def generate_variants(
        self,
        editorial_plan: Any,
        strategy_config: Any,
    ) -> list:
        """Generate three packaging variants for A/B testing.

        Each variant tests a different emotional angle (curiosity / conflict /
        simple) derived from the same core idea, so the test measures *style*
        not just text.

        Returns a list of :class:`~src.ab_testing_engine.ABTestVariant`.
        """
        from src.ab_testing_engine import ABTestVariant

        plan_entry = self._extract_plan_entry(editorial_plan)
        idea       = self._extract_core_idea(plan_entry)

        return [
            self._variant_curiosity(idea, strategy_config, ABTestVariant),
            self._variant_conflict(idea,  strategy_config, ABTestVariant),
            self._variant_simple(idea,    strategy_config, ABTestVariant),
        ]

    # ── Variant builders ───────────────────────────────────────────────────────

    def _variant_curiosity(self, idea: str, strategy_config: Any, cls: type) -> Any:
        """Open-loop curiosity variant — builds anticipation, withholds the answer."""
        return cls(
            id        = "A",
            type      = "curiosity",
            title     = "This changes everything...",
            thumbnail = {"text": "WAIT WHAT?", "style": "clean", "emotion": "curiosity"},
            hook      = f"Wait — {idea}... and nobody saw this coming",
        )

    def _variant_conflict(self, idea: str, strategy_config: Any, cls: type) -> Any:
        """Explicit conflict variant — names the problem, maximises urgency."""
        return cls(
            id        = "B",
            type      = "conflict",
            title     = "The problem nobody talks about",
            thumbnail = {"text": "THE PROBLEM", "style": "high_contrast", "emotion": "shock"},
            hook      = f"{idea}... but here's what nobody tells you",
        )

    def _variant_simple(self, idea: str, strategy_config: Any, cls: type) -> Any:
        """Clean explainer variant — clarity over urgency, lower bounce risk."""
        return cls(
            id        = "C",
            type      = "simple",
            title     = "What you need to know",
            thumbnail = {"text": "EXPLAINED", "style": "neutral", "emotion": "curiosity"},
            hook      = f"Let's break this down clearly: {idea.lower()}",
        )



    def _extract_core_idea(self, plan_entry: dict[str, Any]) -> str:
        """Derive a single conflict-rooted sentence from the editorial plan entry.

        Lookup order:
        1. ``angle_key`` field (if stored directly in plan)
        2. ``conflict`` field  → ``_CONFLICT_TO_IDEA``
        3. ``angle`` label     → keyword scan → ``_ANGLE_KEY_TO_IDEA``
        4. Default fallback
        """
        # 1. Explicit angle_key (not always present — editorial_brain stores label)
        angle_key = plan_entry.get("angle_key", "")
        if angle_key and angle_key in _ANGLE_KEY_TO_IDEA:
            return _ANGLE_KEY_TO_IDEA[angle_key]

        # 2. Conflict string
        conflict = plan_entry.get("conflict", "").lower().strip()
        if conflict in _CONFLICT_TO_IDEA:
            return _CONFLICT_TO_IDEA[conflict]

        # 3. Angle label keyword scan — hype/problem signals take priority
        angle_label = plan_entry.get("angle", "").lower()
        if any(w in angle_label for w in ("overhype", "overhyped", "hype", "hidden", "but might", "catch")):
            return _ANGLE_KEY_TO_IDEA["hype_vs_reality"]
        if any(w in angle_label for w in ("job", "replac", "threat", "danger")):
            return _ANGLE_KEY_TO_IDEA["risk"]

        # 4. Generic angle key fragment scan
        for key, idea in _ANGLE_KEY_TO_IDEA.items():
            if key.replace("_", " ") in angle_label or key in angle_label:
                return idea

        # 5. Remaining keyword scan
        if any(w in angle_label for w in ("break", "chang", "reshape", "transform")):
            return _ANGLE_KEY_TO_IDEA["breakthrough"]
        if any(w in angle_label for w in ("risk", "wrong", "problem")):
            return _ANGLE_KEY_TO_IDEA["risk"]

        return _DEFAULT_IDEA

    # ── Hook ───────────────────────────────────────────────────────────────────

    def _generate_hook(self, idea: str, strategy_config: Any) -> str:
        """Produce an opening hook line tuned to the channel strategy mode."""
        mode = self._extract_mode(strategy_config)

        if mode == "growth":
            return f"{idea}... but here's what nobody tells you"

        if mode == "safe":
            return f"Let's break this down clearly: {idea.lower()}"

        if mode == "experimental":
            return f"What if everything you know about this is wrong? {idea}"

        # fallback (unknown mode)
        return idea

    # ── Title ──────────────────────────────────────────────────────────────────

    def _generate_title(self, idea: str, hook: str) -> str:
        """Return a YouTube title: 5–10 words, conflict-first, plain language."""
        family = self._idea_to_family(idea)
        pattern_key, title = _TITLE_PATTERNS.get(family, _TITLE_PATTERNS["default"])
        return title

    # ── Thumbnail concept ──────────────────────────────────────────────────────

    def _generate_thumbnail(self, idea: str, hook: str) -> dict[str, str]:
        """Return a thumbnail concept dict: text, style, emotion.

        Deliberately does NOT generate the image — that is ThumbnailGenerator's job.
        """
        family = self._idea_to_family(idea)
        return dict(_THUMBNAIL_CONCEPTS.get(family, _THUMBNAIL_CONCEPTS["default"]))

    # ── Consistency check ──────────────────────────────────────────────────────

    def check_consistency(self, result: PackagingResult) -> bool:
        """Verify that hook, title, and thumbnail share the same emotional register.

        Returns True if aligned; logs a warning and returns False if not.
        The check is heuristic: all three must belong to the same idea family.
        """
        hook_family      = self._idea_to_family(result.hook)
        title_family     = self._idea_to_family(result.title)
        thumbnail_family = _THUMBNAIL_CONCEPTS.get(
            next(
                (k for k, v in _THUMBNAIL_CONCEPTS.items()
                 if v["text"] == result.thumbnail.get("text", "")),
                "default",
            ),
            None,
        )
        families = {hook_family, title_family}
        if len(families) > 1:
            logger.warning(
                f"PackagingEngine: consistency check FAILED — "
                f"hook_family={hook_family!r}, title_family={title_family!r}"
            )
            return False
        return True

    # ── Helpers ────────────────────────────────────────────────────────────────

    def _extract_plan_entry(self, editorial_plan: Any) -> dict[str, Any]:
        """Normalise editorial_plan to a single dict."""
        try:
            # EditorialPlan object
            plans = editorial_plan.editorial_plan
            if isinstance(plans, list) and plans:
                return dict(plans[0])
        except AttributeError:
            pass
        if isinstance(editorial_plan, list) and editorial_plan:
            return dict(editorial_plan[0])
        if isinstance(editorial_plan, dict):
            return dict(editorial_plan)
        return {}

    def _extract_mode(self, strategy_config: Any) -> str:
        """Return the mode string from a StrategyConfig object or dict."""
        try:
            return strategy_config.mode
        except AttributeError:
            pass
        if isinstance(strategy_config, dict):
            return strategy_config.get("mode", "growth")
        return "growth"

    def _idea_to_family(self, text: str) -> str:
        """Map any text to one of the five thumbnail/title families."""
        t = text.lower()
        if any(w in t for w in ("problem", "hidden", "catch", "hype", "nobody")):
            return "hype_problem"
        if any(w in t for w in ("changes everything", "game changer", "game chang", "transform", "reshape")):
            return "game_changer"
        if any(w in t for w in ("job", "career", "replac", "disappear", "cost people")):
            return "jobs_risk"
        if any(w in t for w in ("means for you", "what this", "what you", "practical", "actually")):
            return "practical"
        return "default"

    def _log_result(self, result: PackagingResult) -> None:
        logger.info(
            f"PackagingEngine: idea={result.idea!r} | "
            f"hook={result.hook[:60]!r}… | "
            f"title={result.title!r} | "
            f"thumbnail={result.thumbnail}"
        )


# ── PackagingStore ────────────────────────────────────────────────────────────

class PackagingStore:
    """Append-only JSON log of packaging results + CTR for feedback learning."""

    def __init__(self, data_dir: Path | None = None) -> None:
        self._path = (data_dir or settings.data_dir) / "packaging_performance.json"

    def record(
        self,
        video_id: str,
        result: PackagingResult,
        ctr: float,
    ) -> None:
        """Persist a post-publish CTR observation for *result*."""
        entries = self._load()
        entries.append({
            "video_id":        video_id,
            "timestamp":       datetime.now(timezone.utc).isoformat(),
            "title":           result.title,
            "title_pattern":   result.title_pattern,
            "thumbnail_style": result.thumbnail.get("style", "neutral"),
            "thumbnail_emotion": result.thumbnail.get("emotion", "curiosity"),
            "ctr":             ctr,
        })
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(json.dumps(entries, indent=2))
        logger.debug(f"PackagingStore: saved CTR={ctr:.4f} for {video_id!r}")

    def get_best_patterns(self, top_n: int = 3) -> dict[str, float]:
        """Return the top *top_n* title patterns ranked by average CTR."""
        entries = self._load()
        if not entries:
            return {}
        groups: dict[str, list[float]] = defaultdict(list)
        for e in entries:
            groups[e.get("title_pattern", "default")].append(float(e.get("ctr", 0)))
        avg = {p: round(sum(v) / len(v), 4) for p, v in groups.items()}
        ranked = sorted(avg.items(), key=lambda x: x[1], reverse=True)
        return dict(ranked[:top_n])

    def get_best_thumbnail_styles(self, top_n: int = 3) -> dict[str, float]:
        """Return the top *top_n* thumbnail styles ranked by average CTR."""
        entries = self._load()
        if not entries:
            return {}
        groups: dict[str, list[float]] = defaultdict(list)
        for e in entries:
            groups[e.get("thumbnail_style", "neutral")].append(float(e.get("ctr", 0)))
        avg = {s: round(sum(v) / len(v), 4) for s, v in groups.items()}
        ranked = sorted(avg.items(), key=lambda x: x[1], reverse=True)
        return dict(ranked[:top_n])

    def _load(self) -> list[dict[str, Any]]:
        try:
            if self._path.exists():
                data = json.loads(self._path.read_text())
                if isinstance(data, list):
                    return data
        except Exception as exc:
            logger.warning(f"PackagingStore: could not read store ({exc})")
        return []
