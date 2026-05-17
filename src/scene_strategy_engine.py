"""Scene Strategy Engine — intent detection + performance-weighted type selection.

Sits between the editorial plan and SceneVarietyEngineV2:

    editorial_plan + scenes + performance_data
        ↓
    SceneStrategyEngine.build_strategy()   → SceneStrategy
        ↓
    SceneVarietyEngineV2.assign()          → mutated scenes (scene_type + scene_intent)
        ↓
    video_generator                        → clips

Key concepts
------------
Intent
    Per-scene semantic goal: hook | explain | shock | data | reaction | transition.
    Detected from content, position, and editorial-plan angle.

INTENT_TO_SCENE
    Each intent maps to an ordered list of candidate scene types.
    SceneStrategyEngine.pick_best() selects the highest-scoring candidate
    using historical retention data from data/scene_performance.json.

Performance data (optional dict passed by caller)
    low_retention_positions : list[int]   scene indices where viewers drop
    high_retention_types    : dict[str, float]  types with above-avg retention
    avg_retention           : float        channel average (0-1)

Scene memory
    data/scene_performance.json — per-type EMA-smoothed retention scores.
    Updated via update_scores(); read at construction time.
    Missing file falls back to _DEFAULT_SCORES silently.
"""
from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import settings

# ── Intent → scene type candidates (ordered: best first) ──────────────────────

INTENT_TO_SCENE: dict[str, list[str]] = {
    "hook":       ["image", "text_overlay"],       # presenter handled separately
    "explain":    ["diagram", "image"],
    "shock":      ["text_overlay", "cutaway"],
    "data":       ["infographic", "image"],
    "reaction":   ["cutaway", "image"],
    "transition": ["image", "cutaway"],
}

# ── Default performance scores (updated by update_scores) ─────────────────────

_DEFAULT_SCORES: dict[str, float] = {
    "image":        0.55,
    "text_overlay": 0.60,
    "infographic":  0.63,
    "diagram":      0.48,
    "cutaway":      0.52,
}

# ── Content detectors ──────────────────────────────────────────────────────────

_DATA_RE = re.compile(
    r"\b\d[\d,\.]*\s*(?:%|percent|million|billion|trillion|x faster|x more|times)"
    r"|\b(?:revenue|growth|users|market|valuation|funding|raised|price)\b",
    re.IGNORECASE,
)
_EXPLAIN_RE = re.compile(
    r"\b(?:why|how|works?|process|step|mechanism|architecture|under the hood|explain|understand)\b",
    re.IGNORECASE,
)
_SHOCK_RE = re.compile(
    r"\b(?:but|problem|shocking|surprising|nobody|no one|secretly|quietly|"
    r"hidden|banned|collapsed|failed|fired|sued)\b",
    re.IGNORECASE,
)
_REACTION_RE = re.compile(
    r"\b(?:crazy|insane|wild|incredible|unbelievable|mind.blowing|massive|stunning)\b",
    re.IGNORECASE,
)

# Angle key → default intent when angle is known
_ANGLE_INTENT: dict[str, str] = {
    "technical_breakthrough": "explain",
    "industry_impact":         "data",
    "threat_to_jobs":          "shock",
    "overhyped_vs_reality":    "shock",
    "what_this_means_for_you": "reaction",
}


# ── Data classes ───────────────────────────────────────────────────────────────

@dataclass
class SceneStrategyItem:
    index: int
    intent: str
    weight_boost: float = 0.0   # added to scene-type scores during pick_best


@dataclass
class SceneStrategy:
    items: list[SceneStrategyItem] = field(default_factory=list)

    def get(self, index: int) -> SceneStrategyItem | None:
        for item in self.items:
            if item.index == index:
                return item
        return None


# ── Engine ─────────────────────────────────────────────────────────────────────

class SceneStrategyEngine:
    """Produce a SceneStrategy and select scene types by performance score."""

    def __init__(self, data_dir: Path | None = None) -> None:
        self._data_dir = data_dir or settings.data_dir
        self._scores: dict[str, float] = self._load_scores()
        self._session_boosts: dict[str, float] = {}   # set by apply_strategy()

    # ── Public API ─────────────────────────────────────────────────────────────

    def apply_strategy(self, strategy_config: Any) -> None:
        """Translate a StrategyConfig.scene_mix into session-level score boosts.

        Higher desired frequency relative to equal-share baseline → positive boost
        (engine favours that type).  Lower → negative boost.  Called once per run
        before build_strategy().

        Example: scene_mix={"text_overlay": 0.25, "image": 0.20, ...}  with 5 types,
        baseline = 0.20.  text_overlay gets +0.025, image gets 0.0 (at baseline).
        """
        scene_mix: dict[str, float] = getattr(strategy_config, "scene_mix", {})
        if not scene_mix:
            self._session_boosts = {}
            return
        baseline = 1.0 / max(len(scene_mix), 1)
        self._session_boosts = {
            t: round((freq - baseline) * 0.5, 4)
            for t, freq in scene_mix.items()
        }
        logger.debug(f"SceneStrategy session boosts: {self._session_boosts}")

    def build_strategy(
        self,
        scenes: list[Any],                      # list[Scene]
        performance_data: dict[str, Any] | None = None,
        editorial_plan: Any | None = None,
        strategy_config: Any | None = None,
    ) -> SceneStrategy:
        """Assign an intent (and optional weight boost) to every scene.

        *performance_data* shapes decisions at retention drop-points and lifts
        proven scene types.  All keys are optional; omit the dict entirely for
        a content-only strategy.
        *strategy_config* (StrategyConfig) is applied via apply_strategy() when
        provided, so callers don't need to call it separately.
        """
        if strategy_config is not None:
            self.apply_strategy(strategy_config)

        performance_data = performance_data or {}
        plan_angle = self._extract_angle(editorial_plan)

        strategy = SceneStrategy()
        for i, scene in enumerate(scenes):
            intent = self._detect_intent(scene, i, plan_angle)
            strategy.items.append(SceneStrategyItem(index=i, intent=intent))

        self._optimize_with_feedback(strategy, performance_data)

        summary = ", ".join(f"[{s.index}]{s.intent}" for s in strategy.items)
        logger.info(f"SceneStrategy: {summary}")
        return strategy

    def pick_best(self, intent: str, weight_boost: float = 0.0) -> str:
        """Return the highest-scoring scene type for *intent*.

        Score = base (from scene_performance.json)
              + session_boost (from apply_strategy scene_mix)
              + weight_boost (from per-scene feedback, NOT applied to "image")
        """
        options = INTENT_TO_SCENE.get(intent, ["image"])

        def _score(t: str) -> float:
            base    = self._scores.get(t, 0.5)
            session = self._session_boosts.get(t, 0.0)
            item_b  = weight_boost if t != "image" else 0.0
            return base + session + item_b

        return max(options, key=_score)

    def update_scores(self, observed: dict[str, float]) -> None:
        """Merge *observed* retention scores using EMA (α = 0.3) and persist.

        Typical caller: analytics pipeline after a video's retention curve
        has been downloaded from YouTube Analytics.

        Example::

            engine.update_scores({"infographic": 0.78, "text_overlay": 0.71})
        """
        alpha = 0.3
        for scene_type, score in observed.items():
            old = self._scores.get(scene_type, 0.5)
            self._scores[scene_type] = round((1 - alpha) * old + alpha * float(score), 4)

        path = self._data_dir / "scene_performance.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self._scores, indent=2))
        logger.info(f"SceneStrategy: scores persisted → {path.name}")

    def get_scores(self) -> dict[str, float]:
        """Return a copy of the current scene-type performance scores."""
        return dict(self._scores)

    # ── Intent detection ───────────────────────────────────────────────────────

    def _detect_intent(self, scene: Any, index: int, plan_angle: str) -> str:
        if index == 0:
            return "hook"

        # infographic_data set by script generator is an unambiguous data signal
        if scene.infographic_data:
            return "data"

        text = f"{scene.heading} {scene.narration}"

        if _DATA_RE.search(text):
            return "data"
        if _EXPLAIN_RE.search(text):
            return "explain"
        if _SHOCK_RE.search(text):
            return "shock"
        if _REACTION_RE.search(text):
            return "reaction"

        # Angle hint (applies when no strong content signal found)
        if plan_angle and plan_angle in _ANGLE_INTENT:
            return _ANGLE_INTENT[plan_angle]

        # Every third non-zero scene without a strong signal → transition
        if index % 3 == 0:
            return "transition"

        return "explain"

    # ── Feedback optimization ──────────────────────────────────────────────────

    def _optimize_with_feedback(
        self, strategy: SceneStrategy, performance_data: dict[str, Any]
    ) -> None:
        """Mutate strategy items in-place using historical retention data."""
        low_retention: set[int] = set(performance_data.get("low_retention_positions", []))
        high_types: dict[str, float] = performance_data.get("high_retention_types", {})
        avg_retention: float = float(performance_data.get("avg_retention", 1.0))

        for item in strategy.items:
            # At known drop points → inject shock/pattern-break intent
            if item.index in low_retention:
                old = item.intent
                item.intent = "shock"
                logger.debug(
                    f"SceneStrategy: scene {item.index} '{old}' → 'shock' "
                    f"(retention drop at position {item.index})"
                )

            # Low channel retention → boost dynamic (non-image) types
            if avg_retention < 0.5 and item.intent == "explain":
                item.weight_boost += 0.1

        # Merge runtime high-retention data into live scores (session only)
        for t, score in high_types.items():
            if t in self._scores:
                self._scores[t] = max(self._scores[t], float(score))

    # ── Scene memory ───────────────────────────────────────────────────────────

    def _load_scores(self) -> dict[str, float]:
        path = self._data_dir / "scene_performance.json"
        try:
            if path.exists():
                raw = json.loads(path.read_text())
                merged = {**_DEFAULT_SCORES, **{k: float(v) for k, v in raw.items()}}
                logger.debug(f"SceneStrategy: loaded scores from {path.name}")
                return merged
        except Exception as exc:
            logger.warning(f"SceneStrategy: could not load scores ({exc}), using defaults")
        return dict(_DEFAULT_SCORES)

    # ── Editorial plan helpers ─────────────────────────────────────────────────

    def _extract_angle(self, editorial_plan: Any) -> str:
        try:
            plans = editorial_plan.editorial_plan
            if plans:
                raw = str(plans[0].get("angle", "")).lower().replace(" ", "_")
                for key in _ANGLE_INTENT:
                    if key in raw:
                        return key
        except Exception as exc:
            logger.debug(f"SceneStrategyEngine: key lookup failed: {exc}")
        return ""
