"""Scene Variety Engine — assigns visual type and intent to each scene.

v1 (SceneVarietyEngine)   — rule-based: text keywords + position → scene_type
v2 (SceneVarietyEngineV2) — strategy-driven: SceneStrategy (from SceneStrategyEngine)
                            → pick_best(intent) using historical scores → scene_type

Both engines share:
- SCENE_TYPES / SCENE_INTENTS constants
- _enforce_variety() pattern-interrupt logic

Typical v2 usage::

    from src.scene_strategy_engine import SceneStrategyEngine
    from src.scene_variety_engine  import SceneVarietyEngineV2

    engine   = SceneStrategyEngine()
    strategy = engine.build_strategy(scenes, performance_data, editorial_plan)
    SceneVarietyEngineV2(engine).assign(scenes, strategy)

Scene types:
    image        – standard DALL-E + Ken Burns (default)
    text_overlay – large bold-text card rendered in PIL (no image API cost)
    infographic  – animated chart/stat via infographic_generator
    cutaway      – standard image with alternate Ken Burns variant (pattern break)
    diagram      – DALL-E prompted toward whiteboard/explainer style

Scene intents:
    hook       – opening grab (scene 0)
    explain    – educational, step-by-step
    shock      – provocative, surprise reveal
    data       – numeric, statistical
    reaction   – emotional, personal takeaway
    transition – pacing break between dense segments
"""
from __future__ import annotations

import re
from typing import Any, TYPE_CHECKING

from loguru import logger

from src.script_generator import Scene

if TYPE_CHECKING:
    from src.scene_strategy_engine import SceneStrategy, SceneStrategyEngine

SCENE_TYPES   = ("image", "text_overlay", "infographic", "cutaway", "diagram")
SCENE_INTENTS = ("hook", "explain", "shock", "data", "reaction", "transition")

_DATA_RE = re.compile(
    r"\b\d[\d,\.]*\s*(?:%|percent|million|billion|trillion|x faster|x more|times)"
    r"|\b(?:revenue|growth|users|market|valuation|funding|raised|price)\b",
    re.IGNORECASE,
)
_EXPLAIN_RE = re.compile(
    r"\b(?:how|works?|process|step|mechanism|architecture|under the hood|explain|understand)\b",
    re.IGNORECASE,
)
_SHOCK_RE = re.compile(
    r"\b(?:shocking|surprising|nobody|no one|secretly|quietly|hidden|banned|collapsed|failed|fired|sued)\b",
    re.IGNORECASE,
)

# Intent implied by editorial angle key
_ANGLE_INTENT: dict[str, str] = {
    "technical_breakthrough": "explain",
    "industry_impact":         "data",
    "threat_to_jobs":          "shock",
    "overhyped_vs_reality":    "shock",
    "what_this_means_for_you": "reaction",
}

# Type implied by scene_type (fallback when no angle)
_TYPE_INTENT: dict[str, str] = {
    "infographic":  "data",
    "diagram":      "explain",
    "text_overlay": "shock",
    "cutaway":      "reaction",
    "image":        "explain",
}


class SceneVarietyEngine:
    """Mutate ``scene.scene_type`` and ``scene.scene_intent`` on every scene."""

    def assign_scene_types(
        self,
        scenes: list[Scene],
        editorial_plan: Any | None = None,
    ) -> list[Scene]:
        """Assign ``scene_type`` and ``scene_intent`` to every scene in *scenes*.

        Mutates in-place and returns the same list for chaining.
        *editorial_plan* is optional; passing an :class:`~src.editorial_brain.EditorialPlan`
        enables format- and angle-aware decisions.
        """
        plan_format = self._extract_format(editorial_plan)
        plan_angle  = self._extract_angle(editorial_plan)

        for i, scene in enumerate(scenes):
            scene.scene_type   = self._decide_type(scene, i, plan_format)
            scene.scene_intent = self._decide_intent(scene, plan_angle)

        self._enforce_variety(scenes)

        if scenes:
            summary = ", ".join(f"[{s.idx}]{s.scene_type}" for s in scenes)
            logger.info(f"SceneVarietyEngine assigned: {summary}")

        return scenes

    # ── Type selection ─────────────────────────────────────────────────────────

    def _decide_type(self, scene: Scene, index: int, plan_format: str) -> str:
        # Infographic_data already set by script generator → honour it
        if scene.infographic_data:
            return "infographic"

        # First scene = hook; image works best (presenter is prepended separately)
        if index == 0:
            return "image"

        text = f"{scene.heading} {scene.narration}"

        if _DATA_RE.search(text):
            return "infographic"

        if _EXPLAIN_RE.search(text):
            return "diagram"

        if _SHOCK_RE.search(text):
            return "text_overlay"

        # hot_take format → cutaway on even non-zero indices
        if plan_format == "hot_take" and index % 2 == 0:
            return "cutaway"

        return "image"

    # ── Intent selection ───────────────────────────────────────────────────────

    def _decide_intent(self, scene: Scene, plan_angle: str) -> str:
        if plan_angle in _ANGLE_INTENT:
            return _ANGLE_INTENT[plan_angle]
        return _TYPE_INTENT.get(scene.scene_type, "explain")

    # ── Pattern interrupt ──────────────────────────────────────────────────────

    def _enforce_variety(self, scenes: list[Scene]) -> None:
        """Replace consecutive duplicate types with 'cutaway'.

        Only the type is changed; intent set by _decide_intent (from angle) is preserved.
        """
        for i in range(1, len(scenes)):
            if (
                scenes[i].scene_type == scenes[i - 1].scene_type
                and scenes[i].scene_type != "cutaway"
            ):
                scenes[i].scene_type = "cutaway"
                logger.debug(
                    f"SceneVarietyEngine: scene {scenes[i].idx} → cutaway (pattern interrupt)"
                )

    # ── Editorial plan helpers ─────────────────────────────────────────────────

    def _extract_format(self, editorial_plan: Any) -> str:
        try:
            plans = editorial_plan.editorial_plan
            if plans:
                return str(plans[0].get("format", ""))
        except Exception:
            pass
        return ""

    def _extract_angle(self, editorial_plan: Any) -> str:
        try:
            plans = editorial_plan.editorial_plan
            if plans:
                raw = str(plans[0].get("angle", "")).lower().replace(" ", "_")
                for key in _ANGLE_INTENT:
                    if key in raw:
                        return key
        except Exception:
            pass
        return ""


# ── v2: Strategy-driven engine ─────────────────────────────────────────────────

class SceneVarietyEngineV2:
    """Assign scene_type + scene_intent using a pre-built SceneStrategy.

    The strategy engine detects intent per scene (content + position + feedback).
    This engine maps intent → scene_type via performance-weighted selection, then
    enforces variety constraints (no same type on consecutive scenes).
    """

    def __init__(self, strategy_engine: "SceneStrategyEngine | None" = None) -> None:
        if strategy_engine is None:
            from src.scene_strategy_engine import SceneStrategyEngine
            strategy_engine = SceneStrategyEngine()
        self._engine = strategy_engine

    def assign(self, scenes: list[Scene], strategy: "SceneStrategy") -> list[Scene]:
        """Mutate ``scene.scene_type`` and ``scene.scene_intent`` from *strategy*.

        Returns the same list for chaining.
        """

        for scene in scenes:
            item = strategy.get(scene.idx)
            if item is None:
                scene.scene_type   = "image"
                scene.scene_intent = "explain"
                continue

            scene.scene_intent = item.intent
            scene.scene_type   = self._engine.pick_best(item.intent, item.weight_boost)

            # Honour infographic_data regardless of strategy (already rendered)
            if scene.infographic_data and scene.scene_type != "infographic":
                scene.scene_type = "infographic"

        self._enforce_variety(scenes)

        if scenes:
            summary = ", ".join(
                f"[{s.idx}]{s.scene_intent}→{s.scene_type}" for s in scenes
            )
            logger.info(f"SceneVarietyEngineV2 assigned: {summary}")

        return scenes

    def assign_from_config(
        self,
        scenes: list[Scene],
        strategy_config: Any,
        editorial_plan: Any | None = None,
    ) -> list[Scene]:
        """Full v2 pipeline driven by StrategyConfig (from DecisionEngineV2).

        Unlike assign(), which requires a pre-built SceneStrategy, this method
        builds everything from the config's ScenePolicy and scene content.

        Steps:
        1. apply_strategy()             – session boosts from scene_mix
        2. _assign_intents()            – content + position + editorial plan → intent
        3. _apply_scene_mix()           – intent-filtered weighted random draw
        4. _apply_pattern_interrupts()  – cutaway every N scenes (policy-driven)
        5. _enforce_variety()           – no consecutive same types
        6. _enforce_avatar_frequency()  – avatar as managed resource
        7. honour infographic_data      – final override
        """
        policy = strategy_config.scene_policy
        self._engine.apply_strategy(strategy_config)

        self._assign_intents(scenes, editorial_plan)
        self._apply_scene_mix(scenes, policy)
        self._apply_pattern_interrupts(scenes, policy)
        if policy.avoid_repetition:
            self._enforce_variety(scenes)
        self._enforce_avatar_frequency(scenes, strategy_config)

        # infographic_data is an unambiguous renderer contract — always honour it
        for scene in scenes:
            if scene.infographic_data:
                scene.scene_type = "infographic"

        if scenes:
            summary = ", ".join(
                f"[{s.idx}]{s.scene_intent}→{s.scene_type}" for s in scenes
            )
            logger.info(f"SceneVarietyEngineV2.assign_from_config: {summary}")

        return scenes

    # ── assign_from_config helpers ─────────────────────────────────────────────

    def _assign_intents(
        self,
        scenes: list[Scene],
        editorial_plan: Any | None = None,
    ) -> None:
        """Content + position + editorial-plan angle → scene_intent."""
        from src.scene_strategy_engine import _ANGLE_INTENT, _DATA_RE, _EXPLAIN_RE, _SHOCK_RE, _REACTION_RE

        plan_angle = self._extract_angle(editorial_plan)

        for i, scene in enumerate(scenes):
            if i == 0:
                scene.scene_intent = "hook"
                continue
            if scene.infographic_data:
                scene.scene_intent = "data"
                continue
            text = f"{scene.heading} {scene.narration}"
            if _DATA_RE.search(text):
                scene.scene_intent = "data"
            elif _EXPLAIN_RE.search(text):
                scene.scene_intent = "explain"
            elif _SHOCK_RE.search(text):
                scene.scene_intent = "shock"
            elif _REACTION_RE.search(text):
                scene.scene_intent = "reaction"
            elif plan_angle and plan_angle in _ANGLE_INTENT:
                scene.scene_intent = _ANGLE_INTENT[plan_angle]
            elif i % 3 == 0:
                scene.scene_intent = "transition"
            else:
                scene.scene_intent = "explain"

    def _apply_scene_mix(self, scenes: list[Scene], policy: Any) -> None:
        """Weighted random scene_type draw, filtered to each intent's candidates.

        Priority intents (hook/shock by default) bypass the random draw and use
        pick_best() so the most important scenes always get the strongest type.
        """
        import random as _random
        from src.scene_strategy_engine import INTENT_TO_SCENE

        for scene in scenes:
            # Priority intents → deterministic best pick
            if scene.scene_intent in policy.priority_intents:
                scene.scene_type = self._engine.pick_best(scene.scene_intent)
                continue

            if not policy.mix:
                scene.scene_type = self._engine.pick_best(scene.scene_intent)
                continue

            # Weighted random draw within intent's candidate types
            candidates = INTENT_TO_SCENE.get(scene.scene_intent, ["image"])
            weights    = [policy.mix.get(t, 0.0) for t in candidates]
            total      = sum(weights)
            if total <= 0:
                scene.scene_type = candidates[0]
                continue

            r, cumulative = _random.random() * total, 0.0
            chosen = candidates[0]
            for t, w in zip(candidates, weights):
                cumulative += w
                if r <= cumulative:
                    chosen = t
                    break
            scene.scene_type = chosen

    def _apply_pattern_interrupts(self, scenes: list[Scene], policy: Any) -> None:
        """Force 'cutaway' every pattern_interrupt_frequency scenes (skipping scene 0)."""
        freq = policy.pattern_interrupt_frequency
        if freq <= 0:
            return
        for i, scene in enumerate(scenes):
            if i == 0:
                continue
            if i % freq == 0:
                scene.scene_type = "cutaway"
                logger.debug(
                    f"SceneVarietyEngineV2: scene {scene.idx} → cutaway "
                    f"(pattern interrupt at position {i})"
                )

    def _enforce_avatar_frequency(
        self, scenes: list[Scene], strategy_config: Any
    ) -> None:
        """Replace a share of non-hook scenes with 'avatar' per avatar_frequency.

        Avatar scenes are spread evenly across eligible (non-hook) scenes.
        """
        freq = getattr(strategy_config, "avatar_frequency", 0.0)
        if freq <= 0.0:
            return

        eligible = [s for s in scenes if s.scene_intent != "hook"]
        if not eligible:
            return

        n_avatar = max(1, round(len(eligible) * freq))
        step     = max(1, len(eligible) // n_avatar)

        count = 0
        for i, scene in enumerate(eligible):
            if count >= n_avatar:
                break
            if i % step == 0:
                scene.scene_type = "avatar"
                count += 1

    def _extract_angle(self, editorial_plan: Any) -> str:
        try:
            from src.scene_strategy_engine import _ANGLE_INTENT
            plans = editorial_plan.editorial_plan
            if plans:
                raw = str(plans[0].get("angle", "")).lower().replace(" ", "_")
                for key in _ANGLE_INTENT:
                    if key in raw:
                        return key
        except Exception:
            pass
        return ""

    def _enforce_variety(self, scenes: list[Scene]) -> None:
        """No same scene_type on consecutive scenes — break with 'cutaway'."""
        for i in range(1, len(scenes)):
            if (
                scenes[i].scene_type == scenes[i - 1].scene_type
                and scenes[i].scene_type != "cutaway"
            ):
                scenes[i].scene_type = "cutaway"
                logger.debug(
                    f"SceneVarietyEngineV2: scene {scenes[i].idx} → cutaway "
                    f"(pattern interrupt after {scenes[i-1].scene_type})"
                )
