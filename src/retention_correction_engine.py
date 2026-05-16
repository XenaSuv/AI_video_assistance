"""Retention Correction Engine — drop-point detection and scene-level fixes.

YouTube viewers leave at specific moments, not "on average".  This module
closes that feedback loop: analyse the retention curve, map each drop to the
scene responsible, generate a targeted fix, and apply it before the next
video is generated.

Data flow
---------
1.  YouTube Analytics → retention_data dict
2.  detect_drops()           → list[DropPoint]
3.  RetentionCorrectionEngine.analyze() → list[Correction]
4.  apply_corrections()      → modifies Scene objects in place
5.  suggest_strategy_adjustments() → hints for DecisionEngineV2
6.  record_outcome()         → trains the learning store

Scene map
---------
Maps retention-curve timestamp indices to scene metadata::

    scene_map = {
        0: {"scene_idx": 0, "scene_type": "avatar",       "intent": "hook"},
        1: {"scene_idx": 1, "scene_type": "image",        "intent": "explain"},
        2: {"scene_idx": 1, "scene_type": "image",        "intent": "explain"},
        3: {"scene_idx": 2, "scene_type": "text_overlay", "intent": "transition"},
    }

Multiple timestamp indices may map to the same scene (long scenes span several
30-second buckets).  Corrections are deduplicated by scene_idx.

Fix generation rules
--------------------
Priority order:
1.  Learned fix  — if CorrectionStore has a proven winner for this
                   (scene_type, intent) context, use it.
2.  Rule-based   — deterministic fallback:

    scene_type == "image"          → REPLACE_WITH_AVATAR
    intent     == "explain"        → SHORTEN_AND_ADD_HOOK
    intent     == "transition"     → CUT_OR_SPEED_UP
    intent     == "hook"           → INCREASE_PACING
    *fallback*                     → ADD_MICRO_HOOK

apply_corrections
-----------------
Modifies Scene objects (from src/script_generator.py) in place:

    REPLACE_WITH_AVATAR  → scene.scene_type = "avatar"
    INSERT_AVATAR        → scene.scene_type = "avatar"  (same effect on existing scene)
    SHORTEN_AND_ADD_HOOK → scene.duration_sec *= 0.7
    CUT_OR_SPEED_UP      → scene.duration_sec *= 0.6
    INCREASE_PACING      → scene.duration_sec *= 0.75
    ADD_MICRO_HOOK       → scene.visual_prompt prepended with micro-hook hint
    REMOVE_SCENE         → scene.duration_sec = 0  (caller decides whether to drop it)

suggest_strategy_adjustments
-----------------------------
Returns a dict for DecisionEngineV2:

    "pace"                   → "fast" when early drops detected
    "hook_aggressiveness"    → float delta to add
    "force_avatar_intro"     → True when scene-0 drop detected

Learning
--------
CorrectionStore records each (fix, scene_type, intent, delta_retention) tuple.
get_best_fix() returns the fix with the highest mean delta_retention for a
given context; returns None when fewer than MIN_SAMPLES observations exist.
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import settings


# ── Fix types ─────────────────────────────────────────────────────────────────

class FixType(str, Enum):
    ADD_MICRO_HOOK       = "add_micro_hook"
    SHORTEN              = "shorten"
    SHORTEN_AND_ADD_HOOK = "shorten_and_add_hook"
    REPLACE_WITH_AVATAR  = "replace_with_avatar"
    INSERT_AVATAR        = "insert_avatar"
    INCREASE_PACING      = "increase_pacing"
    CUT_OR_SPEED_UP      = "cut_or_speed_up"
    REMOVE_SCENE         = "remove_scene"


# ── Data types ────────────────────────────────────────────────────────────────

@dataclass
class DropPoint:
    """A detected drop in the retention curve."""
    timestamp_idx: int    # index into retention_data["curve"]
    delta:         float  # magnitude of the drop (positive)

    def to_dict(self) -> dict[str, Any]:
        return {"timestamp_idx": self.timestamp_idx, "delta": self.delta}


@dataclass
class Correction:
    """A fix recommendation for one scene."""
    scene_idx:     int
    fix:           FixType
    drop:          DropPoint
    scene_type:    str = ""
    intent:        str = ""
    confidence:    float = 1.0  # 1.0 = rule-based; higher = learning-backed

    def to_dict(self) -> dict[str, Any]:
        return {
            "scene_idx":  self.scene_idx,
            "fix":        self.fix.value,
            "drop":       self.drop.to_dict(),
            "scene_type": self.scene_type,
            "intent":     self.intent,
            "confidence": round(self.confidence, 4),
        }


# ── Drop detection ─────────────────────────────────────────────────────────────

def detect_drops(curve: list[float], threshold: float = 0.15) -> list[DropPoint]:
    """Return DropPoints where retention falls more than *threshold* in one step.

    Args:
        curve:     retention values in [0, 1], ordered by time (index 0 = start).
        threshold: minimum fractional drop to qualify (default 0.15 = 15 pp).
    """
    drops = []
    for i in range(1, len(curve)):
        delta = curve[i - 1] - curve[i]
        if delta > threshold:
            drops.append(DropPoint(timestamp_idx=i, delta=round(delta, 4)))
    return drops


# ── Learning store ─────────────────────────────────────────────────────────────

class CorrectionStore:
    """JSON-backed record of applied fixes and their measured outcomes.

    Records are grouped by (fix_type, scene_type, intent) context.
    get_best_fix() returns the fix with the highest mean delta_retention
    for a given (scene_type, intent) pair when MIN_SAMPLES are available.
    """

    MIN_SAMPLES = 3   # minimum records before learning overrides rules

    def __init__(self, data_dir: Path | None = None) -> None:
        self._path = (data_dir or settings.data_dir) / "correction_history.json"
        self._records: list[dict[str, Any]] = self._load()

    def _load(self) -> list[dict[str, Any]]:
        try:
            if self._path.exists():
                return json.loads(self._path.read_text())
        except Exception as exc:
            logger.warning(f"CorrectionStore: could not load ({exc}), starting fresh")
        return []

    def _save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(json.dumps(self._records, indent=2))

    def record(
        self,
        fix_type:        FixType | str,
        scene_type:      str,
        intent:          str,
        delta_retention: float,
    ) -> None:
        """Save the measured outcome of an applied fix.

        *delta_retention* is the change in retention at the fixed scene's
        timestamp between the video where the fix was applied and the prior
        video (positive = improvement).
        """
        self._records.append({
            "fix":             fix_type.value if isinstance(fix_type, FixType) else fix_type,
            "scene_type":      scene_type,
            "intent":          intent,
            "delta_retention": round(delta_retention, 4),
        })
        self._save()
        logger.info(
            f"CorrectionStore: recorded fix={fix_type!r} "
            f"scene_type={scene_type!r} intent={intent!r} "
            f"delta_retention={delta_retention:+.4f}"
        )

    def get_best_fix(self, scene_type: str, intent: str) -> FixType | None:
        """Return the fix with the best mean delta_retention for this context.

        Returns None if fewer than MIN_SAMPLES observations exist for any fix
        in this (scene_type, intent) bucket.
        """
        relevant = [
            r for r in self._records
            if r["scene_type"] == scene_type and r["intent"] == intent
        ]
        if not relevant:
            return None

        # Aggregate by fix type
        totals: dict[str, list[float]] = {}
        for r in relevant:
            totals.setdefault(r["fix"], []).append(r["delta_retention"])

        # Only consider fix types with enough samples
        qualified = {
            fix: deltas
            for fix, deltas in totals.items()
            if len(deltas) >= self.MIN_SAMPLES
        }
        if not qualified:
            return None

        best_fix = max(qualified, key=lambda f: sum(qualified[f]) / len(qualified[f]))
        try:
            return FixType(best_fix)
        except ValueError:
            return None

    def all_records(self) -> list[dict[str, Any]]:
        return list(self._records)


# ── Main engine ────────────────────────────────────────────────────────────────

_HOOK_MICRO = "[HOOK] "   # prepended to visual_prompt for ADD_MICRO_HOOK


class RetentionCorrectionEngine:
    """Analyses retention curves, maps drops to scenes, and generates fixes.

    Typical lifecycle per published video::

        engine = RetentionCorrectionEngine(data_dir=run_dir)

        corrections = engine.analyze(retention_data, scene_map)
        engine.apply_corrections(script.scenes, corrections)

        adjustments = engine.suggest_strategy_adjustments(corrections, scene_map,
                                                          retention_data["curve"])

        # After next video is published and analytics are collected:
        engine.record_outcome(fix_type, scene_type, intent, delta_retention)
    """

    DEFAULT_DROP_THRESHOLD = 0.15

    def __init__(
        self,
        data_dir:       Path | None = None,
        drop_threshold: float       = DEFAULT_DROP_THRESHOLD,
    ) -> None:
        self._store         = CorrectionStore(data_dir)
        self.drop_threshold = drop_threshold

    # ── Analysis ───────────────────────────────────────────────────────────────

    def analyze(
        self,
        retention_data: dict[str, Any],
        scene_map:      dict[int, dict[str, Any]],
    ) -> list[Correction]:
        """Detect drops, map to scenes, and generate fix recommendations.

        Args:
            retention_data: dict with keys "curve" (list[float]) and optionally
                            "timestamps" (list[int | float]).
            scene_map:      maps timestamp_index → scene info dict with at least
                            "scene_idx", "scene_type", "intent".

        Returns a deduplicated list of Correction objects (one per unique scene).
        """
        curve = retention_data.get("curve", [])
        drops = detect_drops(curve, threshold=self.drop_threshold)

        if not drops:
            logger.info("RetentionCorrectionEngine: no drops detected")
            return []

        seen_scene_idx: set[int] = set()
        corrections: list[Correction] = []

        for drop in drops:
            scene_info = scene_map.get(drop.timestamp_idx)
            if scene_info is None:
                logger.debug(
                    f"RetentionCorrectionEngine: no scene mapped to "
                    f"timestamp_idx={drop.timestamp_idx}, skipping"
                )
                continue

            scene_idx  = scene_info.get("scene_idx", drop.timestamp_idx)
            scene_type = scene_info.get("scene_type", "image")
            intent     = scene_info.get("intent",     "explain")

            if scene_idx in seen_scene_idx:
                continue
            seen_scene_idx.add(scene_idx)

            fix, confidence = self._generate_fix(scene_type, intent)
            corrections.append(
                Correction(
                    scene_idx  = scene_idx,
                    fix        = fix,
                    drop       = drop,
                    scene_type = scene_type,
                    intent     = intent,
                    confidence = confidence,
                )
            )
            logger.info(
                f"RetentionCorrectionEngine: scene={scene_idx} "
                f"type={scene_type!r} intent={intent!r} "
                f"drop={drop.delta:.2f} → fix={fix.value!r} "
                f"(confidence={confidence:.2f})"
            )

        return corrections

    def _generate_fix(self, scene_type: str, intent: str) -> tuple[FixType, float]:
        """Return (fix_type, confidence).

        Checks the learning store first; falls back to rule-based logic.
        """
        learned = self._store.get_best_fix(scene_type, intent)
        if learned is not None:
            return learned, 1.5   # higher confidence for learned fix

        # Rule-based fallback
        if scene_type == "image":
            return FixType.REPLACE_WITH_AVATAR, 1.0
        if intent == "explain":
            return FixType.SHORTEN_AND_ADD_HOOK, 1.0
        if intent == "transition":
            return FixType.CUT_OR_SPEED_UP, 1.0
        if intent == "hook":
            return FixType.INCREASE_PACING, 1.0
        return FixType.ADD_MICRO_HOOK, 1.0

    # ── Applying corrections ───────────────────────────────────────────────────

    def apply_corrections(
        self,
        scenes:      list[Any],   # list[Scene] — typed as Any to avoid circular import
        corrections: list[Correction],
    ) -> list[Any]:
        """Apply fix recommendations to Scene objects in place.

        Returns the same list (modified in place) so callers can chain.
        """
        scene_by_idx = {s.idx: s for s in scenes}

        for correction in corrections:
            scene = scene_by_idx.get(correction.scene_idx)
            if scene is None:
                logger.warning(
                    f"RetentionCorrectionEngine.apply_corrections: "
                    f"scene_idx={correction.scene_idx} not found, skipping"
                )
                continue

            fix = correction.fix
            self._apply_single(scene, fix)
            logger.info(
                f"RetentionCorrectionEngine: applied {fix.value!r} "
                f"to scene {correction.scene_idx}"
            )

        return scenes

    @staticmethod
    def _apply_single(scene: Any, fix: FixType) -> None:
        if fix in (FixType.REPLACE_WITH_AVATAR, FixType.INSERT_AVATAR):
            scene.scene_type = "avatar"
        elif fix == FixType.SHORTEN_AND_ADD_HOOK:
            if scene.duration_sec > 0:
                scene.duration_sec = int(scene.duration_sec * 0.7)
            scene.visual_prompt = _HOOK_MICRO + scene.visual_prompt
        elif fix == FixType.CUT_OR_SPEED_UP:
            if scene.duration_sec > 0:
                scene.duration_sec = int(scene.duration_sec * 0.6)
        elif fix == FixType.INCREASE_PACING:
            if scene.duration_sec > 0:
                scene.duration_sec = int(scene.duration_sec * 0.75)
        elif fix == FixType.ADD_MICRO_HOOK:
            scene.visual_prompt = _HOOK_MICRO + scene.visual_prompt
        elif fix == FixType.REMOVE_SCENE:
            scene.duration_sec = 0

    # ── Strategy adjustments ───────────────────────────────────────────────────

    def suggest_strategy_adjustments(
        self,
        corrections: list[Correction],
        scene_map:   dict[int, dict[str, Any]],
        curve:       list[float],
    ) -> dict[str, Any]:
        """Derive global strategy hints for DecisionEngineV2.

        Rules:
        - Early drops (timestamp_idx <= 2)  → pace="fast", hook_aggressiveness += 0.20
        - Drop at scene 0                   → force_avatar_intro = True
        - Many drops (>= 3)                 → hook_aggressiveness += 0.10
        - No drops                          → empty dict
        """
        if not corrections:
            return {}

        adjustments: dict[str, Any] = {}
        hook_delta = 0.0

        early_drop_scenes = {
            c.scene_idx for c in corrections if c.drop.timestamp_idx <= 2
        }
        drop_at_scene_zero = any(c.scene_idx == 0 for c in corrections)

        if early_drop_scenes:
            adjustments["pace"]  = "fast"
            hook_delta          += 0.20

        if drop_at_scene_zero:
            adjustments["force_avatar_intro"] = True

        if len(corrections) >= 3:
            hook_delta += 0.10

        if hook_delta:
            adjustments["hook_aggressiveness_delta"] = round(hook_delta, 2)

        adjustments["drop_count"]      = len(corrections)
        adjustments["early_drop_count"] = len(early_drop_scenes)

        logger.info(f"RetentionCorrectionEngine: strategy adjustments → {adjustments}")
        return adjustments

    # ── Learning ───────────────────────────────────────────────────────────────

    def record_outcome(
        self,
        fix_type:        FixType | str,
        scene_type:      str,
        intent:          str,
        delta_retention: float,
    ) -> None:
        """Record the measured outcome of an applied fix for future learning."""
        self._store.record(fix_type, scene_type, intent, delta_retention)
