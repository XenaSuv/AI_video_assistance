"""Multi-Armed Bandit Engine — UCB1 exploration/exploitation for packaging variants.

Instead of the sequential A → B → compare pattern (ABTestingEngine), the bandit
continuously picks the variant with the highest UCB score, naturally drifting
toward the best performer while still exploring underused arms.

UCB1 formula
------------
    score = mean_reward + sqrt( (2 × ln(total_impressions)) / arm_impressions )

The confidence bonus shrinks as an arm accumulates impressions, making the
algorithm self-tuning: heavily-tested arms are selected only when they clearly
outperform alternatives.

Reward
------
    reward = CTR × retention_30s   (prevents clickbait winning on clicks alone)
    reward = CTR                   (fallback when retention data is unavailable)

Rate limiting
-------------
YouTube penalises frequent metadata changes.  ``can_switch()`` enforces a
configurable minimum gap (default 2 hours) between variant applications.

Data store
----------
data/bandit_state.json — single JSON file with per-arm statistics.
Persisted after every update() and register call so restarts are seamless.

Integration
-----------
PackagingEngine.generate_variants() → list[ABTestVariant]
    ↓
BanditEngine.register_variants(variants)      ← one-time setup
    ↓
BanditEngine.select_variant()                 ← called each update cycle
    ↓ (apply to YouTube)
BanditEngine.update(variant_id, impressions, clicks, retention_30s)
    ↓
BanditEngine.suggest_strategy_adjustments()   ← DecisionEngineV2 feedback
"""
from __future__ import annotations

import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import settings
from src.constants import VARIANT_TYPE_DELTAS, RateLimitMixin
from src.state_io import atomic_json_write

if TYPE_CHECKING:
    from src.ab_testing_engine import ABTestVariant


# ── Data types ────────────────────────────────────────────────────────────────

@dataclass
class BanditArm:
    """Per-variant statistics tracked by the bandit.

    *reward* is recomputed after every update() call and is the primary signal
    for UCB1 selection.
    """
    variant_id:     str              = "A"
    type:           str              = "curiosity"  # curiosity | conflict | simple
    title:          str              = ""
    thumbnail:      dict[str, str]   = field(default_factory=dict)
    hook:           str              = ""
    impressions:    int              = 0
    clicks:         int              = 0
    retention_sum:  float            = 0.0   # sum of (retention_30s * impressions) batches
    reward:         float            = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "variant_id":    self.variant_id,
            "type":          self.type,
            "title":         self.title,
            "thumbnail":     self.thumbnail,
            "hook":          self.hook,
            "impressions":   self.impressions,
            "clicks":        self.clicks,
            "retention_sum": self.retention_sum,
            "reward":        self.reward,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "BanditArm":
        return cls(
            variant_id    = d.get("variant_id",    "A"),
            type          = d.get("type",          "curiosity"),
            title         = d.get("title",         ""),
            thumbnail     = d.get("thumbnail",     {}),
            hook          = d.get("hook",          ""),
            impressions   = int(d.get("impressions",   0)),
            clicks        = int(d.get("clicks",        0)),
            retention_sum = float(d.get("retention_sum", 0.0)),
            reward        = float(d.get("reward",        0.0)),
        )

    @classmethod
    def from_variant(cls, variant: "ABTestVariant") -> "BanditArm":
        """Construct a fresh arm from an ABTestVariant (zero impressions)."""
        return cls(
            variant_id = variant.id,
            type       = variant.type,
            title      = variant.title,
            thumbnail  = dict(variant.thumbnail),
            hook       = variant.hook,
        )


@dataclass
class BanditState:
    """Full persisted bandit state: all arms + last-switch timestamp."""
    arms:           list[BanditArm]  = field(default_factory=list)
    last_switch_at: str | None       = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "arms":           [a.to_dict() for a in self.arms],
            "last_switch_at": self.last_switch_at,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "BanditState":
        return cls(
            arms           = [BanditArm.from_dict(a) for a in d.get("arms", [])],
            last_switch_at = d.get("last_switch_at"),
        )


# ── BanditStore ───────────────────────────────────────────────────────────────

class BanditStore:
    """Single-file JSON store for bandit state."""

    def __init__(self, data_dir: Path | None = None) -> None:
        self._path = (data_dir or settings.data_dir) / "bandit_state.json"

    def save(self, state: BanditState) -> None:
        atomic_json_write(self._path, state.to_dict())
        logger.debug("BanditStore: state persisted")

    def load(self) -> BanditState:
        try:
            if self._path.exists():
                data = json.loads(self._path.read_text())
                return BanditState.from_dict(data)
        except Exception as exc:
            logger.warning(f"BanditStore: could not load state ({exc}), starting fresh")
        return BanditState()


# ── BanditEngine ──────────────────────────────────────────────────────────────

class BanditEngine(RateLimitMixin):
    """UCB1 multi-armed bandit for continuous packaging variant optimisation.

    Typical lifecycle per video::

        engine = BanditEngine(data_dir=run_dir)
        engine.register_variants(packaging_engine.generate_variants(...))

        # Each update cycle (every 2-4 h):
        if engine.can_switch():
            arm = engine.select_variant()
            youtube.update_video(video_id, arm.title, arm.thumbnail)
            engine.record_switch()
        engine.update(arm.variant_id, impressions=120, clicks=9, retention_30s=0.65)

        adjustments = engine.suggest_strategy_adjustments()
    """

    def __init__(
        self,
        data_dir: Path | None = None,
        min_switch_hours: float = 2.0,
    ) -> None:
        self._store            = BanditStore(data_dir)
        self._state            = self._store.load()
        self.min_switch_hours  = min_switch_hours
        self._label            = "BanditEngine"

    # ── Arm management ─────────────────────────────────────────────────────────

    def register_variants(self, variants: list["ABTestVariant"]) -> None:
        """Register ABTestVariant objects as bandit arms (skips duplicates).

        Call once when variants are first generated for a video.
        New variants not yet in state are added with zero impressions.
        """
        existing_ids = {a.variant_id for a in self._state.arms}
        added = 0
        for v in variants:
            if v.id not in existing_ids:
                self._state.arms.append(BanditArm.from_variant(v))
                added += 1
        if added:
            self._store.save(self._state)
            logger.info(f"BanditEngine: registered {added} arm(s) — total {len(self._state.arms)}")

    def reset(self, keep_history: bool = False) -> None:
        """Clear arm statistics (and optionally all arms).

        Useful when generating a completely new set of variants for a new video.
        *keep_history=True* zeroes metrics but keeps arm definitions.
        """
        if keep_history:
            for arm in self._state.arms:
                arm.impressions = arm.clicks = 0
                arm.retention_sum = arm.reward = 0.0
        else:
            self._state = BanditState()
        self._store.save(self._state)
        logger.info("BanditEngine: state reset")

    # ── UCB1 selection ─────────────────────────────────────────────────────────

    def select_variant(self) -> BanditArm | None:
        """Return the arm with the highest UCB1 score.

        UCB1:  score = mean_reward + sqrt( (2 × ln(total)) / impressions )

        Any arm with 0 impressions is returned immediately (pure exploration).
        Returns None if no arms are registered.
        """
        arms = self._state.arms
        if not arms:
            logger.warning("BanditEngine: no arms registered")
            return None

        # Pure exploration: prefer any unvisited arm
        for arm in arms:
            if arm.impressions == 0:
                logger.debug(f"BanditEngine: exploring unvisited arm {arm.variant_id!r}")
                return arm

        total = sum(a.impressions for a in arms)
        log_total = math.log(total)

        best_score  = -1.0
        best_arm: BanditArm | None = None

        for arm in arms:
            exploration = math.sqrt((2 * log_total) / arm.impressions)
            ucb         = arm.reward + exploration
            if ucb > best_score:
                best_score = ucb
                best_arm   = arm

        assert best_arm is not None
        logger.debug(
            f"BanditEngine: selected {best_arm.variant_id!r} ({best_arm.type}) "
            f"ucb={best_score:.4f} reward={best_arm.reward:.4f}"
        )
        return best_arm

    # ── Statistics update ──────────────────────────────────────────────────────

    def update(
        self,
        variant_id:   str,
        impressions:  int,
        clicks:       int,
        retention_30s: float = 0.0,
    ) -> None:
        """Ingest new performance data for *variant_id* and recompute reward.

        *retention_30s* is the average 30-second retention rate (0–1) for the
        batch.  Pass 0.0 when retention data is not yet available (reward falls
        back to CTR only).
        """
        arm = self._find_arm(variant_id)
        if arm is None:
            logger.warning(f"BanditEngine.update: unknown variant_id {variant_id!r}")
            return

        arm.impressions   += impressions
        arm.clicks        += clicks
        arm.retention_sum += retention_30s * impressions

        arm.reward = self._compute_reward(arm)
        self._store.save(self._state)

        logger.info(
            f"BanditEngine: updated {variant_id!r} → "
            f"imp={arm.impressions}  ctr={arm.clicks/max(arm.impressions,1):.3f}  "
            f"reward={arm.reward:.4f}"
        )

    # ── Rate limiting — provided by RateLimitMixin ────────────────────────────

    # ── Decision Engine feedback ───────────────────────────────────────────────

    def suggest_strategy_adjustments(self) -> dict[str, Any]:
        """Analyse current arm rewards and suggest StrategyConfig adjustments.

        Returns a dict the caller can apply to DecisionEngineV2::

            adj = engine.suggest_strategy_adjustments()
            # e.g. {"preferred_variant_type": "conflict",
            #        "hook_aggressiveness_delta": 0.1,
            #        "best_arm_id": "B",
            #        "best_reward": 0.063}
        """
        arms_with_data = [a for a in self._state.arms if a.impressions > 0]
        if not arms_with_data:
            return {}

        best = max(arms_with_data, key=lambda a: a.reward)
        delta = VARIANT_TYPE_DELTAS.get(best.type, 0.0)

        adjustments = {
            "preferred_variant_type":    best.type,
            "hook_aggressiveness_delta": delta,
            "best_arm_id":               best.variant_id,
            "best_reward":               best.reward,
        }
        logger.info(f"BanditEngine: strategy adjustments → {adjustments}")
        return adjustments

    # ── Introspection ──────────────────────────────────────────────────────────

    def get_arm_stats(self) -> list[dict[str, Any]]:
        """Return current statistics for all registered arms (for logging/debug)."""
        return [a.to_dict() for a in self._state.arms]

    def get_state(self) -> BanditState:
        """Return a reference to the current in-memory state."""
        return self._state

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _find_arm(self, variant_id: str) -> BanditArm | None:
        for arm in self._state.arms:
            if arm.variant_id == variant_id:
                return arm
        return None

    def _compute_reward(self, arm: BanditArm) -> float:
        """reward = CTR × avg_retention_30s; fallback to CTR when retention = 0."""
        if arm.impressions == 0:
            return 0.0
        ctr = arm.clicks / arm.impressions
        if arm.retention_sum > 0:
            avg_ret = arm.retention_sum / arm.impressions
            return round(ctr * avg_ret, 6)
        return round(ctr, 6)
