"""Thompson Sampling Bandit — Bayesian explore/exploit for packaging variants.

Each variant (title + thumbnail) is a bandit arm with a Beta posterior over its
"success probability".  At every selection step we sample once from each arm's
distribution and pick the arm with the highest draw.  This automatically balances
exploration (uncertain arms have wide distributions → occasional big draws) and
exploitation (proven winners have narrow peaks → consistently high draws).

Beta distribution
-----------------
    Beta(α, β)
    α = successes accumulated (quality-adjusted clicks)
    β = failures accumulated (impressions without quality clicks)
    Start: α = β = 1  (uniform prior — equal chance, maximum uncertainty)

Reward (quality-adjusted)
-------------------------
When retention_30s data is available:
    effective_clicks = clicks × retention_30s
    α += effective_clicks          (quality clicks — click AND viewer stayed)
    β += impressions − effective_clicks

When retention data is missing (retention_30s = 0):
    α += clicks                    (raw CTR mode)
    β += impressions − clicks

This prevents high-CTR / low-retention "clickbait" variants from winning.

Rate limiting
-------------
YouTube penalises frequent metadata changes.  Default minimum gap: 2 hours.
can_switch() / record_switch() enforce this.

Data store
----------
data/thompson_state.json — single JSON file, overwritten on every update.
"""
from __future__ import annotations

import json
import random as _stdlib_random
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import settings
from src.constants import VARIANT_TYPE_DELTAS, RateLimitMixin
from src.state_io import atomic_json_write


# ── ABTestVariant ─────────────────────────────────────────────────────────────

@dataclass
class ABTestVariant:
    """Packaging variant created by PackagingEngine and selected by ThompsonBandit.

    Content fields (type, title, thumbnail, hook) are set at creation.
    Metric fields (ctr, retention_30s, score) are populated by deferred
    feedback collection.
    """
    id: str                    = "A"          # "A" | "B" | "C"
    type: str                  = "curiosity"  # curiosity | conflict | simple
    title: str                 = ""
    thumbnail: dict[str, str]  = field(default_factory=dict)
    hook: str                  = ""
    start_time: str | None     = None
    ctr: float                 = 0.0
    retention_30s: float       = 0.0
    score: float               = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "id":            self.id,
            "type":          self.type,
            "title":         self.title,
            "thumbnail":     self.thumbnail,
            "hook":          self.hook,
            "start_time":    self.start_time,
            "ctr":           self.ctr,
            "retention_30s": self.retention_30s,
            "score":         self.score,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ABTestVariant":
        return cls(
            id            = d.get("id",            "A"),
            type          = d.get("type",          "curiosity"),
            title         = d.get("title",         ""),
            thumbnail     = d.get("thumbnail",     {}),
            hook          = d.get("hook",          ""),
            start_time    = d.get("start_time"),
            ctr           = float(d.get("ctr",           0.0)),
            retention_30s = float(d.get("retention_30s", 0.0)),
            score         = float(d.get("score",         0.0)),
        )

@dataclass
class ThompsonArm:
    """Beta-posterior statistics for one packaging variant.

    *alpha* and *beta* are the Beta distribution parameters.  They start at
    (1.0, 1.0) — a uniform prior — and are updated after each observation.
    *impressions* / *clicks* are raw counts kept for reporting only.
    """
    variant_id:  str             = "A"
    type:        str             = "curiosity"  # curiosity | conflict | simple
    title:       str             = ""
    thumbnail:   dict[str, str]  = field(default_factory=dict)
    hook:        str             = ""
    alpha:       float           = 1.0   # successes (quality-adjusted)
    beta:        float           = 1.0   # failures
    impressions: int             = 0     # raw count, for reporting
    clicks:      int             = 0     # raw count, for reporting

    @property
    def mean(self) -> float:
        """Posterior mean = α / (α + β) — best point estimate of success prob."""
        return self.alpha / (self.alpha + self.beta)

    def to_dict(self) -> dict[str, Any]:
        return {
            "variant_id":  self.variant_id,
            "type":        self.type,
            "title":       self.title,
            "thumbnail":   self.thumbnail,
            "hook":        self.hook,
            "alpha":       self.alpha,
            "beta":        self.beta,
            "impressions": self.impressions,
            "clicks":      self.clicks,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ThompsonArm":
        return cls(
            variant_id  = d.get("variant_id",  "A"),
            type        = d.get("type",        "curiosity"),
            title       = d.get("title",       ""),
            thumbnail   = d.get("thumbnail",   {}),
            hook        = d.get("hook",        ""),
            alpha       = float(d.get("alpha",       1.0)),
            beta        = float(d.get("beta",        1.0)),
            impressions = int(d.get("impressions",   0)),
            clicks      = int(d.get("clicks",        0)),
        )

    @classmethod
    def from_variant(cls, variant: ABTestVariant) -> "ThompsonArm":
        """Construct a fresh arm from an ABTestVariant (uniform prior)."""
        return cls(
            variant_id = variant.id,
            type       = variant.type,
            title      = variant.title,
            thumbnail  = dict(variant.thumbnail),
            hook       = variant.hook,
        )


@dataclass
class ThompsonState:
    """Full persisted Thompson bandit state."""
    arms:           list[ThompsonArm]  = field(default_factory=list)
    last_switch_at: str | None         = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "arms":           [a.to_dict() for a in self.arms],
            "last_switch_at": self.last_switch_at,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ThompsonState":
        return cls(
            arms           = [ThompsonArm.from_dict(a) for a in d.get("arms", [])],
            last_switch_at = d.get("last_switch_at"),
        )


# ── ThompsonStore ─────────────────────────────────────────────────────────────

class ThompsonStore:
    """Single-file JSON persistence for Thompson bandit state."""

    def __init__(self, data_dir: Path | None = None) -> None:
        self._path = (data_dir or settings.data_dir) / "thompson_state.json"

    def save(self, state: ThompsonState) -> None:
        atomic_json_write(self._path, state.to_dict())
        logger.debug("ThompsonStore: state persisted")

    def load(self) -> ThompsonState:
        try:
            if self._path.exists():
                data = json.loads(self._path.read_text())
                return ThompsonState.from_dict(data)
        except Exception as exc:
            logger.warning(f"ThompsonStore: could not load state ({exc}), starting fresh")
        return ThompsonState()


# ── ThompsonBandit ────────────────────────────────────────────────────────────

class ThompsonBandit(RateLimitMixin):
    """Thompson Sampling bandit — Beta-posterior exploration/exploitation.

    Typical lifecycle per video::

        bandit = ThompsonBandit(data_dir=run_dir)
        bandit.register_variants(packaging_engine.generate_variants(...))

        # Each update cycle (every 2-4 h):
        if bandit.can_switch():
            arm = bandit.select_variant()
            youtube.update_video(video_id, arm.title, arm.thumbnail)
            bandit.record_switch()
        bandit.update(arm.variant_id, impressions=120, clicks=9, retention_30s=0.65)

        adjustments = bandit.suggest_strategy_adjustments()
    """

    def __init__(
        self,
        data_dir: Path | None = None,
        min_switch_hours: float = 2.0,
        seed: int | None = None,
    ) -> None:
        self._store           = ThompsonStore(data_dir)
        self._state           = self._store.load()
        self.min_switch_hours = min_switch_hours
        self._rng             = _stdlib_random.Random(seed)   # reproducible when seed set
        self._label           = "ThompsonBandit"

    # ── Arm management ─────────────────────────────────────────────────────────

    def register_variants(self, variants: list[ABTestVariant]) -> None:
        """Register ABTestVariant objects as Thompson arms (skips duplicates)."""
        existing_ids = {a.variant_id for a in self._state.arms}
        added = 0
        for v in variants:
            if v.id not in existing_ids:
                self._state.arms.append(ThompsonArm.from_variant(v))
                added += 1
        if added:
            self._store.save(self._state)
            logger.info(
                f"ThompsonBandit: registered {added} arm(s) — total {len(self._state.arms)}"
            )

    def reset(self, keep_history: bool = False) -> None:
        """Clear state.  *keep_history=True* resets posteriors but keeps arm defs."""
        if keep_history:
            for arm in self._state.arms:
                arm.alpha = arm.beta = 1.0
                arm.impressions = arm.clicks = 0
        else:
            self._state = ThompsonState()
        self._store.save(self._state)
        logger.info("ThompsonBandit: state reset")

    # ── Thompson Sampling selection ────────────────────────────────────────────

    def select_variant(self) -> ThompsonArm | None:
        """Sample once from each arm's Beta posterior; return the highest draw.

        Arms with fewer observations have wider distributions and therefore
        occasionally produce large samples — this drives natural exploration.
        """
        arms = self._state.arms
        if not arms:
            logger.warning("ThompsonBandit: no arms registered")
            return None

        best_sample = -1.0
        best_arm: ThompsonArm | None = None

        for arm in arms:
            sample = self._rng.betavariate(arm.alpha, arm.beta)
            if sample > best_sample:
                best_sample = sample
                best_arm    = arm

        assert best_arm is not None
        logger.debug(
            f"ThompsonBandit: selected {best_arm.variant_id!r} ({best_arm.type}) "
            f"sample={best_sample:.4f}  mean={best_arm.mean:.4f}"
        )
        return best_arm

    # ── Update posterior ───────────────────────────────────────────────────────

    def update(
        self,
        variant_id:    str,
        impressions:   int,
        clicks:        int,
        retention_30s: float = 0.0,
    ) -> None:
        """Ingest new observations and update the Beta posterior.

        Quality-adjusted mode (retention_30s > 0):
            effective_clicks = clicks × retention_30s
            α += effective_clicks
            β += impressions − effective_clicks

        CTR-only mode (retention_30s = 0):
            α += clicks
            β += impressions − clicks
        """
        arm = self._find_arm(variant_id)
        if arm is None:
            logger.warning(f"ThompsonBandit.update: unknown variant_id {variant_id!r}")
            return

        if retention_30s > 0:
            effective_clicks = clicks * retention_30s
        else:
            effective_clicks = float(clicks)

        arm.alpha       += effective_clicks
        arm.beta        += impressions - effective_clicks
        arm.impressions += impressions
        arm.clicks      += clicks

        self._store.save(self._state)

        logger.info(
            f"ThompsonBandit: updated {variant_id!r} → "
            f"α={arm.alpha:.2f}  β={arm.beta:.2f}  "
            f"mean={arm.mean:.4f}  imp={arm.impressions}"
        )

    # ── Rate limiting — provided by RateLimitMixin ────────────────────────────

    # ── Decision Engine feedback ───────────────────────────────────────────────

    def suggest_strategy_adjustments(self) -> dict[str, Any]:
        """Return StrategyConfig adjustment hints based on current posteriors.

        The arm with the highest posterior mean (α / (α + β)) is the winner.
        Maps variant type to hook_aggressiveness_delta::

            conflict  → +0.10  (explicit tension works — push harder)
            curiosity → +0.05  (open loops work — moderate increase)
            simple    → -0.05  (clarity wins — dial back aggression)
        """
        arms_with_data = [a for a in self._state.arms if a.impressions > 0]
        if not arms_with_data:
            return {}

        best  = max(arms_with_data, key=lambda a: a.mean)
        delta = VARIANT_TYPE_DELTAS.get(best.type, 0.0)

        adjustments = {
            "preferred_variant_type":    best.type,
            "hook_aggressiveness_delta": delta,
            "best_arm_id":               best.variant_id,
            "best_mean":                 round(best.mean, 4),
        }
        logger.info(f"ThompsonBandit: strategy adjustments → {adjustments}")
        return adjustments

    # ── Introspection ──────────────────────────────────────────────────────────

    def get_arm_stats(self) -> list[dict[str, Any]]:
        """Current stats for all registered arms (for logging / dashboards)."""
        return [a.to_dict() for a in self._state.arms]

    def get_state(self) -> ThompsonState:
        return self._state

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _find_arm(self, variant_id: str) -> ThompsonArm | None:
        for arm in self._state.arms:
            if arm.variant_id == variant_id:
                return arm
        return None
