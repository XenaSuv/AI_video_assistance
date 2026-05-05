"""Scene-policy Thompson Sampling bandit.

Optimises the *visual composition* of videos by treating each scene policy
(a distribution over scene types) as a bandit arm.  Unlike the packaging
bandit (ThompsonBandit), which optimises CTR via title/thumbnail, the scene
bandit optimises viewer *retention* — the metric that reflects whether the
visual pacing kept people watching.

Scene Policy
------------
A scene policy is a dict of scene-type weights that is injected into
strategy.scene_policy.mix and consumed by SceneVarietyEngineV2::

    policy_A = {"avatar": 0.3, "text_overlay": 0.3, "image": 0.2, "diagram": 0.2}
    policy_B = {"image": 0.5, "diagram": 0.3, "avatar": 0.1, "text_overlay": 0.1}

Reward
------
    reward = clamp((retention_30s + avg_watch_pct) / 2, 0, 1)

When only one metric is available the other defaults to 0.0, so the reward
is a conservative half of the available metric.

Scene-Packaging Interaction
---------------------------
High packaging CTR + low retention → visual content is the bottleneck.
SceneBandit.penalize_for_low_retention() adds an extra failure update in
this case, accelerating convergence away from the poor scene policy.

Three built-in policies are registered automatically on first use.  Custom
policies can be added via register_policy().

Integration with Decision Engine
---------------------------------
Pass the bandit to DecisionEngineV2.decide()::

    bandit   = SceneBandit(data_dir=run_dir)
    config   = engine.decide(metrics, scene_bandit=bandit)
    # config.scene_policy.mix is now the Thompson-selected scene mix

After publishing and collecting feedback::

    bandit.update(selected_policy_id, retention_30s=0.62, avg_watch_pct=0.45)
    bandit.penalize_for_low_retention(selected_policy_id, ctr=0.08, retention=0.18)
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
from src.constants import RateLimitMixin


# ── Built-in scene policies ───────────────────────────────────────────────────

BUILTIN_POLICIES: list[dict[str, Any]] = [
    {
        "id":        "policy_A",
        "label":     "balanced",
        "scene_mix": {
            "avatar":       0.30,
            "text_overlay": 0.30,
            "image":        0.20,
            "diagram":      0.20,
        },
    },
    {
        "id":        "policy_B",
        "label":     "visual_heavy",
        "scene_mix": {
            "image":        0.50,
            "diagram":      0.30,
            "avatar":       0.10,
            "text_overlay": 0.10,
        },
    },
    {
        "id":        "policy_C",
        "label":     "presenter_led",
        "scene_mix": {
            "avatar":       0.50,
            "image":        0.30,
            "text_overlay": 0.10,
            "diagram":      0.10,
        },
    },
]


# ── Data types ────────────────────────────────────────────────────────────────

@dataclass
class ScenePolicyArm:
    """Beta-posterior statistics for one scene policy."""
    id:           str              = "policy_A"
    label:        str              = "balanced"
    scene_mix:    dict[str, float] = field(default_factory=dict)
    alpha:        float            = 1.0
    beta:         float            = 1.0
    uses:         int              = 0
    total_reward: float            = 0.0

    @property
    def mean(self) -> float:
        """Posterior mean = α / (α + β) — best point estimate of success prob."""
        return self.alpha / (self.alpha + self.beta)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id":           self.id,
            "label":        self.label,
            "scene_mix":    dict(self.scene_mix),
            "alpha":        self.alpha,
            "beta":         self.beta,
            "uses":         self.uses,
            "total_reward": self.total_reward,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ScenePolicyArm":
        return cls(
            id           = d.get("id",           "policy_A"),
            label        = d.get("label",        "balanced"),
            scene_mix    = d.get("scene_mix",    {}),
            alpha        = float(d.get("alpha",        1.0)),
            beta         = float(d.get("beta",         1.0)),
            uses         = int(d.get("uses",           0)),
            total_reward = float(d.get("total_reward", 0.0)),
        )


@dataclass
class SceneBanditState:
    """Full persisted scene bandit state."""
    arms:           list[ScenePolicyArm] = field(default_factory=list)
    last_switch_at: str | None           = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "arms":           [a.to_dict() for a in self.arms],
            "last_switch_at": self.last_switch_at,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "SceneBanditState":
        return cls(
            arms           = [ScenePolicyArm.from_dict(a) for a in d.get("arms", [])],
            last_switch_at = d.get("last_switch_at"),
        )


# ── Persistence ───────────────────────────────────────────────────────────────

class SceneBanditStore:
    """Single-file JSON persistence for scene bandit state."""

    def __init__(self, data_dir: Path | None = None) -> None:
        self._path = (data_dir or settings.data_dir) / "scene_bandit_state.json"

    def save(self, state: SceneBanditState) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(json.dumps(state.to_dict(), indent=2))
        logger.debug("SceneBanditStore: state persisted")

    def load(self) -> SceneBanditState:
        try:
            if self._path.exists():
                data = json.loads(self._path.read_text())
                return SceneBanditState.from_dict(data)
        except Exception as exc:
            logger.warning(f"SceneBanditStore: could not load state ({exc}), starting fresh")
        return SceneBanditState()


# ── SceneBandit ───────────────────────────────────────────────────────────────

class SceneBandit(RateLimitMixin):
    """Thompson Sampling bandit over scene policies.

    Typical lifecycle::

        bandit = SceneBandit(data_dir=run_dir)

        # Integrated into decision engine:
        config = decision_engine.decide(metrics, scene_bandit=bandit)

        # After receiving viewer feedback:
        bandit.update(selected_policy_id, retention_30s=0.62, avg_watch_pct=0.45)

        # If packaging CTR was high but retention was low:
        bandit.penalize_for_low_retention(selected_policy_id, packaging_ctr=0.08, retention=0.18)
    """

    HIGH_CTR_THRESHOLD   = 0.05
    LOW_RETENTION_THRESH = 0.30

    def __init__(
        self,
        data_dir:         Path | None = None,
        min_switch_hours: float       = 2.0,
        seed:             int | None  = None,
    ) -> None:
        self._store           = SceneBanditStore(data_dir)
        self._state           = self._store.load()
        self.min_switch_hours = min_switch_hours
        self._rng             = _stdlib_random.Random(seed)
        self._label           = "SceneBandit"
        self._ensure_builtins()

    # ── Policy management ──────────────────────────────────────────────────────

    def _ensure_builtins(self) -> None:
        """Register the three built-in policies if no state exists yet."""
        if self._state.arms:
            return
        for spec in BUILTIN_POLICIES:
            self._state.arms.append(
                ScenePolicyArm(
                    id        = spec["id"],
                    label     = spec["label"],
                    scene_mix = dict(spec["scene_mix"]),
                )
            )
        self._store.save(self._state)
        logger.info(f"SceneBandit: registered {len(BUILTIN_POLICIES)} built-in policies")

    def register_policy(
        self,
        policy_id: str,
        scene_mix: dict[str, float],
        label:     str = "",
    ) -> None:
        """Add a custom scene policy arm (skips if id already exists)."""
        if any(a.id == policy_id for a in self._state.arms):
            return
        self._state.arms.append(
            ScenePolicyArm(id=policy_id, label=label or policy_id, scene_mix=dict(scene_mix))
        )
        self._store.save(self._state)
        logger.info(f"SceneBandit: registered policy {policy_id!r}")

    def reset(self, keep_history: bool = False) -> None:
        """Clear state.  keep_history=True resets posteriors but retains arm defs."""
        if keep_history:
            for arm in self._state.arms:
                arm.alpha = arm.beta = 1.0
                arm.uses = 0
                arm.total_reward = 0.0
        else:
            self._state = SceneBanditState()
        self._store.save(self._state)
        logger.info("SceneBandit: state reset")

    # ── Thompson Sampling ──────────────────────────────────────────────────────

    def select_policy(self) -> ScenePolicyArm | None:
        """Sample once from each arm's Beta posterior; return the highest draw.

        Arms with fewer observations have wider distributions and therefore
        occasionally produce large samples — this drives natural exploration.
        """
        arms = self._state.arms
        if not arms:
            logger.warning("SceneBandit: no policies registered")
            return None

        best_sample = -1.0
        best_arm: ScenePolicyArm | None = None

        for arm in arms:
            sample = self._rng.betavariate(arm.alpha, arm.beta)
            if sample > best_sample:
                best_sample = sample
                best_arm    = arm

        assert best_arm is not None
        logger.debug(
            f"SceneBandit: selected {best_arm.id!r} ({best_arm.label}) "
            f"sample={best_sample:.4f}  mean={best_arm.mean:.4f}"
        )
        return best_arm

    # ── Update posterior ───────────────────────────────────────────────────────

    def update(
        self,
        policy_id:     str,
        retention_30s: float = 0.0,
        avg_watch_pct: float = 0.0,
    ) -> None:
        """Ingest viewer-retention feedback and update the Beta posterior.

        reward = clamp((retention_30s + avg_watch_pct) / 2, 0, 1)

        α += reward       (success: viewers stayed)
        β += 1 - reward   (failure: viewers left)
        """
        arm = self._find_arm(policy_id)
        if arm is None:
            logger.warning(f"SceneBandit.update: unknown policy_id {policy_id!r}")
            return

        reward = max(0.0, min(1.0, (retention_30s + avg_watch_pct) / 2))
        arm.alpha        += reward
        arm.beta         += 1.0 - reward
        arm.uses         += 1
        arm.total_reward += reward

        self._store.save(self._state)
        logger.info(
            f"SceneBandit: updated {policy_id!r} → "
            f"reward={reward:.4f}  α={arm.alpha:.2f}  β={arm.beta:.2f}  "
            f"mean={arm.mean:.4f}  uses={arm.uses}"
        )

    def penalize_for_low_retention(
        self,
        policy_id:     str,
        packaging_ctr: float,
        retention:     float,
    ) -> None:
        """Extra beta penalty when CTR is high but retention is low.

        A high CTR confirms the packaging (title/thumbnail) is working.
        If retention is still low, the bottleneck is the scene policy, so
        an extra failure is added to accelerate exploration away from it.
        """
        if packaging_ctr <= self.HIGH_CTR_THRESHOLD or retention >= self.LOW_RETENTION_THRESH:
            return
        arm = self._find_arm(policy_id)
        if arm is None:
            return
        arm.beta += 1.0
        self._store.save(self._state)
        logger.info(
            f"SceneBandit: penalized {policy_id!r} "
            f"(CTR={packaging_ctr:.3f} > {self.HIGH_CTR_THRESHOLD}, "
            f"retention={retention:.3f} < {self.LOW_RETENTION_THRESH})"
        )

    # ── Rate limiting — provided by RateLimitMixin ────────────────────────────

    # ── Introspection ──────────────────────────────────────────────────────────

    def get_arm_stats(self) -> list[dict[str, Any]]:
        """Current stats for all registered policies."""
        return [a.to_dict() for a in self._state.arms]

    def get_best_policy(self) -> ScenePolicyArm | None:
        """Return the arm with the highest posterior mean among used arms."""
        arms = [a for a in self._state.arms if a.uses > 0]
        return max(arms, key=lambda a: a.mean) if arms else None

    def suggest_strategy_adjustments(self) -> dict[str, Any]:
        """Return the winning scene_mix for use by the Decision Engine."""
        best = self.get_best_policy()
        if best is None:
            return {}
        adjustments = {
            "preferred_policy_id":    best.id,
            "preferred_policy_label": best.label,
            "scene_mix":              dict(best.scene_mix),
            "best_mean":              round(best.mean, 4),
        }
        logger.info(f"SceneBandit: strategy adjustments → {adjustments}")
        return adjustments

    def get_state(self) -> SceneBanditState:
        return self._state

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _find_arm(self, policy_id: str) -> ScenePolicyArm | None:
        for arm in self._state.arms:
            if arm.id == policy_id:
                return arm
        return None
