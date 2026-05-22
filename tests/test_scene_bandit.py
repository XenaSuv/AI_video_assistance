"""Tests for SceneBandit (Thompson Sampling over scene policies).

Covers:
- ScenePolicyArm / SceneBanditState serialisation + mean property
- SceneBanditStore persistence
- SceneBandit._ensure_builtins() — auto-registration on first init
- SceneBandit.register_policy() — deduplication
- SceneBandit.select_policy() — returns arm, prefers high posterior mean
- SceneBandit.update() — alpha/beta/uses/total_reward updates
- SceneBandit.penalize_for_low_retention() — conditional extra failure
- SceneBandit.can_switch() / record_switch() / time_until_switch()
- SceneBandit.reset() — full and keep_history modes
- SceneBandit.get_best_policy() — returns best-mean used arm
- SceneBandit.suggest_strategy_adjustments() — Decision Engine feedback
- Integration: DecisionEngineV2.decide(scene_bandit=...) overrides scene_mix
- Convergence: best arm dominates after many observations
"""
from __future__ import annotations

import json
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.scene_bandit import (
    BUILTIN_POLICIES,
    SceneBandit,
    SceneBanditState,
    SceneBanditStore,
    ScenePolicyArm,
)

# ── Helpers ────────────────────────────────────────────────────────────────────

def _tmp() -> Path:
    return Path(tempfile.mkdtemp())


def _arm(
    policy_id: str = "policy_A",
    label:     str = "balanced",
    alpha:     float = 1.0,
    beta:      float = 1.0,
    uses:      int   = 0,
) -> ScenePolicyArm:
    return ScenePolicyArm(
        id=policy_id, label=label,
        scene_mix={"avatar": 0.3, "text_overlay": 0.3, "image": 0.2, "diagram": 0.2},
        alpha=alpha, beta=beta, uses=uses,
    )


def _bandit(*arms, min_switch_hours: float = 0.0, seed: int = 0) -> SceneBandit:
    """Create a bandit pre-loaded with given arms (bypasses built-in registration)."""
    b = SceneBandit(data_dir=_tmp(), min_switch_hours=min_switch_hours, seed=seed)
    b._state.arms = list(arms)
    return b


# ── ScenePolicyArm ─────────────────────────────────────────────────────────────

class TestScenePolicyArm:
    def test_defaults(self):
        arm = ScenePolicyArm()
        assert arm.alpha == 1.0
        assert arm.beta  == 1.0
        assert arm.uses  == 0
        assert arm.total_reward == 0.0

    def test_mean_uniform_prior(self):
        arm = ScenePolicyArm()
        assert abs(arm.mean - 0.5) < 1e-9

    def test_mean_high_alpha(self):
        arm = ScenePolicyArm(alpha=99.0, beta=1.0)
        assert arm.mean > 0.98

    def test_mean_high_beta(self):
        arm = ScenePolicyArm(alpha=1.0, beta=99.0)
        assert arm.mean < 0.02

    def test_to_dict_round_trip(self):
        arm = _arm("policy_X", alpha=3.5, beta=7.2, uses=5)
        arm.total_reward = 1.75
        d   = arm.to_dict()
        arm2 = ScenePolicyArm.from_dict(d)
        assert arm2.id           == "policy_X"
        assert abs(arm2.alpha - 3.5) < 1e-9
        assert abs(arm2.beta  - 7.2) < 1e-9
        assert arm2.uses         == 5
        assert abs(arm2.total_reward - 1.75) < 1e-9

    def test_from_dict_defaults(self):
        arm = ScenePolicyArm.from_dict({})
        assert arm.id    == "policy_A"
        assert arm.alpha == 1.0
        assert arm.beta  == 1.0

    def test_scene_mix_deep_copied_in_to_dict(self):
        arm = _arm()
        d   = arm.to_dict()
        d["scene_mix"]["extra"] = 0.99
        assert "extra" not in arm.scene_mix


# ── SceneBanditState ───────────────────────────────────────────────────────────

class TestSceneBanditState:
    def test_to_dict_round_trip(self):
        state = SceneBanditState(
            arms=[_arm("policy_A"), _arm("policy_B")],
            last_switch_at="2026-01-01T00:00:00+00:00",
        )
        d      = state.to_dict()
        state2 = SceneBanditState.from_dict(d)
        assert len(state2.arms)         == 2
        assert state2.arms[0].id        == "policy_A"
        assert state2.last_switch_at    == "2026-01-01T00:00:00+00:00"

    def test_empty_round_trip(self):
        state = SceneBanditState()
        d     = state.to_dict()
        state2 = SceneBanditState.from_dict(d)
        assert state2.arms           == []
        assert state2.last_switch_at is None


# ── SceneBanditStore ───────────────────────────────────────────────────────────

class TestSceneBanditStore:
    def test_save_and_load(self):
        tmp   = _tmp()
        store = SceneBanditStore(data_dir=tmp)
        state = SceneBanditState(arms=[_arm("policy_A")])
        store.save(state)
        loaded = store.load()
        assert loaded.arms[0].id == "policy_A"

    def test_load_missing_file_returns_fresh(self):
        store  = SceneBanditStore(data_dir=_tmp())
        loaded = store.load()
        assert loaded.arms == []

    def test_load_corrupted_file_returns_fresh(self):
        tmp   = _tmp()
        store = SceneBanditStore(data_dir=tmp)
        store._path.parent.mkdir(parents=True, exist_ok=True)
        store._path.write_text("{bad json{{")
        loaded = store.load()
        assert loaded.arms == []


# ── _ensure_builtins ───────────────────────────────────────────────────────────

class TestEnsureBuiltins:
    def test_three_builtin_policies_registered(self):
        b = SceneBandit(data_dir=_tmp())
        assert len(b._state.arms) == 3

    def test_builtin_ids(self):
        b    = SceneBandit(data_dir=_tmp())
        ids  = {a.id for a in b._state.arms}
        assert ids == {"policy_A", "policy_B", "policy_C"}

    def test_builtins_not_re_registered_on_reload(self):
        tmp = _tmp()
        b1  = SceneBandit(data_dir=tmp)
        b2  = SceneBandit(data_dir=tmp)
        assert len(b2._state.arms) == 3

    def test_existing_state_not_overwritten(self):
        b = SceneBandit(data_dir=_tmp())
        b._state.arms[0].alpha = 99.0
        b._store.save(b._state)
        tmp_path = b._store._path.parent
        b2 = SceneBandit(data_dir=tmp_path)
        assert b2._state.arms[0].alpha == 99.0


# ── register_policy ────────────────────────────────────────────────────────────

class TestRegisterPolicy:
    def test_adds_new_policy(self):
        b = _bandit()
        b.register_policy("custom", {"image": 0.8, "diagram": 0.2})
        ids = [a.id for a in b._state.arms]
        assert "custom" in ids

    def test_skips_duplicate(self):
        b = _bandit(_arm("policy_A"))
        b.register_policy("policy_A", {"image": 1.0})
        assert len(b._state.arms) == 1

    def test_label_defaults_to_id(self):
        b = _bandit()
        b.register_policy("my_policy", {"image": 1.0})
        arm = b._find_arm("my_policy")
        assert arm.label == "my_policy"

    def test_custom_label(self):
        b = _bandit()
        b.register_policy("my_policy", {"image": 1.0}, label="My Custom Policy")
        assert b._find_arm("my_policy").label == "My Custom Policy"


# ── select_policy ──────────────────────────────────────────────────────────────

class TestSelectPolicy:
    def test_returns_arm(self):
        b = _bandit(_arm("policy_A"))
        assert b.select_policy().id == "policy_A"

    def test_returns_none_when_no_arms(self):
        b = SceneBandit(data_dir=_tmp())
        b._state.arms = []
        assert b.select_policy() is None

    def test_selects_one_of_registered(self):
        b = _bandit(_arm("policy_A"), _arm("policy_B"), _arm("policy_C"))
        result = b.select_policy()
        assert result.id in {"policy_A", "policy_B", "policy_C"}

    def test_prefers_high_alpha_arm_statistically(self):
        good = _arm("policy_A", alpha=100.0, beta=1.0)
        bad  = _arm("policy_B", alpha=1.0,   beta=100.0)
        b    = _bandit(good, bad)
        wins = sum(1 for _ in range(200) if b.select_policy().id == "policy_A")
        assert wins >= 195, f"Expected policy_A to dominate; got {wins}/200"

    def test_exploration_across_seeds(self):
        """Thompson explores all arms with equal Beta(1,1) priors in most runs."""
        explore_count = 0
        for seed in range(20):
            b = _bandit(_arm("policy_A"), _arm("policy_B"), seed=seed)
            ids = {b.select_policy().id for _ in range(50)}
            if "policy_A" in ids and "policy_B" in ids:
                explore_count += 1
        assert explore_count >= 18, f"Only {explore_count}/20 seeds explored both arms"

    def test_deterministic_with_seed(self):
        arms = [_arm("policy_A"), _arm("policy_B")]
        b1   = _bandit(*arms, seed=7)
        b2   = _bandit(*arms, seed=7)
        seq1 = [b1.select_policy().id for _ in range(10)]
        seq2 = [b2.select_policy().id for _ in range(10)]
        assert seq1 == seq2


# ── update ─────────────────────────────────────────────────────────────────────

class TestUpdate:
    def test_alpha_increases_by_reward(self):
        b = _bandit(_arm("policy_A"))
        b.update("policy_A", retention_30s=0.6, avg_watch_pct=0.4)
        arm = b._state.arms[0]
        # reward = (0.6 + 0.4) / 2 = 0.5; alpha = 1 + 0.5 = 1.5
        assert abs(arm.alpha - 1.5) < 1e-9

    def test_beta_increases_by_one_minus_reward(self):
        b = _bandit(_arm("policy_A"))
        b.update("policy_A", retention_30s=0.6, avg_watch_pct=0.4)
        arm = b._state.arms[0]
        # beta = 1 + (1 - 0.5) = 1.5
        assert abs(arm.beta - 1.5) < 1e-9

    def test_uses_incremented(self):
        b = _bandit(_arm("policy_A"))
        b.update("policy_A", retention_30s=0.5, avg_watch_pct=0.5)
        assert b._state.arms[0].uses == 1

    def test_total_reward_accumulated(self):
        b = _bandit(_arm("policy_A"))
        b.update("policy_A", retention_30s=0.8, avg_watch_pct=0.8)
        b.update("policy_A", retention_30s=0.4, avg_watch_pct=0.4)
        arm = b._state.arms[0]
        assert abs(arm.total_reward - (0.8 + 0.4)) < 1e-9

    def test_reward_clipped_to_zero_minimum(self):
        b = _bandit(_arm("policy_A"))
        b.update("policy_A", retention_30s=0.0, avg_watch_pct=0.0)
        arm = b._state.arms[0]
        assert abs(arm.alpha - 1.0) < 1e-9   # 1 + 0 = 1
        assert abs(arm.beta  - 2.0) < 1e-9   # 1 + 1 = 2

    def test_reward_clipped_to_one_maximum(self):
        b = _bandit(_arm("policy_A"))
        b.update("policy_A", retention_30s=1.0, avg_watch_pct=1.0)
        arm = b._state.arms[0]
        assert abs(arm.alpha - 2.0) < 1e-9   # 1 + 1 = 2
        assert abs(arm.beta  - 1.0) < 1e-9   # 1 + 0 = 1

    def test_only_retention_available(self):
        b = _bandit(_arm("policy_A"))
        b.update("policy_A", retention_30s=0.6)
        arm = b._state.arms[0]
        # reward = (0.6 + 0.0) / 2 = 0.3
        assert abs(arm.alpha - 1.3) < 1e-9

    def test_unknown_policy_id_noop(self):
        b = _bandit(_arm("policy_A"))
        b.update("does_not_exist", retention_30s=0.5)
        assert b._state.arms[0].alpha == 1.0   # unchanged

    def test_state_persisted_after_update(self):
        tmp = _tmp()
        b   = SceneBandit(data_dir=tmp)
        pid = b._state.arms[0].id
        b.update(pid, retention_30s=0.7, avg_watch_pct=0.6)
        b2 = SceneBandit(data_dir=tmp)
        arm2 = b2._find_arm(pid)
        assert arm2.uses == 1


# ── penalize_for_low_retention ─────────────────────────────────────────────────

class TestPenalizeForLowRetention:
    def test_penalizes_when_ctr_high_and_retention_low(self):
        b   = _bandit(_arm("policy_A"))
        b.penalize_for_low_retention("policy_A", packaging_ctr=0.08, retention=0.20)
        assert b._state.arms[0].beta == 2.0   # 1 + 1 extra failure

    def test_no_penalty_when_ctr_low(self):
        b = _bandit(_arm("policy_A"))
        b.penalize_for_low_retention("policy_A", packaging_ctr=0.03, retention=0.10)
        assert b._state.arms[0].beta == 1.0   # unchanged

    def test_no_penalty_when_retention_ok(self):
        b = _bandit(_arm("policy_A"))
        b.penalize_for_low_retention("policy_A", packaging_ctr=0.10, retention=0.50)
        assert b._state.arms[0].beta == 1.0   # unchanged

    def test_no_penalty_at_threshold_boundary_ctr(self):
        b = _bandit(_arm("policy_A"))
        b.penalize_for_low_retention("policy_A", packaging_ctr=0.05, retention=0.10)
        assert b._state.arms[0].beta == 1.0   # exactly at threshold → no penalty

    def test_unknown_policy_id_noop(self):
        b = _bandit(_arm("policy_A"))
        b.penalize_for_low_retention("no_such_id", packaging_ctr=0.10, retention=0.10)
        assert b._state.arms[0].beta == 1.0


# ── can_switch / record_switch / time_until_switch ─────────────────────────────

class TestRateLimiting:
    def test_can_switch_initially_true(self):
        b = _bandit(_arm())
        assert b.can_switch() is True

    def test_cannot_switch_immediately_after_record(self):
        b = SceneBandit(data_dir=_tmp(), min_switch_hours=2.0)
        b.record_switch()
        assert b.can_switch() is False

    def test_can_switch_after_elapsed(self):
        b = SceneBandit(data_dir=_tmp(), min_switch_hours=2.0)
        past = (datetime.now(timezone.utc) - timedelta(hours=3)).isoformat()
        b._state.last_switch_at = past
        assert b.can_switch() is True

    def test_time_until_switch_zero_when_ready(self):
        b = _bandit(_arm())
        assert b.time_until_switch() == 0.0

    def test_time_until_switch_positive_after_record(self):
        b = SceneBandit(data_dir=_tmp(), min_switch_hours=2.0)
        b.record_switch()
        remaining = b.time_until_switch()
        assert 0.0 < remaining <= 2.0

    def test_record_switch_persists(self):
        tmp = _tmp()
        b   = SceneBandit(data_dir=tmp, min_switch_hours=2.0)
        b.record_switch()
        b2  = SceneBandit(data_dir=tmp, min_switch_hours=2.0)
        assert b2.can_switch() is False


# ── reset ──────────────────────────────────────────────────────────────────────

class TestReset:
    def test_full_reset_clears_arms(self):
        b = SceneBandit(data_dir=_tmp())
        b.reset()
        assert b._state.arms == []

    def test_history_reset_zeroes_posteriors(self):
        b = SceneBandit(data_dir=_tmp())
        b._state.arms[0].alpha = 50.0
        b._state.arms[0].uses  = 10
        b.reset(keep_history=True)
        assert b._state.arms[0].alpha == 1.0
        assert b._state.arms[0].uses  == 0

    def test_history_reset_preserves_arm_ids(self):
        b    = SceneBandit(data_dir=_tmp())
        ids  = [a.id for a in b._state.arms]
        b.reset(keep_history=True)
        ids2 = [a.id for a in b._state.arms]
        assert ids == ids2

    def test_history_reset_preserves_scene_mix(self):
        b    = SceneBandit(data_dir=_tmp())
        mix0 = dict(b._state.arms[0].scene_mix)
        b.reset(keep_history=True)
        assert b._state.arms[0].scene_mix == mix0


# ── get_best_policy / suggest_strategy_adjustments ────────────────────────────

class TestIntrospection:
    def test_get_best_policy_none_when_no_uses(self):
        b = _bandit(_arm("policy_A"), _arm("policy_B"))
        assert b.get_best_policy() is None

    def test_get_best_policy_returns_highest_mean(self):
        a = _arm("policy_A", alpha=80.0, beta=20.0, uses=5)  # mean 0.8
        c = _arm("policy_B", alpha=20.0, beta=80.0, uses=5)  # mean 0.2
        b = _bandit(a, c)
        assert b.get_best_policy().id == "policy_A"

    def test_get_best_policy_ignores_unused(self):
        a = _arm("policy_A", alpha=80.0, beta=20.0, uses=5)
        c = _arm("policy_B", alpha=1.0, beta=1.0, uses=0)    # unused — excluded
        b = _bandit(a, c)
        assert b.get_best_policy().id == "policy_A"

    def test_suggest_adjustments_empty_when_no_uses(self):
        b = _bandit(_arm("policy_A"))
        assert b.suggest_strategy_adjustments() == {}

    def test_suggest_adjustments_fields(self):
        a = _arm("policy_A", alpha=80.0, beta=20.0, uses=5)
        b = _bandit(a)
        adj = b.suggest_strategy_adjustments()
        assert adj["preferred_policy_id"]    == "policy_A"
        assert adj["preferred_policy_label"] == "balanced"
        assert "scene_mix"   in adj
        assert "best_mean"   in adj

    def test_suggest_adjustments_best_mean_rounded(self):
        a = _arm("policy_A", alpha=1.0, beta=3.0, uses=1)  # mean = 0.25
        b = _bandit(a)
        adj = b.suggest_strategy_adjustments()
        assert adj["best_mean"] == 0.25

    def test_get_arm_stats_returns_all(self):
        b = _bandit(_arm("policy_A"), _arm("policy_B"))
        stats = b.get_arm_stats()
        assert len(stats) == 2
        assert stats[0]["id"] == "policy_A"


# ── Convergence ────────────────────────────────────────────────────────────────

class TestConvergence:
    def test_best_policy_dominates_after_observations(self):
        """After many high-retention updates to policy_B, it should dominate."""
        a = _arm("policy_A")
        c = _arm("policy_B")
        b = SceneBandit(data_dir=_tmp())
        b._state.arms = [a, c]
        # policy_A: low retention (2% effective)
        for _ in range(50):
            b.update("policy_A", retention_30s=0.02, avg_watch_pct=0.02)
        # policy_B: high retention (90% effective)
        for _ in range(50):
            b.update("policy_B", retention_30s=0.90, avg_watch_pct=0.90)
        wins = sum(1 for _ in range(200) if b.select_policy().id == "policy_B")
        assert wins > 180, f"policy_B should dominate; got {wins}/200"

    def test_penalized_policy_falls_behind(self):
        """Repeatedly penalizing policy_A shifts selection toward policy_B."""
        a = _arm("policy_A")
        c = _arm("policy_B", alpha=2.0, beta=1.0)   # slight head-start
        b = _bandit(a, c)
        for _ in range(20):
            b.penalize_for_low_retention("policy_A", packaging_ctr=0.10, retention=0.10)
        wins_b = sum(1 for _ in range(100) if b.select_policy().id == "policy_B")
        assert wins_b > 80, f"policy_B should dominate after penalizing A; got {wins_b}/100"


# ── Full pipeline ──────────────────────────────────────────────────────────────

class TestFullPipeline:
    def test_register_select_update_adjust(self):
        b = SceneBandit(data_dir=_tmp())

        policy = b.select_policy()
        assert policy is not None
        assert policy.scene_mix

        b.update(policy.id, retention_30s=0.65, avg_watch_pct=0.50)
        arm = b._find_arm(policy.id)
        assert arm.uses == 1

        adj = b.suggest_strategy_adjustments()
        assert adj["preferred_policy_id"] == policy.id

    def test_two_bandits_are_independent(self):
        """Packaging bandit and scene bandit operate independently."""
        from src.thompson_bandit import ABTestVariant
        from src.thompson_bandit import ThompsonBandit

        tmp      = _tmp()
        scene_b  = SceneBandit(data_dir=tmp)
        pkg_b    = ThompsonBandit(data_dir=tmp)

        # They persist to different files — no collision
        assert scene_b._store._path.name == "scene_bandit_state.json"
        assert pkg_b._store._path.name   == "thompson_state.json"


class TestGetState:
    def test_get_state_returns_state(self):
        b = SceneBandit(data_dir=_tmp())
        state = b.get_state()
        assert state is b._state
