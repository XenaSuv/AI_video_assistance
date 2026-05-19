"""Tests for ThompsonBandit (Beta-posterior Thompson Sampling).

Covers:
- ThompsonArm / ThompsonState serialisation + mean property
- ThompsonStore persistence
- ThompsonArm.from_variant() integration with ABTestVariant
- ThompsonBandit.register_variants() — deduplication, stat preservation
- ThompsonBandit.select_variant() — returns arm, prefers high posterior mean
- ThompsonBandit.update() — alpha/beta increments, retention weighting
- ThompsonBandit.can_switch() / record_switch() / time_until_switch()
- ThompsonBandit.suggest_strategy_adjustments() — Decision Engine feedback
- ThompsonBandit.reset() — full and keep_history modes
- Convergence: after many observations, best arm wins majority of draws
- Full pipeline: register → select → update → adjust
"""
from __future__ import annotations

import tempfile
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

import pytest

from src.ab_testing_engine import ABTestVariant
from src.thompson_bandit import (
    ThompsonArm,
    ThompsonBandit,
    ThompsonState,
    ThompsonStore,
)

# ── Helpers ────────────────────────────────────────────────────────────────────

def _tmp() -> Path:
    return Path(tempfile.mkdtemp())


def _arm(variant_id="A", type_="curiosity", alpha=1.0, beta=1.0,
         impressions=0, clicks=0) -> ThompsonArm:
    return ThompsonArm(
        variant_id=variant_id, type=type_,
        title=f"Title {variant_id}", hook=f"Hook {variant_id}",
        thumbnail={"text": variant_id, "style": "clean", "emotion": "curiosity"},
        alpha=alpha, beta=beta,
        impressions=impressions, clicks=clicks,
    )


def _variant(id_="A", type_="curiosity") -> ABTestVariant:
    return ABTestVariant(
        id=id_, type=type_,
        title=f"Title {id_}", hook=f"Hook {id_}",
        thumbnail={"text": id_, "style": "clean", "emotion": "curiosity"},
    )


def _bandit(*arms, min_switch_hours=0.0, seed=42) -> ThompsonBandit:
    """Create a bandit pre-loaded with given arms (bypasses store)."""
    b = ThompsonBandit(data_dir=_tmp(), min_switch_hours=min_switch_hours, seed=seed)
    b._state.arms = list(arms)
    return b


# ── ThompsonArm ───────────────────────────────────────────────────────────────

class TestThompsonArm:
    def test_defaults(self):
        arm = ThompsonArm()
        assert arm.alpha == 1.0
        assert arm.beta  == 1.0
        assert arm.impressions == 0

    def test_mean_uniform_prior(self):
        arm = ThompsonArm()
        assert abs(arm.mean - 0.5) < 1e-9

    def test_mean_high_alpha(self):
        arm = _arm(alpha=99.0, beta=1.0)
        assert arm.mean > 0.95

    def test_mean_high_beta(self):
        arm = _arm(alpha=1.0, beta=99.0)
        assert arm.mean < 0.05

    def test_to_dict_round_trip(self):
        arm = _arm("B", "conflict", alpha=15.0, beta=90.0, impressions=100, clicks=15)
        d   = arm.to_dict()
        a2  = ThompsonArm.from_dict(d)
        assert a2.variant_id  == "B"
        assert a2.alpha       == 15.0
        assert a2.beta        == 90.0
        assert a2.impressions == 100

    def test_from_dict_defaults(self):
        arm = ThompsonArm.from_dict({})
        assert arm.alpha == 1.0
        assert arm.beta  == 1.0

    def test_from_variant(self):
        v   = _variant("C", "simple")
        arm = ThompsonArm.from_variant(v)
        assert arm.variant_id == "C"
        assert arm.type       == "simple"
        assert arm.alpha      == 1.0
        assert arm.beta       == 1.0

    def test_from_variant_thumbnail_deep_copied(self):
        v   = _variant("A")
        arm = ThompsonArm.from_variant(v)
        v.thumbnail["text"] = "CHANGED"
        assert arm.thumbnail.get("text") != "CHANGED"


# ── ThompsonState ─────────────────────────────────────────────────────────────

class TestThompsonState:
    def test_to_dict_round_trip(self):
        state = ThompsonState(
            arms=[_arm("A"), _arm("B", alpha=5.0, beta=20.0)],
            last_switch_at="2025-01-01T08:00:00+00:00",
        )
        d  = state.to_dict()
        s2 = ThompsonState.from_dict(d)
        assert len(s2.arms) == 2
        assert abs(s2.arms[1].beta - 20.0) < 1e-9
        assert s2.last_switch_at == "2025-01-01T08:00:00+00:00"

    def test_empty_round_trip(self):
        s  = ThompsonState()
        s2 = ThompsonState.from_dict(s.to_dict())
        assert s2.arms == []
        assert s2.last_switch_at is None


# ── ThompsonStore ─────────────────────────────────────────────────────────────

class TestThompsonStore:
    def test_save_and_load(self):
        store = ThompsonStore(data_dir=_tmp())
        state = ThompsonState(arms=[_arm("A", alpha=10.0, beta=5.0)])
        store.save(state)
        loaded = store.load()
        assert abs(loaded.arms[0].alpha - 10.0) < 1e-9

    def test_load_missing_file_returns_empty(self):
        store = ThompsonStore(data_dir=_tmp())
        state = store.load()
        assert state.arms == []

    def test_overwrite_on_save(self):
        tmp   = _tmp()
        store = ThompsonStore(data_dir=tmp)
        store.save(ThompsonState(arms=[_arm("A")]))
        store.save(ThompsonState(arms=[_arm("B")]))
        loaded = store.load()
        assert loaded.arms[0].variant_id == "B"


# ── register_variants() ───────────────────────────────────────────────────────

class TestRegisterVariants:
    def test_registers_all_variants(self):
        b = ThompsonBandit(data_dir=_tmp())
        b.register_variants([_variant("A"), _variant("B"), _variant("C")])
        assert len(b._state.arms) == 3

    def test_deduplication(self):
        b = ThompsonBandit(data_dir=_tmp())
        b.register_variants([_variant("A"), _variant("B")])
        b.register_variants([_variant("B"), _variant("C")])
        assert len(b._state.arms) == 3

    def test_existing_arm_stats_preserved(self):
        b = ThompsonBandit(data_dir=_tmp())
        b.register_variants([_variant("A")])
        b.update("A", impressions=100, clicks=10)
        b.register_variants([_variant("A")])  # duplicate — must not reset
        assert b._state.arms[0].impressions == 100

    def test_fresh_arms_start_at_uniform_prior(self):
        b = ThompsonBandit(data_dir=_tmp())
        b.register_variants([_variant("A")])
        arm = b._state.arms[0]
        assert arm.alpha == 1.0
        assert arm.beta  == 1.0


# ── select_variant() ──────────────────────────────────────────────────────────

class TestSelectVariant:
    def test_returns_none_when_no_arms(self):
        b = ThompsonBandit(data_dir=_tmp())
        assert b.select_variant() is None

    def test_returns_arm_with_single_arm(self):
        b = _bandit(_arm("A"))
        assert b.select_variant().variant_id == "A"

    def test_selects_one_of_registered_arms(self):
        b = _bandit(_arm("A"), _arm("B"), _arm("C"))
        selected = b.select_variant()
        assert selected.variant_id in ("A", "B", "C")

    def test_prefers_high_posterior_mean_statistically(self):
        """After many draws, arm with Beta(100, 1) beats Beta(1, 100)."""
        good = _arm("A", alpha=100.0, beta=1.0)   # mean ≈ 0.99
        bad  = _arm("B", alpha=1.0,   beta=100.0) # mean ≈ 0.01
        b    = _bandit(good, bad)
        wins = sum(1 for _ in range(200) if b.select_variant().variant_id == "A")
        assert wins >= 195, f"Expected A to dominate; got {wins}/200"

    def test_uniform_prior_explores_both_arms(self):
        """Thompson explores both arms with equal Beta(1,1) priors in most runs."""
        explore_count = 0
        for seed in range(20):
            a = _arm("A")
            c = _arm("C")
            b = ThompsonBandit(data_dir=_tmp(), seed=seed)
            b._state.arms = [a, c]
            ids = {b.select_variant().variant_id for _ in range(50)}
            if "A" in ids and "C" in ids:
                explore_count += 1
        assert explore_count >= 18, f"Only {explore_count}/20 seeds explored both arms"

    def test_deterministic_with_seed(self):
        """Same seed → same sequence of selections."""
        arms = [_arm("A"), _arm("B")]
        b1 = _bandit(*arms, seed=99)
        b2 = _bandit(*arms, seed=99)
        seq1 = [b1.select_variant().variant_id for _ in range(10)]
        seq2 = [b2.select_variant().variant_id for _ in range(10)]
        assert seq1 == seq2


# ── update() ─────────────────────────────────────────────────────────────────

class TestUpdate:
    def test_alpha_increases_by_clicks(self):
        b = _bandit(_arm("A"))
        b.update("A", impressions=100, clicks=10)
        arm = b._state.arms[0]
        assert abs(arm.alpha - 11.0) < 1e-9   # 1 + 10

    def test_beta_increases_by_non_clicks(self):
        b = _bandit(_arm("A"))
        b.update("A", impressions=100, clicks=10)
        arm = b._state.arms[0]
        assert abs(arm.beta - 91.0) < 1e-9    # 1 + 90

    def test_impressions_and_clicks_accumulated(self):
        b = _bandit(_arm("A"))
        b.update("A", impressions=100, clicks=8)
        b.update("A", impressions=50,  clicks=3)
        arm = b._state.arms[0]
        assert arm.impressions == 150
        assert arm.clicks      == 11

    def test_retention_weighting_reduces_effective_clicks(self):
        """With retention=0.5, 10 clicks count as 5 effective successes."""
        b = _bandit(_arm("A"))
        b.update("A", impressions=100, clicks=10, retention_30s=0.5)
        arm = b._state.arms[0]
        # alpha = 1.0 + 10*0.5 = 6.0
        assert abs(arm.alpha - 6.0) < 1e-9

    def test_retention_beta_complements_effective_clicks(self):
        b = _bandit(_arm("A"))
        b.update("A", impressions=100, clicks=10, retention_30s=0.5)
        arm = b._state.arms[0]
        # beta = 1.0 + (100 - 10*0.5) = 1.0 + 95.0 = 96.0
        assert abs(arm.beta - 96.0) < 1e-9

    def test_zero_retention_falls_back_to_raw_ctr(self):
        b = _bandit(_arm("A"))
        b.update("A", impressions=100, clicks=10, retention_30s=0.0)
        arm = b._state.arms[0]
        assert abs(arm.alpha - 11.0) < 1e-9  # same as no-retention case

    def test_high_retention_more_successes_than_low_retention(self):
        b_high = _bandit(_arm("A"), seed=0)
        b_low  = _bandit(_arm("A"), seed=0)
        b_high.update("A", impressions=100, clicks=10, retention_30s=0.90)
        b_low.update( "A", impressions=100, clicks=10, retention_30s=0.10)
        assert b_high._state.arms[0].alpha > b_low._state.arms[0].alpha

    def test_unknown_variant_id_does_not_crash(self):
        b = _bandit(_arm("A"))
        b.update("MISSING", impressions=10, clicks=1)

    def test_update_persists_to_store(self):
        tmp = _tmp()
        b   = ThompsonBandit(data_dir=tmp)
        b._state.arms.append(_arm("A"))
        b.update("A", impressions=50, clicks=5)
        loaded = ThompsonStore(data_dir=tmp).load()
        assert abs(loaded.arms[0].alpha - 6.0) < 1e-9

    def test_multiple_updates_accumulate(self):
        b = _bandit(_arm("A"))
        for _ in range(5):
            b.update("A", impressions=20, clicks=2)
        arm = b._state.arms[0]
        assert abs(arm.alpha - 11.0) < 1e-9  # 1 + 5*2
        assert abs(arm.beta  - 91.0) < 1e-9  # 1 + 5*18


# ── Rate limiting ─────────────────────────────────────────────────────────────

class TestRateLimiting:
    def test_can_switch_when_never_switched(self):
        b = ThompsonBandit(data_dir=_tmp(), min_switch_hours=2.0)
        assert b.can_switch() is True

    def test_cannot_switch_immediately_after_record(self):
        b = ThompsonBandit(data_dir=_tmp(), min_switch_hours=2.0)
        b.record_switch()
        assert b.can_switch() is False

    def test_can_switch_after_enough_time(self):
        b = ThompsonBandit(data_dir=_tmp(), min_switch_hours=2.0)
        past = datetime.now(timezone.utc) - timedelta(hours=3)
        b._state.last_switch_at = past.isoformat()
        assert b.can_switch() is True

    def test_cannot_switch_before_min_hours(self):
        b = ThompsonBandit(data_dir=_tmp(), min_switch_hours=4.0)
        recent = datetime.now(timezone.utc) - timedelta(hours=1)
        b._state.last_switch_at = recent.isoformat()
        assert b.can_switch() is False

    def test_time_until_switch_zero_when_ready(self):
        b = ThompsonBandit(data_dir=_tmp())
        assert b.time_until_switch() == 0.0

    def test_time_until_switch_positive_when_blocked(self):
        b = ThompsonBandit(data_dir=_tmp(), min_switch_hours=4.0)
        recent = datetime.now(timezone.utc) - timedelta(hours=1)
        b._state.last_switch_at = recent.isoformat()
        remaining = b.time_until_switch()
        assert 2.9 < remaining < 3.1

    def test_record_switch_persists(self):
        tmp = _tmp()
        b   = ThompsonBandit(data_dir=tmp)
        b.record_switch()
        b2  = ThompsonBandit(data_dir=tmp)
        assert b2._state.last_switch_at is not None


# ── suggest_strategy_adjustments() ───────────────────────────────────────────

class TestSuggestStrategyAdjustments:
    def test_empty_when_no_observations(self):
        b = _bandit(_arm("A", impressions=0))
        assert b.suggest_strategy_adjustments() == {}

    def test_conflict_best_gives_positive_delta(self):
        a = _arm("A", "curiosity", alpha=5.0,  beta=50.0)   # mean ≈ 0.09
        b_arm = _arm("B", "conflict",  alpha=50.0, beta=5.0)   # mean ≈ 0.91
        b_arm.impressions = 100
        a.impressions     = 100
        b = _bandit(a, b_arm)
        adj = b.suggest_strategy_adjustments()
        assert adj["preferred_variant_type"]    == "conflict"
        assert adj["hook_aggressiveness_delta"] == +0.10

    def test_curiosity_best_moderate_delta(self):
        a = _arm("A", "curiosity", alpha=40.0, beta=10.0, impressions=100)
        b_arm = _arm("B", "simple",    alpha=10.0, beta=40.0, impressions=100)
        b = _bandit(a, b_arm)
        adj = b.suggest_strategy_adjustments()
        assert adj["hook_aggressiveness_delta"] == +0.05

    def test_simple_best_negative_delta(self):
        a = _arm("A", "simple",   alpha=80.0, beta=5.0, impressions=100)
        b_arm = _arm("B", "conflict", alpha=5.0,  beta=80.0, impressions=100)
        b = _bandit(a, b_arm)
        adj = b.suggest_strategy_adjustments()
        assert adj["hook_aggressiveness_delta"] == -0.05

    def test_best_arm_id_and_mean_returned(self):
        a = _arm("A", "conflict", alpha=50.0, beta=5.0, impressions=100)
        b = _bandit(a)
        adj = b.suggest_strategy_adjustments()
        assert adj["best_arm_id"] == "A"
        assert abs(adj["best_mean"] - round(a.mean, 4)) < 1e-4

    def test_zero_impression_arms_ignored(self):
        a = _arm("A", "conflict",  alpha=40.0, beta=5.0, impressions=100)
        c = _arm("C", "curiosity", impressions=0)   # not observed
        b = _bandit(a, c)
        adj = b.suggest_strategy_adjustments()
        assert adj["preferred_variant_type"] == "conflict"


# ── reset() ──────────────────────────────────────────────────────────────────

class TestReset:
    def test_full_reset_clears_arms(self):
        b = _bandit(_arm("A"), _arm("B"))
        b.reset(keep_history=False)
        assert b._state.arms == []

    def test_history_reset_zeroes_posteriors(self):
        b = _bandit(_arm("A", alpha=50.0, beta=20.0, impressions=100))
        b.reset(keep_history=True)
        arm = b._state.arms[0]
        assert arm.alpha == 1.0
        assert arm.beta  == 1.0
        assert arm.impressions == 0

    def test_history_reset_preserves_arm_definitions(self):
        b = _bandit(_arm("A", "conflict"))
        b.reset(keep_history=True)
        assert len(b._state.arms) == 1
        assert b._state.arms[0].type == "conflict"


# ── Convergence test ──────────────────────────────────────────────────────────

class TestConvergence:
    def test_best_arm_dominates_after_observations(self):
        """After many observations, best arm wins majority of draws."""
        a     = _arm("A")
        b_arm = _arm("B")
        engine = ThompsonBandit(data_dir=_tmp())  # no fixed seed
        engine._state.arms = [a, b_arm]
        # A: 2% CTR → alpha=11, beta=491
        # B: 20% CTR → alpha=101, beta=401 — massive difference
        engine.update("A", impressions=500, clicks=10)
        engine.update("B", impressions=500, clicks=100)

        wins_b = sum(1 for _ in range(200) if engine.select_variant().variant_id == "B")
        assert wins_b > 180, f"B should dominate; got {wins_b}/200 wins"

    def test_retention_weighted_winner_differs_from_ctr_winner(self):
        """High CTR but low retention should lose to lower CTR with high retention."""
        b = _bandit(_arm("A"), _arm("B"))
        # A: high CTR (0.15) but terrible retention (0.10) → quality mean ≈ 0.015
        # B: moderate CTR (0.08) with great retention (0.80) → quality mean ≈ 0.064
        b.update("A", impressions=100, clicks=15, retention_30s=0.10)
        b.update("B", impressions=100, clicks=8,  retention_30s=0.80)
        # After quality adjustment B should win
        assert b._state.arms[1].mean > b._state.arms[0].mean


# ── Full pipeline ─────────────────────────────────────────────────────────────

class TestFullPipeline:
    def test_generate_register_select_update_adjust(self):
        from src.packaging_engine import PackagingEngine

        @dataclass
        class _FakeConfig:
            mode: str = "growth"

        tmp     = _tmp()
        bandit  = ThompsonBandit(data_dir=tmp, seed=7)
        pe      = PackagingEngine()
        variants = pe.generate_variants({"conflict": "hype vs reality"}, _FakeConfig())

        bandit.register_variants(variants)
        assert len(bandit._state.arms) == 3

        # Simulate 3 update cycles — each arm explored once
        for arm_name in ("A", "B", "C"):
            bandit.update(arm_name, impressions=100, clicks=7, retention_30s=0.65)

        adj = bandit.suggest_strategy_adjustments()
        assert "preferred_variant_type" in adj
        assert "hook_aggressiveness_delta" in adj

    def test_thompson_vs_ucb1_same_interface(self):
        """ThompsonBandit exposes the same key methods as BanditEngine."""
        from src.bandit_engine import BanditEngine
        shared_methods = [
            "register_variants", "select_variant", "update",
            "can_switch", "record_switch", "suggest_strategy_adjustments", "reset",
        ]
        b = ThompsonBandit(data_dir=_tmp())
        u = BanditEngine(data_dir=_tmp())
        for method in shared_methods:
            assert hasattr(b, method), f"ThompsonBandit missing {method}"
            assert hasattr(u, method), f"BanditEngine missing {method}"
