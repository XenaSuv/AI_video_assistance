"""Tests for BanditEngine (UCB1 multi-armed bandit).

Covers:
- BanditArm / BanditState serialisation
- BanditStore persistence
- BanditArm.from_variant() integration with ABTestVariant
- BanditEngine.register_variants() — deduplication
- BanditEngine.select_variant() — exploration, UCB1 exploitation, edge cases
- BanditEngine.update() — CTR, retention, reward formula
- BanditEngine.can_switch() / record_switch() / time_until_switch() — rate limiting
- BanditEngine.suggest_strategy_adjustments() — Decision Engine feedback
- BanditEngine.reset() — full and history-preserving reset
- Full pipeline: register → select → update → suggest
"""
from __future__ import annotations

import math
import tempfile
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from src.ab_testing_engine import ABTestVariant
from src.bandit_engine import BanditArm, BanditEngine, BanditState, BanditStore

# ── Helpers ────────────────────────────────────────────────────────────────────

def _tmp() -> Path:
    return Path(tempfile.mkdtemp())


def _arm(variant_id="A", type_="curiosity", impressions=0, clicks=0,
         retention_sum=0.0, reward=0.0) -> BanditArm:
    return BanditArm(
        variant_id=variant_id, type=type_,
        title=f"Title {variant_id}", hook=f"Hook {variant_id}",
        thumbnail={"text": variant_id, "style": "clean", "emotion": "curiosity"},
        impressions=impressions, clicks=clicks,
        retention_sum=retention_sum, reward=reward,
    )


def _variant(id_="A", type_="curiosity") -> ABTestVariant:
    return ABTestVariant(
        id=id_, type=type_,
        title=f"Title {id_}", hook=f"Hook {id_}",
        thumbnail={"text": id_, "style": "clean", "emotion": "curiosity"},
    )


def _engine_with_arms(*arms, min_switch_hours=0.0) -> BanditEngine:
    """Create an engine pre-loaded with given arms (bypasses store)."""
    tmp = _tmp()
    engine = BanditEngine(data_dir=tmp, min_switch_hours=min_switch_hours)
    engine._state.arms = list(arms)
    return engine


# ── BanditArm ─────────────────────────────────────────────────────────────────

class TestBanditArm:
    def test_defaults(self):
        arm = BanditArm()
        assert arm.impressions == 0
        assert arm.reward == 0.0

    def test_to_dict_round_trip(self):
        arm = _arm("B", "conflict", impressions=100, clicks=8, retention_sum=70.0, reward=0.056)
        d   = arm.to_dict()
        arm2 = BanditArm.from_dict(d)
        assert arm2.variant_id    == "B"
        assert arm2.impressions   == 100
        assert arm2.clicks        == 8
        assert arm2.retention_sum == 70.0
        assert arm2.reward        == 0.056

    def test_from_dict_empty_defaults(self):
        arm = BanditArm.from_dict({})
        assert arm.variant_id == "A"
        assert arm.impressions == 0

    def test_from_variant(self):
        v   = _variant("B", "conflict")
        arm = BanditArm.from_variant(v)
        assert arm.variant_id == "B"
        assert arm.type       == "conflict"
        assert arm.title      == "Title B"
        assert arm.impressions == 0
        assert arm.reward     == 0.0

    def test_from_variant_thumbnail_copied(self):
        v   = _variant("A")
        arm = BanditArm.from_variant(v)
        # Mutating the original thumbnail must not affect the arm
        v.thumbnail["text"] = "CHANGED"
        assert arm.thumbnail.get("text") != "CHANGED"


# ── BanditState ───────────────────────────────────────────────────────────────

class TestBanditState:
    def test_to_dict_round_trip(self):
        state = BanditState(
            arms=[_arm("A"), _arm("B")],
            last_switch_at="2025-01-01T10:00:00+00:00",
        )
        d      = state.to_dict()
        state2 = BanditState.from_dict(d)
        assert len(state2.arms) == 2
        assert state2.last_switch_at == "2025-01-01T10:00:00+00:00"

    def test_empty_state_round_trip(self):
        state = BanditState()
        d     = state.to_dict()
        s2    = BanditState.from_dict(d)
        assert s2.arms == []
        assert s2.last_switch_at is None


# ── BanditStore ───────────────────────────────────────────────────────────────

class TestBanditStore:
    def test_save_and_load(self):
        store = BanditStore(data_dir=_tmp())
        state = BanditState(arms=[_arm("A", impressions=50, clicks=4)])
        store.save(state)
        loaded = store.load()
        assert len(loaded.arms) == 1
        assert loaded.arms[0].impressions == 50

    def test_load_missing_file_returns_empty(self):
        store = BanditStore(data_dir=_tmp())
        state = store.load()
        assert state.arms == []
        assert state.last_switch_at is None

    def test_overwrite_on_save(self):
        tmp   = _tmp()
        store = BanditStore(data_dir=tmp)
        store.save(BanditState(arms=[_arm("A")]))
        store.save(BanditState(arms=[_arm("B")]))
        loaded = store.load()
        assert len(loaded.arms) == 1
        assert loaded.arms[0].variant_id == "B"


# ── register_variants() ────────────────────────────────────────────────────────

class TestRegisterVariants:
    def test_registers_all_variants(self):
        engine = BanditEngine(data_dir=_tmp())
        engine.register_variants([_variant("A"), _variant("B"), _variant("C")])
        assert len(engine._state.arms) == 3

    def test_deduplication(self):
        engine = BanditEngine(data_dir=_tmp())
        engine.register_variants([_variant("A"), _variant("B")])
        engine.register_variants([_variant("B"), _variant("C")])  # B already exists
        assert len(engine._state.arms) == 3

    def test_existing_arm_stats_preserved_on_reregister(self):
        engine = BanditEngine(data_dir=_tmp())
        engine.register_variants([_variant("A")])
        engine.update("A", impressions=100, clicks=8)
        engine.register_variants([_variant("A")])  # duplicate — must not reset stats
        assert engine._state.arms[0].impressions == 100

    def test_persists_to_store(self):
        tmp    = _tmp()
        engine = BanditEngine(data_dir=tmp)
        engine.register_variants([_variant("A")])
        loaded = BanditStore(data_dir=tmp).load()
        assert len(loaded.arms) == 1


# ── select_variant() — exploration ────────────────────────────────────────────

class TestSelectVariantExploration:
    def test_returns_none_with_no_arms(self):
        engine = _engine_with_arms()
        assert engine.select_variant() is None

    def test_unvisited_arm_selected_immediately(self):
        a = _arm("A", impressions=100, reward=0.08)
        b = _arm("B", impressions=0)   # unvisited → must be picked
        engine = _engine_with_arms(a, b)
        selected = engine.select_variant()
        assert selected.variant_id == "B"

    def test_all_unvisited_returns_first(self):
        a = _arm("A", impressions=0)
        b = _arm("B", impressions=0)
        engine = _engine_with_arms(a, b)
        # First unvisited arm wins (linear scan)
        assert engine.select_variant().variant_id == "A"

    def test_single_arm_always_selected(self):
        engine = _engine_with_arms(_arm("A", impressions=50, reward=0.05))
        assert engine.select_variant().variant_id == "A"


# ── select_variant() — UCB1 exploitation ─────────────────────────────────────

class TestSelectVariantUCB1:
    def test_higher_reward_preferred_with_equal_impressions(self):
        a = _arm("A", impressions=100, reward=0.04)
        b = _arm("B", impressions=100, reward=0.09)  # higher reward
        engine = _engine_with_arms(a, b)
        assert engine.select_variant().variant_id == "B"

    def test_less_explored_arm_preferred_despite_lower_reward(self):
        # A: many impressions, higher mean — exploitation suggests A
        # B: few impressions — UCB exploration bonus is large → B wins
        a = _arm("A", impressions=500, reward=0.08)
        b = _arm("B", impressions=10,  reward=0.05)
        engine = _engine_with_arms(a, b)

        total    = 510
        log_t    = math.log(total)
        ucb_a    = 0.08 + math.sqrt(2 * log_t / 500)
        ucb_b    = 0.05 + math.sqrt(2 * log_t / 10)
        # Verify test premise
        assert ucb_b > ucb_a, "UCB of B should exceed A for this test to be meaningful"

        assert engine.select_variant().variant_id == "B"

    def test_heavily_tested_arm_not_boosted(self):
        # A has explored a lot — tiny exploration bonus; B has higher reward → B wins
        a = _arm("A", impressions=10000, reward=0.07)
        b = _arm("B", impressions=10000, reward=0.09)
        engine = _engine_with_arms(a, b)
        assert engine.select_variant().variant_id == "B"

    def test_ucb1_formula_score_correct(self):
        """Verify the engine's score matches the expected UCB1 value."""
        a = _arm("A", impressions=100, reward=0.05)
        b = _arm("B", impressions=200, reward=0.05)
        engine = _engine_with_arms(a, b)
        total = 300
        expected_ucb_a = 0.05 + math.sqrt(2 * math.log(total) / 100)
        expected_ucb_b = 0.05 + math.sqrt(2 * math.log(total) / 200)
        # A should be selected because it has fewer impressions (larger bonus)
        assert expected_ucb_a > expected_ucb_b
        assert engine.select_variant().variant_id == "A"


# ── update() — reward computation ─────────────────────────────────────────────

class TestUpdate:
    def test_impressions_and_clicks_accumulated(self):
        engine = _engine_with_arms(_arm("A"))
        engine.update("A", impressions=100, clicks=8)
        arm = engine._state.arms[0]
        assert arm.impressions == 100
        assert arm.clicks == 8

    def test_reward_is_ctr_without_retention(self):
        engine = _engine_with_arms(_arm("A"))
        engine.update("A", impressions=100, clicks=8, retention_30s=0.0)
        arm = engine._state.arms[0]
        assert abs(arm.reward - 0.08) < 1e-6

    def test_reward_is_ctr_times_retention(self):
        engine = _engine_with_arms(_arm("A"))
        engine.update("A", impressions=100, clicks=10, retention_30s=0.70)
        arm = engine._state.arms[0]
        expected = 0.10 * 0.70
        assert abs(arm.reward - expected) < 1e-6

    def test_retention_correctly_accumulated_across_batches(self):
        engine = _engine_with_arms(_arm("A"))
        # Batch 1: 100 impressions, retention=0.60
        engine.update("A", impressions=100, clicks=8, retention_30s=0.60)
        # Batch 2: 100 impressions, retention=0.80
        engine.update("A", impressions=100, clicks=12, retention_30s=0.80)
        arm = engine._state.arms[0]
        assert arm.impressions   == 200
        assert arm.clicks        == 20
        assert abs(arm.retention_sum - 140.0) < 1e-6  # 0.60*100 + 0.80*100
        avg_ret  = 140.0 / 200
        expected = (20 / 200) * avg_ret
        assert abs(arm.reward - expected) < 1e-6

    def test_multiple_arms_updated_independently(self):
        a = _arm("A")
        b = _arm("B")
        engine = _engine_with_arms(a, b)
        engine.update("A", impressions=50,  clicks=4)
        engine.update("B", impressions=100, clicks=12)
        assert engine._state.arms[0].impressions == 50
        assert engine._state.arms[1].impressions == 100

    def test_unknown_variant_id_does_not_crash(self):
        engine = _engine_with_arms(_arm("A"))
        engine.update("MISSING", impressions=10, clicks=1)  # should log warning, not crash

    def test_update_persists_to_store(self):
        tmp    = _tmp()
        engine = BanditEngine(data_dir=tmp)
        engine._state.arms.append(_arm("A"))
        engine.update("A", impressions=20, clicks=2)
        loaded = BanditStore(data_dir=tmp).load()
        assert loaded.arms[0].impressions == 20


# ── can_switch() / record_switch() / time_until_switch() ─────────────────────

class TestRateLimiting:
    def test_can_switch_true_when_never_switched(self):
        engine = BanditEngine(data_dir=_tmp(), min_switch_hours=2.0)
        assert engine.can_switch() is True

    def test_cannot_switch_immediately_after_record(self):
        engine = BanditEngine(data_dir=_tmp(), min_switch_hours=2.0)
        engine.record_switch()
        assert engine.can_switch() is False

    def test_can_switch_after_min_hours(self):
        engine = BanditEngine(data_dir=_tmp(), min_switch_hours=2.0)
        past = datetime.now(timezone.utc) - timedelta(hours=3)
        engine._state.last_switch_at = past.isoformat()
        assert engine.can_switch() is True

    def test_cannot_switch_before_min_hours(self):
        engine = BanditEngine(data_dir=_tmp(), min_switch_hours=4.0)
        recent = datetime.now(timezone.utc) - timedelta(hours=1)
        engine._state.last_switch_at = recent.isoformat()
        assert engine.can_switch() is False

    def test_time_until_switch_zero_when_ready(self):
        engine = BanditEngine(data_dir=_tmp(), min_switch_hours=2.0)
        assert engine.time_until_switch() == 0.0

    def test_time_until_switch_positive_when_blocked(self):
        engine = BanditEngine(data_dir=_tmp(), min_switch_hours=4.0)
        recent = datetime.now(timezone.utc) - timedelta(hours=1)
        engine._state.last_switch_at = recent.isoformat()
        remaining = engine.time_until_switch()
        assert 2.9 < remaining < 3.1  # ~3 hours remaining

    def test_record_switch_persists(self):
        tmp    = _tmp()
        engine = BanditEngine(data_dir=tmp, min_switch_hours=2.0)
        engine.record_switch()
        loaded = BanditEngine(data_dir=tmp, min_switch_hours=2.0)
        assert loaded._state.last_switch_at is not None


# ── suggest_strategy_adjustments() ────────────────────────────────────────────

class TestSuggestStrategyAdjustments:
    def test_returns_empty_with_no_data(self):
        engine = _engine_with_arms(_arm("A", impressions=0))
        assert engine.suggest_strategy_adjustments() == {}

    def test_conflict_winner_positive_delta(self):
        a = _arm("A", "curiosity", impressions=100, reward=0.04)
        b = _arm("B", "conflict",  impressions=100, reward=0.09)
        engine = _engine_with_arms(a, b)
        adj = engine.suggest_strategy_adjustments()
        assert adj["preferred_variant_type"]    == "conflict"
        assert adj["hook_aggressiveness_delta"] == +0.10

    def test_curiosity_winner_moderate_delta(self):
        a = _arm("A", "curiosity", impressions=100, reward=0.07)
        b = _arm("B", "simple",    impressions=100, reward=0.04)
        engine = _engine_with_arms(a, b)
        adj = engine.suggest_strategy_adjustments()
        assert adj["hook_aggressiveness_delta"] == +0.05

    def test_simple_winner_negative_delta(self):
        a = _arm("A", "simple",   impressions=100, reward=0.08)
        b = _arm("B", "conflict", impressions=100, reward=0.03)
        engine = _engine_with_arms(a, b)
        adj = engine.suggest_strategy_adjustments()
        assert adj["hook_aggressiveness_delta"] == -0.05

    def test_best_arm_id_in_result(self):
        a = _arm("A", "curiosity", impressions=100, reward=0.05)
        b = _arm("B", "conflict",  impressions=100, reward=0.09)
        engine = _engine_with_arms(a, b)
        adj = engine.suggest_strategy_adjustments()
        assert adj["best_arm_id"]  == "B"
        assert adj["best_reward"]  == 0.09

    def test_ignores_arms_with_zero_impressions(self):
        a = _arm("A", "conflict",  impressions=100, reward=0.09)
        b = _arm("B", "curiosity", impressions=0,   reward=0.0)
        engine = _engine_with_arms(a, b)
        adj = engine.suggest_strategy_adjustments()
        assert adj["preferred_variant_type"] == "conflict"


# ── reset() ───────────────────────────────────────────────────────────────────

class TestReset:
    def test_full_reset_clears_all_arms(self):
        engine = _engine_with_arms(_arm("A", impressions=100), _arm("B"))
        engine.reset(keep_history=False)
        assert engine._state.arms == []

    def test_history_reset_zeroes_metrics(self):
        engine = _engine_with_arms(
            _arm("A", impressions=100, clicks=8, retention_sum=60.0, reward=0.048)
        )
        engine.reset(keep_history=True)
        arm = engine._state.arms[0]
        assert arm.impressions   == 0
        assert arm.clicks        == 0
        assert arm.retention_sum == 0.0
        assert arm.reward        == 0.0

    def test_history_reset_preserves_arm_definitions(self):
        engine = _engine_with_arms(_arm("A", "conflict"))
        engine.reset(keep_history=True)
        assert len(engine._state.arms) == 1
        assert engine._state.arms[0].type == "conflict"


# ── Full pipeline integration ──────────────────────────────────────────────────

class TestFullPipeline:
    def test_register_select_update_cycle(self):
        from src.packaging_engine import PackagingEngine

        @dataclass
        class _FakePlan:
            editorial_plan: list

        @dataclass
        class _FakeConfig:
            mode: str = "growth"

        tmp    = _tmp()
        engine = BanditEngine(data_dir=tmp, min_switch_hours=0.0)
        pe     = PackagingEngine()
        variants = pe.generate_variants({"conflict": "hype vs reality"}, _FakeConfig())

        engine.register_variants(variants)
        assert len(engine._state.arms) == 3

        # Exploration phase: all arms get selected once
        selected_ids = set()
        for _ in range(3):
            arm = engine.select_variant()
            assert arm is not None
            selected_ids.add(arm.variant_id)
            engine.update(arm.variant_id, impressions=100, clicks=8, retention_30s=0.65)

        assert selected_ids == {"A", "B", "C"}

    def test_bandit_converges_to_best_arm(self):
        """After many updates, bandit reliably selects the best arm."""
        engine = _engine_with_arms(
            _arm("A", "curiosity", impressions=1000, reward=0.04),
            _arm("B", "conflict",  impressions=1000, reward=0.09),  # clearly best
            _arm("C", "simple",    impressions=1000, reward=0.05),
        )
        # With lots of impressions, exploration bonus is tiny → pure exploitation
        # All arms visited: just use reward ordering
        selected = engine.select_variant()
        assert selected.variant_id == "B"

    def test_suggest_adjustments_after_updates(self):
        engine = _engine_with_arms(_arm("A"), _arm("B", "conflict"))
        engine.update("A", impressions=100, clicks=4,  retention_30s=0.60)
        engine.update("B", impressions=100, clicks=10, retention_30s=0.75)
        adj = engine.suggest_strategy_adjustments()
        assert adj["preferred_variant_type"] == "conflict"
        assert adj["hook_aggressiveness_delta"] == +0.10
