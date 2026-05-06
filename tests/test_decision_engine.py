"""Tests for DecisionEngine — pure logic, no external dependencies."""
import math
import pytest
from datetime import datetime, timedelta, timezone
from src.decision_engine import DecisionEngine
from src.shared_types import ContentStrategy


@pytest.fixture
def engine():
    return DecisionEngine()


@pytest.fixture
def engine_low_min_samples():
    return DecisionEngine(config={"min_samples": 2})


# ── decide() ──────────────────────────────────────────────────────────────────

class TestDecide:
    def test_empty_history_returns_default(self, engine):
        result = engine.decide([])
        assert result.mode == "balanced"
        assert result.angle_weights == {}
        assert result.format_weights == {}
        assert result.confidence == 0.3
        assert result.exploration_rate == engine.base_exploration

    def test_returns_content_strategy_with_all_fields(self, engine, feedback_history):
        result = engine.decide(feedback_history)
        assert isinstance(result, ContentStrategy)
        assert hasattr(result, "angle_weights")
        assert hasattr(result, "format_weights")
        assert hasattr(result, "exploration_rate")
        assert hasattr(result, "mode")
        assert hasattr(result, "confidence")

    def test_low_retention_triggers_growth_mode(self, engine, low_performing_feedback):
        result = engine.decide(low_performing_feedback)
        assert result.mode == "growth"

    def test_high_retention_triggers_safe_mode(self, engine, high_performing_feedback):
        result = engine.decide(high_performing_feedback)
        assert result.mode == "safe"

    def test_medium_retention_stays_balanced(self, engine, feedback_history):
        result = engine.decide(feedback_history)
        assert result.mode == "balanced"

    def test_exploration_rate_is_bounded(self, engine, feedback_history):
        result = engine.decide(feedback_history)
        assert 0.1 <= result.exploration_rate <= 0.4

    def test_confidence_increases_with_sample_size(self, engine):
        small = [{"hook_score": 0.7, "avg_view_percentage": 0.6, "angle": "a", "format": "b"}] * 3
        large = [{"hook_score": 0.7, "avg_view_percentage": 0.6, "angle": "a", "format": "b"}] * 10
        assert engine.decide(small).confidence <= engine.decide(large).confidence


# ── _compute_weights() ────────────────────────────────────────────────────────

class TestComputeWeights:
    def test_skips_categories_below_min_samples(self, engine):
        feedback = [
            {"angle": "technical_breakthrough", "hook_score": 0.8},
            {"angle": "technical_breakthrough", "hook_score": 0.7},
        ]
        weights = engine._compute_weights(feedback, "angle")
        assert "technical_breakthrough" not in weights

    def test_includes_categories_above_min_samples(self, engine_low_min_samples):
        feedback = [
            {"angle": "technical_breakthrough", "hook_score": 0.8},
            {"angle": "technical_breakthrough", "hook_score": 0.6},
        ]
        weights = engine_low_min_samples._compute_weights(feedback, "angle")
        assert "technical_breakthrough" in weights

    def test_weight_is_average_hook_score(self, engine_low_min_samples):
        feedback = [
            {"angle": "hot_angle", "hook_score": 0.8},
            {"angle": "hot_angle", "hook_score": 0.6},
        ]
        weights = engine_low_min_samples._compute_weights(feedback, "angle")
        assert abs(weights["hot_angle"] - 0.7) < 0.01

    def test_weight_clamped_to_valid_range(self, engine_low_min_samples):
        feedback = [
            {"angle": "x", "hook_score": 0.0},
            {"angle": "x", "hook_score": 0.0},
        ]
        weights = engine_low_min_samples._compute_weights(feedback, "angle")
        assert weights["x"] >= 0.1

    def test_high_retention_bonus_applied(self, engine_low_min_samples):
        feedback_no_bonus = [
            {"angle": "x", "hook_score": 0.5, "avg_view_percentage": 0.4},
            {"angle": "x", "hook_score": 0.5, "avg_view_percentage": 0.4},
        ]
        feedback_with_bonus = [
            {"angle": "x", "hook_score": 0.5, "avg_view_percentage": 0.8},
            {"angle": "x", "hook_score": 0.5, "avg_view_percentage": 0.8},
        ]
        w_no = engine_low_min_samples._compute_weights(feedback_no_bonus, "angle")
        w_yes = engine_low_min_samples._compute_weights(feedback_with_bonus, "angle")
        assert w_yes["x"] > w_no["x"]

    def test_items_missing_hook_score_are_ignored(self, engine_low_min_samples):
        feedback = [
            {"angle": "x"},
            {"angle": "x"},
        ]
        weights = engine_low_min_samples._compute_weights(feedback, "angle")
        assert "x" not in weights


# ── _compute_exploration() ────────────────────────────────────────────────────

class TestComputeExploration:
    def test_low_hooks_increase_exploration(self, engine):
        feedback = [{"hook_score": 0.3}] * 10
        rate = engine._compute_exploration(feedback)
        assert rate > engine.base_exploration

    def test_excellent_hooks_decrease_exploration(self, engine):
        feedback = [{"hook_score": 0.9}] * 10
        rate = engine._compute_exploration(feedback)
        assert rate < engine.base_exploration

    def test_medium_hooks_keep_base_rate(self, engine):
        feedback = [{"hook_score": 0.65}] * 10
        rate = engine._compute_exploration(feedback)
        assert rate == engine.base_exploration

    def test_empty_recent_feedback_returns_base(self, engine):
        rate = engine._compute_exploration([])
        assert rate == engine.base_exploration


# ── _select_mode() ────────────────────────────────────────────────────────────

class TestSelectMode:
    def test_growth_mode_when_retention_below_threshold(self, engine):
        feedback = [{"avg_view_percentage": 0.30}] * 5
        assert engine._select_mode(feedback) == "growth"

    def test_safe_mode_when_retention_above_threshold(self, engine):
        feedback = [{"avg_view_percentage": 0.80}] * 5
        assert engine._select_mode(feedback) == "safe"

    def test_balanced_mode_for_middle_range(self, engine):
        feedback = [{"avg_view_percentage": 0.55}] * 5
        assert engine._select_mode(feedback) == "balanced"

    def test_empty_feedback_returns_balanced(self, engine):
        assert engine._select_mode([]) == "balanced"


# ── _compute_confidence() ─────────────────────────────────────────────────────

class TestComputeConfidence:
    def test_low_confidence_with_few_samples(self, engine):
        feedback = [{"hook_score": 0.7}] * 3
        assert engine._compute_confidence(feedback) == 0.3

    def test_high_confidence_with_consistent_performance(self, engine):
        feedback = [{"hook_score": 0.70 + i * 0.01} for i in range(10)]
        assert engine._compute_confidence(feedback) == 0.8

    def test_moderate_confidence_with_variable_performance(self, engine):
        feedback = [{"hook_score": s} for s in [0.3, 0.5, 0.8, 0.4, 0.9] * 2]
        assert engine._compute_confidence(feedback) == 0.6


# ── get_recommended_angle() ───────────────────────────────────────────────────

def _strategy(**kwargs) -> ContentStrategy:
    defaults = dict(angle_weights={}, format_weights={}, exploration_rate=0.2, mode="balanced", confidence=0.5)
    defaults.update(kwargs)
    return ContentStrategy(**defaults)


class TestGetRecommendedAngle:
    def test_exploits_highest_weight(self, engine):
        strategy = _strategy(
            angle_weights={"technical_breakthrough": 0.9, "hot_take": 0.3},
            exploration_rate=0.0,
        )
        result = engine.get_recommended_angle(strategy, ["technical_breakthrough", "hot_take"])
        assert result == "technical_breakthrough"

    def test_returns_valid_angle_always(self, engine):
        strategy = _strategy(angle_weights={}, exploration_rate=1.0)
        angles = ["a", "b", "c"]
        assert engine.get_recommended_angle(strategy, angles) in angles

    def test_ignores_weights_not_in_available_list(self, engine):
        strategy = _strategy(angle_weights={"unknown_angle": 0.99}, exploration_rate=0.0)
        angles = ["technical_breakthrough"]
        assert engine.get_recommended_angle(strategy, angles) in angles


# ── _adjust_for_mode() ────────────────────────────────────────────────────────

class TestAdjustForMode:
    def test_growth_boosts_hot_take(self, engine):
        weights = {"hot_take_format": 0.5, "deep_dive_format": 0.5}
        adjusted = engine._adjust_for_mode(weights, growth_boost=0.2)
        assert adjusted["hot_take_format"] > adjusted["deep_dive_format"]

    def test_safe_boosts_deep_dive(self, engine):
        weights = {"hot_take_format": 0.5, "deep_dive_format": 0.5}
        adjusted = engine._adjust_for_mode(weights, safe_boost=0.2)
        assert adjusted["deep_dive_format"] > adjusted["hot_take_format"]

    def test_weight_capped_at_1_0(self, engine):
        weights = {"hot_take_format": 0.95}
        adjusted = engine._adjust_for_mode(weights, growth_boost=0.5)
        assert adjusted["hot_take_format"] <= 1.0


# ── _decay_weight() ───────────────────────────────────────────────────────────

def _ts(days_ago: float) -> str:
    """Return an ISO timestamp N days in the past."""
    return (datetime.now(timezone.utc) - timedelta(days=days_ago)).isoformat()


class TestDecayWeight:
    def test_fresh_item_weight_near_one(self, engine):
        now = datetime.now(timezone.utc)
        w = engine._decay_weight({"timestamp": _ts(0)}, now)
        assert abs(w - 1.0) < 0.01

    def test_half_life_item_weight_near_half(self, engine):
        now = datetime.now(timezone.utc)
        w = engine._decay_weight({"timestamp": _ts(engine.decay_half_life_days)}, now)
        assert abs(w - 0.5) < 0.02

    def test_double_half_life_item_weight_near_quarter(self, engine):
        now = datetime.now(timezone.utc)
        w = engine._decay_weight({"timestamp": _ts(engine.decay_half_life_days * 2)}, now)
        assert abs(w - 0.25) < 0.02

    def test_no_timestamp_returns_one(self, engine):
        assert engine._decay_weight({}, datetime.now(timezone.utc)) == 1.0

    def test_invalid_timestamp_returns_one(self, engine):
        assert engine._decay_weight({"timestamp": "not-a-date"}, datetime.now(timezone.utc)) == 1.0

    def test_older_item_has_lower_weight_than_newer(self, engine):
        now = datetime.now(timezone.utc)
        w_new = engine._decay_weight({"timestamp": _ts(1)}, now)
        w_old = engine._decay_weight({"timestamp": _ts(30)}, now)
        assert w_new > w_old


# ── decay integration: weights favour recent data ─────────────────────────────

class TestDecayIntegration:
    def test_recent_good_data_outweighs_old_bad_data(self):
        """With short half-life, recent high scores dominate old low scores."""
        engine = DecisionEngine(config={"min_samples": 2, "decay_half_life_days": 7})

        old_bad = [
            {"angle": "x", "hook_score": 0.2, "timestamp": _ts(60)},
            {"angle": "x", "hook_score": 0.2, "timestamp": _ts(60)},
            {"angle": "x", "hook_score": 0.2, "timestamp": _ts(60)},
            {"angle": "x", "hook_score": 0.2, "timestamp": _ts(60)},
            {"angle": "x", "hook_score": 0.2, "timestamp": _ts(60)},
        ]
        recent_good = [
            {"angle": "x", "hook_score": 0.9, "timestamp": _ts(1)},
            {"angle": "x", "hook_score": 0.9, "timestamp": _ts(1)},
            {"angle": "x", "hook_score": 0.9, "timestamp": _ts(1)},
            {"angle": "x", "hook_score": 0.9, "timestamp": _ts(1)},
            {"angle": "x", "hook_score": 0.9, "timestamp": _ts(1)},
        ]
        # Simple average of all would be ~0.55; decay should push it toward 0.9
        weights_with_decay = engine._compute_weights(old_bad + recent_good, "angle")
        plain_avg = 0.55  # naive equal-weight average
        assert weights_with_decay["x"] > plain_avg

    def test_old_good_data_outweighed_by_recent_bad(self):
        """Decay makes recent bad data dominate old good data."""
        engine = DecisionEngine(config={"min_samples": 2, "decay_half_life_days": 7})

        old_good = [
            {"angle": "y", "hook_score": 0.9, "timestamp": _ts(60)},
            {"angle": "y", "hook_score": 0.9, "timestamp": _ts(60)},
            {"angle": "y", "hook_score": 0.9, "timestamp": _ts(60)},
            {"angle": "y", "hook_score": 0.9, "timestamp": _ts(60)},
            {"angle": "y", "hook_score": 0.9, "timestamp": _ts(60)},
        ]
        recent_bad = [
            {"angle": "y", "hook_score": 0.2, "timestamp": _ts(1)},
            {"angle": "y", "hook_score": 0.2, "timestamp": _ts(1)},
            {"angle": "y", "hook_score": 0.2, "timestamp": _ts(1)},
            {"angle": "y", "hook_score": 0.2, "timestamp": _ts(1)},
            {"angle": "y", "hook_score": 0.2, "timestamp": _ts(1)},
        ]
        plain_avg = 0.55
        weights_with_decay = engine._compute_weights(old_good + recent_bad, "angle")
        assert weights_with_decay["y"] < plain_avg

    def test_no_timestamps_behaves_like_plain_average(self):
        """Items without timestamps get weight=1.0 → identical to original behaviour."""
        engine = DecisionEngine(config={"min_samples": 2})
        feedback = [
            {"angle": "z", "hook_score": 0.6},
            {"angle": "z", "hook_score": 0.8},
            {"angle": "z", "hook_score": 0.6},
            {"angle": "z", "hook_score": 0.8},
            {"angle": "z", "hook_score": 0.7},
        ]
        weights = engine._compute_weights(feedback, "angle")
        assert abs(weights["z"] - 0.7) < 0.01

    def test_mode_reflects_recent_retention_not_old(self):
        """Mode selection uses decay — recent low retention wins over old high."""
        engine = DecisionEngine(config={"decay_half_life_days": 7})
        old_high = [{"avg_view_percentage": 0.9, "timestamp": _ts(60)}] * 5
        recent_low = [{"avg_view_percentage": 0.3, "timestamp": _ts(1)}] * 5
        mode = engine._select_mode(old_high + recent_low)
        assert mode == "growth"

    def test_exploration_reflects_recent_hooks_not_old(self):
        """Exploration rate uses decay — recent low hooks win over old high."""
        engine = DecisionEngine(config={"decay_half_life_days": 7})
        old_great = [{"hook_score": 0.95, "timestamp": _ts(60)}] * 5
        recent_poor = [{"hook_score": 0.3, "timestamp": _ts(1)}] * 5
        rate = engine._compute_exploration(old_great + recent_poor)
        assert rate > engine.base_exploration
