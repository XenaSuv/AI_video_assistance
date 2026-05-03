"""Decision engine for strategic content generation based on performance feedback."""
from __future__ import annotations

import math
import random
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any

from loguru import logger

sys_path_insert = __import__("sys").path.insert
from pathlib import Path as _Path

import sys
sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))
from config import settings
from src.shared_types import ContentStrategy


class DecisionEngine:
    """Strategic decision engine that analyzes feedback history to optimize content generation."""

    def __init__(self, config: dict[str, Any] | None = None):
        self.config = config or {}

        # Configuration parameters
        self.min_samples = self.config.get("min_samples", 5)
        self.base_exploration = self.config.get("exploration_rate", 0.2)
        self.growth_threshold = self.config.get("growth_threshold", 0.45)
        self.safe_threshold = self.config.get("safe_threshold", 0.65)
        self.hook_success_threshold = self.config.get("hook_success_threshold", 0.6)
        self.hook_excellent_threshold = self.config.get("hook_excellent_threshold", 0.75)
        # Exponential decay: data older than half_life_days has half the influence.
        self.decay_half_life_days = self.config.get("decay_half_life_days", 14)

    def decide(self, feedback_history: list[dict[str, Any]]) -> ContentStrategy:
        """Make strategic decisions based on feedback history."""
        logger.info(f"Making strategic decisions based on {len(feedback_history)} feedback items")

        if not feedback_history:
            logger.info("No feedback history - using default decisions")
            return self._default_decision()

        angle_weights = self._compute_weights(feedback_history, key="angle")
        format_weights = self._compute_weights(feedback_history, key="format")
        exploration_rate = self._compute_exploration(feedback_history)
        mode = self._select_mode(feedback_history)

        strategy = ContentStrategy(
            angle_weights=angle_weights,
            format_weights=format_weights,
            exploration_rate=exploration_rate,
            mode=mode,
            confidence=self._compute_confidence(feedback_history),
        )
        logger.info(f"Decision: mode={mode}, exploration={exploration_rate:.2f}, {len(angle_weights)} angle weights")
        return strategy

    def _decay_weight(self, item: dict[str, Any], now: datetime) -> float:
        """Return exponential decay weight for a feedback item based on its age.

        Items without a timestamp get weight 1.0 (backward compatible).
        Half-life is controlled by self.decay_half_life_days.
        """
        ts_str = item.get("timestamp") or item.get("published_at")
        if not ts_str:
            return 1.0
        try:
            ts = datetime.fromisoformat(str(ts_str).replace("Z", "+00:00"))
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            age_days = (now - ts).total_seconds() / 86400.0
            return math.exp(-math.log(2) * age_days / self.decay_half_life_days)
        except Exception:
            return 1.0

    def _decay_weighted_mean(
        self,
        feedback: list[dict[str, Any]],
        value_key: str,
        now: datetime,
    ) -> float | None:
        """Return the decay-weighted mean of value_key across feedback items.

        Returns None when no items have the field populated.
        """
        total_w = 0.0
        total_v = 0.0
        for item in feedback:
            v = item.get(value_key)
            if v is None:
                continue
            w = self._decay_weight(item, now)
            total_v += v * w
            total_w += w
        return total_v / total_w if total_w > 0 else None

    def _compute_weights(self, feedback: list[dict[str, Any]], key: str) -> dict[str, float]:
        """Compute decay-weighted performance weights for angles or formats."""
        now = datetime.now(timezone.utc)

        # Collect (hook_score, decay_weight) pairs per category
        scores: defaultdict[str, list[tuple[float, float]]] = defaultdict(list)
        for item in feedback:
            if key in item and "hook_score" in item and item["hook_score"] is not None:
                scores[item[key]].append((item["hook_score"], self._decay_weight(item, now)))

        weights = {}
        for category, pairs in scores.items():
            if len(pairs) < self.min_samples:
                continue

            total_w = sum(w for _, w in pairs)
            avg_hook = sum(s * w for s, w in pairs) / total_w

            # Retention bonus: decay-weighted fraction of high-retention videos × 0.2
            w_high = sum(
                self._decay_weight(item, now)
                for item in feedback
                if item.get(key) == category
                and item.get("avg_view_percentage", 0) > 0.7
            )
            w_all_ret = sum(
                self._decay_weight(item, now)
                for item in feedback
                if item.get(key) == category and "avg_view_percentage" in item
            )
            retention_bonus = (w_high / w_all_ret * 0.2) if w_all_ret > 0 else 0

            weight = round(avg_hook + retention_bonus, 2)
            weights[category] = max(0.1, min(1.0, weight))

        logger.debug(f"Computed {key} weights: {weights}")
        return weights

    def _compute_exploration(self, feedback: list[dict[str, Any]]) -> float:
        """Compute adaptive exploration rate based on decay-weighted recent performance."""
        if not feedback:
            return self.base_exploration

        now = datetime.now(timezone.utc)
        avg_hook = self._decay_weighted_mean(feedback[-10:], "hook_score", now)
        if avg_hook is None:
            return self.base_exploration

        exploration = self.base_exploration
        if avg_hook < self.hook_success_threshold:
            exploration = min(0.4, exploration + 0.1)
            logger.info(f"Low hook performance ({avg_hook:.2f}) - increasing exploration to {exploration}")
        elif avg_hook > self.hook_excellent_threshold:
            exploration = max(0.1, exploration - 0.1)
            logger.info(f"Excellent hook performance ({avg_hook:.2f}) - reducing exploration to {exploration}")

        return round(exploration, 2)

    def _select_mode(self, feedback: list[dict[str, Any]]) -> str:
        """Select operational mode based on decay-weighted average retention."""
        recent = feedback[-10:]
        if not recent:
            return "balanced"

        now = datetime.now(timezone.utc)
        avg_retention = self._decay_weighted_mean(recent, "avg_view_percentage", now)
        if avg_retention is None:
            return "balanced"

        if avg_retention < self.growth_threshold:
            mode = "growth"
            logger.info(f"Low retention ({avg_retention:.2f}) - switching to growth mode")
        elif avg_retention > self.safe_threshold:
            mode = "safe"
            logger.info(f"High retention ({avg_retention:.2f}) - switching to safe mode")
        else:
            mode = "balanced"
            logger.info(f"Balanced retention ({avg_retention:.2f}) - staying in balanced mode")

        return mode

    def _compute_confidence(self, feedback: list[dict[str, Any]]) -> float:
        """Compute confidence level in decisions based on sample size and consistency.

        Returns:
            Confidence score between 0.0 and 1.0
        """
        if len(feedback) < self.min_samples:
            return 0.3  # Low confidence with small sample

        # Check consistency of recent performance
        recent = feedback[-5:]
        if len(recent) < 3:
            return 0.5

        hook_scores = [x.get("hook_score", 0) for x in recent if x.get("hook_score") is not None]
        if len(hook_scores) < 3:
            return 0.5

        # High confidence if consistent performance
        score_range = max(hook_scores) - min(hook_scores)
        if score_range < 0.2:  # Consistent results
            return 0.8

        return 0.6  # Moderate confidence

    def _default_decision(self) -> ContentStrategy:
        """Return default decision when no feedback is available."""
        return ContentStrategy(
            angle_weights={},
            format_weights={},
            exploration_rate=self.base_exploration,
            mode="balanced",
            confidence=0.3,
        )

    def get_recommended_angle(self, strategy: ContentStrategy, available_angles: list[str]) -> str:
        """Get recommended angle based on strategy and exploration."""
        available_weights = {k: v for k, v in strategy.angle_weights.items() if k in available_angles}

        if not available_weights or random.random() < strategy.exploration_rate:
            return random.choice(available_angles)

        return max(available_weights, key=available_weights.get)

    def get_recommended_format(self, strategy: ContentStrategy, available_formats: list[str]) -> str:
        """Get recommended format based on strategy and exploration."""
        mode = strategy.mode

        available_weights = {k: v for k, v in strategy.format_weights.items() if k in available_formats}

        if mode == "growth":
            available_weights = self._adjust_for_mode(available_weights, growth_boost=0.1)
        elif mode == "safe":
            available_weights = self._adjust_for_mode(available_weights, safe_boost=0.2)

        if not available_weights or random.random() < strategy.exploration_rate:
            return random.choice(available_formats)

        return max(available_weights, key=available_weights.get)

    def _adjust_for_mode(self, weights: dict[str, float], growth_boost: float = 0.0, safe_boost: float = 0.0) -> dict[str, float]:
        """Adjust weights based on operational mode."""
        adjusted = {}

        for format_name, weight in weights.items():
            if "hot_take" in format_name.lower() or "controversial" in format_name.lower():
                # Riskier formats get growth boost
                adjusted[format_name] = min(1.0, weight + growth_boost)
            elif "deep_dive" in format_name.lower() or "explainer" in format_name.lower():
                # Safer formats get safe boost
                adjusted[format_name] = min(1.0, weight + safe_boost)
            else:
                adjusted[format_name] = weight

        return adjusted