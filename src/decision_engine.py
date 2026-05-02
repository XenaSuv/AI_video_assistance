"""Decision engine for strategic content generation based on performance feedback."""
from __future__ import annotations

import random
from collections import defaultdict
from typing import Any

from loguru import logger

sys_path_insert = __import__("sys").path.insert
from pathlib import Path as _Path

import sys
sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))
from config import settings


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

    def decide(self, feedback_history: list[dict[str, Any]]) -> dict[str, Any]:
        """Make strategic decisions based on feedback history.

        Args:
            feedback_history: List of feedback analysis results

        Returns:
            Decision dictionary with weights and strategy
        """
        logger.info(f"Making strategic decisions based on {len(feedback_history)} feedback items")

        if not feedback_history:
            logger.info("No feedback history - using default decisions")
            return self._default_decision()

        # Compute performance-based weights
        angle_weights = self._compute_weights(feedback_history, key="angle")
        format_weights = self._compute_weights(feedback_history, key="format")

        # Adaptive exploration rate
        exploration_rate = self._compute_exploration(feedback_history)

        # Select operational mode
        mode = self._select_mode(feedback_history)

        decision = {
            "angle_weights": angle_weights,
            "format_weights": format_weights,
            "exploration_rate": exploration_rate,
            "mode": mode,
            "confidence": self._compute_confidence(feedback_history),
        }

        logger.info(f"Decision: mode={mode}, exploration={exploration_rate:.2f}, {len(angle_weights)} angle weights")
        return decision

    def _compute_weights(self, feedback: list[dict[str, Any]], key: str) -> dict[str, float]:
        """Compute performance weights for angles or formats.

        Args:
            feedback: Feedback history
            key: Key to analyze ("angle" or "format")

        Returns:
            Dictionary mapping keys to performance weights
        """
        scores = defaultdict(list)

        # Collect scores for each category
        for item in feedback:
            if key in item and "hook_score" in item and item["hook_score"] is not None:
                category = item[key]
                score = item["hook_score"]
                scores[category].append(score)

        weights = {}

        for category, vals in scores.items():
            if len(vals) >= self.min_samples:
                # Use hook score as primary metric, but also consider retention
                avg_hook = sum(vals) / len(vals)

                # Bonus for high retention videos
                retention_bonus = 0
                for item in feedback:
                    if item.get(key) == category and "avg_view_percentage" in item:
                        if item["avg_view_percentage"] > 0.7:
                            retention_bonus += 0.1

                weight = round(avg_hook + min(retention_bonus, 0.2), 2)
                weights[category] = max(0.1, min(1.0, weight))  # Clamp between 0.1-1.0

        logger.debug(f"Computed {key} weights: {weights}")
        return weights

    def _compute_exploration(self, feedback: list[dict[str, Any]]) -> float:
        """Compute adaptive exploration rate based on recent performance.

        Returns:
            Exploration rate between 0.1 and 0.4
        """
        recent = feedback[-10:]  # Last 10 videos

        if not recent:
            return self.base_exploration

        # Calculate average hook performance
        hook_scores = [x.get("hook_score", 0) for x in recent if x.get("hook_score") is not None]
        if not hook_scores:
            return self.base_exploration

        avg_hook = sum(hook_scores) / len(hook_scores)

        exploration = self.base_exploration

        # If hooks are underperforming, increase exploration
        if avg_hook < self.hook_success_threshold:
            exploration = min(0.4, exploration + 0.1)
            logger.info(f"Low hook performance ({avg_hook:.2f}) - increasing exploration to {exploration}")

        # If hooks are excellent, reduce exploration to exploit winners
        elif avg_hook > self.hook_excellent_threshold:
            exploration = max(0.1, exploration - 0.1)
            logger.info(f"Excellent hook performance ({avg_hook:.2f}) - reducing exploration to {exploration}")

        return round(exploration, 2)

    def _select_mode(self, feedback: list[dict[str, Any]]) -> str:
        """Select operational mode based on overall performance.

        Returns:
            "growth", "safe", or "balanced"
        """
        recent = feedback[-10:]

        if not recent:
            return "balanced"

        # Calculate average retention
        retentions = [x.get("avg_view_percentage", 0) for x in recent if x.get("avg_view_percentage") is not None]
        if not retentions:
            return "balanced"

        avg_retention = sum(retentions) / len(retentions)

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

    def _default_decision(self) -> dict[str, Any]:
        """Return default decision when no feedback is available."""
        return {
            "angle_weights": {},
            "format_weights": {},
            "exploration_rate": self.base_exploration,
            "mode": "balanced",
            "confidence": 0.3,
        }

    def get_recommended_angle(self, strategy: dict[str, Any], available_angles: list[str]) -> str:
        """Get recommended angle based on strategy and exploration.

        Args:
            strategy: Decision strategy from decide()
            available_angles: List of available angle options

        Returns:
            Recommended angle string
        """
        weights = strategy.get("angle_weights", {})
        exploration_rate = strategy.get("exploration_rate", self.base_exploration)

        # Filter to available angles
        available_weights = {k: v for k, v in weights.items() if k in available_angles}

        # Exploration: random choice
        if not available_weights or random.random() < exploration_rate:
            return random.choice(available_angles)

        # Exploitation: choose highest weighted
        return max(available_weights, key=available_weights.get)

    def get_recommended_format(self, strategy: dict[str, Any], available_formats: list[str]) -> str:
        """Get recommended format based on strategy and exploration.

        Args:
            strategy: Decision strategy from decide()
            available_formats: List of available format options

        Returns:
            Recommended format string
        """
        weights = strategy.get("format_weights", {})
        exploration_rate = strategy.get("exploration_rate", self.base_exploration)
        mode = strategy.get("mode", "balanced")

        # Filter to available formats
        available_weights = {k: v for k, v in weights.items() if k in available_formats}

        # Apply mode-specific adjustments
        if mode == "growth":
            # In growth mode, slightly favor riskier formats
            available_weights = self._adjust_for_mode(available_weights, growth_boost=0.1)
        elif mode == "safe":
            # In safe mode, strongly favor proven formats
            available_weights = self._adjust_for_mode(available_weights, safe_boost=0.2)

        # Exploration: random choice
        if not available_weights or random.random() < exploration_rate:
            return random.choice(available_formats)

        # Exploitation: choose highest weighted
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