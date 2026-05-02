"""Hook optimizer for improving hook effectiveness based on feedback history."""
from __future__ import annotations

import json
import random
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from loguru import logger

sys_path_insert = __import__("sys").path.insert
from pathlib import Path as _Path

import sys
sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))
from config import settings


@dataclass
class HookPattern:
    """Represents a hook pattern with performance metrics."""
    pattern: str
    avg_score: float
    count: int
    success_rate: float


class HookOptimizer:
    """Optimizes hook generation based on historical performance patterns."""

    def __init__(self, feedback_path: str | None = None, exploration_rate: float = 0.2):
        self.feedback_path = feedback_path or str(Path(settings.data_dir) / "feedback_history.json")
        self.exploration_rate = exploration_rate  # 20% of time try new patterns
        self.hook_history = Path(settings.data_dir) / "hook_patterns.json"

    def run(self, editorial_context: dict[str, Any]) -> dict[str, Any]:
        """Generate optimized hook recommendations based on context."""
        logger.info("Running hook optimization")

        feedback = self._load_feedback()
        if not feedback:
            logger.warning("No feedback history available")
            return self._default_recommendations()

        patterns = self._extract_patterns(feedback)
        if not patterns:
            logger.warning("No hook patterns extracted")
            return self._default_recommendations()

        scored = self._score_patterns(patterns)
        filtered = self._filter_by_context(scored, editorial_context)

        recommended = self._select_top_patterns(filtered, top_n=5)
        avoid = self._select_worst_patterns(filtered, bottom_n=3)

        result = {
            "recommended_patterns": recommended,
            "avoid_patterns": avoid,
            "context": editorial_context,
            "exploration_enabled": random.random() < self.exploration_rate,
        }

        # Calculate style bias
        result["style_bias"] = self._calculate_style_bias(recommended, editorial_context)

        self._save_patterns(result)
        return result

    def _load_feedback(self) -> list[dict[str, Any]]:
        """Load feedback history."""
        try:
            if not Path(self.feedback_path).exists():
                return []
            return json.loads(Path(self.feedback_path).read_text())
        except Exception as exc:
            logger.warning(f"Failed to load feedback: {exc}")
            return []

    def _extract_patterns(self, feedback: list[dict[str, Any]]) -> dict[str, list[float]]:
        """Extract hook patterns from feedback."""
        patterns = defaultdict(list)

        for item in feedback:
            hook_text = item.get("hook_text") or item.get("hook")
            hook_score = item.get("hook_score", 0)

            if not hook_text:
                continue

            # Normalize the hook text
            normalized = self._normalize_hook(hook_text)

            # Extract key phrases
            key_phrases = self._extract_key_phrases(normalized)

            # Store scores
            for phrase in key_phrases:
                patterns[phrase].append(hook_score)

        logger.info(f"Extracted {len(patterns)} unique hook patterns")
        return patterns

    def _normalize_hook(self, hook: str) -> str:
        """Normalize hook text for pattern matching."""
        hook = hook.lower().strip()
        # Remove special characters but keep structure
        hook = re.sub(r"[^\w\s]", "", hook)
        return hook

    def _extract_key_phrases(self, hook: str) -> list[str]:
        """Extract key phrases from hook text."""
        phrases = []

        # Add full hook
        if len(hook) > 5:
            phrases.append(hook)

        # Add first few words (pattern start)
        words = hook.split()
        if len(words) >= 3:
            phrases.append(" ".join(words[:3]))

        # Add interesting patterns
        patterns = [
            r"but (here|now|this|thats)",
            r"(however|yet|still|unless|though)",
            r"(problem|issue|catch|twist|secret|hidden)",
            r"(nobody|no one|never|surprising|unexpected)",
            r"(actually|really|honestly|surprisingly)",
        ]

        for pattern in patterns:
            if re.search(pattern, hook):
                match = re.search(pattern, hook)
                if match:
                    phrases.append(match.group(0))

        return phrases

    def _score_patterns(self, patterns: dict[str, list[float]]) -> list[HookPattern]:
        """Score patterns based on historical performance."""
        scored = []

        for pattern, scores in patterns.items():
            if len(scores) < 2:  # Minimum threshold
                continue

            avg_score = sum(scores) / len(scores)
            success_count = sum(1 for s in scores if s > 0.6)
            success_rate = success_count / len(scores)

            scored.append(HookPattern(
                pattern=pattern,
                avg_score=avg_score,
                count=len(scores),
                success_rate=success_rate,
            ))

        # Sort by average score, with success rate as tiebreaker
        scored.sort(key=lambda x: (x.avg_score, x.success_rate), reverse=True)

        logger.debug(f"Scored {len(scored)} patterns")
        return scored

    def _filter_by_context(self, scored: list[HookPattern], context: dict[str, Any]) -> list[HookPattern]:
        """Filter patterns to match editorial context."""
        angle = context.get("angle", "").lower()
        format_type = context.get("format", "").lower()
        persona = context.get("persona", {})

        if isinstance(persona, dict):
            persona_style = persona.get("style", "").lower()
        else:
            persona_style = str(persona).lower()

        filtered = []

        for pattern in scored:
            pattern_text = pattern.pattern.lower()

            # Angle-specific filtering
            if "hype" in angle:
                # For hype angles, prefer contrast patterns
                if any(w in pattern_text for w in ["problem", "no one", "but", "yet"]):
                    filtered.append(pattern)
                    continue

            if "conflict" in angle or "threat" in angle:
                # For conflict, prefer tension patterns
                if any(w in pattern_text for w in ["problem", "issue", "catch", "unless"]):
                    filtered.append(pattern)
                    continue

            if "twist" in angle:
                # For twist, prefer surprise patterns
                if any(w in pattern_text for w in ["never", "unexpected", "actually", "surprisingly"]):
                    filtered.append(pattern)
                    continue

            if "curiosity" in angle or "personal" in angle:
                # For curiosity/personal, prefer question patterns
                if any(w in pattern_text for w in ["but here", "thats where", "honestly"]):
                    filtered.append(pattern)
                    continue

            # Persona-specific filtering
            if "skeptical" in persona_style or "cynical" in persona_style:
                if any(w in pattern_text for w in ["honestly", "not convinced", "feels off"]):
                    filtered.append(pattern)
                    continue

            # Format filtering
            if "hot_take" in format_type:
                # Hot takes need more aggressive hooks
                if pattern.avg_score > 0.65:
                    filtered.append(pattern)
                    continue

            if "deep_dive" in format_type:
                # Deep dives can use educational patterns
                if pattern.count >= 3:
                    filtered.append(pattern)
                    continue

            # Default: include if score is decent
            if pattern.avg_score > 0.5:
                filtered.append(pattern)

        logger.debug(f"Filtered to {len(filtered)} context-appropriate patterns")
        return filtered if filtered else scored[:10]

    def _select_top_patterns(self, scored: list[HookPattern], top_n: int = 5) -> list[str]:
        """Select top performing patterns."""
        return [p.pattern for p in scored[:top_n]]

    def _select_worst_patterns(self, scored: list[HookPattern], bottom_n: int = 3) -> list[str]:
        """Select worst performing patterns to avoid."""
        if len(scored) <= bottom_n:
            return []
        return [p.pattern for p in scored[-bottom_n:]]

    def _calculate_style_bias(self, patterns: list[str], context: dict[str, Any]) -> dict[str, float]:
        """Calculate style bias based on recommended patterns."""
        bias = {
            "controversy": 0.5,
            "curiosity": 0.5,
            "emotion": 0.5,
            "evidence": 0.5,
        }

        pattern_text = " ".join(patterns).lower()

        # Detect controversy patterns
        if any(w in pattern_text for w in ["problem", "issue", "fail", "catch", "but"]):
            bias["controversy"] = 0.75

        # Detect curiosity patterns
        if any(w in pattern_text for w in ["but here", "where", "what", "honestly", "actually"]):
            bias["curiosity"] = 0.8

        # Detect emotional patterns
        if any(w in pattern_text for w in ["never", "surprising", "unexpected", "no one"]):
            bias["emotion"] = 0.7

        # Detect evidence patterns
        if any(w in pattern_text for w in ["data", "shows", "proves", "research", "study"]):
            bias["evidence"] = 0.6

        return bias

    def _save_patterns(self, result: dict[str, Any]) -> None:
        """Save pattern analysis for debugging."""
        try:
            history = []
            if self.hook_history.exists():
                history = json.loads(self.hook_history.read_text())

            history.append({
                "timestamp": str(__import__("datetime").datetime.now()),
                "recommended_patterns": result["recommended_patterns"],
                "avoid_patterns": result["avoid_patterns"],
                "context": result["context"],
                "style_bias": result["style_bias"],
            })

            # Keep last 100 entries
            history = history[-100:]

            self.hook_history.write_text(json.dumps(history, indent=2))
        except Exception as exc:
            logger.warning(f"Failed to save patterns: {exc}")

    def _default_recommendations(self) -> dict[str, Any]:
        """Return default recommendations when no history is available."""
        return {
            "recommended_patterns": [
                "but here is the problem",
                "and this is where it gets interesting",
                "no one is talking about this",
                "if you are building anything with ai",
                "this might affect you more than you think",
            ],
            "avoid_patterns": [
                "in this video we will",
                "today we discuss",
                "as you can see",
            ],
            "context": {},
            "exploration_enabled": True,
            "style_bias": {
                "controversy": 0.6,
                "curiosity": 0.7,
                "emotion": 0.5,
                "evidence": 0.5,
            },
        }


def get_optimized_hooks(
    editorial_context: dict[str, Any],
    feedback_path: str | None = None,
) -> dict[str, Any]:
    """Convenience function to get optimized hooks."""
    optimizer = HookOptimizer(feedback_path=feedback_path)
    return optimizer.run(editorial_context)
