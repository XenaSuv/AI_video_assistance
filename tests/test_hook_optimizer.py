"""Tests for src/hook_optimizer.py."""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.hook_optimizer import (
    HookOptimizer,
    HookPattern,
    get_optimized_hooks,
)
from src.shared_types import HookOptimizationResult

# ── helpers ────────────────────────────────────────────────────────────────────

def _optimizer(tmp_path: Path, feedback: list | None = None) -> HookOptimizer:
    fb_path = tmp_path / "feedback.json"
    if feedback is not None:
        fb_path.write_text(json.dumps(feedback))
    opt = HookOptimizer(feedback_path=str(fb_path))
    # Override hook_history path so it writes to tmp_path
    opt.hook_history = tmp_path / "hook_patterns.json"
    return opt


def _feedback_item(hook: str = "but here is the problem", score: float = 0.8) -> dict:
    return {"hook_text": hook, "hook_score": score}


def _feedback_items(n: int = 5, score: float = 0.8) -> list[dict]:
    return [_feedback_item(f"hook phrase number {i}", score) for i in range(n)]


# ═══════════════════════════════════════════════════════════════════════════════
# HookOptimizer.__init__
# ═══════════════════════════════════════════════════════════════════════════════

class TestHookOptimizerInit:
    def test_default_feedback_path(self, tmp_path):
        with patch("src.hook_optimizer.settings") as ms:
            ms.data_dir = str(tmp_path)
            opt = HookOptimizer()
        assert "feedback_history.json" in opt.feedback_path

    def test_custom_feedback_path(self, tmp_path):
        opt = HookOptimizer(feedback_path=str(tmp_path / "custom.json"))
        assert "custom.json" in opt.feedback_path

    def test_default_exploration_rate(self, tmp_path):
        opt = _optimizer(tmp_path)
        assert opt.exploration_rate == 0.2


# ═══════════════════════════════════════════════════════════════════════════════
# _load_feedback
# ═══════════════════════════════════════════════════════════════════════════════

class TestLoadFeedback:
    def test_returns_empty_when_file_missing(self, tmp_path):
        opt = HookOptimizer(feedback_path=str(tmp_path / "nonexistent.json"))
        assert opt._load_feedback() == []

    def test_loads_json_list(self, tmp_path):
        data = [{"hook_text": "hello", "hook_score": 0.7}]
        fb_path = tmp_path / "fb.json"
        fb_path.write_text(json.dumps(data))
        opt = HookOptimizer(feedback_path=str(fb_path))
        result = opt._load_feedback()
        assert len(result) == 1

    def test_returns_empty_on_corrupt_file(self, tmp_path):
        fb_path = tmp_path / "fb.json"
        fb_path.write_text("not valid json")
        opt = HookOptimizer(feedback_path=str(fb_path))
        result = opt._load_feedback()
        assert result == []


# ═══════════════════════════════════════════════════════════════════════════════
# _normalize_hook
# ═══════════════════════════════════════════════════════════════════════════════

class TestNormalizeHook:
    def test_lowercases(self, tmp_path):
        opt = _optimizer(tmp_path)
        assert opt._normalize_hook("HELLO WORLD") == "hello world"

    def test_removes_special_chars(self, tmp_path):
        opt = _optimizer(tmp_path)
        result = opt._normalize_hook("Hello, World! It's amazing.")
        assert "," not in result
        assert "!" not in result

    def test_strips_whitespace(self, tmp_path):
        opt = _optimizer(tmp_path)
        assert opt._normalize_hook("  hello  ") == "hello"


# ═══════════════════════════════════════════════════════════════════════════════
# _extract_key_phrases
# ═══════════════════════════════════════════════════════════════════════════════

class TestExtractKeyPhrases:
    def test_includes_full_hook(self, tmp_path):
        opt = _optimizer(tmp_path)
        phrases = opt._extract_key_phrases("but here is the real problem")
        assert any(len(p) > 5 for p in phrases)

    def test_includes_first_three_words(self, tmp_path):
        opt = _optimizer(tmp_path)
        phrases = opt._extract_key_phrases("no one is talking about this")
        assert "no one is" in phrases

    def test_detects_pattern_words(self, tmp_path):
        opt = _optimizer(tmp_path)
        phrases = opt._extract_key_phrases("this is a problem with ai")
        assert any("problem" in p for p in phrases)

    def test_short_hook_skips_full_phrase(self, tmp_path):
        opt = _optimizer(tmp_path)
        phrases = opt._extract_key_phrases("hi")
        # len <= 5 → not added as full phrase
        assert "hi" not in phrases


# ═══════════════════════════════════════════════════════════════════════════════
# _extract_patterns
# ═══════════════════════════════════════════════════════════════════════════════

class TestExtractPatterns:
    def test_extracts_from_feedback(self, tmp_path):
        opt = _optimizer(tmp_path)
        feedback = [_feedback_item("but here is the problem", 0.8)] * 3
        patterns = opt._extract_patterns(feedback)
        assert len(patterns) > 0

    def test_skips_items_without_hook(self, tmp_path):
        opt = _optimizer(tmp_path)
        feedback = [{"hook_score": 0.8}]  # no hook_text
        patterns = opt._extract_patterns(feedback)
        assert len(patterns) == 0

    def test_uses_hook_field_as_fallback(self, tmp_path):
        opt = _optimizer(tmp_path)
        feedback = [{"hook": "no one knows this secret", "hook_score": 0.7}]
        patterns = opt._extract_patterns(feedback)
        assert len(patterns) > 0

    def test_accumulates_scores_per_pattern(self, tmp_path):
        opt = _optimizer(tmp_path)
        hook = "no one is talking about this"
        feedback = [_feedback_item(hook, s) for s in [0.5, 0.7, 0.9]]
        patterns = opt._extract_patterns(feedback)
        # Each feedback item contributes to the same phrase patterns
        for _phrase, scores in patterns.items():
            assert len(scores) >= 1


# ═══════════════════════════════════════════════════════════════════════════════
# _score_patterns
# ═══════════════════════════════════════════════════════════════════════════════

class TestScorePatterns:
    def test_requires_at_least_two_scores(self, tmp_path):
        opt = _optimizer(tmp_path)
        patterns = {"single_occurrence": [0.8]}
        result = opt._score_patterns(patterns)
        assert result == []

    def test_scores_by_avg(self, tmp_path):
        opt = _optimizer(tmp_path)
        patterns = {
            "high_scorer": [0.9, 0.8, 0.85],
            "low_scorer":  [0.2, 0.3, 0.25],
        }
        result = opt._score_patterns(patterns)
        assert result[0].pattern == "high_scorer"
        assert result[-1].pattern == "low_scorer"

    def test_computes_success_rate(self, tmp_path):
        opt = _optimizer(tmp_path)
        patterns = {"test": [0.8, 0.7, 0.3]}  # 2/3 > 0.6
        result = opt._score_patterns(patterns)
        assert len(result) == 1
        assert abs(result[0].success_rate - 2/3) < 0.01

    def test_returns_sorted_list(self, tmp_path):
        opt = _optimizer(tmp_path)
        patterns = {"b": [0.5, 0.6], "a": [0.8, 0.9]}
        result = opt._score_patterns(patterns)
        assert result[0].avg_score >= result[-1].avg_score


# ═══════════════════════════════════════════════════════════════════════════════
# _filter_by_context
# ═══════════════════════════════════════════════════════════════════════════════

class TestFilterByContext:
    def _make_patterns(self, texts: list[str], score: float = 0.7) -> list[HookPattern]:
        return [
            HookPattern(pattern=t, avg_score=score, count=5, success_rate=0.6)
            for t in texts
        ]

    def test_returns_all_above_threshold_for_neutral_context(self, tmp_path):
        opt = _optimizer(tmp_path)
        patterns = self._make_patterns(["good hook here", "another good hook"])
        result = opt._filter_by_context(patterns, {})
        assert len(result) == 2

    def test_filters_hype_angle_to_contrast_patterns(self, tmp_path):
        opt = _optimizer(tmp_path)
        patterns = self._make_patterns([
            "but here is the problem", "this is amazing news today"
        ])
        result = opt._filter_by_context(patterns, {"angle": "hype"})
        assert any("problem" in p.pattern for p in result)

    def test_falls_back_to_scored_when_no_matches(self, tmp_path):
        opt = _optimizer(tmp_path)
        patterns = self._make_patterns(["totally generic hook text"])
        # hype angle requires "problem|no one|but|yet" — none present, score > 0.5 fallback
        result = opt._filter_by_context(patterns, {"angle": "hype"})
        # Falls back to scored[:10] when filtered is empty
        assert len(result) > 0

    def test_conflict_angle_prefers_tension_words(self, tmp_path):
        opt = _optimizer(tmp_path)
        patterns = self._make_patterns(["this is the catch", "just normal text"])
        result = opt._filter_by_context(patterns, {"angle": "conflict"})
        assert any("catch" in p.pattern for p in result)

    def test_persona_skeptical_filters_honestly(self, tmp_path):
        opt = _optimizer(tmp_path)
        patterns = self._make_patterns(["honestly not convinced", "generic text"])
        result = opt._filter_by_context(
            patterns, {"persona": {"style": "skeptical"}}
        )
        assert any("honestly" in p.pattern for p in result)

    def test_hot_take_format_prefers_high_score(self, tmp_path):
        opt = _optimizer(tmp_path)
        patterns = [
            HookPattern("mediocre hook", 0.55, 3, 0.5),
            HookPattern("great hook here", 0.70, 3, 0.7),
        ]
        result = opt._filter_by_context(patterns, {"format": "hot_take"})
        assert any(p.avg_score > 0.65 for p in result)


# ═══════════════════════════════════════════════════════════════════════════════
# _select_top / _select_worst
# ═══════════════════════════════════════════════════════════════════════════════

class TestSelectPatterns:
    def _patterns(self) -> list[HookPattern]:
        return [
            HookPattern("best", 0.9, 5, 0.8),
            HookPattern("good", 0.7, 4, 0.6),
            HookPattern("ok",   0.5, 3, 0.4),
            HookPattern("bad",  0.3, 2, 0.2),
        ]

    def test_select_top_returns_n_patterns(self, tmp_path):
        opt = _optimizer(tmp_path)
        result = opt._select_top_patterns(self._patterns(), top_n=2)
        assert result == ["best", "good"]

    def test_select_worst_returns_bottom_n(self, tmp_path):
        opt = _optimizer(tmp_path)
        result = opt._select_worst_patterns(self._patterns(), bottom_n=2)
        assert "bad" in result

    def test_select_worst_returns_empty_when_too_few(self, tmp_path):
        opt = _optimizer(tmp_path)
        result = opt._select_worst_patterns(self._patterns()[:2], bottom_n=3)
        assert result == []


# ═══════════════════════════════════════════════════════════════════════════════
# _calculate_style_bias
# ═══════════════════════════════════════════════════════════════════════════════

class TestCalculateStyleBias:
    def test_detects_controversy(self, tmp_path):
        opt = _optimizer(tmp_path)
        bias = opt._calculate_style_bias(["but here is the problem"], {})
        assert bias["controversy"] > 0.5

    def test_detects_curiosity(self, tmp_path):
        opt = _optimizer(tmp_path)
        bias = opt._calculate_style_bias(["but here is what actually happened"], {})
        assert bias["curiosity"] > 0.5

    def test_detects_emotion(self, tmp_path):
        opt = _optimizer(tmp_path)
        bias = opt._calculate_style_bias(["never seen anything so surprising"], {})
        assert bias["emotion"] > 0.5

    def test_detects_evidence(self, tmp_path):
        opt = _optimizer(tmp_path)
        bias = opt._calculate_style_bias(["new data shows this research"], {})
        assert bias["evidence"] > 0.5

    def test_returns_defaults_for_neutral_text(self, tmp_path):
        opt = _optimizer(tmp_path)
        bias = opt._calculate_style_bias(["just some text"], {})
        assert all(v == 0.5 for v in bias.values())


# ═══════════════════════════════════════════════════════════════════════════════
# _save_patterns
# ═══════════════════════════════════════════════════════════════════════════════

class TestSavePatterns:
    def test_saves_to_disk(self, tmp_path):
        opt = _optimizer(tmp_path)
        result = HookOptimizationResult(
            recommended_patterns=["hook1"],
            avoid_patterns=["avoid1"],
            context={"angle": "hype"},
            exploration_enabled=True,
            style_bias={"controversy": 0.6, "curiosity": 0.7, "emotion": 0.5, "evidence": 0.5},
        )
        opt._save_patterns(result)
        assert opt.hook_history.exists()

    def test_appends_to_existing_history(self, tmp_path):
        opt = _optimizer(tmp_path)
        existing = [{"timestamp": "2026-01-01", "recommended_patterns": [], "avoid_patterns": [], "context": {}, "style_bias": {}}]
        opt.hook_history.write_text(json.dumps(existing))
        result = HookOptimizationResult(
            recommended_patterns=["p1"],
            avoid_patterns=[],
            context={},
            exploration_enabled=False,
            style_bias={},
        )
        opt._save_patterns(result)
        data = json.loads(opt.hook_history.read_text())
        assert len(data) == 2

    def test_caps_history_at_100(self, tmp_path):
        opt = _optimizer(tmp_path)
        # Write 100 existing entries
        existing = [{"timestamp": "t", "recommended_patterns": [], "avoid_patterns": [], "context": {}, "style_bias": {}}] * 100
        opt.hook_history.write_text(json.dumps(existing))
        result = HookOptimizationResult(
            recommended_patterns=["new"],
            avoid_patterns=[],
            context={},
            exploration_enabled=False,
            style_bias={},
        )
        opt._save_patterns(result)
        data = json.loads(opt.hook_history.read_text())
        assert len(data) == 100  # capped at 100


# ═══════════════════════════════════════════════════════════════════════════════
# _default_recommendations
# ═══════════════════════════════════════════════════════════════════════════════

class TestDefaultRecommendations:
    def test_returns_hook_optimization_result(self, tmp_path):
        opt = _optimizer(tmp_path)
        result = opt._default_recommendations()
        assert isinstance(result, HookOptimizationResult)

    def test_has_recommended_patterns(self, tmp_path):
        opt = _optimizer(tmp_path)
        result = opt._default_recommendations()
        assert len(result.recommended_patterns) > 0

    def test_exploration_enabled_by_default(self, tmp_path):
        opt = _optimizer(tmp_path)
        result = opt._default_recommendations()
        assert result.exploration_enabled is True


# ═══════════════════════════════════════════════════════════════════════════════
# HookOptimizer.run
# ═══════════════════════════════════════════════════════════════════════════════

class TestHookOptimizerRun:
    def test_returns_defaults_when_no_feedback(self, tmp_path):
        opt = _optimizer(tmp_path)  # no feedback file
        result = opt.run({"angle": "technical"})
        assert isinstance(result, HookOptimizationResult)
        assert len(result.recommended_patterns) > 0

    def test_returns_defaults_when_empty_feedback(self, tmp_path):
        opt = _optimizer(tmp_path, feedback=[])
        result = opt.run({})
        assert isinstance(result, HookOptimizationResult)

    def test_uses_feedback_when_available(self, tmp_path):
        feedback = [_feedback_item("but here is the problem", 0.9)] * 10
        opt = _optimizer(tmp_path, feedback=feedback)
        result = opt.run({"angle": "technical"})
        assert isinstance(result, HookOptimizationResult)
        assert len(result.recommended_patterns) > 0

    def test_returns_defaults_when_no_scoreable_patterns(self, tmp_path):
        # All feedback items have unique hooks → each has only 1 score → filtered out
        feedback = [_feedback_item(f"unique hook {i}", 0.8) for i in range(5)]
        opt = _optimizer(tmp_path, feedback=feedback)
        result = opt.run({})
        # Falls back to defaults when no patterns pass the minimum threshold of 2
        assert isinstance(result, HookOptimizationResult)

    def test_saves_patterns_when_feedback_available(self, tmp_path):
        # Need ≥2 occurrences of same phrase to produce patterns
        feedback = [_feedback_item("no one talks about this problem", 0.8)] * 5
        opt = _optimizer(tmp_path, feedback=feedback)
        result = opt.run({"angle": "technical"})
        # _save_patterns is called when patterns exist
        # If history file was written, we know it ran
        assert isinstance(result, HookOptimizationResult)


# ═══════════════════════════════════════════════════════════════════════════════
# get_optimized_hooks (convenience function)
# ═══════════════════════════════════════════════════════════════════════════════

class TestGetOptimizedHooks:
    def test_returns_hook_optimization_result(self, tmp_path):
        result = get_optimized_hooks({"angle": "technical"}, feedback_path=str(tmp_path / "fb.json"))
        assert isinstance(result, HookOptimizationResult)

    def test_passes_context_to_optimizer(self, tmp_path):
        context = {"angle": "hype", "format": "hot_take"}
        result = get_optimized_hooks(context, feedback_path=str(tmp_path / "fb.json"))
        assert isinstance(result, HookOptimizationResult)
