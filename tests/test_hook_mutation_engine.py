"""Tests for HookMutationEngine."""
from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import pytest

from src.hook_mutation_engine import HookMutationEngine, mutate_hooks

# ── __init__ ──────────────────────────────────────────────────────────────────

class TestInit:
    def test_no_llm_by_default(self):
        engine = HookMutationEngine()
        assert engine.llm is None

    def test_stores_provided_llm(self):
        fake_llm = MagicMock()
        engine = HookMutationEngine(llm=fake_llm)
        assert engine.llm is fake_llm

    def test_max_mutations_defaults_to_8(self):
        engine = HookMutationEngine()
        # settings is a frozen dataclass without .get(), so the else branch fires
        assert engine.max_mutations == 8


# ── run() ─────────────────────────────────────────────────────────────────────

class TestRun:
    def test_returns_dict_with_mutated_hooks_key(self):
        engine = HookMutationEngine()
        result = engine.run(["Here's why this matters"], {})
        assert "mutated_hooks" in result

    def test_empty_top_hooks_returns_empty_list(self):
        engine = HookMutationEngine()
        result = engine.run([], {})
        assert result["mutated_hooks"] == []

    def test_single_hook_produces_variants(self):
        engine = HookMutationEngine()
        result = engine.run(["But here is the problem no one talks about"], {"angle": "technical_breakthrough"})
        # rule mutations should fire for "but" and "problem" keywords
        assert isinstance(result["mutated_hooks"], list)

    def test_max_mutations_caps_output(self):
        engine = HookMutationEngine()
        # Provide many hooks so output is capped at max_mutations
        hooks = [
            "But the problem is hidden here",
            "No one talks about this problem",
            "Here's why no one admits this",
            "But this is what they're hiding",
            "The problem no one wants to hear",
        ]
        result = engine.run(hooks, {})
        assert len(result["mutated_hooks"]) <= engine.max_mutations

    def test_run_with_context_dict(self):
        engine = HookMutationEngine()
        context = {"angle": "industry_impact", "persona": "expert", "format": "deep_dive"}
        result = engine.run(["Here's what no one is saying about this"], context)
        assert "mutated_hooks" in result

    def test_run_with_llm_calls_llm(self):
        fake_llm = MagicMock(return_value="Variant one\nVariant two\nVariant three")
        engine = HookMutationEngine(llm=fake_llm)
        engine.run(["Here's why it matters"], {"angle": "hot_take"})
        fake_llm.assert_called()


# ── _select_base_hooks() ──────────────────────────────────────────────────────

class TestSelectBaseHooks:
    def test_empty_returns_empty(self):
        engine = HookMutationEngine()
        assert engine._select_base_hooks([]) == []

    def test_fewer_than_5_returns_all(self):
        engine = HookMutationEngine()
        hooks = ["a", "b", "c"]
        assert engine._select_base_hooks(hooks) == hooks

    def test_more_than_5_returns_first_5(self):
        engine = HookMutationEngine()
        hooks = [f"hook_{i}" for i in range(10)]
        result = engine._select_base_hooks(hooks)
        assert result == hooks[:5]
        assert len(result) == 5

    def test_exactly_5_returns_all(self):
        engine = HookMutationEngine()
        hooks = [f"hook_{i}" for i in range(5)]
        assert engine._select_base_hooks(hooks) == hooks


# ── _rule_mutations() ─────────────────────────────────────────────────────────

class TestRuleMutations:
    def test_problem_keyword_triggers_variants(self):
        engine = HookMutationEngine()
        result = engine._rule_mutations("Here is the core problem with AI")
        assert len(result) > 0
        assert any("problem" in r.lower() or "break" in r.lower() or "no one" in r.lower() for r in result)

    def test_no_one_keyword_triggers_variants(self):
        engine = HookMutationEngine()
        result = engine._rule_mutations("This is what no one is admitting")
        assert any("missing" in r.lower() or "ignored" in r.lower() or "admit" in r.lower() for r in result)

    def test_heres_why_keyword_triggers_variants(self):
        engine = HookMutationEngine()
        result = engine._rule_mutations("Here's why this is important")
        assert len(result) > 0

    def test_heres_what_keyword_triggers_variants(self):
        engine = HookMutationEngine()
        result = engine._rule_mutations("Here's what you need to know")
        assert len(result) > 0

    def test_but_without_problem_triggers_variant(self):
        engine = HookMutationEngine()
        result = engine._rule_mutations("But this changes everything about AI")
        assert any("nobody" in r.lower() or "notices" in r.lower() or "thing" in r.lower() for r in result)

    def test_hook_with_no_keywords_returns_empty(self):
        engine = HookMutationEngine()
        result = engine._rule_mutations("AI is transforming industries worldwide today")
        assert result == []

    def test_returns_normalized_list(self):
        engine = HookMutationEngine()
        result = engine._rule_mutations("The problem is bigger than expected")
        # All results should be at least 10 chars
        assert all(len(r) >= 10 for r in result)


# ── _llm_mutations() ─────────────────────────────────────────────────────────

class TestLlmMutations:
    def test_returns_empty_when_no_llm(self):
        engine = HookMutationEngine()
        result = engine._llm_mutations("Some hook text here", {})
        assert result == []

    def test_calls_llm_with_prompt(self):
        fake_llm = MagicMock(return_value="Variant A\nVariant B\nVariant C")
        engine = HookMutationEngine(llm=fake_llm)
        result = engine._llm_mutations("Here's the hook", {"angle": "technical"})
        fake_llm.assert_called_once()
        assert isinstance(result, list)

    def test_parses_string_response(self):
        fake_llm = MagicMock(return_value="This is variant one\nThis is variant two\nThis is variant three")
        engine = HookMutationEngine(llm=fake_llm)
        result = engine._llm_mutations("Sample hook text", {})
        assert len(result) <= 3
        assert all(isinstance(r, str) for r in result)

    def test_parses_list_response(self):
        fake_llm = MagicMock(return_value=["First variant hook here", "Second variant hook text", "Third variant hook now"])
        engine = HookMutationEngine(llm=fake_llm)
        result = engine._llm_mutations("Sample hook", {})
        assert isinstance(result, list)
        assert len(result) <= 3

    def test_handles_llm_exception_gracefully(self):
        fake_llm = MagicMock(side_effect=RuntimeError("LLM API error"))
        engine = HookMutationEngine(llm=fake_llm)
        result = engine._llm_mutations("Sample hook text", {})
        assert result == []

    def test_unknown_response_type_returns_empty(self):
        fake_llm = MagicMock(return_value=42)  # int — unexpected type
        engine = HookMutationEngine(llm=fake_llm)
        result = engine._llm_mutations("Sample hook text", {})
        assert result == []


# ── _normalize_hooks() ────────────────────────────────────────────────────────

class TestNormalizeHooks:
    def test_removes_short_hooks(self):
        engine = HookMutationEngine()
        result = engine._normalize_hooks(["hi", "yo", "This is a long enough hook text"])
        assert "hi" not in result
        assert "yo" not in result

    def test_strips_trailing_period(self):
        engine = HookMutationEngine()
        result = engine._normalize_hooks(["Here is a hook with period."])
        assert result[0] == "Here is a hook with period"

    def test_strips_whitespace(self):
        engine = HookMutationEngine()
        result = engine._normalize_hooks(["  Padded hook text here  "])
        assert result[0] == "Padded hook text here"

    def test_keeps_valid_hooks(self):
        engine = HookMutationEngine()
        hooks = ["This is a valid hook", "Another valid hook here"]
        result = engine._normalize_hooks(hooks)
        assert len(result) == 2

    def test_empty_input_returns_empty(self):
        engine = HookMutationEngine()
        assert engine._normalize_hooks([]) == []


# ── _deduplicate() ────────────────────────────────────────────────────────────

class TestDeduplicate:
    def test_removes_exact_duplicates(self):
        engine = HookMutationEngine()
        hooks = ["Same hook text", "Same hook text", "Different hook text"]
        result = engine._deduplicate(hooks)
        assert len(result) == 2

    def test_removes_case_insensitive_duplicates(self):
        engine = HookMutationEngine()
        hooks = ["Same Hook Text", "same hook text"]
        result = engine._deduplicate(hooks)
        assert len(result) == 1

    def test_preserves_order(self):
        engine = HookMutationEngine()
        hooks = ["first hook", "second hook", "third hook"]
        result = engine._deduplicate(hooks)
        assert result == hooks

    def test_empty_input_returns_empty(self):
        engine = HookMutationEngine()
        assert engine._deduplicate([]) == []


# ── _diversity_filter() ───────────────────────────────────────────────────────

class TestDiversityFilter:
    def test_returns_all_when_5_or_fewer(self):
        engine = HookMutationEngine()
        hooks = ["hook one", "hook two", "hook three"]
        result = engine._diversity_filter(hooks)
        assert len(result) == 3

    def test_caps_at_5_when_more_than_5(self):
        engine = HookMutationEngine()
        hooks = [f"hook number {i} text here" for i in range(10)]
        result = engine._diversity_filter(hooks)
        assert len(result) == 5

    def test_exactly_5_returns_all(self):
        engine = HookMutationEngine()
        hooks = [f"hook {i} here now" for i in range(5)]
        result = engine._diversity_filter(hooks)
        assert len(result) == 5


# ── _score_mutations() ────────────────────────────────────────────────────────

class TestScoreMutations:
    def test_returns_list_of_dicts(self):
        engine = HookMutationEngine()
        hooks = ["no one knows this", "but this is important now"]
        result = engine._score_mutations(hooks, {})
        assert all("hook" in item and "score" in item for item in result)

    def test_sorted_by_score_descending(self):
        engine = HookMutationEngine()
        hooks = ["no one admits this hidden secret", "just a plain statement"]
        result = engine._score_mutations(hooks, {})
        assert result[0]["score"] >= result[-1]["score"]

    def test_high_value_keywords_boost_score(self):
        engine = HookMutationEngine()
        high = "no one knows this hidden secret"
        low = "just a plain statement about things"
        result = engine._score_mutations([high, low], {})
        high_item = next(r for r in result if r["hook"] == high)
        low_item = next(r for r in result if r["hook"] == low)
        assert high_item["score"] > low_item["score"]

    def test_short_hook_gets_length_bonus(self):
        engine = HookMutationEngine()
        short_hook = "no one knows why"  # 4 words
        long_hook = "no one knows why this is happening over here in this scenario right"  # >10 words
        result = engine._score_mutations([short_hook, long_hook], {})
        short_item = next(r for r in result if r["hook"] == short_hook)
        long_item = next(r for r in result if r["hook"] == long_hook)
        assert short_item["score"] > long_item["score"]

    def test_context_angle_match_adds_bonus(self):
        engine = HookMutationEngine()
        hook = "technical_breakthrough is happening now"
        result = engine._score_mutations([hook], {"angle": "technical_breakthrough"})
        assert result[0]["score"] > 0

    def test_empty_input_returns_empty(self):
        engine = HookMutationEngine()
        assert engine._score_mutations([], {}) == []


# ── mutate_hooks() (module-level function) ────────────────────────────────────

class TestMutateHooksFunction:
    def test_delegates_to_engine(self):
        result = mutate_hooks(["Here's why no one admits this"], {"angle": "industry_impact"})
        assert "mutated_hooks" in result

    def test_empty_hooks_returns_empty_list(self):
        result = mutate_hooks([], {})
        assert result["mutated_hooks"] == []

    def test_accepts_llm_argument(self):
        fake_llm = MagicMock(return_value="Hook variant one text\nHook variant two text")
        result = mutate_hooks(["Here's what nobody is saying"], {}, llm=fake_llm)
        assert "mutated_hooks" in result
        fake_llm.assert_called()
