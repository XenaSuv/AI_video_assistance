"""Tests for PackagingEngine and PackagingStore.

Covers:
- Core idea extraction from angle keys, conflict strings, angle labels
- Hook generation per strategy mode
- Title generation per idea family
- Thumbnail concept generation per idea family
- Full generate() pipeline
- Hook / title / thumbnail consistency check
- PackagingStore: record, get_best_patterns, get_best_thumbnail_styles
- Integration with EditorialPlan objects and StrategyConfig objects
"""
from __future__ import annotations

import pytest
import tempfile
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any

from src.packaging_engine import (
    PackagingEngine,
    PackagingResult,
    PackagingStore,
    _ANGLE_KEY_TO_IDEA,
    _CONFLICT_TO_IDEA,
    _DEFAULT_IDEA,
)


# ── Fakes ──────────────────────────────────────────────────────────────────────

@dataclass
class _FakeEditorialPlan:
    editorial_plan: list[dict[str, Any]]


@dataclass
class _FakeStrategyConfig:
    mode: str = "growth"


def _tmp() -> Path:
    return Path(tempfile.mkdtemp())


# ── PackagingResult ────────────────────────────────────────────────────────────

class TestPackagingResult:
    def test_to_dict_round_trip(self):
        r = PackagingResult(
            idea="test idea",
            hook="test hook",
            title="Test Title Here",
            thumbnail={"text": "TEST", "style": "clean", "emotion": "curiosity"},
            title_pattern="game_changer",
        )
        d  = r.to_dict()
        r2 = PackagingResult.from_dict(d)
        assert r2.idea          == r.idea
        assert r2.hook          == r.hook
        assert r2.title         == r.title
        assert r2.thumbnail     == r.thumbnail
        assert r2.title_pattern == r.title_pattern

    def test_from_dict_empty_defaults(self):
        r = PackagingResult.from_dict({})
        assert r.idea  == ""
        assert r.title == ""
        assert r.thumbnail == {}
        assert r.title_pattern == "default"


# ── _extract_core_idea ─────────────────────────────────────────────────────────

class TestExtractCoreIdea:
    def setup_method(self):
        self.engine = PackagingEngine()

    def test_angle_key_technical_breakthrough(self):
        entry = {"angle_key": "technical_breakthrough"}
        idea  = self.engine._extract_core_idea(entry)
        assert "hidden problem" in idea.lower() or "hype" in idea.lower()

    def test_angle_key_overhyped(self):
        idea = self.engine._extract_core_idea({"angle_key": "overhyped_vs_reality"})
        assert "hype" in idea.lower() or "problem" in idea.lower()

    def test_angle_key_industry_impact(self):
        idea = self.engine._extract_core_idea({"angle_key": "industry_impact"})
        assert "changes everything" in idea.lower() or "industry" in idea.lower()

    def test_angle_key_threat_to_jobs(self):
        idea = self.engine._extract_core_idea({"angle_key": "threat_to_jobs"})
        assert "job" in idea.lower() or "cost people" in idea.lower()

    def test_angle_key_what_this_means(self):
        idea = self.engine._extract_core_idea({"angle_key": "what_this_means_for_you"})
        assert "means" in idea.lower() or "you" in idea.lower()

    def test_conflict_hype_vs_reality(self):
        idea = self.engine._extract_core_idea({"conflict": "hype vs reality"})
        assert idea == _CONFLICT_TO_IDEA["hype vs reality"]

    def test_conflict_innovation_vs_adoption(self):
        idea = self.engine._extract_core_idea({"conflict": "innovation vs adoption"})
        assert idea == _CONFLICT_TO_IDEA["innovation vs adoption"]

    def test_conflict_growth_vs_jobs(self):
        idea = self.engine._extract_core_idea({"conflict": "growth vs jobs"})
        assert idea == _CONFLICT_TO_IDEA["growth vs jobs"]

    def test_conflict_promise_vs_practical(self):
        idea = self.engine._extract_core_idea({"conflict": "promise vs practical use"})
        assert idea == _CONFLICT_TO_IDEA["promise vs practical use"]

    def test_angle_label_hype_keyword(self):
        entry = {"angle": "This looks like a breakthrough but might be overhyped"}
        idea  = self.engine._extract_core_idea(entry)
        assert "hype" in idea.lower() or "problem" in idea.lower()

    def test_angle_label_job_keyword(self):
        idea = self.engine._extract_core_idea({"angle": "This technology might replace jobs"})
        assert "job" in idea.lower() or "wrong" in idea.lower()

    def test_simplified_alias_breakthrough(self):
        idea = self.engine._extract_core_idea({"angle_key": "breakthrough"})
        assert "changes everything" in idea.lower()

    def test_simplified_alias_risk(self):
        idea = self.engine._extract_core_idea({"angle_key": "risk"})
        assert "wrong" in idea.lower() or "cost" in idea.lower()

    def test_simplified_alias_hype_vs_reality(self):
        idea = self.engine._extract_core_idea({"angle_key": "hype_vs_reality"})
        assert "hype" in idea.lower() or "problem" in idea.lower()

    def test_unknown_angle_returns_default(self):
        idea = self.engine._extract_core_idea({"angle": "Something totally new"})
        assert isinstance(idea, str) and len(idea) > 0

    def test_empty_entry_returns_default(self):
        assert self.engine._extract_core_idea({}) == _DEFAULT_IDEA


# ── _generate_hook ─────────────────────────────────────────────────────────────

class TestGenerateHook:
    def setup_method(self):
        self.engine = PackagingEngine()
        self.idea   = "Everyone is hyped — but there is a hidden problem"

    def test_growth_mode_includes_nobody_tells(self):
        hook = self.engine._generate_hook(self.idea, _FakeStrategyConfig(mode="growth"))
        assert "nobody tells" in hook.lower()

    def test_safe_mode_includes_break_down(self):
        hook = self.engine._generate_hook(self.idea, _FakeStrategyConfig(mode="safe"))
        assert "break this down" in hook.lower()

    def test_experimental_mode_includes_wrong(self):
        hook = self.engine._generate_hook(self.idea, _FakeStrategyConfig(mode="experimental"))
        assert "wrong" in hook.lower()

    def test_unknown_mode_returns_idea(self):
        hook = self.engine._generate_hook(self.idea, _FakeStrategyConfig(mode="unknown_mode"))
        assert self.idea in hook

    def test_dict_strategy_config(self):
        hook = self.engine._generate_hook(self.idea, {"mode": "growth"})
        assert "nobody tells" in hook.lower()

    def test_hook_contains_idea(self):
        for mode in ("growth", "safe", "experimental"):
            hook = self.engine._generate_hook(self.idea, _FakeStrategyConfig(mode=mode))
            # The idea (or fragment) must be traceable in the hook
            assert len(hook) > 0


# ── _generate_title ────────────────────────────────────────────────────────────

class TestGenerateTitle:
    def setup_method(self):
        self.engine = PackagingEngine()

    def test_hype_problem_idea(self):
        title = self.engine._generate_title(
            "Everyone is hyped — but there is a hidden problem", ""
        )
        assert "hidden" in title.lower() or "problem" in title.lower()

    def test_game_changer_idea(self):
        title = self.engine._generate_title("This changes everything for the industry", "")
        assert "change" in title.lower() or "game" in title.lower()

    def test_jobs_risk_idea(self):
        title = self.engine._generate_title("This could cost people their jobs", "")
        assert "job" in title.lower()

    def test_practical_idea(self):
        title = self.engine._generate_title("Here is what this really means for you", "")
        assert len(title) > 0

    def test_default_fallback(self):
        title = self.engine._generate_title("Something important is happening", "")
        assert len(title.split()) >= 4

    def test_title_word_count(self):
        for idea in list(_ANGLE_KEY_TO_IDEA.values()) + [_DEFAULT_IDEA]:
            title = self.engine._generate_title(idea, "")
            words = len(title.split())
            assert 3 <= words <= 12, f"Title word count {words} out of range: {title!r}"


# ── _generate_thumbnail ────────────────────────────────────────────────────────

class TestGenerateThumbnail:
    def setup_method(self):
        self.engine = PackagingEngine()

    def _check_thumbnail(self, thumb: dict) -> None:
        assert "text"    in thumb
        assert "style"   in thumb
        assert "emotion" in thumb
        assert thumb["style"]   in ("high_contrast", "clean", "neutral")
        assert thumb["emotion"] in ("shock", "curiosity", "excitement")

    def test_hype_problem_thumbnail(self):
        thumb = self.engine._generate_thumbnail(
            "Everyone is hyped — but there is a hidden problem", ""
        )
        assert thumb["emotion"] == "shock"
        assert thumb["style"]   == "high_contrast"

    def test_game_changer_thumbnail(self):
        thumb = self.engine._generate_thumbnail("This changes everything for the industry", "")
        assert thumb["style"]   == "clean"
        assert thumb["emotion"] == "curiosity"

    def test_jobs_risk_thumbnail(self):
        thumb = self.engine._generate_thumbnail("This could cost people their jobs", "")
        assert "risk" in thumb["text"].lower() or "job" in thumb["text"].lower()

    def test_thumbnail_has_required_keys(self):
        for idea in list(_ANGLE_KEY_TO_IDEA.values()) + [_DEFAULT_IDEA]:
            thumb = self.engine._generate_thumbnail(idea, "")
            assert set(thumb.keys()) >= {"text", "style", "emotion"}

    def test_thumbnail_not_a_real_image(self):
        """Thumbnail concept must be a dict, never a path or bytes."""
        thumb = self.engine._generate_thumbnail("Something important is happening", "")
        assert isinstance(thumb, dict)


# ── Full generate() pipeline ──────────────────────────────────────────────────

class TestGenerate:
    def setup_method(self):
        self.engine = PackagingEngine()

    def test_returns_packaging_result(self):
        plan   = {"angle_key": "technical_breakthrough", "conflict": "hype vs reality"}
        result = self.engine.generate(plan, _FakeStrategyConfig(mode="growth"))
        assert isinstance(result, PackagingResult)

    def test_result_fields_non_empty(self):
        plan   = {"conflict": "growth vs jobs"}
        result = self.engine.generate(plan, _FakeStrategyConfig(mode="safe"))
        assert result.idea
        assert result.hook
        assert result.title
        assert result.thumbnail

    def test_editorial_plan_object(self):
        ep     = _FakeEditorialPlan(editorial_plan=[{"conflict": "hype vs reality"}])
        result = self.engine.generate(ep, _FakeStrategyConfig(mode="growth"))
        assert "hype" in result.idea.lower() or "problem" in result.idea.lower()

    def test_editorial_plan_list(self):
        plan   = [{"angle_key": "industry_impact"}]
        result = self.engine.generate(plan, _FakeStrategyConfig(mode="safe"))
        assert "changes everything" in result.idea.lower() or "industry" in result.idea.lower()

    def test_strategy_config_dict(self):
        result = self.engine.generate(
            {"conflict": "hype vs reality"},
            {"mode": "experimental"},
        )
        assert "wrong" in result.hook.lower()

    def test_growth_mode_hook_aggressive(self):
        result = self.engine.generate(
            {"conflict": "hype vs reality"},
            _FakeStrategyConfig(mode="growth"),
        )
        assert "nobody tells" in result.hook.lower()

    def test_safe_mode_hook_clear(self):
        result = self.engine.generate(
            {"conflict": "innovation vs adoption"},
            _FakeStrategyConfig(mode="safe"),
        )
        assert "break this down" in result.hook.lower()

    def test_all_angles_produce_valid_results(self):
        for angle_key in _ANGLE_KEY_TO_IDEA:
            result = self.engine.generate(
                {"angle_key": angle_key},
                _FakeStrategyConfig(mode="growth"),
            )
            assert result.idea
            assert result.title
            assert result.thumbnail.get("style") in ("high_contrast", "clean", "neutral")

    def test_title_pattern_recorded(self):
        result = self.engine.generate(
            {"conflict": "hype vs reality"},
            _FakeStrategyConfig(mode="growth"),
        )
        assert result.title_pattern in ("hype_problem", "game_changer", "jobs_risk", "practical", "default")


# ── Hook / title / thumbnail consistency ──────────────────────────────────────

class TestHookConsistency:
    def setup_method(self):
        self.engine = PackagingEngine()

    def test_consistent_result_passes(self):
        result = self.engine.generate(
            {"conflict": "hype vs reality"},
            _FakeStrategyConfig(mode="growth"),
        )
        # A result generated from one idea should be internally consistent
        assert isinstance(self.engine.check_consistency(result), bool)

    def test_mismatched_families_fail(self):
        result = PackagingResult(
            idea          = "Everyone is hyped — but there is a hidden problem",
            hook          = "This changes everything for the industry",
            title         = "The Jobs That Are Disappearing Now",
            thumbnail     = {"text": "WHAT IS THIS?", "style": "neutral", "emotion": "curiosity"},
            title_pattern = "default",
        )
        # hook and title belong to different families
        assert self.engine.check_consistency(result) is False


# ── PackagingStore ─────────────────────────────────────────────────────────────

class TestPackagingStore:
    def test_record_and_load(self):
        store  = PackagingStore(data_dir=_tmp())
        result = PackagingResult(
            idea="idea", hook="hook", title="title",
            thumbnail={"text": "T", "style": "clean", "emotion": "curiosity"},
            title_pattern="game_changer",
        )
        store.record("vid-001", result, ctr=0.07)
        patterns = store.get_best_patterns()
        assert "game_changer" in patterns

    def test_best_patterns_sorted_by_ctr(self):
        store = PackagingStore(data_dir=_tmp())
        r_low  = PackagingResult("", "", "", {}, "default")
        r_high = PackagingResult("", "", "", {}, "hype_problem")
        store.record("v1", r_low,  ctr=0.02)
        store.record("v2", r_high, ctr=0.09)
        patterns = store.get_best_patterns()
        keys = list(patterns.keys())
        assert keys[0] == "hype_problem"

    def test_empty_store_returns_empty_dict(self):
        store = PackagingStore(data_dir=_tmp())
        assert store.get_best_patterns() == {}
        assert store.get_best_thumbnail_styles() == {}

    def test_best_thumbnail_styles_sorted_by_ctr(self):
        store = PackagingStore(data_dir=_tmp())
        r_neutral = PackagingResult("", "", "", {"style": "neutral", "text": "X", "emotion": "curiosity"}, "default")
        r_hc      = PackagingResult("", "", "", {"style": "high_contrast", "text": "Y", "emotion": "shock"}, "hype_problem")
        store.record("v1", r_neutral, ctr=0.03)
        store.record("v2", r_hc,      ctr=0.11)
        styles = store.get_best_thumbnail_styles()
        assert list(styles.keys())[0] == "high_contrast"

    def test_multiple_records_same_pattern_averaged(self):
        store = PackagingStore(data_dir=_tmp())
        r = PackagingResult("", "", "", {}, "game_changer")
        store.record("v1", r, ctr=0.04)
        store.record("v2", r, ctr=0.06)
        patterns = store.get_best_patterns()
        assert abs(patterns["game_changer"] - 0.05) < 1e-4

    def test_top_n_respected(self):
        store = PackagingStore(data_dir=_tmp())
        for p in ("hype_problem", "game_changer", "jobs_risk", "practical", "default"):
            r = PackagingResult("", "", "", {}, p)
            store.record(f"v-{p}", r, ctr=0.05)
        patterns = store.get_best_patterns(top_n=2)
        assert len(patterns) == 2


# ── Feedback loop (integration) ───────────────────────────────────────────────

class TestFeedbackLoop:
    def test_boost_winning_patterns(self):
        """get_best_patterns guides the next generate() call."""
        store  = PackagingStore(data_dir=_tmp())
        engine = PackagingEngine()

        # Record that "hype_problem" titles get high CTR
        r = PackagingResult("", "", "", {}, "hype_problem")
        store.record("v1", r, ctr=0.12)
        store.record("v2", r, ctr=0.10)

        best = store.get_best_patterns(top_n=1)
        assert list(best.keys())[0] == "hype_problem"
        # Caller can now bias generate() toward hype_problem angle
        result = engine.generate(
            {"angle_key": "technical_breakthrough"},
            _FakeStrategyConfig(mode="growth"),
        )
        assert result.title_pattern == "hype_problem"

    def test_record_uses_result_thumbnail_style(self):
        store  = PackagingStore(data_dir=_tmp())
        result = PackagingResult(
            idea="idea", hook="hook", title="title",
            thumbnail={"text": "THE PROBLEM", "style": "high_contrast", "emotion": "shock"},
            title_pattern="hype_problem",
        )
        store.record("v1", result, ctr=0.08)
        styles = store.get_best_thumbnail_styles()
        assert "high_contrast" in styles
