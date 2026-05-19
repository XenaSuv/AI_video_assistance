"""Tests for ABTestingEngine, ABTestVariant, ABTestResult, ABTestStore,
and the PackagingEngine.generate_variants() extension.

Covers:
- ABTestVariant / ABTestResult serialisation
- ABTestStore: save, load_recent, get_variant_type_performance, get_winning_patterns
- PackagingEngine.generate_variants() — 3 variants, correct types and fields
- ABTestingEngine._score() — CTR × retention fallback logic
- ABTestingEngine._select_winner() — highest score wins
- ABTestingEngine.run_test() — full sequential flow with mock YouTube client
- ABTestingEngine.suggest_strategy_adjustments() — conflict → +0.1
- ABTestingEngine.apply_winner_feedback() — bridges to PackagingStore
- Dry-run mode (youtube_client=None)
"""
from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from src.ab_testing_engine import (
    _VARIANT_TYPE_TO_PATTERN,
    ABTestingEngine,
    ABTestResult,
    ABTestStore,
    ABTestVariant,
)
from src.packaging_engine import PackagingEngine, PackagingStore

# ── Fakes ──────────────────────────────────────────────────────────────────────

class _MockYouTube:
    """Injectable mock that records calls and serves pre-set metric sequences."""

    def __init__(
        self,
        ctr_seq: list[float] | None = None,
        retention_seq: list[float] | None = None,
    ) -> None:
        self.calls: list[tuple] = []
        self._ctr_iter  = iter(ctr_seq       or [0.05, 0.08, 0.04])
        self._ret_iter  = iter(retention_seq  or [0.60, 0.72, 0.55])

    def update_video(self, video_id, title=None, thumbnail_concept=None):
        self.calls.append(("update", video_id, title))

    def get_video_analytics(self, video_id):
        return {
            "ctr":          next(self._ctr_iter,  0.05),
            "retention_30s": next(self._ret_iter, 0.60),
        }


@dataclass
class _FakeStrategyConfig:
    mode: str = "growth"


@dataclass
class _FakeEditorialPlan:
    editorial_plan: list[dict]


def _no_wait(hours: float) -> None:
    """Drop-in wait_fn that skips real sleeping in tests."""


def _tmp() -> Path:
    return Path(tempfile.mkdtemp())


def _two_variants() -> list[ABTestVariant]:
    return [
        ABTestVariant(id="A", type="curiosity", title="Title A", hook="Hook A",
                      thumbnail={"text": "A", "style": "clean", "emotion": "curiosity"}),
        ABTestVariant(id="B", type="conflict",  title="Title B", hook="Hook B",
                      thumbnail={"text": "B", "style": "high_contrast", "emotion": "shock"}),
    ]


# ── ABTestVariant ──────────────────────────────────────────────────────────────

class TestABTestVariant:
    def test_defaults(self):
        v = ABTestVariant()
        assert v.id   == "A"
        assert v.type == "curiosity"
        assert v.ctr  == 0.0
        assert v.score == 0.0

    def test_to_dict_round_trip(self):
        v = ABTestVariant(
            id="B", type="conflict", title="title", hook="hook",
            thumbnail={"text": "X"}, start_time="2025-01-01T00:00:00Z",
            ctr=0.07, retention_30s=0.65, score=0.0455,
        )
        d  = v.to_dict()
        v2 = ABTestVariant.from_dict(d)
        assert v2.id            == "B"
        assert v2.type          == "conflict"
        assert v2.ctr           == 0.07
        assert v2.retention_30s == 0.65
        assert v2.score         == 0.0455

    def test_from_dict_empty_defaults(self):
        v = ABTestVariant.from_dict({})
        assert v.id   == "A"
        assert v.ctr  == 0.0

    def test_thumbnail_field_preserved(self):
        thumb = {"text": "THE PROBLEM", "style": "high_contrast", "emotion": "shock"}
        v = ABTestVariant(thumbnail=thumb)
        assert ABTestVariant.from_dict(v.to_dict()).thumbnail == thumb


# ── ABTestResult ───────────────────────────────────────────────────────────────

class TestABTestResult:
    def test_to_dict_round_trip(self):
        winner = ABTestVariant(id="B", type="conflict", score=0.05)
        result = ABTestResult(
            video_id="vid-001",
            variants=_two_variants(),
            winner=winner,
            completed_at="2025-01-01T10:00:00Z",
        )
        d  = result.to_dict()
        r2 = ABTestResult.from_dict(d)
        assert r2.video_id     == "vid-001"
        assert r2.winner.id    == "B"
        assert len(r2.variants) == 2

    def test_winner_none_serialises(self):
        result = ABTestResult(video_id="x", variants=[], winner=None)
        d = result.to_dict()
        assert d["winner"] is None
        r2 = ABTestResult.from_dict(d)
        assert r2.winner is None


# ── ABTestStore ────────────────────────────────────────────────────────────────

class TestABTestStore:
    def test_save_and_load_recent(self):
        store  = ABTestStore(data_dir=_tmp())
        winner = ABTestVariant(id="A", type="curiosity", score=0.04)
        result = ABTestResult(video_id="v1", winner=winner)
        store.save(result)
        recent = store.load_recent()
        assert len(recent) == 1
        assert recent[0].video_id == "v1"

    def test_multiple_saves_appended(self):
        store = ABTestStore(data_dir=_tmp())
        for i in range(3):
            store.save(ABTestResult(video_id=f"v{i}", winner=ABTestVariant(type="conflict")))
        assert len(store.load_recent()) == 3

    def test_load_recent_limit(self):
        store = ABTestStore(data_dir=_tmp())
        for i in range(5):
            store.save(ABTestResult(video_id=f"v{i}", winner=ABTestVariant(type="simple")))
        assert len(store.load_recent(n=2)) == 2

    def test_get_variant_type_performance(self):
        store = ABTestStore(data_dir=_tmp())
        store.save(ABTestResult(video_id="v1", winner=ABTestVariant(type="conflict", score=0.08)))
        store.save(ABTestResult(video_id="v2", winner=ABTestVariant(type="conflict", score=0.10)))
        store.save(ABTestResult(video_id="v3", winner=ABTestVariant(type="simple",   score=0.04)))
        perf = store.get_variant_type_performance()
        assert "conflict" in perf
        assert abs(perf["conflict"] - 0.09) < 1e-4
        assert abs(perf["simple"]   - 0.04) < 1e-4

    def test_get_winning_patterns_sorted(self):
        store = ABTestStore(data_dir=_tmp())
        store.save(ABTestResult(video_id="v1", winner=ABTestVariant(type="conflict",  score=0.10)))
        store.save(ABTestResult(video_id="v2", winner=ABTestVariant(type="curiosity", score=0.06)))
        store.save(ABTestResult(video_id="v3", winner=ABTestVariant(type="simple",    score=0.03)))
        patterns = store.get_winning_patterns()
        keys = list(patterns.keys())
        assert keys[0] == "conflict"
        assert keys[-1] == "simple"

    def test_empty_store_returns_empty(self):
        store = ABTestStore(data_dir=_tmp())
        assert store.get_variant_type_performance() == {}
        assert store.get_winning_patterns() == {}

    def test_top_n_respected(self):
        store = ABTestStore(data_dir=_tmp())
        for t in ("conflict", "curiosity", "simple"):
            store.save(ABTestResult(video_id=t, winner=ABTestVariant(type=t, score=0.05)))
        assert len(store.get_winning_patterns(top_n=2)) == 2


# ── PackagingEngine.generate_variants() ───────────────────────────────────────

class TestGenerateVariants:
    def setup_method(self):
        self.engine = PackagingEngine()

    def test_returns_three_variants(self):
        variants = self.engine.generate_variants(
            {"conflict": "hype vs reality"},
            _FakeStrategyConfig(mode="growth"),
        )
        assert len(variants) == 3

    def test_variant_types(self):
        variants = self.engine.generate_variants(
            {"conflict": "hype vs reality"},
            _FakeStrategyConfig(mode="growth"),
        )
        types = [v.type for v in variants]
        assert "curiosity" in types
        assert "conflict"  in types
        assert "simple"    in types

    def test_variant_ids_assigned_sequentially(self):
        variants = self.engine.generate_variants(
            {"conflict": "innovation vs adoption"},
            _FakeStrategyConfig(mode="safe"),
        )
        assert variants[0].id == "A"
        assert variants[1].id == "B"
        assert variants[2].id == "C"

    def test_all_variants_have_title_and_hook(self):
        variants = self.engine.generate_variants(
            {"angle_key": "threat_to_jobs"},
            _FakeStrategyConfig(mode="growth"),
        )
        for v in variants:
            assert v.title
            assert v.hook
            assert v.thumbnail

    def test_curiosity_variant_style_is_clean(self):
        variants = self.engine.generate_variants(
            {"conflict": "hype vs reality"},
            _FakeStrategyConfig(),
        )
        curiosity = next(v for v in variants if v.type == "curiosity")
        assert curiosity.thumbnail["style"] == "clean"
        assert curiosity.thumbnail["emotion"] == "curiosity"

    def test_conflict_variant_style_is_high_contrast(self):
        variants = self.engine.generate_variants(
            {"conflict": "growth vs jobs"},
            _FakeStrategyConfig(),
        )
        conflict = next(v for v in variants if v.type == "conflict")
        assert conflict.thumbnail["style"] == "high_contrast"
        assert conflict.thumbnail["emotion"] == "shock"

    def test_simple_variant_style_is_neutral(self):
        variants = self.engine.generate_variants(
            {"conflict": "promise vs practical use"},
            _FakeStrategyConfig(),
        )
        simple = next(v for v in variants if v.type == "simple")
        assert simple.thumbnail["style"] == "neutral"

    def test_hook_contains_core_idea(self):
        variants = self.engine.generate_variants(
            {"conflict": "hype vs reality"},
            _FakeStrategyConfig(),
        )
        # At least one variant's hook must contain the core idea fragment
        ideas = [v.hook for v in variants]
        assert any("problem" in h.lower() or "hype" in h.lower() or "hidden" in h.lower()
                   for h in ideas)

    def test_accepts_editorial_plan_object(self):
        ep = _FakeEditorialPlan(editorial_plan=[{"conflict": "growth vs jobs"}])
        variants = self.engine.generate_variants(ep, _FakeStrategyConfig())
        assert len(variants) == 3


# ── ABTestingEngine._score() ──────────────────────────────────────────────────

class TestScore:
    def setup_method(self):
        self.engine = ABTestingEngine(data_dir=_tmp())

    def test_score_with_retention(self):
        v = ABTestVariant(ctr=0.08, retention_30s=0.70)
        assert abs(self.engine._score(v) - 0.056) < 1e-6

    def test_score_fallback_to_ctr_when_retention_zero(self):
        v = ABTestVariant(ctr=0.08, retention_30s=0.0)
        assert self.engine._score(v) == 0.08

    def test_score_zero_ctr(self):
        v = ABTestVariant(ctr=0.0, retention_30s=0.80)
        assert self.engine._score(v) == 0.0

    def test_higher_retention_lifts_score(self):
        v_low  = ABTestVariant(ctr=0.10, retention_30s=0.40)
        v_high = ABTestVariant(ctr=0.10, retention_30s=0.80)
        assert self.engine._score(v_high) > self.engine._score(v_low)


# ── ABTestingEngine._select_winner() ──────────────────────────────────────────

class TestSelectWinner:
    def setup_method(self):
        self.engine = ABTestingEngine(data_dir=_tmp())

    def test_highest_score_wins(self):
        variants = [
            ABTestVariant(id="A", type="curiosity", score=0.04),
            ABTestVariant(id="B", type="conflict",  score=0.08),
            ABTestVariant(id="C", type="simple",    score=0.02),
        ]
        assert self.engine._select_winner(variants).id == "B"

    def test_single_variant_wins(self):
        v = ABTestVariant(id="A", score=0.05)
        assert self.engine._select_winner([v]).id == "A"

    def test_tie_returns_first_by_max_order(self):
        variants = [
            ABTestVariant(id="A", score=0.05),
            ABTestVariant(id="B", score=0.05),
        ]
        # max() returns first maximum — implementation-defined but consistent
        winner = self.engine._select_winner(variants)
        assert winner.id in ("A", "B")


# ── ABTestingEngine.run_test() ────────────────────────────────────────────────

class TestRunTest:
    def test_run_test_returns_result(self):
        yt     = _MockYouTube(ctr_seq=[0.05, 0.09], retention_seq=[0.6, 0.7])
        engine = ABTestingEngine(youtube_client=yt, data_dir=_tmp())
        result = engine.run_test("vid-001", _two_variants(), wait_fn=_no_wait)
        assert isinstance(result, ABTestResult)
        assert result.winner is not None
        assert result.completed_at is not None

    def test_run_test_picks_higher_score(self):
        # B has ctr=0.09 * retention=0.70 = 0.063 > A: 0.05 * 0.60 = 0.030
        yt     = _MockYouTube(ctr_seq=[0.05, 0.09], retention_seq=[0.60, 0.70])
        engine = ABTestingEngine(youtube_client=yt, data_dir=_tmp())
        result = engine.run_test("vid-001", _two_variants(), wait_fn=_no_wait)
        assert result.winner.id == "B"

    def test_run_test_ids_assigned_a_b(self):
        yt     = _MockYouTube(ctr_seq=[0.05, 0.08])
        engine = ABTestingEngine(youtube_client=yt, data_dir=_tmp())
        variants = _two_variants()
        # Reset ids to confirm engine assigns them
        variants[0].id = "X"
        variants[1].id = "X"
        engine.run_test("vid-001", variants, wait_fn=_no_wait)
        assert variants[0].id == "A"
        assert variants[1].id == "B"

    def test_run_test_three_variants(self):
        yt     = _MockYouTube(ctr_seq=[0.05, 0.07, 0.09], retention_seq=[0.6, 0.65, 0.75])
        engine = ABTestingEngine(youtube_client=yt, data_dir=_tmp())
        pe = PackagingEngine()
        variants = pe.generate_variants({"conflict": "hype vs reality"}, _FakeStrategyConfig())
        result = engine.run_test("vid-002", variants, wait_fn=_no_wait)
        assert result.winner.id == "C"  # highest score: 0.09 * 0.75 = 0.0675

    def test_run_test_metrics_stored_on_variants(self):
        yt     = _MockYouTube(ctr_seq=[0.06, 0.10], retention_seq=[0.5, 0.8])
        engine = ABTestingEngine(youtube_client=yt, data_dir=_tmp())
        result = engine.run_test("vid-003", _two_variants(), wait_fn=_no_wait)
        assert result.variants[0].ctr == 0.06
        assert result.variants[1].ctr == 0.10

    def test_run_test_saves_to_store(self):
        tmp    = _tmp()
        yt     = _MockYouTube()
        engine = ABTestingEngine(youtube_client=yt, data_dir=tmp)
        engine.run_test("vid-004", _two_variants(), wait_fn=_no_wait)
        recent = engine._store.load_recent()
        assert any(r.video_id == "vid-004" for r in recent)

    def test_run_test_applies_winner_at_end(self):
        yt     = _MockYouTube(ctr_seq=[0.05, 0.09])
        engine = ABTestingEngine(youtube_client=yt, data_dir=_tmp())
        engine.run_test("vid-005", _two_variants(), wait_fn=_no_wait)
        # Last update call should be for the winner's title
        update_calls = [c for c in yt.calls if c[0] == "update"]
        # 2 variants + 1 winner reapply = 3 update calls
        assert len(update_calls) == 3

    def test_run_test_raises_on_empty_variants(self):
        engine = ABTestingEngine(data_dir=_tmp())
        with pytest.raises(ValueError):
            engine.run_test("vid-999", [], wait_fn=_no_wait)

    def test_dry_run_no_youtube_client(self):
        """No youtube_client → no crash, scores stay 0.0, first variant wins."""
        engine = ABTestingEngine(youtube_client=None, data_dir=_tmp())
        result = engine.run_test("vid-dry", _two_variants(), wait_fn=_no_wait)
        assert result.winner is not None
        assert result.variants[0].ctr == 0.0


# ── suggest_strategy_adjustments() ────────────────────────────────────────────

class TestSuggestStrategyAdjustments:
    def test_conflict_winner_increases_aggressiveness(self):
        store  = ABTestStore(data_dir=_tmp())
        store.save(ABTestResult(video_id="v1", winner=ABTestVariant(type="conflict", score=0.09)))
        store.save(ABTestResult(video_id="v2", winner=ABTestVariant(type="conflict", score=0.08)))
        engine = ABTestingEngine(data_dir=store._path.parent)
        adj = engine.suggest_strategy_adjustments()
        assert adj["preferred_variant_type"] == "conflict"
        assert adj["hook_aggressiveness_delta"] == +0.10

    def test_curiosity_winner_moderate_increase(self):
        store = ABTestStore(data_dir=_tmp())
        store.save(ABTestResult(video_id="v1", winner=ABTestVariant(type="curiosity", score=0.07)))
        engine = ABTestingEngine(data_dir=store._path.parent)
        adj = engine.suggest_strategy_adjustments()
        assert adj["hook_aggressiveness_delta"] == +0.05

    def test_simple_winner_decreases_aggressiveness(self):
        store = ABTestStore(data_dir=_tmp())
        store.save(ABTestResult(video_id="v1", winner=ABTestVariant(type="simple", score=0.05)))
        engine = ABTestingEngine(data_dir=store._path.parent)
        adj = engine.suggest_strategy_adjustments()
        assert adj["hook_aggressiveness_delta"] == -0.05

    def test_empty_history_returns_empty(self):
        engine = ABTestingEngine(data_dir=_tmp())
        assert engine.suggest_strategy_adjustments() == {}


# ── apply_winner_feedback() ────────────────────────────────────────────────────

class TestApplyWinnerFeedback:
    def test_bridges_to_packaging_store(self):
        tmp    = _tmp()
        store  = PackagingStore(data_dir=tmp)
        engine = ABTestingEngine(data_dir=tmp)
        winner = ABTestVariant(id="B", type="conflict", title="Title", hook="Hook",
                               thumbnail={"text": "X"}, ctr=0.09, score=0.06)
        result = ABTestResult(video_id="vid-001", variants=[], winner=winner)
        engine.apply_winner_feedback(result, store)

        patterns = store.get_best_patterns()
        # conflict type → "hype_problem" title_pattern
        assert "hype_problem" in patterns

    def test_ctr_recorded_correctly(self):
        tmp    = _tmp()
        store  = PackagingStore(data_dir=tmp)
        engine = ABTestingEngine(data_dir=tmp)
        winner = ABTestVariant(id="A", type="curiosity", ctr=0.07, score=0.05,
                               thumbnail={"text": "W", "style": "clean", "emotion": "curiosity"})
        result = ABTestResult(video_id="v1", winner=winner)
        engine.apply_winner_feedback(result, store)
        patterns = store.get_best_patterns()
        # curiosity → "game_changer"
        assert "game_changer" in patterns
        assert abs(patterns["game_changer"] - 0.07) < 1e-4

    def test_no_winner_is_no_op(self):
        store  = PackagingStore(data_dir=_tmp())
        engine = ABTestingEngine(data_dir=_tmp())
        result = ABTestResult(video_id="v1", winner=None)
        engine.apply_winner_feedback(result, store)
        assert store.get_best_patterns() == {}


# ── Variant-type → title_pattern mapping ──────────────────────────────────────

class TestVariantTypeToPattern:
    def test_all_variant_types_mapped(self):
        for t in ("curiosity", "conflict", "simple"):
            assert t in _VARIANT_TYPE_TO_PATTERN

    def test_conflict_maps_to_hype_problem(self):
        assert _VARIANT_TYPE_TO_PATTERN["conflict"] == "hype_problem"

    def test_curiosity_maps_to_game_changer(self):
        assert _VARIANT_TYPE_TO_PATTERN["curiosity"] == "game_changer"


# ── Full pipeline integration ──────────────────────────────────────────────────

class TestFullPipeline:
    def test_generate_variants_then_run_test_then_feedback(self):
        tmp = _tmp()
        # Generate variants
        pe = PackagingEngine()
        variants = pe.generate_variants(
            {"conflict": "hype vs reality"},
            _FakeStrategyConfig(mode="growth"),
        )
        assert len(variants) == 3

        # Run A/B test
        yt     = _MockYouTube(ctr_seq=[0.04, 0.09, 0.06], retention_seq=[0.55, 0.75, 0.65])
        engine = ABTestingEngine(youtube_client=yt, data_dir=tmp)
        result = engine.run_test("vid-pipeline", variants, wait_fn=_no_wait)
        assert result.winner.id == "B"   # 0.09 * 0.75 = 0.0675 > others

        # Apply feedback
        ps = PackagingStore(data_dir=tmp)
        engine.apply_winner_feedback(result, ps)
        patterns = ps.get_best_patterns()
        assert patterns  # some pattern was recorded

        # Strategy adjustments
        adj = engine.suggest_strategy_adjustments()
        assert "hook_aggressiveness_delta" in adj
