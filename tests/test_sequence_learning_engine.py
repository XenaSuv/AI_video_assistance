"""Tests for SequenceLearningEngine.

Covers:
- extract_patterns(): window sizes, edge cases (too short, exact length)
- _intent_from(): dict scenes, object scenes, missing fields
- _set_intent(): mutates dict and object scenes
- SequenceLearningStore: append / load / empty file
- store_sequence(): extracts sequence, skips without performance
- store_sequence(): empty scene list skipped
- aggregate_patterns(): groups by pattern, correct averages, min_count filter
- aggregate_patterns(): window parameter respected
- get_top_patterns(): sorted desc, threshold filter, min_count filter
- get_bad_patterns(): sorted asc, threshold
- bad_pattern_set(): O(1) frozenset lookup
- best_next(): data-backed recommendation (highest retention)
- best_next(): no matching prefix → random fallback from known set
- best_next(): prefix with one element
- generate_sequence(): starts with "hook" by default, correct length
- generate_sequence(): custom start list
- generate_sequence(): respects best_next when data exists
- auto_correct_sequence(): replaces last element of bad 3-gram with "twist"
- auto_correct_sequence(): no bad patterns → 0 corrections
- auto_correct_sequence(): works with Scene objects (duck typing)
- auto_correct_sequence(): sequence shorter than window → no correction
- auto_correct_sequence(): returns count of corrections
- get_risk_boost(): bad pattern → BAD_PATTERN_RISK_BOOST, good → 0.0
- get_sequence_bias(): returns top-N pattern tuples
"""
from __future__ import annotations

import json
import tempfile
from dataclasses import dataclass
from pathlib import Path

import pytest

from src.sequence_learning_engine import (
    BAD_PATTERN_RISK_BOOST,
    SequenceLearningEngine,
    SequenceLearningStore,
    _intent_from,
    _set_intent,
    extract_patterns,
)

# ── Helpers ────────────────────────────────────────────────────────────────────

def _tmp() -> Path:
    return Path(tempfile.mkdtemp())


def _engine(
    data_dir:  Path | None = None,
    min_count: int         = 1,   # low default for fast tests
    window:    int         = 3,
    seed:      int         = 42,
) -> SequenceLearningEngine:
    return SequenceLearningEngine(
        data_dir=data_dir or _tmp(),
        min_count=min_count,
        window=window,
        seed=seed,
    )


def _run(
    intents:   list[str] = None,
    retention: float      = 0.6,
    drops:     list[int]  = None,
    has_perf:  bool       = True,
) -> dict:
    scenes = [{"index": i, "type": "image", "intent": intent}
              for i, intent in enumerate(intents or ["hook", "explain", "twist"])]
    perf   = {"avg_watch_pct": retention, "ctr": 0.06, "drops": drops or []} if has_perf else None
    return {"scenes": scenes, "performance": perf}


@dataclass
class _SceneObj:
    scene_intent: str = "explain"


# ── extract_patterns ───────────────────────────────────────────────────────────

class TestExtractPatterns:
    def test_window_3_basic(self):
        result = extract_patterns(["hook", "explain", "twist", "example"], window=3)
        assert result == [("hook", "explain", "twist"), ("explain", "twist", "example")]

    def test_window_2(self):
        result = extract_patterns(["a", "b", "c"], window=2)
        assert result == [("a", "b"), ("b", "c")]

    def test_exact_window_length(self):
        result = extract_patterns(["a", "b", "c"], window=3)
        assert result == [("a", "b", "c")]

    def test_shorter_than_window_returns_empty(self):
        assert extract_patterns(["hook", "explain"], window=3) == []

    def test_empty_sequence_returns_empty(self):
        assert extract_patterns([], window=3) == []

    def test_window_1_returns_singletons(self):
        result = extract_patterns(["hook", "explain"], window=1)
        assert result == [("hook",), ("explain",)]

    def test_all_elements_covered(self):
        seq    = ["a", "b", "c", "d", "e"]
        pats   = extract_patterns(seq, window=3)
        assert len(pats) == 3
        # Verify every triplet is present
        assert ("a", "b", "c") in pats
        assert ("b", "c", "d") in pats
        assert ("c", "d", "e") in pats


# ── _intent_from / _set_intent ─────────────────────────────────────────────────

class TestIntentHelpers:
    def test_intent_from_dict_intent_key(self):
        assert _intent_from({"intent": "hook"}) == "hook"

    def test_intent_from_dict_scene_intent_key(self):
        assert _intent_from({"scene_intent": "explain"}) == "explain"

    def test_intent_from_dict_missing_key(self):
        assert _intent_from({}) == "unknown"

    def test_intent_from_scene_object(self):
        assert _intent_from(_SceneObj(scene_intent="twist")) == "twist"

    def test_set_intent_dict(self):
        d = {"intent": "explain"}
        _set_intent(d, "hook")
        assert d["intent"] == "hook"

    def test_set_intent_object(self):
        obj = _SceneObj(scene_intent="explain")
        _set_intent(obj, "twist")
        assert obj.scene_intent == "twist"


# ── SequenceLearningStore ─────────────────────────────────────────────────────

class TestSequenceLearningStore:
    def test_append_and_load(self):
        d = _tmp()
        store = SequenceLearningStore(d)
        store.append({"sequence": ["hook", "explain"], "retention": 0.6})
        records = store.load()
        assert len(records) == 1
        assert records[0]["sequence"] == ["hook", "explain"]

    def test_load_empty_returns_empty(self):
        assert SequenceLearningStore(_tmp()).load() == []

    def test_multiple_appends(self):
        d = _tmp()
        store = SequenceLearningStore(d)
        for i in range(5):
            store.append({"retention": 0.5 + i * 0.05})
        assert len(store.load()) == 5

    def test_each_line_valid_json(self):
        d = _tmp()
        store = SequenceLearningStore(d)
        store.append({"sequence": ["a", "b"], "retention": 0.5})
        store.append({"sequence": ["b", "c"], "retention": 0.7})
        for line in (d / "sequence_learning.jsonl").read_text().splitlines():
            json.loads(line)


# ── store_sequence ─────────────────────────────────────────────────────────────

class TestStoreSequence:
    def test_happy_path_stores_record(self):
        d = _tmp()
        e = _engine(d)
        e.store_sequence(_run())
        records = e._store.load()
        assert len(records) == 1
        assert records[0]["sequence"] == ["hook", "explain", "twist"]

    def test_retention_stored(self):
        d = _tmp()
        e = _engine(d)
        e.store_sequence(_run(retention=0.75))
        assert abs(e._store.load()[0]["retention"] - 0.75) < 1e-9

    def test_drops_stored(self):
        d = _tmp()
        e = _engine(d)
        e.store_sequence(_run(drops=[1, 2]))
        assert e._store.load()[0]["drops"] == [1, 2]

    def test_skips_without_performance(self):
        d = _tmp()
        e = _engine(d)
        e.store_sequence(_run(has_perf=False))
        assert len(e._store.load()) == 0

    def test_skips_empty_scene_list(self):
        d = _tmp()
        e = _engine(d)
        e.store_sequence({"scenes": [], "performance": {"avg_watch_pct": 0.5, "drops": []}})
        assert len(e._store.load()) == 0

    def test_record_has_timestamp(self):
        d = _tmp()
        e = _engine(d)
        e.store_sequence(_run())
        assert "_ts" in e._store.load()[0]


# ── aggregate_patterns ────────────────────────────────────────────────────────

class TestAggregatePatterns:
    def test_single_sequence_correct_patterns(self):
        d = _tmp()
        e = _engine(d, min_count=1)
        e.store_sequence(_run(["hook", "explain", "twist", "example"], retention=0.8))
        agg = e.aggregate_patterns()
        pats = {x["pattern"] for x in agg}
        assert ("hook", "explain", "twist")  in pats
        assert ("explain", "twist", "example") in pats

    def test_avg_retention_correct(self):
        d = _tmp()
        e = _engine(d, min_count=1)
        e.store_sequence(_run(["hook", "explain", "twist"], retention=0.60))
        e.store_sequence(_run(["hook", "explain", "twist"], retention=0.80))
        agg = {x["pattern"]: x for x in e.aggregate_patterns()}
        pat = ("hook", "explain", "twist")
        assert abs(agg[pat]["avg_retention"] - 0.70) < 1e-4

    def test_min_count_filter(self):
        d = _tmp()
        e = _engine(d, min_count=3)
        # Only 2 occurrences — should be filtered
        e.store_sequence(_run(["hook", "explain", "twist"], retention=0.8))
        e.store_sequence(_run(["hook", "explain", "twist"], retention=0.8))
        assert e.aggregate_patterns(min_count=3) == []

    def test_window_respected(self):
        d = _tmp()
        e = _engine(d, min_count=1, window=2)
        e.store_sequence(_run(["hook", "explain", "twist"], retention=0.7))
        agg = e.aggregate_patterns(window=2)
        pats = {x["pattern"] for x in agg}
        assert ("hook", "explain") in pats
        assert ("explain", "twist") in pats

    def test_empty_store_returns_empty(self):
        assert _engine().aggregate_patterns() == []

    def test_count_field_correct(self):
        d = _tmp()
        e = _engine(d, min_count=1)
        for _ in range(4):
            e.store_sequence(_run(["hook", "explain", "twist"], retention=0.6))
        agg = {x["pattern"]: x for x in e.aggregate_patterns()}
        assert agg[("hook", "explain", "twist")]["count"] == 4


# ── get_top_patterns ──────────────────────────────────────────────────────────

class TestGetTopPatterns:
    def _populate(self, data_dir: Path, records: list[tuple[list[str], float]]):
        e = _engine(data_dir, min_count=1)
        for seq, ret in records:
            e.store_sequence(_run(seq, retention=ret))
        return e

    def test_sorted_desc(self):
        d = _tmp()
        e = self._populate(d, [
            (["hook", "explain", "twist"], 0.5),
            (["hook", "avatar",  "twist"], 0.8),
            (["hook", "data",    "twist"], 0.65),
        ])
        top = e.get_top_patterns()
        retentions = [x["avg_retention"] for x in top]
        assert retentions == sorted(retentions, reverse=True)

    def test_threshold_filter(self):
        d = _tmp()
        e = self._populate(d, [
            (["hook", "explain", "twist"], 0.3),
            (["hook", "avatar",  "twist"], 0.8),
        ])
        top = e.get_top_patterns(threshold=0.6)
        for x in top:
            assert x["avg_retention"] >= 0.6

    def test_empty_returns_empty(self):
        assert _engine().get_top_patterns() == []


# ── get_bad_patterns ──────────────────────────────────────────────────────────

class TestGetBadPatterns:
    def test_bad_patterns_below_threshold(self):
        d = _tmp()
        e = _engine(d, min_count=1)
        e.store_sequence(_run(["hook", "explain", "explain"], retention=0.2))
        e.store_sequence(_run(["hook", "avatar",  "twist"],   retention=0.8))
        bad = e.get_bad_patterns(threshold=0.4)
        assert len(bad) == 1
        pats = {x["pattern"] for x in bad}
        assert ("hook", "explain", "explain") in pats

    def test_sorted_asc(self):
        d = _tmp()
        e = _engine(d, min_count=1)
        for seq, ret in [
            (["hook", "explain", "explain"], 0.2),
            (["hook", "data",    "explain"], 0.35),
        ]:
            e.store_sequence(_run(seq, retention=ret))
        bad = e.get_bad_patterns(threshold=0.4)
        retentions = [x["avg_retention"] for x in bad]
        assert retentions == sorted(retentions)


# ── bad_pattern_set ────────────────────────────────────────────────────────────

class TestBadPatternSet:
    def test_returns_frozenset_of_tuples(self):
        d = _tmp()
        e = _engine(d, min_count=1)
        e.store_sequence(_run(["hook", "explain", "explain"], retention=0.2))
        bps = e.bad_pattern_set(threshold=0.4)
        assert isinstance(bps, frozenset)
        assert ("hook", "explain", "explain") in bps

    def test_good_pattern_not_in_set(self):
        d = _tmp()
        e = _engine(d, min_count=1)
        e.store_sequence(_run(["hook", "avatar", "twist"], retention=0.9))
        bps = e.bad_pattern_set(threshold=0.4)
        assert ("hook", "avatar", "twist") not in bps


# ── best_next ─────────────────────────────────────────────────────────────────

class TestBestNext:
    def _with_data(self, d: Path) -> SequenceLearningEngine:
        e = _engine(d, min_count=1)
        # ("hook", "explain") → "twist" has high retention
        e.store_sequence(_run(["hook", "explain", "twist"],  retention=0.85))
        # ("hook", "explain") → "example" has low retention
        e.store_sequence(_run(["hook", "explain", "example"], retention=0.25))
        return e

    def test_returns_highest_retention_next(self):
        d = _tmp()
        e = self._with_data(d)
        assert e.best_next(["hook", "explain"]) == "twist"

    def test_single_element_prefix(self):
        d = _tmp()
        e = _engine(d, min_count=1)
        e.store_sequence(_run(["hook", "avatar", "twist"], retention=0.9))
        # prefix=["hook"] → look for 2-grams starting with ("hook",)
        result = e.best_next(["hook"])
        assert result == "avatar"

    def test_no_data_returns_fallback(self):
        from src.sequence_learning_engine import _FALLBACK_INTENTS
        result = _engine().best_next(["nonexistent", "prefix"])
        assert result in _FALLBACK_INTENTS

    def test_unmatched_prefix_returns_fallback(self):
        from src.sequence_learning_engine import _FALLBACK_INTENTS
        d = _tmp()
        e = _engine(d, min_count=1)
        e.store_sequence(_run(["hook", "explain", "twist"], retention=0.8))
        result = e.best_next(["twist", "avatar"])  # never seen this prefix
        assert result in _FALLBACK_INTENTS


# ── generate_sequence ─────────────────────────────────────────────────────────

class TestGenerateSequence:
    def test_default_starts_with_hook(self):
        result = _engine().generate_sequence(length=5)
        assert result[0] == "hook"

    def test_correct_length(self):
        for length in (4, 6, 8):
            assert len(_engine().generate_sequence(length=length)) == length

    def test_custom_start(self):
        result = _engine().generate_sequence(length=5, start=["avatar", "explain"])
        assert result[:2] == ["avatar", "explain"]
        assert len(result) == 5

    def test_uses_best_next_when_data_exists(self):
        d = _tmp()
        e = _engine(d, min_count=1)
        # Feed strong signal: after (hook, explain) → twist is best
        for _ in range(3):
            e.store_sequence(_run(["hook", "explain", "twist"], retention=0.9))
            e.store_sequence(_run(["hook", "explain", "data"],  retention=0.2))
        seq = e.generate_sequence(length=4, start=["hook", "explain"])
        # Third element should be "twist"
        assert seq[2] == "twist"


# ── auto_correct_sequence ──────────────────────────────────────────────────────

class TestAutoCorrectSequence:
    def _engine_with_bad(self, d: Path) -> SequenceLearningEngine:
        e = _engine(d, min_count=1)
        for _ in range(2):   # ensure min_count met for both
            e.store_sequence(_run(["hook", "explain", "explain"], retention=0.2))
            e.store_sequence(_run(["hook", "avatar",  "twist"],   retention=0.8))
        return e

    def test_corrects_bad_pattern(self):
        d = _tmp()
        e = self._engine_with_bad(d)
        scenes = [{"intent": "hook"}, {"intent": "explain"}, {"intent": "explain"}]
        corrections = e.auto_correct_sequence(scenes, threshold=0.4)
        assert corrections == 1
        assert scenes[2]["intent"] == "twist"

    def test_no_correction_for_good_pattern(self):
        d = _tmp()
        e = self._engine_with_bad(d)
        scenes = [{"intent": "hook"}, {"intent": "avatar"}, {"intent": "twist"}]
        corrections = e.auto_correct_sequence(scenes, threshold=0.4)
        assert corrections == 0
        assert scenes[2]["intent"] == "twist"   # unchanged

    def test_works_with_scene_objects(self):
        d = _tmp()
        e = self._engine_with_bad(d)
        scenes = [_SceneObj("hook"), _SceneObj("explain"), _SceneObj("explain")]
        corrections = e.auto_correct_sequence(scenes, threshold=0.4)
        assert corrections == 1
        assert scenes[2].scene_intent == "twist"

    def test_sequence_shorter_than_window_no_correction(self):
        d = _tmp()
        e = self._engine_with_bad(d)
        scenes = [{"intent": "hook"}, {"intent": "explain"}]
        corrections = e.auto_correct_sequence(scenes, window=3, threshold=0.4)
        assert corrections == 0

    def test_multiple_corrections(self):
        d = _tmp()
        e = _engine(d, min_count=1)
        # Both ("a","b","c") and ("x","y","z") are bad
        e.store_sequence(_run(["a", "b", "c"], retention=0.1))
        e.store_sequence(_run(["x", "y", "z"], retention=0.1))
        scenes = [
            {"intent": "a"}, {"intent": "b"}, {"intent": "c"},
            {"intent": "x"}, {"intent": "y"}, {"intent": "z"},
        ]
        corrections = e.auto_correct_sequence(scenes, threshold=0.4)
        assert corrections >= 1   # at least one bad pattern corrected

    def test_custom_replacement(self):
        d = _tmp()
        e = self._engine_with_bad(d)
        scenes = [{"intent": "hook"}, {"intent": "explain"}, {"intent": "explain"}]
        e.auto_correct_sequence(scenes, threshold=0.4, replacement="avatar")
        assert scenes[2]["intent"] == "avatar"


# ── get_risk_boost ─────────────────────────────────────────────────────────────

class TestGetRiskBoost:
    def test_bad_pattern_returns_boost(self):
        d = _tmp()
        e = _engine(d, min_count=1)
        e.store_sequence(_run(["hook", "explain", "explain"], retention=0.2))
        boost = e.get_risk_boost(("hook", "explain", "explain"))
        assert boost == BAD_PATTERN_RISK_BOOST

    def test_good_pattern_returns_zero(self):
        d = _tmp()
        e = _engine(d, min_count=1)
        e.store_sequence(_run(["hook", "avatar", "twist"], retention=0.9))
        boost = e.get_risk_boost(("hook", "avatar", "twist"))
        assert boost == 0.0

    def test_unknown_pattern_returns_zero(self):
        assert _engine().get_risk_boost(("a", "b", "c")) == 0.0


# ── get_sequence_bias ──────────────────────────────────────────────────────────

class TestGetSequenceBias:
    def test_returns_list_of_tuples(self):
        d = _tmp()
        e = _engine(d, min_count=1)
        e.store_sequence(_run(["hook", "avatar", "twist"], retention=0.9))
        bias = e.get_sequence_bias(n=3)
        assert isinstance(bias, list)
        for item in bias:
            assert isinstance(item, tuple)

    def test_respects_n(self):
        d = _tmp()
        e = _engine(d, min_count=1)
        for seq in [
            ["hook", "explain", "twist"],
            ["hook", "avatar",  "data"],
            ["hook", "data",    "example"],
        ]:
            e.store_sequence(_run(seq, retention=0.8))
        bias = e.get_sequence_bias(n=2)
        assert len(bias) <= 2

    def test_empty_when_no_data(self):
        assert _engine().get_sequence_bias() == []
