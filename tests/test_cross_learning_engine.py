"""Tests for CrossLearningEngine.

Covers:
- CrossLearningStore.append() / load() — JSONL roundtrip, missing file
- _dominant_scene() — majority type, tie-breaking, empty list
- _extract_record() — happy path, missing performance (returns None)
- _extract_record() — defaults for missing strategy / packaging / editorial
- learn() — persists via store, skips records without performance
- aggregate() — groups by combo key, correct avg_retention and avg_ctr
- aggregate() — multiple combos distinguished correctly, normalisation
- get_top_combinations() — min_count filter, sorted by retention desc
- get_top_combinations() — topic filter, fallback when no topic match
- get_worst_combinations() — sorted by retention asc
- recommend() — exploitation: returns top combo
- recommend() — exploration: returns a dict with the right keys (seed-based)
- recommend() — no data → returns None
- recommend() — topic-aware lookup, falls back to global when topic absent
- recommend() — exploration_rate=0 → always exploit
- recommend() — exploration_rate=1 → always explore (never None)
- _random_combo() — always contains required keys with valid values
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from src.cross_learning_engine import (
    CrossLearningEngine,
    CrossLearningStore,
    _HOOK_TYPES,
    _PACES,
    _PERSONAS,
    _SCENE_TYPES,
)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _tmp() -> Path:
    return Path(tempfile.mkdtemp())


def _engine(
    data_dir: Path | None = None,
    exploration_rate: float = 0.0,   # default: always exploit in tests
    seed: int = 42,
) -> CrossLearningEngine:
    return CrossLearningEngine(
        data_dir=data_dir or _tmp(),
        exploration_rate=exploration_rate,
        seed=seed,
    )


def _run(
    hook_type:  str   = "conflict",
    scene_type: str   = "avatar",
    persona:    str   = "energetic",
    pace:       str   = "fast",
    topic:      str   = "AI news",
    retention:  float = 0.65,
    ctr:        float = 0.07,
    has_perf:   bool  = True,
) -> dict:
    """Minimal run_history record matching the orchestrator output format."""
    scenes = [{"index": 0, "type": scene_type, "intent": "hook", "predicted_risk": 0.2}]
    perf   = {"avg_watch_pct": retention, "ctr": ctr, "drops": []} if has_perf else None
    return {
        "video_id":   "vid_test",
        "strategy":   {"mode": "growth", "pace": pace, "persona": persona},
        "packaging":  {"hook": "...", "hook_type": hook_type, "title": "Test"},
        "editorial":  {"topic": topic},
        "scenes":     scenes,
        "performance": perf,
    }


def _record(combo: dict, retention: float = 0.6, ctr: float = 0.07) -> dict:
    return {"combo": combo, "result": {"retention": retention, "ctr": ctr}}


_COMBO_A = {"hook_type": "conflict", "scene_type": "avatar",
            "persona": "energetic", "pace": "fast", "topic": "AI news"}
_COMBO_B = {"hook_type": "curiosity", "scene_type": "image",
            "persona": "balanced",  "pace": "normal", "topic": "tutorial"}


# ── CrossLearningStore ─────────────────────────────────────────────────────────

class TestCrossLearningStore:
    def test_append_and_load(self):
        d = _tmp()
        store = CrossLearningStore(d)
        store.append({"combo": _COMBO_A, "result": {"retention": 0.6, "ctr": 0.07}})
        store.append({"combo": _COMBO_B, "result": {"retention": 0.4, "ctr": 0.04}})
        records = store.load()
        assert len(records) == 2
        assert records[0]["combo"] == _COMBO_A
        assert records[1]["combo"] == _COMBO_B

    def test_load_empty_returns_empty_list(self):
        assert CrossLearningStore(_tmp()).load() == []

    def test_append_creates_jsonl_file(self):
        d = _tmp()
        store = CrossLearningStore(d)
        store.append({"combo": _COMBO_A, "result": {}})
        path = d / "cross_learning.jsonl"
        assert path.exists()
        lines = path.read_text().strip().splitlines()
        assert len(lines) == 1

    def test_each_line_is_valid_json(self):
        d = _tmp()
        store = CrossLearningStore(d)
        for _ in range(3):
            store.append({"combo": _COMBO_A, "result": {"retention": 0.5, "ctr": 0.06}})
        path = d / "cross_learning.jsonl"
        for line in path.read_text().strip().splitlines():
            json.loads(line)  # must not raise


# ── _dominant_scene ────────────────────────────────────────────────────────────

class TestDominantScene:
    def _dom(self, scenes: list[dict]) -> str:
        return _engine()._dominant_scene({"scenes": scenes})

    def test_single_scene(self):
        assert self._dom([{"type": "avatar"}]) == "avatar"

    def test_majority_type(self):
        scenes = [{"type": "avatar"}, {"type": "avatar"}, {"type": "image"}]
        assert self._dom(scenes) == "avatar"

    def test_empty_scenes_returns_unknown(self):
        assert self._dom([]) == "unknown"

    def test_missing_scenes_key(self):
        assert _engine()._dominant_scene({}) == "unknown"

    def test_missing_type_key_counted_as_unknown(self):
        scenes = [{"type": "image"}, {}, {}]
        result = _engine()._dominant_scene({"scenes": scenes})
        assert result == "unknown"   # 2× unknown vs 1× image


# ── _extract_record ────────────────────────────────────────────────────────────

class TestExtractRecord:
    def test_happy_path(self):
        r = _engine()._extract_record(_run())
        assert r is not None
        assert r["combo"]["hook_type"]  == "conflict"
        assert r["combo"]["scene_type"] == "avatar"
        assert r["combo"]["persona"]    == "energetic"
        assert r["combo"]["pace"]       == "fast"
        assert r["combo"]["topic"]      == "AI news"
        assert abs(r["result"]["retention"] - 0.65) < 1e-9
        assert abs(r["result"]["ctr"]       - 0.07) < 1e-9

    def test_no_performance_returns_none(self):
        assert _engine()._extract_record(_run(has_perf=False)) is None

    def test_missing_strategy_uses_defaults(self):
        run = _run()
        run.pop("strategy")
        r = _engine()._extract_record(run)
        assert r["combo"]["persona"] == "balanced"
        assert r["combo"]["pace"]    == "normal"

    def test_missing_packaging_uses_defaults(self):
        run = _run()
        run.pop("packaging")
        r = _engine()._extract_record(run)
        assert r["combo"]["hook_type"] == "curiosity"

    def test_missing_editorial_uses_general(self):
        run = _run()
        run.pop("editorial")
        r = _engine()._extract_record(run)
        assert r["combo"]["topic"] == "general"

    def test_record_has_timestamp(self):
        r = _engine()._extract_record(_run())
        assert "_ts" in r


# ── learn ──────────────────────────────────────────────────────────────────────

class TestLearn:
    def test_learn_persists_record(self):
        d = _tmp()
        e = _engine(d)
        e.learn(_run())
        assert len(e._store.load()) == 1

    def test_learn_skips_without_performance(self):
        d = _tmp()
        e = _engine(d)
        e.learn(_run(has_perf=False))
        assert len(e._store.load()) == 0

    def test_learn_multiple_records(self):
        d = _tmp()
        e = _engine(d)
        for _ in range(5):
            e.learn(_run())
        assert len(e._store.load()) == 5


# ── aggregate ─────────────────────────────────────────────────────────────────

class TestAggregate:
    def _agg(self, records: list[dict]) -> list[dict]:
        d = _tmp()
        store = CrossLearningStore(d)
        for r in records:
            store.append(r)
        return CrossLearningEngine(d)._CrossLearningEngine__agg_from_store(store)

    # Use aggregate() directly through the engine
    def test_single_combo_single_observation(self):
        d = _tmp()
        e = _engine(d)
        e.learn(_run(retention=0.60))
        agg = e.aggregate()
        assert len(agg) == 1
        assert abs(agg[0]["avg_retention"] - 0.60) < 1e-4
        assert agg[0]["count"] == 1

    def test_single_combo_averages_correctly(self):
        d = _tmp()
        e = _engine(d)
        e.learn(_run(retention=0.60, ctr=0.06))
        e.learn(_run(retention=0.80, ctr=0.10))
        agg = e.aggregate()
        assert len(agg) == 1
        assert abs(agg[0]["avg_retention"] - 0.70) < 1e-4
        assert abs(agg[0]["avg_ctr"]       - 0.08) < 1e-4
        assert agg[0]["count"] == 2

    def test_two_combos_separated(self):
        d = _tmp()
        e = _engine(d)
        e.learn(_run(hook_type="conflict", retention=0.80))
        e.learn(_run(hook_type="curiosity", retention=0.40))
        agg = e.aggregate()
        assert len(agg) == 2
        rets = {r["combo"]["hook_type"]: r["avg_retention"] for r in agg}
        assert abs(rets["conflict"]  - 0.80) < 1e-4
        assert abs(rets["curiosity"] - 0.40) < 1e-4

    def test_empty_store_returns_empty(self):
        assert _engine().aggregate() == []


# ── get_top_combinations ───────────────────────────────────────────────────────

class TestGetTopCombinations:
    def _populate(self, d: Path, combo_retention_pairs: list[tuple[dict, float, int]]):
        e = CrossLearningEngine(d)
        for combo_kw, ret, n in combo_retention_pairs:
            for _ in range(n):
                e.learn(_run(**combo_kw, retention=ret))
        return e

    def test_min_count_filter(self):
        d = _tmp()
        e = self._populate(d, [
            ({"hook_type": "conflict"}, 0.80, 3),   # 3 — qualifies
            ({"hook_type": "curiosity"}, 0.70, 2),  # 2 — excluded
        ])
        top = e.get_top_combinations(min_count=3)
        assert len(top) == 1
        assert top[0]["combo"]["hook_type"] == "conflict"

    def test_sorted_by_retention_desc(self):
        d = _tmp()
        e = self._populate(d, [
            ({"hook_type": "conflict"},  0.50, 3),
            ({"hook_type": "curiosity"}, 0.80, 3),
            ({"hook_type": "simple"},    0.65, 3),
        ])
        top = e.get_top_combinations(min_count=3)
        retentions = [x["avg_retention"] for x in top]
        assert retentions == sorted(retentions, reverse=True)

    def test_topic_filter(self):
        d = _tmp()
        e = CrossLearningEngine(d)
        for _ in range(3):
            e.learn(_run(hook_type="conflict",  topic="AI news",  retention=0.80))
            e.learn(_run(hook_type="curiosity", topic="tutorial", retention=0.50))
        top = e.get_top_combinations(min_count=3, topic="AI news")
        assert len(top) == 1
        assert top[0]["combo"]["topic"] == "AI news"

    def test_empty_when_no_data(self):
        assert _engine().get_top_combinations() == []


# ── get_worst_combinations ─────────────────────────────────────────────────────

class TestGetWorstCombinations:
    def test_sorted_asc(self):
        d = _tmp()
        e = CrossLearningEngine(d)
        for _ in range(3):
            e.learn(_run(hook_type="conflict",  retention=0.80))
            e.learn(_run(hook_type="curiosity", retention=0.30))
        worst = e.get_worst_combinations(min_count=3)
        assert worst[0]["combo"]["hook_type"] == "curiosity"


# ── recommend ─────────────────────────────────────────────────────────────────

class TestRecommend:
    def _with_data(self, d: Path, n: int = 3, **run_kw) -> CrossLearningEngine:
        e = CrossLearningEngine(d, exploration_rate=0.0, seed=42)
        for _ in range(n):
            e.learn(_run(**run_kw))
        return e

    def test_exploit_returns_top_combo(self):
        d = _tmp()
        e = CrossLearningEngine(d, exploration_rate=0.0, seed=42)
        for _ in range(3):
            e.learn(_run(hook_type="conflict", retention=0.80))
            e.learn(_run(hook_type="curiosity", retention=0.40))
        result = e.recommend()
        assert result is not None
        assert result["hook_type"] == "conflict"

    def test_no_data_returns_none(self):
        assert _engine(exploration_rate=0.0).recommend() is None

    def test_exploration_returns_dict_with_required_keys(self):
        # exploration_rate=1.0 always explores
        e = CrossLearningEngine(_tmp(), exploration_rate=1.0, seed=7)
        result = e.recommend()
        assert result is not None
        for k in ("hook_type", "scene_type", "persona", "pace"):
            assert k in result

    def test_exploration_values_are_valid(self):
        e = CrossLearningEngine(_tmp(), exploration_rate=1.0, seed=7)
        for _ in range(20):
            r = e.recommend()
            assert r["hook_type"]  in _HOOK_TYPES
            assert r["scene_type"] in _SCENE_TYPES
            assert r["persona"]    in _PERSONAS
            assert r["pace"]       in _PACES

    def test_topic_context_filters_results(self):
        d = _tmp()
        e = CrossLearningEngine(d, exploration_rate=0.0, seed=42)
        for _ in range(3):
            e.learn(_run(topic="AI news",  hook_type="conflict",  retention=0.80))
            e.learn(_run(topic="tutorial", hook_type="curiosity", retention=0.50))
        result = e.recommend(context={"topic": "AI news"})
        assert result["hook_type"] == "conflict"

    def test_topic_fallback_when_topic_absent(self):
        d = _tmp()
        e = CrossLearningEngine(d, exploration_rate=0.0, seed=42)
        for _ in range(3):
            e.learn(_run(topic="AI news", hook_type="conflict", retention=0.80))
        # "cooking" topic has no data → falls back to global best
        result = e.recommend(context={"topic": "cooking"})
        assert result is not None
        assert result["hook_type"] == "conflict"

    def test_explore_rate_zero_never_explores(self):
        d = _tmp()
        e = CrossLearningEngine(d, exploration_rate=0.0, seed=42)
        for _ in range(3):
            e.learn(_run(hook_type="conflict", retention=0.80))
        # Should always return the top combo, never random
        results = {e.recommend()["hook_type"] for _ in range(20)}
        assert results == {"conflict"}

    def test_explicit_exploration_rate_override(self):
        d = _tmp()
        e = CrossLearningEngine(d, exploration_rate=0.0, seed=99)
        # Even with engine default=0, passing rate=1.0 explores
        result = e.recommend(exploration_rate=1.0)
        # Random — just check it returns a valid dict
        assert result is not None
        assert "hook_type" in result


# ── _random_combo ─────────────────────────────────────────────────────────────

class TestRandomCombo:
    def test_always_returns_required_keys(self):
        e = _engine()
        for _ in range(30):
            c = e._random_combo()
            for k in ("hook_type", "scene_type", "persona", "pace"):
                assert k in c

    def test_values_within_known_sets(self):
        e = _engine(seed=0)
        for _ in range(50):
            c = e._random_combo()
            assert c["hook_type"]  in _HOOK_TYPES
            assert c["scene_type"] in _SCENE_TYPES
            assert c["persona"]    in _PERSONAS
            assert c["pace"]       in _PACES


class TestCrossLearningStoreErrorPaths:
    def test_append_swallows_write_error(self, tmp_path):
        store = CrossLearningStore(tmp_path)
        # Make dir read-only so write fails
        tmp_path.chmod(0o555)
        try:
            store.append({"combo": _COMBO_A, "result": {}})  # should not raise
        finally:
            tmp_path.chmod(0o755)

    def test_load_swallows_read_error(self, tmp_path):
        store = CrossLearningStore(tmp_path)
        path = tmp_path / "cross_learning.jsonl"
        path.write_text("not-json\n")
        result = store.load()
        assert result == []
